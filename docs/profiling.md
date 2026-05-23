# Profiling the meerkat PG extension

Notes and pitfalls collected while attacking the per-query hot path with
`perf`. The bottom line: most "the bench is broken" moments come from
profiling the wrong thing or attributing cycles to the wrong frame.
Internalise the checklist before drawing conclusions from a flame
graph.

## Quick recipe (rekall, single nprobe, steady state)

```bash
# 1. Build the index once
cd ~/repos/rekall
./target/release/rekall run \
    --datasets benchmarks/datasets/cohere-1m.toml \
    --indexes  benchmarks/indexes/mkt-n4000-cctrue-fs16-soar.toml \
    --phase index

# 2. Tiny single-nprobe index config so the search phase dwells long
#    enough at one operating point for a stable perf sample.
cat > /tmp/idx-n5only.toml <<'EOF'
[[index]]
type = "meerkat"
name = "mkt-n4000-cctrue-fs16-soar"
[index.params]
nlist = 4000
centroid_compression = true
fastscan = true
soar_lambda = 1.0
boundary_epsilon = 0.05
[index.gucs]
"mkt.fastscan_bits" = 16

[[search.sweep]]
index = "mkt-n4000-cctrue-fs16-soar"
[search.sweep.params]
"mkt.nprobe" = [5]
EOF

# 3. Start rekall in background; once the active client-backend
#    appears, sleep past warmup, then record for ~20 s.
rm -rf /tmp/results-prof && mkdir -p /tmp/results-prof
./target/release/rekall run \
    --datasets benchmarks/datasets/cohere-1m.toml \
    --indexes  /tmp/idx-n5only.toml \
    --phase search >/tmp/rekall.log 2>&1 &
REK=$!

for i in $(seq 1 60); do
    PID=$(psql -h /var/run/postgresql -U rekall -d rekall -t -c "
        SELECT pid FROM pg_stat_activity
        WHERE state='active' AND datname='rekall'
          AND backend_type='client backend'
          AND query LIKE 'SELECT id FROM cohere_1m%'
          AND pid <> pg_backend_pid();" | tr -d ' \n')
    [ -n "$PID" ] && break
    kill -0 $REK 2>/dev/null || break
    sleep 0.3
done
sleep 4   # let warmup queries pass

sudo perf record -F 1999 -g --call-graph dwarf -o /tmp/n5.perf -p $PID -- sleep 20
sudo chown $USER /tmp/n5.perf
wait $REK

perf report -i /tmp/n5.perf --no-children --stdio --percent-limit 0.5 \
    | grep -E '^[[:space:]]+[0-9]+\.[0-9]+%'
```

## Pitfalls

### Profile warm sessions, not first queries

Most "this function is taking 33%" findings turn out to be artifacts of
a single cold call dominating a short window. Meerkat lazily generates
the RaBitQ rotation matrix on a backend's first query — a one-shot
~800 ms event. If a perf window starts at the same moment and lasts
2.5 seconds, that one call is 32% of the window. We saw exactly this
during the optimisation pass (the "matrix-gen is 32% of every query!"
red herring).

Verify steady state:

- Wait past rekall's `warmup` (default 500 queries — ~250 ms at 2000 QPS).
- Cross-check the "% of cycles" against the call frequency: at QPS=Q
  and µs/call=C, expected % is `Q · C · window_ms / 1000`. If the
  profile says X% but `Q · C ≪ X%`, the call is being attributed too
  much weight (usually DWARF unwinding confusion).
- For one-shot caches, a process-local counter is the cheapest sanity
  check. Diff against the perf %.

### Use rekall, not psycopg2

The rekall Rust client (`rekall-lib/src/backend/pg_vector_type.rs`)
binds query vectors using pgvector's **binary** wire format via
`PgVector: ToSql`. PG calls `vector_recv` — no float parsing.

A psycopg2 script that passes `[0.1,0.2,...]` text gets very
different numbers: ~35% of CPU goes to `__GI_strtof_l_internal` parsing
the input. Conclusions drawn from a psycopg2 bench do not generalise
to the rekall (and pgvector-python) path.

If you have to use a Python harness, use psycopg3 + `pgvector.psycopg.register_vector()`
which also goes binary.

### DWARF unwinding misattributes through inlined functions

Some of meerkat's hot helpers are inlined, and `perf record
--call-graph dwarf` occasionally walks a stack that crosses an inlined
boundary in a way that attributes a parent frame's cycles to the
inlinee. The original "32% in `mkt_random_orthogonal_matrix` via
`gram_schmidt_qr (inlined)`" was a perf bug, not a real hotspot
(confirmed by counter: 3 calls in the entire run).

Mitigations:

- `perf record --call-graph fp` if `-fno-omit-frame-pointer` is in the
  build flags. Frame-pointer unwinding is cheaper and less error-prone.
- Add a one-line `static unsigned long counter; counter++;` to a
  suspected hot function and `fprintf(stderr, ...)` it. Compare actual
  call count against perf's claimed cost.
- Look at `perf report --children` and walk the call tree manually
  rather than trusting the "self" %.

### Pick the right backend

A naive `pg_stat_activity` query returns autovacuum, walwriters, and
your own meta-queries. The filter that actually picks the rekall
search backend is:

```sql
SELECT pid FROM pg_stat_activity
WHERE state='active'
  AND datname='rekall'
  AND backend_type='client backend'
  AND query LIKE 'SELECT id FROM cohere_1m%'   -- or whatever
  AND pid <> pg_backend_pid();
```

Without the `backend_type='client backend'` filter you'll happily
attach perf to an autovacuum worker and waste a 20 s recording on
`heap_vacuum_rel` and `pg_checksum_page`.

### Clear cached rekall results between runs

rekall skips sweep points that already have a result file in the
configured `results_dir`. After a change, **rename or delete**
`results/cohere-1m/` (or whatever the dataset config points at) before
re-running, or rekall will silently report cached numbers and your
"after" looks the same as your "before".

### Index build dominates if you profile the full `rekall run`

Build phase on cohere-1M is ~75 s, search phase per-nprobe is ~5–20 s.
Profiling the *whole* command gets you a flame graph of
`mkt_matrix_transpose_vector_mul` (per-vector encoding during build),
not of the query path. Always restrict to `--phase search` for
query-side profiles.

### Per-backend caches that survive

When sanity-checking "is this expensive thing actually called per
query?", these are the per-backend caches that may be hiding the cost:

| What | Where | Invalidation |
| --- | --- | --- |
| RaBitQ rotation matrix P | `cached_params` in `src/pg/mktann_cache.c` | dim+seed key, survives relcache invalidation |
| Global mean + P·mean | `rd_amcache` via `mktann_cache_get` | freed on relcache invalidation |
| TID dedup buffer (SOAR / boundary) | `cached_dedup_gens` in `src/pg/mktann_scan.c` | grow-on-demand, never freed; generation counter invalidates entries |
| Buffer-cache slot hints (skip BufTable) | `buf_hint[]` in `src/pg/mktann_storage.c` | none — `ReadRecentBuffer` tag-check is the source of truth |

If a profile shows time in any of these, treat it as a startup-cost
amortised over the session, not per-query.

### Tag the perf window

Add a one-line marker so you can correlate `rekall.log` to the perf
recording:

```bash
echo "PERF START $(date -u +%H:%M:%S.%3N)"
sudo perf record ... -- sleep 20
echo "PERF END   $(date -u +%H:%M:%S.%3N)"
```

Then grep rekall.log for which nprobe was active during that window.

## Flame graph

```bash
git clone https://github.com/brendangregg/FlameGraph /tmp/fg
perf script -i /tmp/n5.perf | /tmp/fg/stackcollapse-perf.pl \
    | /tmp/fg/flamegraph.pl > /tmp/n5.svg
```

Open in a browser. For dwarf-recorded perf data, the stacks have full
function names if the binary has debug info (`-g`) — release builds
strip it, but our meson `release` keeps it. Verify with `nm meerkat.so | head`.

## Recoverable side effects

Diagnostic counters added to investigate "is X called per query?":

```c
static unsigned long ncalls;
ncalls++;
fprintf(stderr, "[mkt-debug] X pid=%d call=%lu\n", getpid(), ncalls);
fflush(stderr);
```

This works because PG redirects stderr to its log file. Read with:

```bash
sudo grep 'mkt-debug' /var/log/postgresql/postgresql-18-main.log
```

Remember to **remove** the counter before benchmarking the new code —
the `fprintf` itself is non-trivial and pollutes timing.

## What we measured

For posterity. Steady-state perf profile (rekall, cohere-1M,
mkt-n4000-cctrue-fs16-soar, nprobe=5, post-warmup, 20 s window) on
Graviton 4:

| % | Symbol | What |
| --- | --- | --- |
| 19% | `sgemv_n_NEOVERSEV1` (libopenblas) | Per-query rotation P·q |
| 13% | `mkt_rabitq_inner_product_multi_neon` | Centroid scoring |
| 9% | `hash_search_with_hash_value` (postgres) | BufTable lookups |
| 8% | `mkt_fastscan_accumulate_hacc_sve2` | Fastscan kernel |
| 5% | `mkt_distance_cosine_neon` | Rerank |
| 4% | `scan_fastscan_page` | Fastscan glue |

About half the per-query cost is fixed setup (rotation + centroid
descent) that doesn't amortise as nprobe grows. The actual ANN
work (fastscan kernel + glue) scales with nprobe.
