#!/usr/bin/env bash
#
# bench-insert.sh - Benchmark meerkat aminsert on a live PostgreSQL instance.
#
# Measures two things against an already-built index:
#   1. Bulk throughput  - INSERT ... SELECT of COUNT rows (rows/sec).
#   2. Per-row latency   - COUNT single-row inserts in a server-side loop,
#                          timed with clock_timestamp() so the number is the
#                          backend's aminsert cost, not client round-trips.
#                          Reports avg / p50 / p95 / p99 / max (microseconds).
#
# With --profile it additionally attaches `perf` to the serving backend while a
# sustained single-row insert loop runs, and renders a flame graph so excessive
# hotspots in the insert path are visible.
#
# Vectors come from a source table (default: a cohere-wikipedia table loaded
# separately); the first BASE rows are indexed up front and the next COUNT rows
# (distinct ids) are the insert workload, so no duplicate TIDs are produced.
#
# Requirements:
#   - A running PostgreSQL with the meerkat + vector extensions and a source
#     table SRC_TABLE(id int, v vector(DIM)) holding >= BASE + COUNT rows.
#   - For --profile: perf, a FlameGraph checkout (../FlameGraph or
#     /tmp/FlameGraph), and kernel.perf_event_paranoid <= 1
#     (sudo sysctl -w kernel.perf_event_paranoid=1).
#   - Benchmark a RELEASE build of the extension (LTO off for honest symbol
#     attribution); install it before running.
#
# Usage:
#   scripts/bench-insert.sh [--base N] [--count N] [--nlist N]
#                           [--src TABLE] [--dim D] [--rebuild] [--profile]
#
# Connection is taken from the standard libpq env vars (PGHOST/PGPORT/
# PGDATABASE) or their defaults below; override as needed.

set -euo pipefail

# ---- Connection / dataset defaults (override via env or flags) ----
PGHOST=${PGHOST:-/home/enordstr/ClaudeWorkspace/pg/run/unix-sock}
PGPORT=${PGPORT:-5435}
PGDATABASE=${PGDATABASE:-postgres}
PSQL=${PSQL:-/home/enordstr/ClaudeWorkspace/pg/usr/18.4-release/bin/psql}
export PGHOST PGPORT PGDATABASE

SRC_TABLE=${SRC_TABLE:-cohere_wikipedia_22_12_1m_angular}
DIM=${DIM:-768}
BASE=${BASE:-500000}     # rows indexed before timing
COUNT=${COUNT:-20000}    # rows inserted during the benchmark
NLIST=${NLIST:-1000}
REBUILD=0
PROFILE=0
PROFILE_SECS=${PROFILE_SECS:-20}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILE_DIR="$PROJECT_ROOT/profiles"

BENCH_TBL=ins_bench

while [[ $# -gt 0 ]]; do
	case "$1" in
		--base)    BASE=$2; shift 2 ;;
		--count)   COUNT=$2; shift 2 ;;
		--nlist)   NLIST=$2; shift 2 ;;
		--src)     SRC_TABLE=$2; shift 2 ;;
		--dim)     DIM=$2; shift 2 ;;
		--rebuild) REBUILD=1; shift ;;
		--profile) PROFILE=1; shift ;;
		*) echo "unknown arg: $1" >&2; exit 2 ;;
	esac
done

psql_q() { "$PSQL" -X -v ON_ERROR_STOP=1 -qtA "$@"; }

info() { printf '\033[0;32m[bench-insert]\033[0m %s\n' "$*"; }
die()  { printf '\033[0;31m[bench-insert]\033[0m %s\n' "$*" >&2; exit 1; }

# ---- Preflight -----------------------------------------------------------
"$PSQL" -X -c 'SELECT 1' >/dev/null 2>&1 \
	|| die "cannot connect (PGHOST=$PGHOST PGPORT=$PGPORT). Is the instance up?"

have_src=$(psql_q -c "SELECT to_regclass('$SRC_TABLE') IS NOT NULL;")
[[ "$have_src" == "t" ]] || die "source table '$SRC_TABLE' not found"

src_rows=$(psql_q -c "SELECT count(*) FROM $SRC_TABLE;")
(( src_rows >= BASE + COUNT )) \
	|| die "source has $src_rows rows; need >= $((BASE + COUNT))"

info "src=$SRC_TABLE dim=$DIM base=$BASE count=$COUNT nlist=$NLIST"

# ---- Setup: base table + index, held-out workload set --------------------
needs_build=$REBUILD
if [[ "$(psql_q -c "SELECT to_regclass('$BENCH_TBL') IS NOT NULL;")" != "t" ]]; then
	needs_build=1
fi

if (( needs_build )); then
	info "building base table ($BASE rows) + index (nlist=$NLIST)"
	psql_q <<SQL
DROP TABLE IF EXISTS $BENCH_TBL;
CREATE TABLE $BENCH_TBL (id bigint PRIMARY KEY, v vector($DIM));
INSERT INTO $BENCH_TBL SELECT id, v FROM $SRC_TABLE
	WHERE id <= $BASE ORDER BY id;
CREATE INDEX ${BENCH_TBL}_idx ON $BENCH_TBL USING mktann (v)
	WITH (nlist = $NLIST, centroid_compression = true);
ANALYZE $BENCH_TBL;
SQL
else
	info "reusing existing $BENCH_TBL (use --rebuild to recreate)"
	psql_q -c "DELETE FROM $BENCH_TBL WHERE id > $BASE;"
fi

# The workload rows: distinct ids just past the indexed range.
LO=$((BASE + 1)); HI=$((BASE + COUNT))

# ---- 1. Bulk throughput --------------------------------------------------
info "bulk INSERT ... SELECT of $COUNT rows"
bulk_ms=$(psql_q <<SQL
\\set QUIET on
DELETE FROM $BENCH_TBL WHERE id > $BASE;
DO \$\$
DECLARE t0 timestamptz; dt double precision;
BEGIN
	t0 := clock_timestamp();
	INSERT INTO $BENCH_TBL SELECT id, v FROM $SRC_TABLE
		WHERE id BETWEEN $LO AND $HI;
	dt := extract(epoch FROM clock_timestamp() - t0) * 1000.0;
	RAISE NOTICE 'BULK_MS=%', dt;
END \$\$;
SQL
)
bulk_ms=$(printf '%s\n' "$bulk_ms" | sed -n 's/.*BULK_MS=//p' | head -1)
if [[ -n "$bulk_ms" ]]; then
	rate=$(awk -v c="$COUNT" -v ms="$bulk_ms" 'BEGIN{printf "%.0f", c/(ms/1000.0)}')
	info "bulk: ${bulk_ms} ms for $COUNT rows  ->  ${rate} rows/sec"
fi

# ---- 2. Per-row latency (server-side, no client round-trips) -------------
info "per-row latency: $COUNT single-row inserts (server-side loop)"
psql_q <<SQL
\\set QUIET on
DELETE FROM $BENCH_TBL WHERE id > $BASE;
DROP TABLE IF EXISTS ins_lat;
CREATE TEMP TABLE ins_lat (us double precision);
DO \$\$
DECLARE r record; t0 timestamptz;
BEGIN
	FOR r IN SELECT id, v FROM $SRC_TABLE
			 WHERE id BETWEEN $LO AND $HI ORDER BY id LOOP
		t0 := clock_timestamp();
		INSERT INTO $BENCH_TBL VALUES (r.id, r.v);
		INSERT INTO ins_lat
			VALUES (extract(epoch FROM clock_timestamp() - t0) * 1e6);
	END LOOP;
END \$\$;
SELECT format(
	'latency us  avg=%s  p50=%s  p95=%s  p99=%s  max=%s',
	round(avg(us)::numeric, 1),
	round((percentile_cont(0.50) WITHIN GROUP (ORDER BY us))::numeric, 1),
	round((percentile_cont(0.95) WITHIN GROUP (ORDER BY us))::numeric, 1),
	round((percentile_cont(0.99) WITHIN GROUP (ORDER BY us))::numeric, 1),
	round(max(us)::numeric, 1))
FROM ins_lat;
SQL

# ---- 3. Optional: profile the backend during sustained inserts -----------
if (( PROFILE )); then
	command -v perf >/dev/null || die "perf not found"
	FG=""
	for d in "$PROJECT_ROOT/../FlameGraph" /tmp/FlameGraph; do
		[[ -f "$d/stackcollapse-perf.pl" ]] && FG="$d" && break
	done
	[[ -n "$FG" ]] || die "FlameGraph not found (clone to ../FlameGraph)"
	mkdir -p "$PROFILE_DIR"

	paranoid=$(cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo 99)
	(( paranoid <= 1 )) || \
		info "WARNING: perf_event_paranoid=$paranoid; run: sudo sysctl -w kernel.perf_event_paranoid=1"

	info "profiling a sustained insert loop for ${PROFILE_SECS}s"
	# Background a backend that announces its PID, waits for perf to attach,
	# then inserts in a tight loop (recycling the workload range) until killed.
	out="$PROFILE_DIR/insert_profile_session.txt"
	psql_q > "$out" 2>&1 <<SQL &
SELECT 'BACKEND_PID=' || pg_backend_pid();
SELECT pg_sleep(3);
DELETE FROM $BENCH_TBL WHERE id > $BASE;
DO \$\$
DECLARE r record; deadline timestamptz := clock_timestamp()
	+ ($PROFILE_SECS + 5) * interval '1 second'; nid int := $HI;
BEGIN
	WHILE clock_timestamp() < deadline LOOP
		FOR r IN SELECT v FROM $SRC_TABLE
				 WHERE id BETWEEN $LO AND $HI ORDER BY id LOOP
			nid := nid + 1;
			INSERT INTO $BENCH_TBL VALUES (nid, r.v);
		END LOOP;
	END LOOP;
END \$\$;
SQL
	sess_pid=$!

	pid=""
	for _ in $(seq 1 20); do
		pid=$(sed -n 's/^BACKEND_PID=//p' "$out" 2>/dev/null | head -1)
		[[ -n "$pid" ]] && break
		sleep 0.3
	done
	[[ -n "$pid" ]] || { kill "$sess_pid" 2>/dev/null || true; die "no backend PID"; }
	info "attaching perf to backend $pid"

	perf record -F 999 -g --call-graph dwarf -o "$PROFILE_DIR/insert.perf.data" \
		-p "$pid" -- sleep "$PROFILE_SECS" || true

	kill "$sess_pid" 2>/dev/null || true
	wait "$sess_pid" 2>/dev/null || true

	perf script -i "$PROFILE_DIR/insert.perf.data" \
		| "$FG/stackcollapse-perf.pl" > "$PROFILE_DIR/insert.folded"
	"$FG/flamegraph.pl" --title "meerkat aminsert" \
		"$PROFILE_DIR/insert.folded" > "$PROFILE_DIR/insert-flame.svg"
	info "flame graph: $PROFILE_DIR/insert-flame.svg"
fi

info "done"
