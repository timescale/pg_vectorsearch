---
name: profiling
description: >
  How to profile this repo with perf and flame graphs — release builds
  only, scripts/profile.sh and scripts/profile-bench.sh, where output
  lands, and how to read it. Trigger when asked to profile, find a hot
  path, check a performance regression, or validate that a change is
  actually faster.
---

# Profiling

Profile to validate a performance decision or catch a regression. A read of
the code is not a measurement. Do not profile a debug build: it will not
show where time actually goes.

`docs/development.md` has the longer reference (flame-graph reading, extra
`perf` flags). This skill is the workflow. Machine-specific setup, including
`perf_event_paranoid`, lives in `CLAUDE.local.md` when that file exists.

## When to profile

- Before and after a change on a hot path (scan, distance, quantize,
  quantization, or search). Profile the same workload on both sides and
  compare them. A profile of only the new code is not a comparison.
- When a benchmark moved and it is not obvious which function moved it.
- Not as a substitute for a recall/QPS benchmark. Profiling says where time
  goes; it does not say whether the index got better.

Keep notes on what was measured and what the flame graph showed. Do not put
machine-specific numbers into commit messages or tracked docs.

## What to build

`scripts/profile.sh` records whatever command you pass, but it refuses to
run unless `builddir/vectorsearch` exists. That check does not rebuild
anything, and it does not create a `builddir-profile`. Use a release binary:

```bash
meson setup builddir-release --buildtype=release
meson compile -C builddir-release
```

Point the profiled command at that binary (or a release `builddir`). A
debug `builddir` satisfies the script's existence check and still produces
a useless profile.

The script needs `perf`. If `kernel.perf_event_paranoid` is above 1 it
exits and prints the exact `sysctl` commands. Do not re-run the profile
under `sudo`. Ask the user to lower the setting; on this machine the
standing setup is in `CLAUDE.local.md`.

FlameGraph is taken from `../FlameGraph` if present, otherwise cloned under
`$TMPDIR` (this machine uses `/tmp/claude`).

## Commands

From the repo root:

```bash
# Profile any command
./scripts/profile.sh ./bin/vectorsearch bench distance --dim 768 --count 10000

# Distance microbench shortcut: dim, count, impl
./scripts/profile-bench.sh 768 10000 avx512

# Cache misses, branch misses, higher sample rate
./scripts/profile.sh --events cache-misses --output cache-profile \
  ./bin/vectorsearch bench distance --dim 768 --count 100000
./scripts/profile.sh --events branch-misses --output branch-profile \
  ./bin/vectorsearch bench distance
./scripts/profile.sh --freq 4999 ./bin/vectorsearch bench distance

# Keep perf.data for perf report / perf annotate
./scripts/profile.sh --keep-perf-data ./bin/vectorsearch bench distance
```

`profile.sh` options: `--events` (default `cycles`), `--freq` (default
999), `--output` (default `flamegraph`), `--keep-perf-data`.

Output lands in `profiles/` (gitignored):

- `profiles/<name>.svg` — flame graph, open in a browser
- `profiles/<name>-icicle.svg` — inverted call chains
- `profiles/<name>.folded` — collapsed stacks
- `profiles/perf.data` — only with `--keep-perf-data`

## Reading the result

Width is time in that function including callees. Color is meaningless.
Wide bars are the optimization targets. A tall stack is a deep call chain,
not necessarily a slow function. Search the SVG for the function you
expected to be hot; if it is a sliver, the time is somewhere else.

With `--keep-perf-data`:

```bash
perf report --stdio
perf annotate <symbol>
```

Compare the before and after profiles of the same workload: which
functions got wider or narrower, and whether time moved into a callee
the change introduced. Keep the before output (`--output before`) so it
is not overwritten. Do not claim a speedup from a flame graph alone
when the question was QPS or recall.
