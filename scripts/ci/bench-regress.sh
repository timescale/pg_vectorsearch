#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.
#
# Compare standalone benchmarks on the same machine. Requires git, Meson,
# Ninja, a C compiler, and Python 3. Uses ccache when available.
#
# Environment:
#   BASE_SHA          Target branch or commit (default: origin/main).
#   HEAD_SHA          Commit to benchmark (default: HEAD).
#   BENCH_OUTPUT_DIR  New directory for results (default: a temp directory).
#   BODY              PR description, for the Disable-check: bench trailer.
#   GITHUB_STEP_SUMMARY  Append the comparison report here when set.

set -euo pipefail

suites=(distance idset quantize rabitq-kernel page-score fastscan-kernel
        topk rotation search cluster)

die() {
    printf 'Error: %s\n' "$*" >&2
    if [[ -n "${summary:-}" ]]; then
        printf '\nError: %s\n' "$*" >>"$summary"
    fi
    exit 2
}

finish() {
    local status=$?
    rm -rf -- "$scratch"
    if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
        cat "$summary" >>"$GITHUB_STEP_SUMMARY" || exit 2
    fi
    exit "$status"
}

build_cli() {
    local side=$1 sha=$2 dir="$scratch/$1"

    mkdir -p "$dir/src" || return 1
    git -C "$repo" archive "$sha" | tar -xf - -C "$dir/src" || return 1
    printf '%s\n' "$sha" >"$dir/src/.commit-sha" || return 1

    CCACHE_BASEDIR="$dir" meson setup "$dir/build" "$dir/src" \
        --buildtype=release -Dtools=enabled -Dpostgresql=disabled &&
        CCACHE_BASEDIR="$dir" meson compile -C "$dir/build" \
            vectorsearch:executable
}

inventory() {
    "$1" --help | awk '$1 == "bench" { print $2 }'
}

validate_result() {
    python3 - "$@" <<'PY'
import json
import sys

suite, path = sys.argv[1:]
try:
    with open(path, encoding="utf-8") as f:
        result = json.load(f)
    if not isinstance(result, dict) or result.get("bench") != suite:
        raise ValueError(f"expected a {suite!r} result")
    metrics = result.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("expected a nonempty metrics array")
    if suite == "cluster":
        # A failed implementation can leave no metric yet exit successfully.
        expected = {"lloyd(cblas)", "lloyd", "hamerly", "elkan", "pgvector"}
        missing = expected - {metric["name"] for metric in metrics}
        if missing:
            raise ValueError(f"missing cluster cases: {', '.join(sorted(missing))}")
except (OSError, ValueError, TypeError, KeyError) as e:
    sys.exit(f"Error: {path}: {e}")
PY
}

run_benchmark() {
    local side=$1 cli=$2 suite=$3
    local json="$output/$suite.$side.json" log="$output/$suite.$side.log"

    if "$cli" bench "$suite" --json="$json" >"$log" 2>&1 &&
        validate_result "$suite" "$json" >>"$log" 2>&1; then
        return 0
    fi
    note "Error: $suite ($side) failed; see $log"
    tail -n 20 "$log" >&2
    return 1
}

note() {
    printf '%s\n' "$*" | tee -a "$summary"
}

repo=$(git rev-parse --show-toplevel) || die "run from a git checkout"
target=$(git -C "$repo" rev-parse --verify --end-of-options \
    "${BASE_SHA:-origin/main}^{commit}") || die "cannot resolve BASE_SHA"
head=$(git -C "$repo" rev-parse --verify --end-of-options \
    "${HEAD_SHA:-HEAD}^{commit}") || die "cannot resolve HEAD_SHA"
base=$(git -C "$repo" merge-base "$target" "$head") ||
    die "cannot find the merge base"

if [[ -n "${BENCH_OUTPUT_DIR:-}" ]]; then
    mkdir -- "$BENCH_OUTPUT_DIR" || die "cannot create BENCH_OUTPUT_DIR"
    output=$(cd -- "$BENCH_OUTPUT_DIR" && pwd)
else
    output=$(mktemp -d "${TMPDIR:-/tmp}/bench-results.XXXXXXXX")
fi
printf 'Benchmark results: %s\n' "$output"
summary="$output/summary.md"
printf "## Benchmark comparison\n\nMerge base: \`%s\`  \nHead: \`%s\`\n" \
    "$base" "$head" >"$summary"
scratch=$(mktemp -d "${TMPDIR:-/tmp}/bench-build.XXXXXXXX") ||
    die "cannot create scratch directory"
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -z "${CC:-}" ]] && command -v ccache >/dev/null; then
    export CC='ccache cc'
fi

for side in base head; do
    printf 'Building %s in release mode\n' "$side"
    sha=$base
    [[ "$side" != head ]] || sha=$head
    log="$output/build.$side.log"
    if ! build_cli "$side" "$sha" >"$log" 2>&1; then
        tail -n 40 "$log" >&2
        die "$side build failed; see $log"
    fi
done

base_cli="$scratch/base/build/vectorsearch"
head_cli="$scratch/head/build/vectorsearch"
if ! base_suites=$(inventory "$base_cli") || [[ -z "$base_suites" ]]; then
    die "cannot list baseline benchmark suites"
fi
if ! head_suites=$(inventory "$head_cli") ||
    ! grep -Fxq compare <<<"$head_suites"; then
    die "cannot list head benchmark suites or find bench compare"
fi

errors=0
regressions=0
for suite in "${suites[@]}"; do
    note ""
    note "### $suite"
    note ""
    if ! grep -Fxq "$suite" <<<"$head_suites"; then
        note "Error: head does not list $suite"
        errors=$((errors + 1))
        continue
    fi

    head_result="$output/$suite.head.json"
    base_result="$output/$suite.base.json"
    new=false
    if grep -Fxq "$suite" <<<"$base_suites"; then
        if ! run_benchmark base "$base_cli" "$suite"; then
            errors=$((errors + 1))
            continue
        fi
    else
        new=true
        base_result=$head_result
    fi
    if ! run_benchmark head "$head_cli" "$suite"; then
        errors=$((errors + 1))
        continue
    fi

    report="$output/$suite.compare.txt"
    if [[ "$new" == true ]]; then
        report="$output/$suite.validate.log"
    fi
    status=0
    # For a new suite, self-comparison validates the result but is not
    # evidence of performance. Report it as unpaired, not as a pass.
    "$head_cli" bench compare "$base_result" "$head_result" \
        >"$report" 2>&1 || status=$?
    if [[ "$new" == true && "$status" -eq 0 ]]; then
        note "New suite: the merge base has no $suite benchmark."
        continue
    fi
    cat "$report"
    sed 's/^/    /' "$report" >>"$summary"
    case "$new:$status" in
        false:0) ;;
        false:1) regressions=$((regressions + 1)) ;;
        *) errors=$((errors + 1)) ;;
    esac
done

note ""
note "Benchmark verdict: $regressions regressed suite(s), $errors error(s)."
if (( errors > 0 )); then
    exit 2
fi
if (( regressions > 0 )); then
    if grep -Eqi '^disable-check:[[:blank:]]*bench[[:space:]]*$' \
        <<<"${BODY:-}"; then
        note "The PR description waives benchmark regressions."
    else
        exit 1
    fi
fi
