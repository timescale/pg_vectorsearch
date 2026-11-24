#!/bin/bash
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.

set -uo pipefail

# shellcheck source=test/scripts/vs_test.sh
source "$(dirname "${BASH_SOURCE[0]}")/vs_test.sh"

script="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
script="$script/scripts/ci/bench-regress.sh"
repo="$VS_WORKSPACE/repo"
stubs="$VS_WORKSPACE/stubs"
scratch="$VS_WORKSPACE/scratch"
mkdir -p "$stubs" "$scratch"

# Use real Git history; stub just the build and timing commands.
make_history() (
    set -e
    git init -q -b target "$repo"
    cd "$repo"
    git config user.name "Benchmark Test"
    git config user.email bench@example.invalid
    git config core.hooksPath /dev/null
    printf 'common\n' > marker
    git add marker
    git commit -qm common
    git branch feature
    printf 'target\n' > marker
    git commit -qam target
    git switch -q feature
    printf 'feature\n' > marker
    git commit -qam feature
    git switch -q target
    printf 'dirty\n' > marker
)
make_history || exit 1

cat >"$stubs/meson" <<'STUB'
#!/bin/bash
set -euo pipefail
case "$1" in
    setup)
        [[ "$4" == --buildtype=release && "$5" == -Dtools=enabled &&
           "$6" == -Dpostgresql=disabled ]]
        mkdir -p "$2"
        cp "$BENCH_TEST_CLI" "$2/vectorsearch"
        printf 'build %s %s\n' "$(<"$3/marker")" "$(<"$3/.commit-sha")" \
            >>"$BENCH_TEST_TRACE"
        ;;
    compile)
        [[ "$2" == -C && "$4" == vectorsearch:executable ]]
        ;;
    *) exit 2 ;;
esac
STUB

cat >"$stubs/vectorsearch" <<'STUB'
#!/bin/bash
set -euo pipefail
side=$(basename "$(dirname "$(dirname "$0")")")
scenario=$BENCH_TEST_SCENARIO
if [[ "$1" == --help ]]; then
    if [[ "$side:$scenario" == base:bad-inventory ]]; then
        exit 7
    fi
    for suite in compare distance idset quantize rabitq-kernel page-score \
            fastscan-kernel topk rotation search cluster; do
        case "$side:$scenario:$suite" in
            base:new:rotation | head:removed:rotation) continue ;;
        esac
        printf '    bench %s        Description\n' "$suite"
    done
    exit 0
fi

[[ "$1" == bench ]]
suite=$2
if [[ "$suite" == compare ]]; then
    [[ "$side" == head ]]
    printf 'compare %s %s\n' "$(basename "$3")" "$(basename "$4")" \
        >>"$BENCH_TEST_TRACE"
    printf 'example report\n'
    if [[ "$(basename "$4")" == distance.head.json ]]; then
        case "$scenario" in
            drop | mixed) exit 1 ;;
            compare-error) exit 2 ;;
        esac
    fi
    exit 0
fi

printf '%s %s\n' "$side" "$suite" >>"$BENCH_TEST_TRACE"
output=${3#--json=}
case "$side:$scenario:$suite" in
    base:mixed:idset) exit 4 ;;
    head:missing-json:idset) exit 0 ;;
    head:wrong-suite:idset) suite=search ;;
esac
if [[ "$suite" == cluster ]]; then
    metrics='{"name":"lloyd"},{"name":"hamerly"},{"name":"elkan"},{"name":"pgvector"}'
    if [[ "$side:$scenario" != base:partial-cluster ]]; then
        metrics+=',{"name":"lloyd(cblas)"}'
    fi
else
    metrics='{"name":"qps","unit":"q/s","value":100}'
fi
printf '{"bench":"%s","metrics":[%s]}\n' "$suite" "$metrics" >"$output"
STUB
chmod +x "$stubs/meson" "$stubs/vectorsearch"

run() {
    local name=$1 scenario=$2 body=${3:-}
    env -C "$repo" PATH="$stubs:$PATH" TMPDIR="$scratch" \
        BASE_SHA=target HEAD_SHA=feature BODY="$body" \
        BENCH_OUTPUT_DIR="$VS_WORKSPACE/$name" \
        GITHUB_STEP_SUMMARY="$VS_WORKSPACE/$name.summary" \
        BENCH_TEST_CLI="$stubs/vectorsearch" \
        BENCH_TEST_TRACE="$VS_WORKSPACE/$name.trace" \
        BENCH_TEST_SCENARIO="$scenario" "$script"
}

base=$(git -C "$repo" merge-base target feature)
head=$(git -C "$repo" rev-parse feature)
expect_status "all suites can pass" 0 run ok ok
expect_eq "build the exact merge base and head; ignore dirty checkout" \
    "$(printf 'build common %s\nbuild feature %s' "$base" "$head")" \
    "$(head -n 2 "$VS_WORKSPACE/ok.trace")"
expect_eq "run pairs back to back and compare with the head CLI" \
    $'base distance\nhead distance\ncompare distance.base.json distance.head.json\nbase idset' \
    "$(sed -n '3,6p' "$VS_WORKSPACE/ok.trace")"
expect_status "keep the benchmark JSON" 0 \
    test -s "$VS_WORKSPACE/ok/distance.base.json"
expect_status "copy the report to the GitHub step summary" 0 \
    cmp "$VS_WORKSPACE/ok/summary.md" "$VS_WORKSPACE/ok.summary"

expect_status "run and report a suite that is new on head" 0 run new new
expect_status "show the missing baseline as unpaired, not a pass" 0 \
    grep -Fq "New suite: the merge base has no rotation" \
        "$VS_WORKSPACE/new/summary.md"
expect_status "removing a suite from head is an error" 2 run removed removed
expect_status "a failing baseline inventory is an error" 2 \
    run bad-inventory bad-inventory
expect_status "a stable regression fails" 1 run drop drop
expect_status "the exact trailer waives regressions" 0 \
    run waived drop $'Explanation.\n\nDisable-check: bench'
expect_status "a similar trailer does not waive regressions" 1 \
    run not-waived drop 'Disable-check: bench-other'
expect_status "a supported suite failure cannot be waived" 2 \
    run mixed mixed 'Disable-check: bench'
expect_status "continue through the remaining suites after an error" 0 \
    grep -Fxq 'head cluster' "$VS_WORKSPACE/mixed.trace"
expect_status "a comparator error cannot be waived" 2 \
    run compare-error compare-error 'Disable-check: bench'
expect_status "success without a JSON file is an error" 2 \
    run missing-json missing-json
expect_status "a file for the wrong suite is an error" 2 \
    run wrong-suite wrong-suite
expect_status "cluster must include every default case" 2 \
    run partial-cluster partial-cluster
expect_eq "scratch builds are removed on all outcomes" "" "$(ls -A "$scratch")"
expect_eq "leave the user's checkout untouched" dirty "$(<"$repo/marker")"

tap_finish
