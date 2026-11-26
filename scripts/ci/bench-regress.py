#!/usr/bin/env python3
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.

"""Compare standalone release benchmarks from two Git revisions.

BASE_SHA names the target branch (default origin/main); HEAD_SHA defaults
to HEAD. BENCH_OUTPUT_DIR names a new results directory, or a temporary
results directory is created and retained. BODY is the PR description.
GITHUB_STEP_SUMMARY, when set, receives a copy of the report.
"""

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from textwrap import indent

SUITES = (
    "distance", "idset", "quantize", "rabitq-kernel", "page-score",
    "fastscan-kernel", "topk", "rotation", "search", "cluster",
)
CLUSTER_CASES = {"lloyd(cblas)", "lloyd", "hamerly", "elkan", "pgvector"}
OVERRIDE = re.compile(
    r"^disable-check:[ \t]*bench[ \t\r]*$", re.IGNORECASE | re.MULTILINE
)
REGRESSION = 1
ERROR = 2


def git_output(*args):
    return subprocess.check_output(["git", *args], text=True).rstrip("\n")


def build_cli(repo, sha, directory, log):
    source = directory / "src"
    build = directory / "build"
    archive = directory / "source.tar"
    source.mkdir(parents=True)
    env = dict(os.environ, CCACHE_BASEDIR=str(directory))
    if not env.get("CC") and shutil.which("ccache"):
        env["CC"] = "ccache cc"

    subprocess.run(
        ["git", "-C", repo, "archive", "-o", archive, sha],
        check=True, stderr=log,
    )
    subprocess.run(
        ["tar", "-xf", archive, "-C", source],
        check=True, stdout=log, stderr=subprocess.STDOUT,
    )
    archive.unlink()
    (source / ".commit-sha").write_text(sha + "\n", encoding="utf-8")
    subprocess.run(
        ["meson", "setup", build, source, "--buildtype=release",
         "-Dtools=enabled", "-Dpostgresql=disabled"],
        check=True, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    subprocess.run(
        ["meson", "compile", "-C", build, "vectorsearch:executable"],
        check=True, env=env, stdout=log, stderr=subprocess.STDOUT,
    )
    return build / "vectorsearch"


def inventory(cli):
    help_text = subprocess.check_output([cli, "--help"], text=True)
    return set(re.findall(r"^[ \t]*bench[ \t]+(\S+)", help_text, re.MULTILINE))


def validate_result(suite, path):
    with path.open(encoding="utf-8") as result_file:
        result = json.load(result_file)
    if not isinstance(result, dict) or result.get("bench") != suite:
        raise ValueError(f"expected a {suite!r} result")
    metrics = result.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("expected a nonempty metrics array")
    if suite == "cluster":
        # A failed implementation can leave no metric yet exit successfully.
        names = {m["name"] for m in metrics
                 if isinstance(m, dict) and isinstance(m.get("name"), str)}
        missing = CLUSTER_CASES - names
        if missing:
            raise ValueError(f"missing cluster cases: {', '.join(sorted(missing))}")


def print_log_tail(path, lines):
    with path.open(encoding="utf-8", errors="replace") as log:
        sys.stderr.writelines(deque(log, maxlen=lines))


def run_benchmark(side, cli, suite, output, summary):
    result = output / f"{suite}.{side}.json"
    log_path = output / f"{suite}.{side}.log"
    with log_path.open("w", encoding="utf-8") as log:
        try:
            subprocess.run(
                [cli, "bench", suite, f"--json={result}"],
                check=True, stdout=log, stderr=subprocess.STDOUT,
            )
            validate_result(suite, result)
        except (OSError, ValueError, subprocess.CalledProcessError) as exc:
            print(f"Error: {exc}", file=log)
        else:
            return result
    note(summary, f"Error: {suite} ({side}) failed; see {log_path}")
    print_log_tail(log_path, 20)
    return None


def note(summary, text=""):
    print(text, flush=True)
    print(text, file=summary)


def compare_suite(suite, base_cli, head_cli, base_suites, head_suites,
                  output, summary):
    note(summary, f"\n### {suite}\n")
    if suite not in head_suites:
        note(summary, f"Error: head does not list {suite}")
        return ERROR

    new = suite not in base_suites
    if not new:
        base_result = run_benchmark("base", base_cli, suite, output, summary)
        if base_result is None:
            return ERROR
    head_result = run_benchmark("head", head_cli, suite, output, summary)
    if head_result is None:
        return ERROR

    # Self-comparison checks the format of an unpaired result.
    if new:
        base_result = head_result
    suffix = "validate.log" if new else "compare.txt"
    report = output / f"{suite}.{suffix}"
    with report.open("w", encoding="utf-8") as log:
        status = subprocess.run(
            [head_cli, "bench", "compare", base_result, head_result],
            stdout=log, stderr=subprocess.STDOUT,
        ).returncode
    if new and status == 0:
        note(summary, f"New suite: the merge base has no {suite} benchmark.")
        return 0

    table = report.read_text(encoding="utf-8", errors="replace")
    print(table, end="", flush=True)
    summary.write(indent(table, "    ", lambda line: True))
    return status if not new and status in (0, REGRESSION) else ERROR


def compare_revisions(repo, base, head, output, summary):
    with tempfile.TemporaryDirectory(prefix="bench-build.") as temporary:
        scratch = Path(temporary)
        binaries = {}
        for side, sha in (("base", base), ("head", head)):
            print(f"Building {side} in release mode", flush=True)
            log_path = output / f"build.{side}.log"
            try:
                with log_path.open("w", encoding="utf-8") as log:
                    binaries[side] = build_cli(repo, sha, scratch / side, log)
            except (OSError, subprocess.CalledProcessError) as exc:
                if log_path.exists():
                    print_log_tail(log_path, 40)
                raise RuntimeError(f"{side} build failed; see {log_path}") from exc

        base_cli, head_cli = binaries["base"], binaries["head"]
        base_suites, head_suites = inventory(base_cli), inventory(head_cli)
        if not base_suites:
            raise RuntimeError("cannot list baseline benchmark suites")
        if "compare" not in head_suites:
            raise RuntimeError("cannot find head bench compare")

        statuses = [
            compare_suite(suite, base_cli, head_cli, base_suites, head_suites,
                          output, summary)
            for suite in SUITES
        ]

    regressions = statuses.count(REGRESSION)
    errors = statuses.count(ERROR)
    note(summary, f"\nBenchmark verdict: {regressions} regressed suite(s), "
                  f"{errors} error(s).")
    if errors:
        return ERROR
    if regressions:
        if not OVERRIDE.search(os.environ.get("BODY", "")):
            return REGRESSION
        note(summary, "The PR description waives benchmark regressions.")
    return 0


def main():
    repo = git_output("rev-parse", "--show-toplevel")
    target = os.environ.get("BASE_SHA") or "origin/main"
    head = os.environ.get("HEAD_SHA") or "HEAD"
    target = git_output("rev-parse", "--verify", "--end-of-options",
                        f"{target}^{{commit}}")
    head = git_output("rev-parse", "--verify", "--end-of-options",
                      f"{head}^{{commit}}")
    base = git_output("merge-base", target, head)
    output_dir = os.environ.get("BENCH_OUTPUT_DIR")
    if output_dir:
        output = Path(output_dir).resolve()
        output.mkdir()
    else:
        output = Path(tempfile.mkdtemp(prefix="bench-results.")).resolve()
    print(f"Benchmark results: {output}", flush=True)

    summary_path = output / "summary.md"
    try:
        with summary_path.open("w", encoding="utf-8") as summary:
            print(f"## Benchmark comparison\n\nMerge base: `{base}`  \n"
                  f"Head: `{head}`", file=summary)
            try:
                return compare_revisions(repo, base, head, output, summary)
            except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
                print(f"Error: {exc}", file=sys.stderr)
                print(f"\nError: {exc}", file=summary)
                return ERROR
    finally:
        step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if step_summary and summary_path.exists():
            with summary_path.open("rb") as source, open(step_summary, "ab") as dest:
                shutil.copyfileobj(source, dest)


def terminate(signum, _frame):
    # Unwind the temporary build directory on CI cancellation.
    raise SystemExit(128 + signum)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, terminate)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, subprocess.CalledProcessError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(ERROR)
