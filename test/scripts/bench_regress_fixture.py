#!/usr/bin/env python3
# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.

"""Stand-ins for Meson and vectorsearch; Git snapshots are real."""

import json
import os
import shutil
import sys
from pathlib import Path

SUITES = (
    "compare", "distance", "idset", "quantize", "rabitq-kernel",
    "page-score", "fastscan-kernel", "topk", "rotation", "search", "cluster",
)


def trace(line):
    with open(os.environ["BENCH_TEST_TRACE"], "a", encoding="utf-8") as log:
        print(line, file=log)


def meson(args):
    if args[0] == "setup":
        _, build, source, *options = args
        assert options == [
            "--buildtype=release", "-Dtools=enabled", "-Dpostgresql=disabled",
        ]
        source = Path(source)
        Path(build).mkdir()
        shutil.copy2(__file__, Path(build) / "vectorsearch")
        marker = (source / "marker").read_text().strip()
        sha = (source / ".commit-sha").read_text().strip()
        trace(f"build {marker} {sha}")
    else:
        assert args[0:2] == ["compile", "-C"]
        assert args[3:] == ["vectorsearch:executable"]
    return 0


def vectorsearch(args):
    side = Path(sys.argv[0]).parent.parent.name
    scenario = os.environ["BENCH_TEST_SCENARIO"]
    if args == ["--help"]:
        if (side, scenario) == ("base", "bad-inventory"):
            return 7
        for suite in SUITES:
            if suite == "rotation" and (side, scenario) in {
                ("base", "new"), ("head", "removed"),
            }:
                continue
            print(f"    bench {suite:12} Description")
        return 0

    assert args[0] == "bench"
    suite = args[1]
    if suite == "compare":
        assert side == "head"
        base, head = (Path(path).name for path in args[2:])
        trace(f"compare {base} {head}")
        print("example report")
        if head == "distance.head.json":
            return {"drop": 1, "mixed": 1, "compare-error": 2}.get(scenario, 0)
        return 0

    trace(f"{side} {suite}")
    if (side, scenario, suite) == ("base", "mixed", "idset"):
        return 4
    if (side, scenario, suite) == ("head", "missing-json", "idset"):
        return 0
    if (side, scenario, suite) == ("head", "wrong-suite", "idset"):
        suite = "search"
    if suite == "cluster":
        names = ["lloyd", "hamerly", "elkan", "pgvector"]
        if (side, scenario) != ("base", "partial-cluster"):
            names.append("lloyd(cblas)")
        unit = "vec/s"
    else:
        names = ["qps"]
        unit = "q/s"
    assert args[2].startswith("--json=")
    result = Path(args[2][len("--json="):])
    metrics = [{"name": name, "unit": unit, "value": 100} for name in names]
    result.write_text(json.dumps({"bench": suite, "metrics": metrics}))
    return 0


if __name__ == "__main__":
    command = meson if Path(sys.argv[0]).name == "meson" else vectorsearch
    sys.exit(command(sys.argv[1:]))
