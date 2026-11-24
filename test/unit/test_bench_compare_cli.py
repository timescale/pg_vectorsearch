# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def check(status, *args, stderr=None, close_stdout=False):
    result = subprocess.run(
        [sys.argv[1], "bench", *args],
        capture_output=True,
        text=True,
        preexec_fn=(lambda: os.close(1)) if close_stdout else None,
    )
    if result.returncode != status or (stderr and stderr not in result.stderr):
        raise AssertionError(
            f"{args}: expected exit {status}, got {result.returncode}\n"
            f"{result.stdout}\n{result.stderr}"
        )


def write_result(path, bench, value):
    path.write_text(json.dumps({
        "bench": bench,
        "metrics": [{"name": "qps", "unit": "q/s", "value": value}],
    }))


with tempfile.TemporaryDirectory() as tmp:
    base = Path(tmp) / "base.json"
    head = Path(tmp) / "head.json"
    write_result(base, "search", 100)

    check(0, "compare", "--help")
    check(2, "compare", base, head, stderr="Error:")
    head.write_text("{")
    check(2, "compare", base, head, stderr="Error:")

    write_result(head, "search", 100)
    check(0, "compare", base, head)
    check(2, "compare", "--quiet", base, head, close_stdout=True,
          stderr="cannot save stdout")
    check(2, "compare", "--json=-", base, head, stderr="does not take --json")
    check(2, "compare", "--spinup=bad", base, head, stderr="invalid --spinup")
    check(2, "--min-runtime=bad", "compare", base, head,
          stderr="invalid --min-runtime")
    check(2, "compare", base, head, close_stdout=True,
          stderr="cannot write comparison report")

    write_result(head, "search", 80)
    check(0, "compare", base, head)
    write_result(head, "search", 60)
    check(1, "compare", base, head)
    write_result(head, "cluster", 100)
    check(2, "compare", base, head, stderr="bench names must match")
