# Copyright (c) 2026 Tiger Data, Inc.
# Licensed under the PostgreSQL License. See LICENSE for details.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE.parent.parent / "scripts/ci/bench-regress.py"


class BenchRegressTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.repo = self.workspace / "repo"
        self.repo.mkdir()
        self.stubs = self.workspace / "stubs"
        self.stubs.mkdir()
        self.scratch = self.workspace / "scratch"
        self.scratch.mkdir()
        meson = self.stubs / "meson"
        shutil.copy2(HERE / "bench_regress_fixture.py", meson)
        meson.chmod(0o755)
        self.env = os.environ.copy()
        for name in (
            "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_NAMESPACE",
        ):
            self.env.pop(name, None)

        # Both tips diverge from a common commit, and the checkout is dirty.
        self.git("init", "-q", "-b", "target")
        self.git("config", "user.name", "Benchmark Test")
        self.git("config", "user.email", "bench@example.invalid")
        self.git("config", "core.hooksPath", os.devnull)
        self.marker = self.repo / "marker"
        self.marker.write_text("common\n")
        self.git("add", "marker")
        self.git("commit", "-qm", "common")
        self.git("branch", "feature")
        self.marker.write_text("target\n")
        self.git("commit", "-qam", "target")
        self.git("switch", "-q", "feature")
        self.marker.write_text("feature\n")
        self.git("commit", "-qam", "feature")
        self.git("switch", "-q", "target")
        self.marker.write_text("dirty\n")

    def git(self, *args):
        return subprocess.check_output(
            ["git", *args], cwd=self.repo, env=self.env, text=True,
        ).strip()

    def benchmark(self, scenario, status=0, *, body="", name=None):
        output = self.workspace / (name or scenario)
        trace = output.with_suffix(".trace")
        step_summary = output.with_suffix(".summary")
        env = dict(
            self.env, PATH=f"{self.stubs}{os.pathsep}{self.env['PATH']}",
            TMPDIR=str(self.scratch), BASE_SHA="target", HEAD_SHA="feature",
            BODY=body, BENCH_OUTPUT_DIR=str(output),
            GITHUB_STEP_SUMMARY=str(step_summary), BENCH_TEST_TRACE=str(trace),
            BENCH_TEST_SCENARIO=scenario,
        )
        result = subprocess.run(
            [sys.executable, RUNNER], cwd=self.repo, env=env,
            capture_output=True, text=True,
        )
        self.assertEqual(status, result.returncode, result.stdout + result.stderr)
        self.assertEqual([], list(self.scratch.iterdir()))
        self.assertEqual("dirty\n", self.marker.read_text())
        self.assertEqual((output / "summary.md").read_bytes(),
                         step_summary.read_bytes())
        return output, trace.read_text().splitlines()

    def test_exact_revisions_and_pair_order(self):
        output, trace = self.benchmark("ok")
        base = self.git("merge-base", "target", "feature")
        head = self.git("rev-parse", "feature")
        self.assertEqual([
            f"build common {base}",
            f"build feature {head}",
            "base distance",
            "head distance",
            "compare distance.base.json distance.head.json",
            "base idset",
        ], trace[:6])
        self.assertTrue((output / "distance.base.json").stat().st_size)

    def test_new_suite_is_unpaired(self):
        output, trace = self.benchmark("new")
        self.assertNotIn("base rotation", trace)
        self.assertIn("head rotation", trace)
        self.assertIn("New suite: the merge base has no rotation",
                      (output / "summary.md").read_text())

    def test_regression_override(self):
        self.benchmark("drop", 1)
        self.benchmark("drop", body="Explanation.\n\nDisable-check: bench",
                       name="waived")
        self.benchmark("drop", 1, body="Disable-check: bench-other",
                       name="not-waived")

    def test_errors_cannot_be_waived(self):
        for scenario in (
            "removed", "bad-inventory", "mixed", "compare-error",
            "missing-json", "wrong-suite", "partial-cluster",
        ):
            with self.subTest(scenario=scenario):
                _, trace = self.benchmark(scenario, 2, body="Disable-check: bench")
                if scenario == "mixed":
                    self.assertIn("head cluster", trace)


if __name__ == "__main__":
    unittest.main()
