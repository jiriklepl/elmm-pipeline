"""The pipeline uses an explicit compiler process and its public JSON contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from elmm_pipeline.local_adapter import LocalAdapterError, check


class LocalCompilerProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.snapshot = self.root / "snapshot"
        (self.snapshot / "compiler").mkdir(parents=True)
        (self.snapshot / "compiler/__main__.py").touch()
        self.source = self.root / "source.f90"
        self.source.write_text("module unrelated\nend module\n")
        self.output = self.root / "output"
        self.python = Path(sys.executable)

    def candidate(self, supported=True):
        return {"module": "unrelated", "name": "advance", "qualified_name": "unrelated::advance",
                "path": str(self.source), "line": 3, "annotated": False,
                "supported": supported, "reason": None if supported else "loop-carried RAW dependence"}

    def completed(self, report, returncode=0):
        return subprocess.CompletedProcess([], returncode, json.dumps(report), "")

    def test_preflight_returns_selected_compiler_decision_without_generating(self):
        for supported in (True, False):
            candidate = self.candidate(supported)
            with self.subTest(supported=supported), patch(
                    "elmm_pipeline.local_adapter.subprocess.run", return_value=self.completed([candidate])) as run:
                report = check(self.source, self.snapshot, self.python, "advance", self.output)
            self.assertEqual(report, candidate)
            command = run.call_args.args[0]
            self.assertIn("--list-candidates", command)
            self.assertIn("--json", command)
            self.assertEqual(command[command.index("--fallback") + 1], "error")
            self.assertEqual(command[command.index("--opt-level") + 1], "1")
            self.assertEqual(run.call_args.kwargs["cwd"], self.snapshot)
            self.assertEqual(run.call_args.kwargs["timeout"], 240)
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], str(self.snapshot))
            self.assertEqual({path.name for path in self.output.iterdir()}, {"check.log", "check.json"})

    def test_compiler_timeout_preserves_partial_output_and_rejects_artifacts(self):
        for stdout, stderr in ((b"partial stdout\n", b"partial stderr\n"),
                               ("partial stdout\n", "partial stderr\n"), (None, None)):
            with self.subTest(stdout=stdout), patch(
                    "elmm_pipeline.local_adapter.subprocess.run",
                    side_effect=subprocess.TimeoutExpired(["compiler"], 240, stdout, stderr)), \
                    self.assertRaisesRegex(LocalAdapterError, "240-second phase limit"):
                check(self.source, self.snapshot, self.python, "advance", self.output)
            receipt = json.loads((self.output / "check.log.timeout.json").read_text())
            self.assertEqual(receipt["timeout_seconds"], 240)
            self.assertTrue(receipt["timed_out"])
            self.assertIsNone(receipt["returncode"])
            self.assertEqual(receipt["cwd"], str(self.snapshot))
            self.assertIn("--list-candidates", receipt["argv"])
            self.assertEqual((self.output / "check.log").read_text(),
                             "partial stdout\npartial stderr\n" if stdout else "")
            self.assertFalse((self.output / "check.json").exists())

    def test_preflight_rejects_missing_ambiguous_and_unstructured_reports(self):
        for report in ([], [self.candidate(), self.candidate()], {"supported": True}, [{"name": "advance"}]):
            with self.subTest(report=report), patch(
                    "elmm_pipeline.local_adapter.subprocess.run", return_value=self.completed(report)), \
                    self.assertRaises(LocalAdapterError):
                check(self.source, self.snapshot, self.python, "advance", self.output)

    def test_preflight_accepts_exact_qualified_entry(self):
        other = {**self.candidate(), "module": "other", "qualified_name": "other::advance"}
        with patch("elmm_pipeline.local_adapter.subprocess.run", return_value=self.completed([other, self.candidate()])):
            report = check(self.source, self.snapshot, self.python, "UNRELATED::ADVANCE", self.output)
        self.assertEqual(report["qualified_name"], "unrelated::advance")

    def test_native_baseline_needs_no_scope_facts(self):
        from types import SimpleNamespace
        from elmm_pipeline.cli import scoped_configuration
        args = SimpleNamespace(tools=["baseline"], target="cpu", gpu_policy="sections")
        self.assertIsNone(scoped_configuration(args))
        args.tools = ["baseline", "local"]
        with self.assertRaisesRegex(ValueError, "--target gpu"):
            scoped_configuration(args)


if __name__ == "__main__":
    unittest.main()
