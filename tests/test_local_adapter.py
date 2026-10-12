"""The pipeline uses an explicit compiler process and its public JSON contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elmm_pipeline.gpu_backend import transform as gpu_transform
from elmm_pipeline.local_adapter import LocalAdapterError, check, transform


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
            self.assertEqual(run.call_args.kwargs["env"]["PYTHONPATH"], str(self.snapshot))
            self.assertEqual({path.name for path in self.output.iterdir()}, {"check.log", "check.json"})

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

    def test_generation_retains_authoritative_report_and_compiler_artifacts(self):
        outputs = ["region.cpp", "region.cu", "interface.f90", "runtime.hpp"]
        report = {"kernel": "advance", "supported": True, "reason": None, "parallel_regions": 2,
                  "execution_plan": "ordered dependence proofs", "memory_plan": "one owned workspace", "outputs": outputs}

        def run_compiler(command, **kwargs):
            for name in outputs:
                (self.output / name).write_text("cpp_advance_start_hot cpp_advance_finish_hot\n")
            return self.completed(report)

        with patch("elmm_pipeline.local_adapter.subprocess.run", side_effect=run_compiler):
            metadata = transform(self.source, self.snapshot, self.python, self.output, entry="advance")
        self.assertEqual(metadata["compiler_report"], report)
        self.assertEqual(json.loads((self.output / "compiler-report.json").read_text()), report)
        self.assertIn("--json", metadata["command"])
        self.assertNotIn("--verbose", metadata["command"])
        for name in outputs:
            self.assertEqual((self.output / name).read_text(),
                             "cpp_advance_start_hot cpp_advance_finish_hot\n")

    def test_generation_reports_rejection_and_incompatible_snapshot(self):
        rejection = self.completed({"kernel": "advance", "supported": False, "reason": "unsafe RAW", "outputs": []}, 1)
        legacy = subprocess.CompletedProcess([], 0, "Generated files in outputs/\n", "")
        for response, reason in ((rejection, "unsafe RAW"), (legacy, "refresh the explicit compiler snapshot")):
            with self.subTest(reason=reason), patch("elmm_pipeline.local_adapter.subprocess.run", return_value=response), \
                    self.assertRaisesRegex(LocalAdapterError, reason):
                transform(self.source, self.snapshot, self.python, self.output, entry="advance")

    def test_gpu_uses_compiler_region_count_and_rejects_host_only_generation(self):
        for count in (0, 2):
            def compile_local(*args, count=count, **kwargs):
                # Deliberately no __global__ string: metadata must not count source text.
                (self.output / "region.cu").write_text("kernel_region_0_device\n")
                (self.output / "interface.f90").touch()
                return {"compiler_report": {"parallel_regions": count}}

            with self.subTest(count=count), patch("elmm_pipeline.local_adapter.transform", side_effect=compile_local):
                if count == 0:
                    with self.assertRaisesRegex(ValueError, "no proven parallel regions"):
                        gpu_transform("local", self.source, self.python, self.output,
                                      compiler_root=self.snapshot, entry="advance")
                else:
                    metadata = gpu_transform("local", self.source, self.python, self.output,
                                             compiler_root=self.snapshot, entry="advance")
                    self.assertEqual(metadata["gpu"]["kernel_regions"], 2)
                    self.assertEqual((self.output / "region.cu").read_text(),
                                     "kernel_region_0_device\n")


if __name__ == "__main__":
    unittest.main()
