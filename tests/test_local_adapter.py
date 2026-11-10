"""The pipeline uses an explicit compiler process and its public JSON contract."""

from __future__ import annotations

import hashlib
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

    def test_generation_retains_authoritative_report_and_compiler_artifacts(self):
        report = self.shared_report()
        outputs = report["outputs"]

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

    def shared_report(self):
        contract = {"schema_version": 1, "id": "separate-arithmetic-v1",
                    "required_cuda_options": ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
                    "required_host_options": ["-ffp-contract=off"]}
        contract["identity"] = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        outputs = ["shared_entry.cu", "shared_interface.f90", "runtime.hpp", "scoped_runtime.cu",
                   "fort_scoped_memory.f90"]
        units = [{"path": name, "language": "cuda" if name.endswith(".cu") else "fortran",
                  "role": "shared_entry" if name.startswith("shared_") else "common_runtime",
                  **({"numerical_contract": contract} if name.endswith(".cu") else {})}
                 for name in outputs if name != "runtime.hpp"]
        report = {"artifact_schema_version": 2, "memory_model": "scoped", "supported": True,
                  "kernel": "advance", "parallel_regions": 2, "outputs": outputs, "build_sources": units,
                  "numerical_contract": contract,
                  "scoped": {"cuda_source": "shared_entry.cu", "fortran_source": "shared_interface.f90"}}

        return report

    def test_scoped_generation_consumes_only_the_versioned_shared_abi(self):
        report = self.shared_report()
        outputs, units = report["outputs"], report["build_sources"]

        def run_compiler(command, **kwargs):
            self.assertNotIn("--cuda-output", command)
            self.assertNotIn("--cpp-output", command)
            self.assertNotIn("--fortran-output", command)
            for name in outputs:
                (self.output / name).write_text("shared artifact\n")
            return self.completed(report)

        with patch("elmm_pipeline.local_adapter.subprocess.run", side_effect=run_compiler):
            metadata = transform(self.source, self.snapshot, self.python, self.output, entry="advance",
                                 context={"nested_parallel": True},
                                 gpu_options={"memory_model": "scoped", "gpu_policy": "sections"})
        self.assertEqual(metadata["target"], "scoped")
        self.assertEqual({Path(n).name for n in metadata["extra_sources"]},
                         {"shared_interface.f90", "fort_scoped_memory.f90"})
        self.assertEqual({Path(x["source"]).name for x in metadata["gpu_sources"]},
                         {"shared_entry.cu", "scoped_runtime.cu"})
        for change, reason in (({"artifact_schema_version": 1}, "artifact schema 2"),
                               ({"offload": {}}, "without an ordinary backend"),
                               ({"build_sources": [*units[:-1], {**units[-1], "language": "unknown"}]},
                                "explicit CUDA or Fortran"),
                               ({"build_sources": [*units, {"path": "old.cu", "role": "numerical_entry"}]},
                                "only shared-entry")):
            with self.subTest(change=change), patch("elmm_pipeline.local_adapter.subprocess.run",
                    return_value=self.completed({**report, **change})), self.assertRaisesRegex(LocalAdapterError, reason):
                transform(self.source, self.snapshot, self.python, self.output, entry="advance",
                          gpu_options={"memory_model": "scoped", "gpu_policy": "sections"})

    def test_removed_call_adapter_rejects_before_compilation_or_writes(self):
        from elmm_pipeline.adaptation import adapt_project
        with patch("elmm_pipeline.local_adapter.subprocess.run") as run:
            for operation in (
                lambda: gpu_transform("local", self.source, self.python, self.output,
                                      compiler_root=self.snapshot, entry="advance"),
                lambda: transform(self.source, self.snapshot, self.python, self.output,
                                  entry="advance", gpu_options={"memory_model": "call"}),
                lambda: adapt_project("local", self.root, self.python, self.output,
                                      memory_model="call"),
            ):
                with self.subTest(operation=operation), self.assertRaisesRegex((ValueError, LocalAdapterError), "removed"):
                    operation()
            for policy in ("always", "chunked", "hybrid"):
                with self.subTest(policy=policy), self.assertRaisesRegex(LocalAdapterError, "removed"):
                    transform(self.source, self.snapshot, self.python, self.output,
                              entry="advance", gpu_options={"gpu_policy": policy})
            run.assert_not_called()
        self.assertFalse(self.output.exists())

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
