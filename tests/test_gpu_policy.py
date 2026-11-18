"""Runtime policy dispatch uses public compiler metadata and native fallbacks."""
from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from elmm_pipeline.cli import prepare_build
from elmm_pipeline.gpu_validation import gpu_activity, native_policy_exercised, offload_decisions
from elmm_pipeline.instrumentation import execution_coverage


def contract(*, collective=True, policy="auto"):
    return {"policy": policy, "native_fallback_query": "policy_query", "collective_entry": collective,
            "host_threads": 4, "profile_available": True, "profile_reason": None, "analysis": {}}


class GPUPolicyProtocolTests(unittest.TestCase):
    def test_interleaved_decisions_and_markers_remain_distinct_from_cuda_proof(self):
        region = {"id": "abcd", "entry": "outline_abcd", "transformation": {"compiler_report": {"offload": contract()}}}
        log = ("FORT_RUNTIME ELMM_PIPELINE_REGION abcd\n"
               "FORT_RUNTIME FORT_OFFLOAD entry=outline_abcd mode=native gpu_units=0 cpu_units=2\n")
        activity = gpu_activity(log, "local", [region])
        self.assertFalse(activity["verified"])
        self.assertTrue(native_policy_exercised(activity, [region]))
        self.assertEqual(activity["offload_decisions"]["modes"]["native"], 1)
        self.assertEqual(execution_coverage(log, [region])["executed_regions"], ["abcd"])
        log += "FORT_OFFLOAD entry=outline_abcd mode=hybrid gpu_units=1 cpu_units=1\n"
        claimed = gpu_activity(log, "local", [region])
        self.assertFalse(claimed["verified"])
        self.assertFalse(native_policy_exercised(claimed, [region]))
        traced = gpu_activity(log+"FORT_RUNTIME kernel\n", "local", [region])
        self.assertTrue(traced["verified"])
        self.assertEqual(traced["offload_decisions"]["cpu_units"], 3)
        with self.assertRaisesRegex(ValueError, "absent"):
            offload_decisions("FORT_OFFLOAD entry=unknown mode=native gpu_units=0 cpu_units=1", [region])
        self.assertEqual(offload_decisions("NOTFORT_OFFLOAD entry=outline_abcd mode=native gpu_units=0 cpu_units=1", [region])["calls"], 0)
        slabs = offload_decisions("FORT_OFFLOAD entry=outline_abcd mode=hybrid gpu_units=16 cpu_units=32 unit_kind=slabs", [region])
        self.assertEqual(slabs["unit_counts"], {"slabs": {"gpu_units": 16, "cpu_units": 32}})
        self.assertEqual(slabs["entries"]["outline_abcd"]["unit_kind"], "slabs")

    def test_profile_content_and_policy_are_build_identity_and_copied_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("elmm/src", "PoisFFT/src", "local-compiler/compiler"):
                path = root/".cache/repos"/name
                path.mkdir(parents=True)
                (path/"source.f90").write_text("! source")
            implementation = Path(__file__).resolve().parents[1]/"elmm_pipeline"
            shutil.copytree(implementation, root/"elmm_pipeline")
            profile = root/"hardware.json"
            profile.write_text('{"rates":1}')
            facts = root/"facts.json"
            facts.write_text('{"schema_version":1,"participation":"serial","captures":{}}')
            args = SimpleNamespace(build_arg=[], target="gpu", gpu_toolchain={}, gpu_policy="auto",
                                   calibration_profile=profile, source_pattern=["**/*.f90"], select=[], exclude=[],
                                   marked_only=False, no_coverage=True, jobs=1, threads=4,
                                   tools=["local"],
                                   scope_entry="step", scope_source="source.f90", scope_facts=facts)
            setup = {"pins": {}, "packages": {"local": []}, "native": {}}
            seen = []

            def adapt(*args, **kwargs):
                copied = Path(kwargs["gpu_options"]["calibration_profile"])
                self.assertNotEqual(copied, profile)
                self.assertEqual(copied.read_bytes(), profile.read_bytes())
                seen.append(dict(kwargs["gpu_options"]))
                return {"tool": "local", "regions": [{"id": "one"}], "changed_files": []}

            def build_(*args, **kwargs):
                directory = args[2]
                directory.mkdir(parents=True)
                (directory/"ELMM").write_text("binary")
                (directory/"build-command.json").write_text("{}")
                return directory/"ELMM"

            with patch("elmm_pipeline.cli.ROOT", root), patch("elmm_pipeline.adaptation.adapt_project", side_effect=adapt), \
                    patch("elmm_pipeline.cli.build", side_effect=build_):
                first, metadata = prepare_build("local", args, setup, {})
                self.assertEqual(metadata["identity"]["gpu_policy"], "auto")
                self.assertEqual(metadata["identity"]["calibration_profile"]["sha256"], hashlib.sha256(profile.read_bytes()).hexdigest())
                profile.write_text('{"rates":2}')
                second, _ = prepare_build("local", args, setup, {})
                args.gpu_policy = "sections"
                third, _ = prepare_build("local", args, setup, {})
                calls = root/"elmm_pipeline/scope_extraction.py"
                calls.write_text(calls.read_text()+"\n# changed checked source extraction\n")
                fourth, changed = prepare_build("local", args, setup, {})
                self.assertEqual(changed["identity"]["implementation"]["scope_extraction.py"],
                                 hashlib.sha256(calls.read_bytes()).hexdigest())
            self.assertEqual(len({first.parent, second.parent, third.parent, fourth.parent}), 4)
            self.assertEqual([item["gpu_policy"] for item in seen], ["auto", "auto", "sections", "sections"])


if __name__ == "__main__":
    unittest.main()
