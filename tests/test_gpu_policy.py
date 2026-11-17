"""Runtime policy dispatch uses public compiler metadata and native fallbacks."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import test_local_adapter as local_fixture
from elmm_pipeline.cli import prepare_build
from elmm_pipeline.gpu_validation import gpu_activity, native_policy_exercised, offload_decisions
from elmm_pipeline.instrumentation import execution_coverage
from elmm_pipeline.local_adapter import transform


def contract(*, collective=True, policy="auto"):
    return {"policy": policy, "native_fallback_query": "policy_query", "collective_entry": collective,
            "host_threads": 4, "profile_available": True, "profile_reason": None, "analysis": {}}


class GPUPolicyProtocolTests(unittest.TestCase):
    def test_explicit_public_policy_arguments_and_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root/"snapshot"
            (snapshot/"compiler").mkdir(parents=True)
            (snapshot/"compiler/__main__.py").touch()
            source = root/"source.f90"
            source.write_text("module subject\nend module\n")
            profile = root/"hardware.json"
            profile.write_text("{}")
            output = root/"generated"
            report = local_fixture.LocalCompilerProtocolTests().shared_report()
            report["scoped"]["team"] = {"available": True}
            outputs = report["outputs"]

            def compile_(argv, **kwargs):
                for filename in outputs:
                    (output/filename).write_text("! generated\n")
                return subprocess.CompletedProcess(argv, 0, json.dumps(report), "")

            with patch("elmm_pipeline.local_adapter.subprocess.run", side_effect=compile_):
                result = transform(source, snapshot, Path(sys.executable), output, entry="advance",
                                   context={"gpu_dispatch": "collective_single"},
                                   gpu_options={"gpu_policy": "auto", "host_threads": 4,
                                                "calibration_profile": str(profile)})
            argv = result["command"]
            self.assertEqual(argv[argv.index("--gpu-policy")+1], "auto")
            self.assertEqual(argv[argv.index("--calibration-profile")+1], str(profile))
            self.assertEqual(argv[argv.index("--host-threads")+1], "4")
            self.assertIn("--gpu-collective", argv)
            self.assertEqual(argv[argv.index("--memory-model")+1], "scoped")
            self.assertEqual(result["compiler_report"]["scoped"], report["scoped"])
            self.assertNotIn("offload", result["compiler_report"])

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
                                   tools=["local"], memory_model="scoped", scope_execution="graph",
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

    def test_isolated_checkouts_use_public_policy_and_native_query(self):
        compiler = os.environ.get("ELMM_TEST_COMPILER")
        python = os.environ.get("ELMM_TEST_PYTHON")
        if not compiler or not python or not shutil.which("gfortran") or not shutil.which("g++"):
            self.skipTest("requires explicit compiler/Python and GNU compilers")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pipeline = root / "pipeline"
            snapshot = root / "compiler-checkout"
            ignore = shutil.ignore_patterns("__pycache__", ".pytest_cache", ".ruff_cache")
            implementation = Path(__file__).resolve().parents[1] / "elmm_pipeline"
            shutil.copytree(implementation, pipeline / "elmm_pipeline", ignore=ignore)
            shutil.copytree(Path(compiler) / "compiler", snapshot / "compiler", ignore=ignore)
            source = root / "entry.f90"
            source.write_text("""module policy_subject
contains
subroutine advance(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
 a(i)=a(i)+1.0_8
end do
end subroutine
end module
""")
            request = {"source": str(source), "snapshot": str(snapshot), "python": python,
                       "output": str(root / "generated")}
            (pipeline / "request.json").write_text(json.dumps(request))
            (pipeline / "run.py").write_text("""import json
from pathlib import Path
import elmm_pipeline
from elmm_pipeline.local_adapter import transform
request=json.loads(Path('request.json').read_text())
result=transform(Path(request['source']),Path(request['snapshot']),Path(request['python']),
                 Path(request['output']),entry='advance',gpu_options={'gpu_policy':'auto','host_threads':4})
Path('result.json').write_text(json.dumps({'package':elmm_pipeline.__file__,'report':result}))
""")
            environment = dict(os.environ, PYTHONPATH=str(pipeline))
            run = subprocess.run([python, str(pipeline / "run.py")], cwd=pipeline,
                                 env=environment, capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            result = json.loads((pipeline / "result.json").read_text())
            self.assertTrue(Path(result["package"]).is_relative_to(pipeline))
            report = result["report"]["compiler_report"]
            self.assertEqual(report["offload"]["policy"], "auto")
            self.assertFalse(report["offload"]["profile_available"])
            query = report["offload"]["native_fallback_query"]
            generated = root / "generated"
            (generated / "driver.f90").write_text(f"""program verify
use policy_subject
implicit none
real(8)::a(-2:8)
a=3.0_8
if ({query}(a,size(a))) error stop 'missing calibration selected GPU'
call advance(a,size(a))
if (any(a/=4.0_8)) error stop 'native result differs'
print *, 'INDEPENDENT_POLICY_PASS'
end program
""")
            for command in (
                ["g++", "-O2", "-std=c++17", "-fopenmp", "-c", "region.cpp"],
                ["gfortran", "-O2", "-fopenmp", "-ffree-line-length-none", "interface.f90", "driver.f90",
                 "region.o", "-lstdc++", "-o", "verify"],
                [str(generated / "verify")],
            ):
                run = subprocess.run(command, cwd=generated, capture_output=True, text=True, timeout=60)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            self.assertIn("INDEPENDENT_POLICY_PASS", run.stdout)


if __name__ == "__main__":
    unittest.main()
