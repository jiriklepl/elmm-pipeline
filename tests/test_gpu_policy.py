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

from elmm_pipeline.adaptation import patch_regions
from elmm_pipeline.cli import prepare_build
from elmm_pipeline.gpu_validation import gpu_activity, native_policy_exercised, offload_decisions
from elmm_pipeline.instrumentation import execution_coverage, region_call
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
            outputs = ["region.cpp", "region.cu", "interface.f90", "runtime.hpp"]
            report = {"kernel": "advance", "supported": True, "parallel_regions": 2,
                      "outputs": outputs, "offload": contract()}

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
            self.assertEqual(result["compiler_report"]["offload"], contract())

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
            args = SimpleNamespace(build_arg=[], target="gpu", gpu_toolchain={}, gpu_policy="auto",
                                   calibration_profile=profile, source_pattern=["**/*.f90"], select=[], exclude=[],
                                   marked_only=False, no_coverage=True, jobs=1, threads=4)
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
                args.gpu_policy = "hybrid"
                third, _ = prepare_build("local", args, setup, {})
                calls = root/"elmm_pipeline/gpu_calls.py"
                calls.write_text(calls.read_text()+"\n# changed call normalization\n")
                fourth, changed = prepare_build("local", args, setup, {})
                self.assertEqual(changed["identity"]["implementation"]["gpu_calls.py"],
                                 hashlib.sha256(calls.read_bytes()).hexdigest())
            self.assertEqual(len({first.parent, second.parent, third.parent, fourth.parent}), 4)
            self.assertEqual([item["gpu_policy"] for item in seen], ["auto", "auto", "hybrid", "hybrid"])

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


@unittest.skipUnless(shutil.which("gfortran"), "requires GNU Fortran")
class NativePolicyDispatchTests(unittest.TestCase):
    def test_public_patch_collective_query_barrier_and_native_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = "!$omp do\ndo i=1,n\na(i)=a(i)+1.0_8\nend do\n!$omp end do\n"
            source = """module subject
use policy_module, only: ready_signal
implicit none
real(8), allocatable :: a(:)
contains
subroutine advance(n)
integer, intent(in) :: n
integer :: i,lane
ready_signal=0
!$omp parallel num_threads(4) private(i,lane)
!$omp do schedule(static,1)
do lane=1,4
if(lane==4) call delayed_ready()
end do
!$omp end do nowait
"""+original+"""!$omp end parallel
end subroutine
subroutine delayed_ready()
integer :: first,now,rate
call system_clock(first,rate)
do
 call system_clock(now)
 if(real(now-first)/rate>0.05) exit
end do
ready_signal=7
end subroutine
end module
"""
            path = root/"subject.f90"
            path.write_text(source)
            first = source[:source.index(original)].count("\n")+1
            region = {"id": "policy", "source": "subject.f90", "entry": "offload", "generated_module": "policy_module",
                      "start_line": first, "end_line": first+original.count("\n")-1,
                      "insertion_line": source.splitlines().index("subroutine advance(n)")+2,
                      "original_text": original, "call": "call offload(a,n,lbound(a,1),size(a))",
                      "actual_args": ["a", "n", "lbound(a,1)", "size(a)"],
                      "gpu_scope": {"dispatch": "collective_single", "expected_omp_level": 1,
                                    "allocated_arrays": ["a"], "guard": "size(a)>=n"},
                      "transformation": {"compiler_report": {"offload": contract()}}}
            patch_regions(root, [region], coverage=False)
            patched = path.read_text()
            self.assertIn("only: offload, policy_query", patched)
            self.assertNotIn("!$omp single", region_call(region, coverage=False))
            (root/"policy.f90").write_text("""module policy_module
use omp_lib
implicit none
integer :: ready_signal=0,queries=0,calls=0
logical :: want_gpu=.false.
contains
logical function policy_query(field,n,lower,extent)
real(8), intent(in) :: field(:)
integer, intent(in) :: n,lower,extent
if(ready_signal/=7) error stop 'query ran before native producer finished'
if(extent/=size(field)) error stop 'descriptor changed'
!$omp atomic update
queries=queries+1
policy_query=want_gpu
end function
subroutine offload(field,n,lower,extent)
real(8), intent(inout) :: field(:)
integer, intent(in) :: n,lower,extent
integer :: index
if(omp_get_num_threads()/=4) error stop 'existing team not reused'
!$omp atomic update
calls=calls+1
!$omp do
do index=1,n
 field(index-lower+1)=field(index-lower+1)+1.0_8
end do
!$omp end do
end subroutine
end module
""")
            (root/"driver.f90").write_text("""program verify
use subject
use policy_module
use omp_lib
implicit none
call omp_set_dynamic(.false.)
call advance(0)
if(queries/=0.or.calls/=0) error stop 'unallocated query evaluated'
allocate(a(-2:8)); a=-1.0_8
want_gpu=.false.
call advance(4)
if(queries/=4.or.calls/=0.or.any(a(1:4)/=0.0_8)) error stop 'native fallback'
want_gpu=.true.
call advance(4)
if(queries/=8.or.calls/=4.or.any(a(1:4)/=1.0_8)) error stop 'collective entry'
if(any(a(-2:0)/=-1.0_8).or.any(a(5:8)/=-1.0_8)) error stop 'halo changed'
!$omp parallel if(.false.)
call advance(4)
!$omp end parallel
if(queries/=8.or.calls/=4.or.any(a(1:4)/=2.0_8)) error stop 'nested caller fallback'
print *, 'POLICY_PASS'
end program
""")
            build = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                                    "policy.f90", "subject.f90", "driver.f90", "-o", "verify"],
                                   cwd=root, capture_output=True, text=True)
            self.assertEqual(build.returncode, 0, build.stdout+build.stderr)
            run = subprocess.run([str(root/"verify")], cwd=root, capture_output=True, text=True, timeout=15)
            self.assertEqual(run.returncode, 0, run.stdout+run.stderr)
            self.assertIn("POLICY_PASS", run.stdout)


if __name__ == "__main__":
    unittest.main()
