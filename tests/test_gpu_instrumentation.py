"""Compile the public GPU patching path with native kernels and GNU OpenMP.

Counters identify generated dispatches, so equal numerical output cannot hide
an incorrect GPU call from a nested caller or one call per team member.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from elmm_pipeline.adaptation import patch_regions
from elmm_pipeline.instrumentation import execution_coverage, region_call

HAS_FPARSER = importlib.util.find_spec("fparser") is not None
if HAS_FPARSER:
    from elmm_pipeline.gpu_discovery import discover_sources


SERIAL = """module subject
implicit none
contains
subroutine advance(output,input,low,high,stride)
real(kind=8), intent(inout), contiguous :: output(-3:)
real(kind=8), intent(in), contiguous :: input(-3:)
integer, intent(in) :: low,high,stride
integer :: tile,position
do tile=low,high,stride
 do position=tile,min(tile+stride-1,high)
  output(position)=output(position)+2.0_8*input(position)+position
 end do
end do
end subroutine
end module
"""

COLLECTIVE = """module subject
implicit none
contains
subroutine advance(output,input,observed,n,stride)
real(kind=8), intent(inout), contiguous :: output(:),input(:),observed(:)
integer, intent(in) :: n,stride
integer :: tile,position
!$omp parallel private(tile,position)
!$omp do schedule(static)
do tile=1,n,stride
 do position=tile,min(tile+stride-1,n)
  if (position==n) call delayed_lane()
  input(position)=real(position,8)
 end do
end do
!$omp end do nowait
!$omp do schedule(static)
do tile=1,n,stride
 do position=tile,min(tile+stride-1,n)
  output(position)=output(position)+2.0_8*input(position)
 end do
end do
!$omp end do nowait
call native_boundary()
!$omp do schedule(static)
do tile=1,n,stride
 do position=tile,min(tile+stride-1,n)
  observed(position)=output(position)
 end do
end do
!$omp end do
!$omp end parallel
end subroutine
subroutine delayed_lane()
integer :: begin,now,rate
call system_clock(begin,rate)
do
 call system_clock(now)
 if (real(now-begin)/rate>=0.075) exit
end do
end subroutine
subroutine native_boundary()
end subroutine
end module
"""

SERIAL_DRIVER = """program verify
use subject, only: candidate=>advance
use reference, only: original=>advance
use @KERNEL@, only: test_calls,test_level
use omp_lib, only: omp_set_dynamic,omp_set_num_threads,omp_set_max_active_levels
implicit none
real(kind=8) :: result(-3:11,3),expected(-3:11,3),input(-3:11,3)
integer :: lane,position
call omp_set_dynamic(.false.)
call omp_set_num_threads(3)
call omp_set_max_active_levels(1)
input=1.125_8
result=-17.0_8
expected=result
call original(expected(:,1),input(:,1),-2,9,4)
call candidate(result(:,1),input(:,1),-2,9,4)
if (any(result/=expected)) stop 11
if (test_calls/=1 .or. test_level/=0) stop 12
! A nonpositive tile stride must execute the unchanged native loop.
call original(expected(:,1),input(:,1),-2,9,-2)
call candidate(result(:,1),input(:,1),-2,9,-2)
if (any(result/=expected) .or. test_calls/=1) stop 13
! Each outer lane supplies its own array: discovery cannot know this caller.
do lane=1,3
 call original(expected(:,lane),input(:,lane),-2,9,3)
end do
!$omp parallel do num_threads(3)
do lane=1,3
 call candidate(result(:,lane),input(:,lane),-2,9,3)
end do
!$omp end parallel do
if (any(result/=expected) .or. test_calls/=1) stop 14
! An inactive region still increases OMP_GET_LEVEL and must fall back.
call original(expected(:,1),input(:,1),-2,9,3)
!$omp parallel if(.false.)
call candidate(result(:,1),input(:,1),-2,9,3)
!$omp end parallel
if (any(result/=expected) .or. test_calls/=1) stop 15
print *, 'VERIFIED_SERIAL'
end program
"""

COLLECTIVE_DRIVER = """program verify
use subject, only: candidate=>advance
use reference, only: original=>advance
use @KERNEL@, only: test_calls,test_level
use omp_lib, only: omp_set_dynamic,omp_set_num_threads,omp_set_max_active_levels
implicit none
integer, parameter :: n=27
real(kind=8) :: result(n,3),expected(n,3),input(n,3),ref_input(n,3),observed(n,3),ref_observed(n,3)
integer :: lane
call omp_set_dynamic(.false.)
call omp_set_num_threads(3)
call omp_set_max_active_levels(1)
result=-17.0_8
expected=result
input=-999.0_8
ref_input=input
observed=0
ref_observed=0
call original(expected(:,1),ref_input(:,1),ref_observed(:,1),n,4)
call candidate(result(:,1),input(:,1),observed(:,1),n,4)
if (any(result/=expected) .or. any(input/=ref_input)) stop 21
if (any(observed/=ref_observed)) stop 22
if (test_calls/=1 .or. test_level/=1) stop 23
! Negative stride fallback must leave the worksharing DO collective.
call original(expected(:,1),ref_input(:,1),ref_observed(:,1),n,-2)
call candidate(result(:,1),input(:,1),observed(:,1),n,-2)
if (any(result/=expected) .or. any(observed/=ref_observed)) stop 24
if (test_calls/=1) stop 25
! The routine's lexical parallel now runs at level two inside each caller.
do lane=1,3
 call original(expected(:,lane),ref_input(:,lane),ref_observed(:,lane),n,3)
end do
!$omp parallel do num_threads(3)
do lane=1,3
 call candidate(result(:,lane),input(:,lane),observed(:,lane),n,3)
end do
!$omp end parallel do
if (any(result/=expected) .or. any(observed/=ref_observed)) stop 26
if (test_calls/=1) stop 27
call original(expected(:,1),ref_input(:,1),ref_observed(:,1),n,3)
!$omp parallel if(.false.)
call candidate(result(:,1),input(:,1),observed(:,1),n,3)
!$omp end parallel
if (any(result/=expected) .or. any(observed/=ref_observed)) stop 28
if (test_calls/=1) stop 29
print *, 'VERIFIED_COLLECTIVE'
end program
"""


@unittest.skipUnless(HAS_FPARSER and shutil.which("gfortran"), "requires fparser and GNU Fortran/OpenMP")
class GPUInstrumentationIntegrationTests(unittest.TestCase):
    def compile_and_run(self, source, driver, *, coverage, delay_kernel=False):
        with tempfile.TemporaryDirectory(prefix="gpu-instrumentation-") as directory:
            root = Path(directory)
            native = root / "subject.f90"
            native.write_text(source)
            (root / "reference.f90").write_text(source.replace("module subject", "module reference"))
            discovered = discover_sources([str(native)], str(root), str(root / "generated"))
            regions = [region for region in discovered["regions"] if region["gpu_scope"]["kind"] == "strip_mined"
                       and any(symbol["name"].lower() == "input" for symbol in region["symbols"])]
            self.assertEqual(len(regions), 1, discovered["rejected"])
            region = regions[0]
            changed = patch_regions(root, regions, coverage=coverage)
            self.assertEqual(changed, ["subject.f90"])
            patched = native.read_text()
            self.assertIn("=> omp_get_level", patched)
            self.assertIn(region_call(region, coverage=coverage).strip(), patched)
            # Instrument only the independent generated native kernel. The
            # application still traverses the actual public patching path.
            kernel = region["module_source"].replace(
                "  implicit none", "  use omp_lib, only: omp_get_level\n  implicit none\n"
                "  integer :: test_calls=0, test_level=-1", 1)
            record = ("    !$omp critical(test_kernel_record)\n"
                      "    test_calls=test_calls+1\n    test_level=omp_get_level()\n"
                      "    !$omp end critical(test_kernel_record)\n")
            if delay_kernel:
                record += "    call delay_generated_kernel()\n"
                delay = ("  subroutine delay_generated_kernel()\n"
                         "    integer :: begin,now,rate\n    call system_clock(begin,rate)\n"
                         "    do\n      call system_clock(now)\n"
                         "      if (real(now-begin)/rate>=0.025) exit\n"
                         "    end do\n  end subroutine\n")
                kernel = kernel.replace("end module ", delay + "end module ", 1)
            kernel = re.sub(r"(?m)^    do ", lambda match: record + match[0], kernel, count=1)
            generated = Path(region["input_path"])
            generated.write_text(kernel)
            driver_path = root / "driver.f90"
            driver_path.write_text(driver.replace("@KERNEL@", region["generated_module"]))
            executable = root / "verify"
            compiled = subprocess.run(
                ["gfortran", "-O0", "-fopenmp", "-ffree-line-length-none", "-fcheck=all",
                 str(generated), str(root / "reference.f90"), str(native), str(driver_path),
                 "-o", str(executable)], cwd=root, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            run = subprocess.run([str(executable)], cwd=root, capture_output=True, text=True,
                                 env={**os.environ, "OMP_NUM_THREADS": "3", "OMP_DYNAMIC": "FALSE"}, timeout=20)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            evidence = execution_coverage(run.stdout, regions, enabled=coverage)
            self.assertEqual(evidence["enabled"], coverage)
            self.assertEqual(evidence["executed_regions"], [region["id"]] if coverage else [])
            self.assertEqual(run.stdout.count("ELMM_PIPELINE_REGION "), 1 if coverage else 0)
            return run.stdout, patched, region

    def test_serial_dispatch_and_dynamic_caller_fallback(self):
        for coverage in (False, True):
            with self.subTest(coverage=coverage):
                stdout, patched, region = self.compile_and_run(SERIAL, SERIAL_DRIVER, coverage=coverage)
                self.assertIn("VERIFIED_SERIAL", stdout)
                self.assertEqual(region["gpu_scope"]["dispatch"], "serial")
                self.assertIn("() == 0", patched)

    def test_collective_sync_single_dispatch_and_native_fallback(self):
        for coverage in (False, True):
            with self.subTest(coverage=coverage):
                stdout, patched, region = self.compile_and_run(
                    COLLECTIVE, COLLECTIVE_DRIVER, coverage=coverage, delay_kernel=True)
                self.assertIn("VERIFIED_COLLECTIVE", stdout)
                self.assertEqual(region["gpu_scope"]["dispatch"], "collective_single")
                self.assertIn("() == 1", patched)
                self.assertIn("!$omp barrier\n!$omp single", patched)
                self.assertIn("!$omp end single", patched)
                self.assertIn("!$omp end do nowait", region["original_text"])


if __name__ == "__main__":
    unittest.main()
