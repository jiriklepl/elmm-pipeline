"""Call grouping uses Fortran bindings, with no application-name knowledge."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from elmm_pipeline.adaptation import patch_regions

AVAILABLE = all(importlib.util.find_spec(name) for name in ("fparser", "islpy"))
if AVAILABLE:
    from elmm_pipeline.gpu_discovery import discover_sources

COMPILER = Path(os.environ.get("ELMM_COMPILER_ROOT", Path(__file__).resolve().parents[2]))

SOURCE = """module subject
implicit none
real(8) :: coefficient = 2
contains
subroutine advance(a,b,n)
real(8), contiguous :: a(-3:),b(-3:)
integer, intent(in) :: n
call prepare(a,b,n)
call finish(a,b,n)
end subroutine
subroutine prepare(x,y,n)
real(8), intent(in) :: x(-2:)
real(8), intent(inout) :: y(-2:)
integer, intent(in) :: n
integer :: i
real(8) :: scale
scale = coefficient
!$omp parallel do private(i)
do i=-1,n
y(i)=x(i)+scale+real(i,8)
end do
!$omp end parallel do
end subroutine
subroutine finish(x,y,n)
real(8), intent(inout) :: x(-2:)
real(8), intent(in) :: y(-2:)
integer, intent(in) :: n
integer :: i
do i=0,min(n,size(x)-4)
x(i)=y(i-1)+3
end do
end subroutine
end module
"""


@unittest.skipUnless(AVAILABLE and shutil.which("gfortran") and (COMPILER/"compiler/__main__.py").is_file(),
                     "requires selected compiler, fparser, islpy and GNU Fortran")
class GPUCallTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gpu-calls-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root/"subject.f90"

    def discover(self, text=SOURCE, *, compiler=COMPILER, select=("advance",)):
        self.source.write_text(text)
        return discover_sources([str(self.source)], str(self.root), str(self.root/"generated"),
                                select=list(select), compiler_root=str(compiler), compiler_python=sys.executable)

    def test_calls_are_retained_and_checked_by_independent_compiler(self):
        checkout = self.root/"separate-compiler"
        shutil.copytree(COMPILER/"compiler", checkout/"compiler", ignore=shutil.ignore_patterns("__pycache__"))
        result = self.discover(compiler=checkout)
        region, = result["regions"]
        self.assertEqual(result["rejected"], [])
        self.assertEqual(region["gpu_scope"]["grouped_calls"], 2)
        self.assertEqual(region["module_source"].lower().count("    call outline_"), 2)
        self.assertEqual(region["args"].count("a"), 1)
        self.assertEqual(region["args"].count("coefficient"), 1)
        self.assertTrue(region["compiler_check"]["supported"])
        renamed = SOURCE.replace("prepare", "diffuse").replace("finish", "advect").replace("advance", "evaluate")
        other, = self.discover(renamed, select=("evaluate",))["regions"]
        self.assertEqual(other["gpu_scope"]["grouped_calls"], 2)
        self.assertEqual(other["symbols"], region["symbols"])

    def test_native_agreement_halos_negative_bounds_index_data_and_changing_shapes(self):
        result = self.discover()
        region, = result["regions"]
        (self.root/"reference.f90").write_text(SOURCE.replace("module subject", "module reference"))
        patch_regions(self.root, [region])
        driver = self.root/"main.f90"
        driver.write_text("""program check
use subject
use reference, only: native => advance
implicit none
real(8), allocatable :: a(:),b(:),x(:),y(:)
integer :: extent,i
do extent=3,17,7
 allocate(a(-3:extent+3),b(-3:extent+3),x(-3:extent+3),y(-3:extent+3))
 do i=-3,extent+3
  a(i)=real(i*i,8)
  b(i)=-123
 end do
 x=a
 y=b
 call native(x,y,extent)
 call advance(a,b,extent)
 if (any(a/=x).or.any(b/=y)) stop 1
 deallocate(a,b,x,y)
end do
end program
""")
        binary = self.root/"check"
        built = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                                region["input_path"], str(self.root/"reference.f90"), str(self.source), str(driver),
                                "-o", str(binary)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
        run = subprocess.run([str(binary)], cwd=self.root, capture_output=True, text=True, timeout=20,
                             env={**os.environ, "OMP_NUM_THREADS": "4"})
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)

    def test_opaque_calls_and_control_flow_remain_boundaries(self):
        for gap in ("call opaque()", "nonsense=1", "!$omp barrier", "if (n>0) then"):
            with self.subTest(gap=gap):
                source = SOURCE.replace("call finish(a,b,n)", gap+"\ncall finish(a,b,n)" +
                                        ("\nend if" if gap.startswith("if") else ""))
                self.assertEqual(self.discover(source)["regions"], [])
        for begin, end in (("block\nreal(8) :: a(8),b(8)", "end block"),
                           ("associate(a=>b)", "end associate")):
            with self.subTest(scope=begin):
                source = SOURCE.replace("call prepare(a,b,n)", begin+"\ncall prepare(a,b,n)")
                source = source.replace("call finish(a,b,n)", "call finish(a,b,n)\n"+end)
                self.assertEqual(self.discover(source)["regions"], [])

    def test_unsafe_callees_are_rejected_and_other_loops_survive(self):
        changes = [
            ("scale = coefficient", "coefficient = 3"),
            ("scale = coefficient", "call prepare(x,y,n)"),
            ("real(8), intent(in) :: x(-2:)", "real(8), optional, intent(in) :: x(-2:)"),
            ("!$omp parallel do private(i)", "!$omp parallel do firstprivate(scale) private(i)"),
            ("call prepare(a,b,n)", "call prepare(a,a,n)"),
            ("call prepare(a,b,n)", "call prepare(a(:),b,n)"),
        ]
        for before, after in changes:
            with self.subTest(after=after):
                result = self.discover(SOURCE.replace(before, after))
                # Private scalar preparation is internal and safely retained;
                # firstprivate copied before the loop has the same value.
                if "firstprivate(scale)" in after:
                    self.assertEqual(len(result["regions"]), 1, result["rejected"])
                else:
                    self.assertEqual(result["regions"], [], result)
        source = SOURCE.replace("call finish(a,b,n)", "call finish(a,b,n)\ndo j=0,n\nb(j)=a(j)+1\nend do")
        source = source.replace("integer, intent(in) :: n", "integer, intent(in) :: n\ninteger :: j", 1)
        source = source.replace("scale = coefficient", "call unknown()")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        self.assertNotIn("call ", result["regions"][0]["original_text"])

    def test_import_rename_and_hidden_state_visibility(self):
        helper = SOURCE.replace("module subject", "module implementation")
        (self.root/"implementation.f90").write_text(helper)
        caller = """module subject
use implementation, only: p=>prepare,f=>finish,c=>coefficient
contains
subroutine advance(a,b,n)
real(8),contiguous :: a(:),b(:)
integer :: n
call p(a,b,n)
call f(a,b,n)
end subroutine
end module
"""
        result = self.discover(caller)
        region, = result["regions"]
        self.assertIn("c", region["args"])
        self.assertEqual(self.discover(caller.replace(",c=>coefficient", ""))["regions"], [])
        wildcard = caller.replace("use implementation, only: p=>prepare,f=>finish,c=>coefficient", "use implementation")
        wildcard = wildcard.replace("call p(", "call prepare(").replace("call f(", "call finish(")
        self.assertEqual(len(self.discover(wildcard)["regions"]), 1)
        for private in ("private :: coefficient", "private\npublic :: prepare,finish"):
            (self.root/"implementation.f90").write_text(helper.replace("implicit none", "implicit none\n"+private))
            self.assertEqual(self.discover(wildcard)["regions"], [])
        # Private module procedures cannot be mistaken for an external binding
        # with the same spelling in a USE-associated caller.
        (self.root/"implementation.f90").write_text(helper.replace("implicit none", "implicit none\nprivate :: prepare"))
        self.assertEqual(self.discover(wildcard)["regions"], [])

    def test_lifted_helpers_preserve_setup_negative_strides_and_empty_ranges(self):
        source = SOURCE.replace("real(8) :: coefficient = 2", "real(8) :: coefficient = 2\ninteger :: stride=3")
        source = source.replace("integer :: i", "integer :: i,tile,step")
        source = source.replace("scale = coefficient", "scale = coefficient\nstep=stride")
        source = source.replace("!$omp parallel do private(i)", "!$omp parallel do private(i,tile)")
        source = source.replace("do i=-1,n", "do tile=-1,n,step\ndo i=tile,min(tile+step-1,n)")
        source = source.replace("!$omp end parallel do", "end do\n!$omp end parallel do")
        result = self.discover(source)
        region, = result["regions"]
        self.assertIn("IF ((step) > 0) THEN", region["module_source"])
        self.assertNotIn("step", region["args"])
        (self.root/"reference.f90").write_text(source.replace("module subject", "module reference"))
        patch_regions(self.root, [region])
        driver = self.root/"main.f90"
        driver.write_text("""program check
use subject
use reference, only: native=>advance,other_stride=>stride
implicit none
real(8) :: a(-3:17),b(-3:17),x(-3:17),y(-3:17)
integer :: step,n
do step=-3,7,2
 do n=-4,14,3
  stride=step
  other_stride=step
  a=2
  b=-17
  x=a
  y=b
  call native(x,y,n)
  call advance(a,b,n)
  if(any(a/=x).or.any(b/=y)) stop 1
 end do
end do
end program
""")
        self.compile_and_run(region, driver)
        # A lifted-away iterator cannot supply a later value.
        source = source.replace("!$omp end parallel do", "!$omp end parallel do\ny(0)=tile")
        self.assertEqual(self.discover(source)["regions"], [])

    def test_unallocated_and_empty_captures_keep_native_fallback(self):
        source = """module subject
implicit none
real(8),allocatable :: a(:),b(:)
contains
subroutine advance(n)
integer :: n
call first(n)
call second(n)
end subroutine
subroutine first(n)
integer :: n,i
do i=1,n
b(i)=a(i)+1
end do
end subroutine
subroutine second(n)
integer :: n,i
do i=1,n
a(i)=b(i)+2
end do
end subroutine
end module
"""
        region, = self.discover(source)["regions"]
        self.assertEqual(region["gpu_scope"]["allocated_arrays"], ["a", "b"])
        (self.root/"reference.f90").write_text(source.replace("module subject", "module reference"))
        patch_regions(self.root, [region])
        driver = self.root/"main.f90"
        driver.write_text("""program check
use subject
use reference, only: native=>advance,x=>a,y=>b
implicit none
integer :: n
call advance(0)
call native(0)
do n=0,3
 allocate(a(n),b(n))
 allocate(x(n),y(n))
 a=2
 b=-17
 x=a
 y=b
 call advance(n)
 call native(n)
 if(any(a/=x).or.any(b/=y)) stop 1
 deallocate(a,b)
 deallocate(x,y)
end do
end program
""")
        self.compile_and_run(region, driver)

    def compile_and_run(self, region, driver):
        binary = self.root/"check"
        built = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                                region["input_path"], str(self.root/"reference.f90"), str(self.source), str(driver),
                                "-o", str(binary)], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
        run = subprocess.run([str(binary)], cwd=self.root, capture_output=True, text=True, timeout=20,
                             env={**os.environ, "OMP_NUM_THREADS": "4"})
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)


if __name__ == "__main__":
    unittest.main()
