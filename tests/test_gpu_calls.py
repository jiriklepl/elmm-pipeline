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
