"""Compiler-checked outlining preserves native storage and execution order."""
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

AVAILABLE = all(importlib.util.find_spec(name) is not None for name in ("fparser", "islpy"))
if AVAILABLE:
    from elmm_pipeline.gpu_discovery import discover_sources

COMPILER = Path(os.environ.get("ELMM_COMPILER_ROOT", Path(__file__).resolve().parents[2]))


def subject(body, *, declarations="", header="a,b,n,enabled", array_attributes="intent(inout)"):
    return f"""module subject
implicit none
contains
subroutine advance({header})
real(kind=8), {array_attributes} :: a(-2:),b(-2:)
integer, intent(in) :: n
logical, intent(in) :: enabled
integer :: cell,other,tile
logical :: keep
real(kind=8) :: weight
{declarations}
{body}
end subroutine
end module
"""


@unittest.skipUnless(AVAILABLE and shutil.which("gfortran") and (COMPILER/"compiler/__main__.py").is_file(),
                     "requires explicit local compiler, fparser, islpy and GNU Fortran")
class LocalGPUDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gpu-local-discovery-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def discover(self, source, *, checked=True):
        self.source = self.root/"subject.f90"
        self.source.write_text(source)
        options = {"compiler_root": str(COMPILER), "compiler_python": sys.executable} if checked else {}
        return discover_sources([str(self.source)], str(self.root), str(self.root/"generated"), **options)

    def test_ordered_raw_war_waw_groups_use_authoritative_compiler(self):
        cases = [("a(cell)=a(cell)+1", "b(cell)=a(cell-1)+2"),
                 ("b(cell)=a(cell-1)+1", "a(cell)=a(cell)+2"),
                 ("a(cell)=b(cell)+1", "a(cell+1)=b(cell)+2")]
        for first, second in cases:
            with self.subTest(first=first):
                source = subject(f"do cell=0,n\n{first}\nend do\ndo cell=0,n\n{second}\nend do")
                conservative = self.discover(source, checked=False)
                self.assertEqual([r["loop_nests"] for r in conservative["regions"]], [1, 1])
                result = self.discover(source)
                self.assertEqual([r["loop_nests"] for r in result["regions"]], [2], result["rejected"])
                self.assertTrue(result["regions"][0]["compiler_check"]["supported"])

    def test_compiler_rejection_retains_eligible_siblings(self):
        source = subject("""do cell=0,n
 b(cell)=a(cell)+1
end do
do cell=0,n
 a(cell)=a(cell-1)+1
end do
do cell=0,n
 b(cell)=a(cell)+2
end do""")
        result = self.discover(source)
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])
        self.assertTrue(any("selected compiler" in item["reason"] for item in result["rejected"]))

    def test_logical_temporaries_flags_and_intrinsic_keywords(self):
        source = subject("""do cell=0,min(n,size(a,dim=1)-4)
 keep=enabled
 weight=real(kind=8,a=cell)
 if (keep) then
  a(cell)=b(cell)+weight
 else if (b(cell)>0) then
  a(cell)=b(cell)-weight
 else
  a(cell)=0.0_8
 end if
end do""")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertIn("logical, intent(in) :: enabled", region["module_source"])
        self.assertIn("logical :: keep", region["module_source"])
        self.assertNotIn("dim", region["actual_args"])
        self.assertNotIn("kind", region["actual_args"])
        self.assertTrue(all("SIZE(" not in arg.upper() for arg in region["actual_args"]))

    def test_type_only_inquiry_does_not_capture_an_undefined_scalar(self):
        source = subject("do cell=0,n\na(cell)=b(cell)+tiny(weight)\nend do")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertNotIn("weight", region["actual_args"])
        self.assertIn("real(kind=8) :: weight", region["module_source"])
        source = source.replace("do cell=0,n", "weight=0.25_8\ndo cell=0,n").replace("tiny(weight)", "tiny(weight)+weight")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        self.assertIn("weight", result["regions"][0]["actual_args"])

    def test_explicit_default_integer_and_logical_kinds(self):
        source = subject("do cell=0_4,n\nif (enabled.eqv..true._4) a(cell)=b(cell)+1_4\nend do")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        self.assertNotIn("_4", result["regions"][0]["module_source"].split("DO cell", 1)[1])

    def test_storage_scope_and_liveness_restrictions_remain(self):
        base = "do cell=0,n\na(cell)=b(cell)+1\nend do"
        sources = [subject(base, array_attributes="pointer"),
                   subject(base, array_attributes="target,intent(inout)"),
                   subject(base.replace("b(cell)", "scratch(cell)"), declarations="real(8) :: scratch(-2:50)"),
                   subject("!$omp parallel private(b,cell)\n!$omp do\n"+base+"\n!$omp end do\n!$omp end parallel"),
                   subject(base+"\nb(0)=cell"),
                   subject(base.replace("a(cell)=", "call opaque()\na(cell)="))]
        for source in sources:
            with self.subTest(source=source):
                result = self.discover(source)
                self.assertFalse(result["regions"], result)
                self.assertTrue(result["rejected"])

    def test_calls_and_barriers_remain_group_boundaries(self):
        for gap in ("call opaque()", "!$omp barrier"):
            with self.subTest(gap=gap):
                source = subject(f"""!$omp parallel private(cell)
!$omp do
do cell=0,n
 a(cell)=b(cell)+1
end do
!$omp end do
{gap}
!$omp do
do cell=0,n
 b(cell)=a(cell-1)+2
end do
!$omp end do
!$omp end parallel""")
                result = self.discover(source)
                self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])
                self.assertTrue(all(gap not in region["original_text"] for region in result["regions"]))

    def test_lifted_tile_final_value_cannot_disappear(self):
        source = subject("""tile=2
do tile=1,4,2
 do cell=tile,min(tile+2-1,4)
  a(cell)=a(cell)+1
 end do
end do
do cell=1,tile-1
 b(cell)=b(cell)+1
end do""")
        result = self.discover(source)
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [1], result["rejected"])
        self.assertNotIn("do tile=", result["regions"][0]["original_text"])
        self.assertTrue(any("tile iterator remains live" in item["reason"] for item in result["rejected"]))

    def test_allocatable_bound_only_capture_and_intrinsic_binding(self):
        source = """module subject
implicit none
real(8), allocatable :: a(:),bounds(:)
contains
subroutine advance(n)
integer :: cell,n
do cell=1,n
 a(cell)=real(size(bounds,dim=1),8)
end do
end subroutine
end module
"""
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        self.assertEqual(result["regions"][0]["gpu_scope"]["allocated_arrays"], ["a", "bounds"])
        result = self.discover(source.replace("integer :: cell,n", "integer :: cell,n,allocated"))
        self.assertFalse(result["regions"])
        self.assertTrue(any("ALLOCATED intrinsic binding" in item["reason"] for item in result["rejected"]))
        invalid = subject("do cell=0,n\nif(allocated(a)) a(cell)=b(cell)\nend do")
        result = self.discover(invalid)
        self.assertFalse(result["regions"])
        self.assertTrue(any("ALLOCATED inquiry requires" in item["reason"] for item in result["rejected"]))

    def test_compiled_allocatables_bounds_and_imperfect_conditionals(self):
        source = """module subject
implicit none
real(8), allocatable :: field(:,:),output(:,:)
contains
subroutine advance(last,enabled)
integer, intent(in) :: last
logical, intent(in) :: enabled
integer :: row,column
real(8) :: weight
logical :: keep
!$omp parallel private(row,column,weight,keep)
!$omp do
do row=-1,last
 weight=real(row,kind=8)+0.25_8
 keep=enabled
 do column=lbound(field,dim=1)+1,ubound(array=field,dim=1)-1
  if (keep) then
   output(column,row)=field(column,row)+weight
  else
   output(column,row)=field(column,row)-weight
  end if
 end do
end do
!$omp end do
!$omp end parallel
end subroutine
end module
"""
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertEqual(region["gpu_scope"]["allocated_arrays"], ["field", "output"])
        self.assertTrue(all("UBOUND" not in value.upper() and "SIZE" not in value.upper() for value in region["actual_args"]))
        (self.root/"reference.f90").write_text(source.replace("module subject", "module reference"))
        patch_regions(self.root, [region], coverage=False)
        driver = self.root/"driver.f90"
        driver.write_text("""program verify
use subject, only: candidate=>advance,cf=>field,co=>output
use reference, only: original=>advance,rf=>field,ro=>output
implicit none
integer :: shape,iteration
call original(-2,.true.)
call candidate(-2,.true.)
do shape=1,2
 allocate(cf(-3:5+shape,-2:3),co(-3:5+shape,-2:3))
 allocate(rf(-3:5+shape,-2:3),ro(-3:5+shape,-2:3))
 do iteration=1,2
  cf=real(iteration,8)*2.5_8
  rf=cf
  co=-30
  ro=co
  call original(2,iteration==1)
  call candidate(2,iteration==1)
  if (any(cf/=rf).or.any(co/=ro)) stop 1
 end do
 deallocate(cf,co)
 deallocate(rf,ro)
end do
end program
""")
        binary = self.root/"verify"
        built = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                                region["input_path"], str(self.root/"reference.f90"), str(self.source), str(driver), "-o", str(binary)],
                               cwd=self.root, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout+built.stderr)
        run = subprocess.run([str(binary)], cwd=self.root, capture_output=True, text=True, timeout=20,
                             env={**os.environ, "OMP_NUM_THREADS": "4"})
        self.assertEqual(run.returncode, 0, run.stdout+run.stderr)


if __name__ == "__main__":
    unittest.main()
