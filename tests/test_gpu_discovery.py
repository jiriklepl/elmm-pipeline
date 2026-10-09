"""GPU lifting proves complete domains independently of application names."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

HAS_FPARSER = importlib.util.find_spec("fparser") is not None
if HAS_FPARSER:
    from elmm_pipeline.gpu_discovery import discover_sources


def tiled(rank=2, *, parallel="combined", clause="", step="tile_size"):
    points = [f"axis{d}" for d in range(rank)]
    blocks = [f"block{d}" for d in range(rank)]
    indices = ",".join(points)
    shape = ",".join("-4:" for _ in points)
    private = ",".join(points + blocks + ["temporary"])
    header = {"combined": f"!$omp parallel do private({private}) collapse({rank}) {clause}",
              "collective": f"!$omp parallel private({private}) {clause}\n!$omp do collapse({rank})",
              "serial": "", "orphan": f"!$omp do private({private}) {clause}"}[parallel]
    footer = {"combined": "!$omp end parallel do", "collective": "!$omp end do\n!$omp end parallel", "serial": "", "orphan": "!$omp end do"}[parallel]
    loops = [f"do {block} = low, high, {step}" for block in blocks]
    loops += [f"do {point} = {block}, min(high, {block}+({step})-1)" for point, block in zip(points, blocks)]
    return f"""module unrelated
use iso_fortran_env, only: wp=>real64
implicit none
contains
subroutine work(output, input, low, high, tile_size, factor)
real(wp), intent(inout) :: output({shape})
real(wp), intent(in) :: input({shape})
real(wp), intent(in) :: factor
integer, intent(in) :: low, high, tile_size
integer :: {','.join(points+blocks)}
real(wp) :: temporary
{header}
{chr(10).join(loops)}
temporary = input({indices})*factor + ({'+'.join(points)})*0.125_wp
output({indices}) = temporary
{chr(10).join('end do' for _ in loops)}
{footer}
end subroutine
end module
"""


@unittest.skipUnless(HAS_FPARSER and shutil.which("gfortran"), "requires fparser and gfortran")
class GPUDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gpu-discovery-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def discover(self, source):
        path = self.root / "source.f90"
        path.write_text(source)
        return discover_sources([str(path)], str(self.root), str(self.root / "generated"))

    def test_arbitrary_rank_and_names_full_domain(self):
        for rank in (1, 2, 4):
            with self.subTest(rank=rank):
                result = self.discover(tiled(rank))
                self.assertEqual(len(result["regions"]), 1, result["rejected"])
                region = result["regions"][0]
                self.assertEqual(region["loop_ranks"], [rank])
                self.assertEqual(region["gpu_scope"]["original_loop_rank"], rank*2)
                self.assertEqual(region["gpu_scope"]["dispatch"], "serial")
                self.assertTrue(region["original_text"].lstrip().startswith("!$omp parallel do"))
                self.assertTrue(region["original_text"].rstrip().endswith("!$omp end parallel do"))
                self.assertNotIn("block0", region["module_source"])
                self.assertNotIn("tile_size", region["actual_args"])
                self.assertIn("real(kind=8)", region["module_source"])
                self.assertFalse(region["annotated"])
        original = self.discover(tiled(3))
        renamed = tiled(3)
        for old, new in {"work":"advance", "output":"field", "input":"oldfield", "axis0":"i", "axis1":"j", "axis2":"k", "block0":"t", "block1":"u", "block2":"v", "tile_size":"slab", "factor":"coefficient", "wp":"precision"}.items():
            renamed = re.sub(r"(?<![A-Za-z0-9])"+old+r"\b", new, renamed)
        other = self.discover(renamed)
        self.assertEqual([x["loop_ranks"] for x in other["regions"]], [x["loop_ranks"] for x in original["regions"]])

    def test_collective_scope_consumes_only_worksharing(self):
        result = self.discover(tiled(2, parallel="collective").replace("!$omp end do", "!$omp end do nowait"))
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertEqual(region["gpu_scope"]["dispatch"], "collective_single")
        self.assertEqual(region["gpu_scope"]["expected_omp_level"], 1)
        self.assertTrue(region["original_text"].lstrip().startswith("!$omp do"))
        self.assertNotIn("!$omp parallel", region["original_text"])
        self.assertNotIn("!$omp end parallel", region["original_text"])
        self.assertIn("!$omp end do nowait", region["original_text"])
        self.assertEqual(region["gpu_scope"]["barriers"], "before_and_after")

    def test_later_conditional_loop_definitions_do_not_hide_real_liveouts(self):
        source = """module unrelated
implicit none
contains
subroutine work(a,n,flag)
integer, intent(in) :: n,flag
real, intent(inout) :: a(:,:)
integer :: row,col
do row=1,n
 do col=1,n
  a(col,row)=2*a(col,row)
 end do
end do
if (flag>0) then
 do row=1,n
  do col=1,n
   a(col,row)=3*a(col,row)
  end do
 end do
endif
end subroutine
end module
"""
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 2, result["rejected"])
        # Neither a conditional loop nor a possibly empty outer loop can
        # globally kill the old COL value at a read after its loop body.
        for tail in ("print *, col", "if (flag<0) print *, col"):
            changed = source.replace("end subroutine", tail + "\nend subroutine")
            self.assertFalse(self.discover(changed)["regions"])
        # The later loop's own lower limit is evaluated before its iterator
        # definition; that genuine read of the previous COL stays native.
        changed = source.replace("  do col=1,n", "  do col=col,n")
        self.assertFalse(self.discover(changed)["regions"])

    def test_reject_unproved_parallel_capture_and_contexts(self):
        cases = {
            "private array": tiled(clause="private(input)"),
            "firstprivate scalar": tiled(clause="firstprivate(factor)"),
            "private stride": tiled(clause="firstprivate(tile_size)"),
            "private bound": tiled(clause="firstprivate(high)"),
            "enclosing private": tiled(parallel="collective", clause="firstprivate(factor)"),
            "orphan": tiled(parallel="orphan"),
            "reduction": tiled(clause="reduction(+:factor)"),
            "nested team": tiled(parallel="collective").replace("!$omp do collapse(2)", "!$omp parallel do collapse(2)").replace("!$omp end do", "!$omp end parallel do"),
            "threadprivate": tiled().replace("contains", "real(wp) :: hidden\n!$omp threadprivate(hidden)\ncontains").replace("*factor", "*hidden"),
            "unshared parallel": tiled(parallel="collective").replace("!$omp do collapse(2)", "").replace("!$omp end do", ""),
            "foreign directive": tiled(parallel="serial").replace("do block0", "!$acc parallel loop\ndo block0"),
        }
        for label, source in cases.items():
            with self.subTest(label=label):
                result = self.discover(source)
                self.assertFalse(result["regions"], result)
                self.assertTrue(result["rejected"])

    def test_reject_partial_tiles_and_never_extract_descendants(self):
        source = tiled()
        cases = [source.replace("block0+(tile_size)-1", "block0+(tile_size)"),
                 source.replace("output(axis0,axis1) = temporary", "output(axis0,axis1) = temporary + block0"),
                 source.replace("end subroutine", "print *, block0\nend subroutine"),
                 source.replace("do axis0", "factor = 2\ndo axis0"),
                 source.replace("output(axis0,axis1) = temporary", "output(axis0,axis1) = output(axis0-1,axis1)+temporary")]
        for changed in cases:
            with self.subTest(source=changed):
                self.assertFalse(self.discover(changed)["regions"])

    def test_rectangular_parallel_nest(self):
        source = tiled(1).replace("do block0 = low, high, tile_size\n", "").replace("do axis0 = block0, min(high, block0+(tile_size)-1)", "do axis0 = low, high").replace("end do\nend do", "end do")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        self.assertEqual(result["regions"][0]["gpu_scope"]["kind"], "rectangular")

    def test_native_equivalence_negative_bounds_and_guard_fallback(self):
        for parallel in ("serial", "combined", "collective"):
            with self.subTest(parallel=parallel):
                source = tiled(2, parallel=parallel)
                if parallel == "collective":
                    source = source.replace("!$omp end do", "!$omp end do nowait")
                result = self.discover(source)
                self.assertEqual(len(result["regions"]), 1, result["rejected"])
                region = result["regions"][0]
                scope = region["gpu_scope"]
                # Render just the documented scope contract, independently of
                # GPU availability; the normalized kernel is native Fortran.
                call = region["call"]
                if scope["dispatch"] == "collective_single":
                    call = "!$omp barrier\n!$omp single\n" + call + "\n!$omp end single"
                replacement = f"if ({scope['guard']} .and. omp_get_level()=={scope['expected_omp_level']}) then\n{call}\nelse\n{region['original_text']}endif\n"
                lines = source.splitlines(keepends=True)
                lines[region["start_line"]-1:region["end_line"]] = [replacement]
                lines.insert(region["insertion_line"]-1, region["use"]+"\nuse omp_lib, only: omp_get_level\n")
                (self.root/"adapted.f90").write_text("".join(lines).replace("module unrelated", "module changed"))
                (self.root/"reference.f90").write_text(source)
                (self.root/"driver.f90").write_text("""program check
use unrelated, only: reference=>work
use changed, only: candidate=>work
use iso_fortran_env, only: real64
implicit none
real(real64) :: a(-4:8,-4:8), b(-4:8,-4:8), input(-4:8,-4:8)
integer :: i,j,step,test,lower,upper
input=3.125_real64
do test=1,5
 select case(test)
 case(1)
  step=3;lower=-3;upper=7
 case(2)
  step=20;lower=-3;upper=7
 case(3)
  step=-2;lower=-3;upper=7
 case(4)
  step=-2;lower=7;upper=-3
 case(5)
  step=3;lower=7;upper=-3
 end select
 a=-777; b=a
 call reference(a,input,lower,upper,step,2.0_real64)
 call candidate(b,input,lower,upper,step,2.0_real64)
 if (any(a /= b)) stop 9
end do
end program
""")
                executable=self.root/"check"
                compiled=subprocess.run(["gfortran","-fopenmp","-ffree-line-length-none","-fcheck=all",region["input_path"],str(self.root/"reference.f90"),str(self.root/"adapted.f90"),str(self.root/"driver.f90"),"-o",str(executable)],cwd=self.root,capture_output=True,text=True)
                self.assertEqual(compiled.returncode,0,compiled.stderr)
                run=subprocess.run([str(executable)],cwd=self.root,capture_output=True,text=True,env={**os.environ,"OMP_NUM_THREADS":"3"},timeout=15)
                self.assertEqual(run.returncode,0,run.stdout+run.stderr)


if __name__ == "__main__":
    unittest.main()
