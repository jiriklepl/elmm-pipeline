"""Adjacent GPU entries share captures without crossing native boundaries."""
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

HAS_FPARSER = importlib.util.find_spec("fparser") is not None
if HAS_FPARSER:
    from elmm_pipeline.gpu_discovery import discover_sources


def adjacent(rank=2, *, parallel="collective", tiled=(False, False), gap=""):
    points = [f"coordinate{axis}" for axis in range(rank)]
    tiles = [f"tile{axis}" for axis in range(rank)]
    subscripts = ",".join(points)
    shape = ",".join("-3:" for _ in points)
    blocks = []
    for index, output in enumerate(("first", "second")):
        upper = "high" if index == 0 else "high-1"
        loops = []
        if tiled[index]:
            loops.extend(f"do {tile}=low,{upper},stride" for tile in tiles)
            loops.extend(f"do {point}={tile},min({tile}+stride-1,{upper})" for point, tile in zip(points, tiles))
        else:
            loops.extend(f"do {point}=low,{upper}" for point in points)
        body = f"{output}({subscripts})={output}({subscripts})+{index+2}.0_8*input({subscripts})"
        begin = "!$omp do\n" if parallel == "collective" else ""
        end = "\n!$omp end do" if parallel == "collective" else ""
        blocks.append(begin + "\n".join(loops + [body] + ["end do"]*len(loops)) + end)
    private = ",".join(points + tiles)
    begin = f"!$omp parallel private({private})\n" if parallel == "collective" else ""
    end = "\n!$omp end parallel" if parallel == "collective" else ""
    return f"""module subject
implicit none
contains
subroutine advance(first,second,input,low,high,stride)
real(kind=8), intent(inout), contiguous :: first({shape}),second({shape})
real(kind=8), intent(in), contiguous :: input({shape})
integer, intent(in) :: low,high,stride
integer :: {private}
{begin}{blocks[0]}
{gap}
{blocks[1]}{end}
end subroutine
subroutine native_boundary()
end subroutine
end module
"""


@unittest.skipUnless(HAS_FPARSER and shutil.which("gfortran"), "requires fparser and GNU Fortran")
class GPUGroupingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gpu-grouping-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def discover(self, source):
        self.path = self.root / "subject.f90"
        self.path.write_text(source)
        return discover_sources([str(self.path)], str(self.root), str(self.root / "generated"))

    def test_unrelated_ranks_and_names_share_one_capture(self):
        for rank in (1, 2, 4):
            for tiled in ((False, False), (False, True), (True, True)):
                with self.subTest(rank=rank, tiled=tiled):
                    source = adjacent(rank, tiled=tiled)
                    result = self.discover(source)
                    self.assertEqual(len(result["regions"]), 1, result["rejected"])
                    region = result["regions"][0]
                    self.assertEqual(region["loop_nests"], 2)
                    self.assertEqual(region["loop_ranks"], [rank, rank])
                    self.assertEqual(region["args"].count("input"), 1)
                    self.assertEqual(region["actual_args"].count("input"), 1)
                    self.assertEqual(len(region["gpu_scope"]["member_spans"]), 2)
                    self.assertEqual(region["gpu_scope"]["grouped_nests"], 2)
                    self.assertEqual(region["original_text"].count("!$omp do"), 2)
                    self.assertNotIn("!$omp parallel", region["original_text"])
        source = adjacent(3)
        for old, new in {"advance": "evaluate", "first": "density", "second": "flux", "input": "forcing", "coordinate0": "longitude", "coordinate1": "latitude", "coordinate2": "altitude"}.items():
            source = re.sub(r"\b" + old + r"\b", new, source)
        renamed = self.discover(source)
        self.assertEqual([r["loop_ranks"] for r in renamed["regions"]], [[3, 3]])

    def test_calls_assignments_barriers_and_conditionals_remain_boundaries(self):
        for gap in ("call native_boundary()", "!$omp barrier"):
            with self.subTest(gap=gap):
                result = self.discover(adjacent(gap=gap))
                self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])
                self.assertTrue(all(gap not in r["original_text"] for r in result["regions"]))
        # A scalar assignment and an IF remain native, even though their
        # values are unrelated to these loops and merging could be tempting.
        for gap in ("bookkeeping=7", "if (high<low) bookkeeping=9"):
            source = adjacent(parallel="serial", gap=gap).replace("integer :: coordinate", "integer :: bookkeeping\ninteger :: coordinate", 1)
            result = self.discover(source)
            self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])
        source = adjacent(parallel="serial")
        marker = source.index("do coordinate0=low,high-1")
        source = source[:marker] + "if (high>low) then\n" + source[marker:]
        source = source.replace("end subroutine", "end if\nend subroutine", 1)
        result = self.discover(source)
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])

    def test_separate_parallel_teams_do_not_merge(self):
        source = adjacent(gap="!$omp end parallel\n!$omp parallel")
        result = self.discover(source)
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])
        teams = [r["gpu_scope"]["team_start_line"] for r in result["regions"]]
        self.assertNotEqual(*teams)

    def test_rejected_union_retains_individually_proved_nests(self):
        source = adjacent(1).replace("3.0_8*input(coordinate0)", "first(coordinate0-1)")
        result = self.discover(source)
        # Cross-kernel neighbour dependence is left to separate calls; the
        # existing combined pointwise proof is deliberately not relaxed.
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [1, 1], result["rejected"])
        source = adjacent(1).replace("!$omp do\ndo coordinate0=low,high-1", "!$omp do firstprivate(input)\ndo coordinate0=low,high-1")
        result = self.discover(source)
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [1], result["rejected"])
        self.assertTrue(any("private" in r["reason"] for r in result["rejected"]))
        source = adjacent(1).replace("advance(first,second,input,", "advance(first,second,third,input,")
        source = source.replace("second(-3:)", "second(-3:),third(-3:)", 1)
        source = source.replace("!$omp end parallel", """!$omp do
do coordinate0=low,high-1
 third(coordinate0)=first(coordinate0-1)
end do
!$omp end do
!$omp end parallel""", 1)
        result = self.discover(source)
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [2, 1], result["rejected"])

    def test_group_proves_liveness_after_last_private_loop(self):
        source = adjacent(2).replace("!$omp do\n", "!$omp do private(coordinate0,coordinate1)\n")
        result = self.discover(source)
        # The old singleton scan rejected the first loop because fparser
        # treats the later PRIVATE list as a possible iterator-value read.
        self.assertEqual([r["loop_nests"] for r in result["regions"]], [2], result["rejected"])

    def test_later_bound_observes_native_final_tile_iterator(self):
        source = """module subject
implicit none
contains
subroutine advance(a,b)
real(kind=8), intent(inout) :: a(:),b(:)
integer :: t,i
t=2
do t=1,4,2
 do i=t,min(t+2-1,4)
  a(i)=a(i)+1
 end do
end do
do i=1,t-1
 b(i)=b(i)+1
end do
end subroutine
end module
"""
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertEqual(region["loop_nests"], 1)
        self.assertEqual(region["gpu_scope"]["kind"], "rectangular")
        self.assertTrue(any("tile iterator remains live" in item["reason"] for item in result["rejected"]))
        (self.root / "reference.f90").write_text(source.replace("module subject", "module reference"))
        patch_regions(self.root, [region], coverage=False)
        self.assertIn("do t=1,4,2", self.path.read_text())
        driver = self.root / "driver.f90"
        driver.write_text("""program verify
use subject, only: candidate=>advance
use reference, only: original=>advance
implicit none
real(kind=8) :: a(4),b(4),reference_a(4),reference_b(4)
a=0
b=0
reference_a=0
reference_b=0
call original(reference_a,reference_b)
call candidate(a,b)
if (any(a/=reference_a) .or. any(b/=reference_b)) stop 1
if (any(b/=1)) stop 2
end program
""")
        executable = self.root / "verify"
        compile_ = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
            region["input_path"], str(self.root/"reference.f90"), str(self.path), str(driver), "-o", str(executable)],
            cwd=self.root, capture_output=True, text=True)
        self.assertEqual(compile_.returncode, 0, compile_.stdout + compile_.stderr)
        run = subprocess.run([str(executable)], cwd=self.root, capture_output=True, text=True, timeout=15)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)

    def test_compiled_group_preserves_halos_empty_domains_and_full_fallback(self):
        for parallel in ("serial", "collective"):
            with self.subTest(parallel=parallel):
                source = adjacent(2, parallel=parallel, tiled=(False, True))
                result = self.discover(source)
                self.assertEqual(len(result["regions"]), 1, result["rejected"])
                region = result["regions"][0]
                self.assertEqual(region["loop_nests"], 2)
                (self.root / "reference.f90").write_text(source.replace("module subject", "module reference"))
                patch_regions(self.root, [region], coverage=False)
                driver = self.root / "driver.f90"
                driver.write_text("""program verify
use subject, only: candidate=>advance
use reference, only: original=>advance
implicit none
real(kind=8) :: a(-3:9,-3:9),b(-3:9,-3:9),reference_a(-3:9,-3:9),reference_b(-3:9,-3:9),data(-3:9,-3:9)
integer :: test,stride,low,high
low=-2
high=7
do test=1,4
 stride=3
 if (test==2) stride=20
 if (test==3) stride=-2
 if (test==4) high=-3
 a=-31.0_8
 b=-47.0_8
 reference_a=a
 reference_b=b
 data=0.5_8*test
 call original(reference_a,reference_b,data,low,high,stride)
 call candidate(a,b,data,low,high,stride)
 if (any(a/=reference_a) .or. any(b/=reference_b)) stop 1
end do
end program
""")
                executable = self.root / "verify"
                compile_ = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                    region["input_path"], str(self.root/"reference.f90"), str(self.path), str(driver), "-o", str(executable)],
                    cwd=self.root, capture_output=True, text=True)
                self.assertEqual(compile_.returncode, 0, compile_.stdout + compile_.stderr)
                run = subprocess.run([str(executable)], cwd=self.root, capture_output=True, text=True,
                                     env={**os.environ, "OMP_NUM_THREADS": "3"}, timeout=15)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()
