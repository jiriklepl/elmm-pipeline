"""Generic pass legality and native numerical comparison on unrelated kernels.

Integration tests use explicit ELMM_LOKI_PYTHON/ELMM_PSYCLONE_PYTHON interpreters;
no tool environment outside the portable pipeline is discovered implicitly.
"""
from __future__ import annotations
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from elmm_pipeline.recipes import transform
from elmm_pipeline.tool_workers.generic_analysis import Access, Assignment, Loop, Scope, affine, prove_fusion


ONE_DIMENSION = """module unrelated
implicit none
contains
subroutine flux(n, field, result)
integer, intent(in) :: n
real(kind=8), contiguous, intent(in) :: field(:)
real(kind=8), contiguous, intent(out) :: result(:)
integer :: cell
real(kind=8) :: alpha, beta
alpha = 2.0_8
do cell=1,n
result(cell) = field(cell)*alpha
end do
beta = 3.0_8
do cell=1,n
result(cell) = result(cell)+beta
end do
end subroutine
end module
"""

TWO_DIMENSIONS = """module unrelated
implicit none
contains
subroutine flux(nx, ny, field, result)
integer, intent(in) :: nx, ny
real(kind=4), contiguous, intent(in) :: field(-3:,-1:)
real(kind=4), contiguous, intent(out) :: result(-3:,-1:)
integer :: column, row
do row=0,ny
do column=-2,nx
result(column,row) = field(column-1,row) + field(column,row-1)
end do
end do
do row=0,ny
do column=-2,nx
result(column,row) = result(column,row)*0.5_4
end do
end do
end subroutine
end module
"""

HELPERS = """module unrelated
implicit none
contains
subroutine flux(n, field, result)
integer, intent(in) :: n
real(kind=8), intent(in) :: field(:)
real(kind=8), intent(out) :: result(:)
call initialize(n,field,result)
call rescale(n,result)
end subroutine
subroutine initialize(extent, origin, destination)
integer, intent(in) :: extent
real(kind=8), intent(in) :: origin(:)
real(kind=8), intent(out) :: destination(:)
integer :: point
do point=1,extent
destination(point)=origin(point)*2.0_8
end do
end subroutine
subroutine rescale(extent, values)
integer, intent(in) :: extent
real(kind=8), intent(inout) :: values(:)
integer :: point
do point=1,extent
values(point)=values(point)+3.0_8
end do
end subroutine
end module
"""


class GenericDependenceTests(unittest.TestCase):
    def test_affine_equivalence(self):
        self.assertEqual(affine("point - (-3) + 1"), affine("4 + point"))
        self.assertIsNone(affine("point*point"))

    def test_mutable_affine_index_is_not_invariant(self):
        from elmm_pipeline.tool_workers.generic_analysis import analyze_nest
        scope = Scope({"a"}, {"a"}, set(), {"i", "t"})
        nest = Loop("i", ("1", "4", "1"), [
            Assignment(Access("t"), [Access("i")]),
            Assignment(Access("a", ("i+t+1",)), [Access("i"), Access("t")]),
        ])
        analysis, reason = analyze_nest(nest, scope)
        self.assertIsNone(analysis)
        self.assertIn("loop-written scalar", reason)

    def test_cross_iteration_dependency_rejected(self):
        scope = Scope({"input", "out", "n"}, {"input", "out"}, {"input", "n"}, {"i"})
        first = Loop("i", ("2", "n", "1"), [Assignment(Access("out", ("i",)), [Access("input", ("i",))])])
        second = Loop("i", ("2", "n", "1"), [Assignment(Access("out", ("i",)), [Access("out", ("i-1",))])])
        legal, reason = prove_fusion(first, second, scope)
        self.assertFalse(legal)
        self.assertIn("dependence", reason)


class GenericWorkerIntegrationTests(unittest.TestCase):
    def tools(self):
        tools = [(name, Path(os.environ[f"ELMM_{name.upper()}_PYTHON"]))
                 for name in ("loki", "psyclone") if os.environ.get(f"ELMM_{name.upper()}_PYTHON")]
        if not tools or not shutil.which("gfortran"):
            self.skipTest("Set explicit tool interpreter variables and provide gfortran")
        return tools

    def generate(self, root, tool, python, source):
        path = root / "input.f90"
        path.write_text(source)
        metadata = transform(tool, path, python, root / "artifacts", context={"nested_parallel": True})
        self.assertEqual(path.read_text(), source)
        return Path(metadata["generated_path"]).read_text(), metadata

    def compile_compare(self, root, source, generated, rank):
        (root / "original.f90").write_text(source.replace("module unrelated", "module original"))
        (root / "generated.f90").write_text(generated.replace("UNRELATED", "generated").replace("unrelated", "generated"))
        if rank == 1:
            declarations = "real(kind=8) :: x(31), a(31), b(31)\ninteger :: i,n"
            body = """do i=1,31
x(i)=sin(real(i,8))*real(i,8)
enddo
do n=0,31,31
a=-7.0_8
b=-7.0_8
call old(n,x,a)
call new(n,x,b)
if (any(abs(a-b)>1.e-12_8)) error stop 'mismatch'
enddo"""
        else:
            declarations = "real(kind=4) :: x(-3:11,-1:7),a(-3:11,-1:7),b(-3:11,-1:7)\ninteger :: i,j"
            body = """do j=-1,7
do i=-3,11
x(i,j)=sin(real(i,4))+cos(real(j,4))
enddo
enddo
a=-7.0_4
b=-7.0_4
call old(10,6,x,a)
call new(10,6,x,b)
if (any(abs(a-b)>1.e-6_4)) error stop 'mismatch'"""
        (root / "driver.f90").write_text("program compare\nuse original, only: old=>flux\nuse generated, only: new=>flux\nimplicit none\n" + declarations + "\n" + body + "\nprint *, 'PASS'\nend program\n")
        compile_result = subprocess.run(["gfortran", "-O2", "-fcheck=all", "-ffree-line-length-none", "original.f90", "generated.f90", "driver.f90", "-o", "check"], cwd=root, capture_output=True, text=True)
        self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
        result = subprocess.run([str(root / "check")], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("PASS", result.stdout)

    def test_unannotated_1d_motion_and_fusion(self):
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generated, meta = self.generate(root, tool, python, ONE_DIMENSION)
                self.assertTrue(meta["changed"])
                self.assertEqual(meta["loop_fusions"], 1)
                self.assertEqual(meta["moved_assignments"], 1)
                self.assertNotIn("!$omp", generated.lower())
                self.compile_compare(root, ONE_DIMENSION, generated, 1)

    def test_unannotated_2d_negative_bounds_and_float32(self):
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generated, meta = self.generate(root, tool, python, TWO_DIMENSIONS)
                self.assertEqual(meta["loop_fusions"], 2, meta)
                self.assertEqual(generated.lower().count("contiguous"), 2)
                self.compile_compare(root, TWO_DIMENSIONS, generated, 2)

    def test_same_module_helpers_inlined(self):
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generated, meta = self.generate(root, tool, python, HELPERS)
                self.assertEqual(meta["inlined_calls"], 2, meta)
                self.assertEqual(meta["loop_fusions"], 1, meta)
                self.compile_compare(root, HELPERS, generated, 1)

    def test_mutable_index_fusion_preserves_native_result(self):
        source = """module mutable_index
contains
subroutine entry(a)
integer, intent(inout) :: a(:)
integer :: i, t
do i=1,4
t=-i
a(i+t+1)=i
end do
do i=1,4
t=-i
a(i+t+1)=a(i+t+1)+1
end do
end subroutine
end module
"""
        driver = "program check\nuse mutable_index\ninteger :: a(1)\na=0\ncall entry(a)\nif(a(1)/=8) error stop 'mutable-index fusion changed result'\nprint *, 'PASS'\nend program\n"
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generated, meta = self.generate(root, tool, python, source)
                self.assertFalse(meta["changed"], meta)
                self.assertEqual(meta["loop_fusions"], 0, meta)
                self.assertIn("loop-written scalar", str(meta["routines"]))
                (root / "driver.f90").write_text(driver)
                for name, text in (("native", source), ("processed", generated)):
                    (root / (name + ".f90")).write_text(text)
                    compiled = subprocess.run(["gfortran", "-O0", "-fcheck=all", name + ".f90", "driver.f90", "-o", name], cwd=root, capture_output=True, text=True)
                    self.assertEqual(compiled.returncode, 0, compiled.stderr)
                    result = subprocess.run([str(root / name)], cwd=root, capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn("PASS", result.stdout)

    def test_recurrence_remains_native(self):
        source = ONE_DIMENSION.replace("beta = 3.0_8\n", "").replace("result(cell)+beta", "result(cell-1)+3.0_8").replace("do cell=1,n", "do cell=2,n")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                generated, meta = self.generate(Path(tmp), tool, python, source)
                self.assertFalse(meta["changed"])
                self.assertEqual(generated, source)
                self.assertIn("dependence", str(meta["routines"]))

    def test_side_effect_call_remains_native(self):
        source = ONE_DIMENSION.replace("beta = 3.0_8\n", "").replace("result(cell)+beta", "result(cell)+3.0_8").replace("result(cell) = field(cell)*alpha", "call observe(cell)\nresult(cell) = field(cell)*alpha")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                generated, meta = self.generate(Path(tmp), tool, python, source)
                self.assertFalse(meta["changed"])
                self.assertEqual(generated, source)
                self.assertTrue(meta["routines"][0]["diagnostics"])

    def test_pure_intrinsic_arithmetic_can_fuse(self):
        source = ONE_DIMENSION.replace("field(cell)*alpha", "sin(field(cell))*alpha")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                generated, meta = self.generate(root, tool, python, source)
                self.assertEqual(meta["loop_fusions"], 1, meta)
                self.compile_compare(root, source, generated, 1)

    def test_distinct_induction_variable_observed_after_loop_blocks_fusion(self):
        source = ONE_DIMENSION.replace("beta = 3.0_8\n", "").replace("result(cell)+beta", "result(other)+3.0_8").replace("integer :: cell", "integer :: cell, other")
        source = source.replace("do cell=1,n\nresult(cell) = result(other)", "do other=1,n\nresult(other) = result(other)")
        source = source.replace("end subroutine", "print *, other\nend subroutine")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                generated, meta = self.generate(Path(tmp), tool, python, source)
                self.assertEqual(meta["loop_fusions"], 0, meta)
                self.assertEqual(generated, source)

    def test_nonunit_stride_retained(self):
        source = ONE_DIMENSION.replace("beta = 3.0_8\n", "").replace("result(cell)+beta", "result(cell)+3.0_8").replace("do cell=1,n", "do cell=1,n,2")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                generated, meta = self.generate(Path(tmp), tool, python, source)
                self.assertFalse(meta["changed"])
                self.assertEqual(generated, source)
                self.assertIn("stride", str(meta["routines"]))

    def test_native_parallel_directives_retained(self):
        source = ONE_DIMENSION.replace("do cell=1,n", "!$omp parallel do\ndo cell=1,n").replace("end do", "end do\n!$omp end parallel do")
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                generated, meta = self.generate(Path(tmp), tool, python, source)
                self.assertFalse(meta["changed"])
                self.assertEqual(generated, source)
                self.assertIn("parallel directives retained", str(meta["diagnostics"]))

    def test_single_loop_roundtrip_not_claimed_as_optimization(self):
        source = ONE_DIMENSION[:ONE_DIMENSION.index("beta =")] + "end subroutine\nend module\n"
        for tool, python in self.tools():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as tmp:
                generated, meta = self.generate(Path(tmp), tool, python, source)
                self.assertFalse(meta["changed"])
                self.assertEqual(source, generated)


if __name__ == "__main__":
    unittest.main()
