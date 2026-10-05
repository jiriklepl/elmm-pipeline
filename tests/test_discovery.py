"""Structural discovery must work on unrelated unannotated source programs."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HAS_FPARSER = importlib.util.find_spec("fparser") is not None
if HAS_FPARSER:
    from elmm_pipeline.structural import discover_sources


PLAIN = """module arbitrary_physics
use iso_fortran_env, only: precision_alias => real64
implicit none
real(precision_alias) :: factor = 0.25_precision_alias
contains
subroutine evolve(destination, source, width, height)
real(precision_alias), intent(inout) :: destination(-2:,-1:)
real(precision_alias), intent(in) :: source(0:,3:)
integer, intent(in) :: width, height
integer :: longitude, level
real(precision_alias) :: intermediate
! No selection annotation is present.
do level=3,height
 do longitude=1,width
  intermediate = source(longitude,level) + longitude*0.5_precision_alias
  destination(longitude,level) = intermediate
 end do
end do
do level=3,height
 do longitude=1,width
  destination(longitude,level) = destination(longitude,level)*factor
 end do
end do
end subroutine evolve
end module arbitrary_physics
"""


def one_loop(body: str, *, prefix: str = "", suffix: str = "", declaration: str = "") -> str:
    return f"""module unrelated
implicit none
contains
subroutine update(a,b,n)
real, intent(inout) :: a(:)
real, intent(in) :: b(:)
integer, intent(in) :: n
integer :: position
real :: temporary
{declaration}
{prefix}
do position=2,n-1
{body}
end do
{suffix}
end subroutine
end module
"""


@unittest.skipUnless(HAS_FPARSER and shutil.which("gfortran"), "requires fparser and gfortran in the selected test environment")
class StructuralDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="generic-discovery-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def discover(self, source: str, *, defines=()):
        path = self.root / "source.f90"
        path.write_text(source)
        return discover_sources([str(path)], str(self.root), str(self.root / "generated"), list(defines))

    def test_unannotated_arbitrary_rank_two_and_adjacent_loops(self):
        result = self.discover(PLAIN)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertEqual(region["loop_nests"], 2)
        self.assertEqual(region["loop_ranks"], [2, 2])
        self.assertIn("real(kind=8)", region["module_source"])
        self.assertIn("longitude * 0.5_8", region["module_source"])
        self.assertEqual(region["actual_args"].count("width"), 1)
        self.assertEqual(region["actual_args"].count("height"), 1)
        self.assertFalse(result["selection"]["markers_required"])
        self.assertIn("factor", region["actual_args"])

    def test_renaming_every_algorithm_symbol_changes_no_eligibility(self):
        original = self.discover(PLAIN)
        renamed = PLAIN
        import re
        substitutions = {"arbitrary_physics": "different_subject", "precision_alias": "wp", "factor": "coefficient",
                         "evolve": "process", "destination": "answer", "source": "data", "width": "nx",
                         "height": "ny", "longitude": "col", "level": "row", "intermediate": "work"}
        for old, new in substitutions.items():
            renamed = re.sub(r"(?<![A-Za-z0-9])" + old + r"\b", new, renamed)
        result = self.discover(renamed)
        self.assertEqual([r["loop_ranks"] for r in result["regions"]], [r["loop_ranks"] for r in original["regions"]])
        self.assertEqual(result["regions"][0]["routine"], "process")

    def test_real_kind_comes_from_preprocessed_declared_constant(self):
        source = one_loop("a(position)=b(position)*0.5_wp")
        source = source.replace("implicit none", """use iso_fortran_env, only: real32, real64
implicit none
#ifdef WIDE
integer, parameter :: wp=real64
#else
integer, parameter :: wp=real32
#endif""")
        source = source.replace("real,", "real(wp),")
        for definitions, expected in (((), "real(kind=4)"), (("WIDE",), "real(kind=8)")):
            with self.subTest(definitions=definitions):
                result = self.discover(source, defines=definitions)
                self.assertEqual(len(result["regions"]), 1, result["rejected"])
                self.assertIn(expected, result["regions"][0]["module_source"])

    def test_native_equivalence_different_lower_bounds_index_as_data_and_halos(self):
        result = self.discover(PLAIN)
        region = result["regions"][0]
        lines = PLAIN.splitlines(keepends=True)
        lines[region["start_line"] - 1:region["end_line"]] = [region["call"] + "\n"]
        lines.insert(region["insertion_line"] - 1, region["use"] + "\n")
        adapted = "".join(lines).replace("arbitrary_physics", "adapted_physics")
        (self.root / "reference.f90").write_text(PLAIN)
        (self.root / "adapted.f90").write_text(adapted)
        (self.root / "driver.f90").write_text("""program check
use arbitrary_physics, only: original=>evolve
use adapted_physics, only: outlined=>evolve
implicit none
real(kind=8) :: src(0:10,3:10), a(-2:9,-1:9), b(-2:9,-1:9)
integer :: x,y,w,h
do y=3,10
do x=0,10
src(x,y)=x*0.125d0-y*0.375d0
end do
end do
do h=2,8,3
do w=0,7,3
a=-123.5d0
b=a
call original(a,src,w,h)
call outlined(b,src,w,h)
if(any(a/=b)) error stop 'outline changed interior, empty domain, or halo'
end do
end do
print *, 'PASS'
end program
""")
        command = ["gfortran", "-O2", "-fcheck=all", "-ffree-line-length-none", "reference.f90",
                   region["input_path"], "adapted.f90", "driver.f90", "-o", "check"]
        built = subprocess.run(command, cwd=self.root, capture_output=True, text=True)
        self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
        run = subprocess.run([str(self.root / "check")], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        self.assertIn("PASS", run.stdout)

    def test_recurrences_reductions_unknown_calls_and_scratch_are_rejected(self):
        cases = [
            (one_loop("a(position)=a(position-1)+b(position)"), "neighbor/dependent"),
            (one_loop("temporary=temporary+b(position)\na(position)=temporary"), "read before definition"),
            (one_loop("a(position)=unknown(b(position))"), "opaque function"),
            (one_loop("scratch(position)=b(position)\na(position)=scratch(position)", declaration="real :: scratch(100)"), "local scratch"),
            (one_loop("a(position)=b(position)", suffix="print *, position"), "live after"),
            (one_loop("temporary=b(position)\na(position)=temporary", suffix="print *, temporary"), "live after"),
            (one_loop("a(position)=b(position)").replace("real,", "real(mystery_kind),"), "unresolved integer/kind"),
            (one_loop("a(position)=b(position)").replace("subroutine update", "pure subroutine update"), "PURE/ELEMENTAL"),
            (one_loop("!$omp atomic\na(position)=a(position)+b(position)"), "directive inside"),
            (one_loop("temporary=-position\na(position+temporary+1)=b(position)").replace("real :: temporary", "integer :: temporary"), "region-written scalar"),
            (one_loop("a(position)=b(position)", declaration="integer, external :: lbound"), "LBOUND intrinsic binding"),
        ]
        for source, reason in cases:
            with self.subTest(reason=reason):
                result = self.discover(source)
                self.assertFalse(result["regions"], result["regions"])
                self.assertTrue(any(reason in row["reason"] for row in result["rejected"]), result["rejected"])

    def test_outer_tiles_and_omp_collapse_remain_native(self):
        source = """module tiled
contains
subroutine operation(a,b,nx,ny,tx,ty)
real, intent(inout)::a(:,:)
real, intent(in)::b(:,:)
integer,intent(in)::nx,ny,tx,ty
integer::ib,jb,col,row
!$omp parallel do collapse(2) private(col,row)
do jb=1,ny,ty
do ib=1,nx,tx
do row=jb,min(jb+ty-1,ny)
do col=ib,min(ib+tx-1,nx)
a(col,row)=b(col,row)+col
enddo
enddo
enddo
enddo
!$omp end parallel do
end subroutine
end module
"""
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertEqual(region["loop_ranks"], [2])
        self.assertTrue(region["context"]["nested_parallel"])
        self.assertTrue(region["original_text"].startswith("do row="))
        self.assertNotIn("!$omp", region["original_text"])
        self.assertIn("MIN(jb + ty - 1, ny)", region["actual_args"])

    def test_directed_loop_is_not_replaced_by_call(self):
        result = self.discover(one_loop("a(position)=b(position)", prefix="!$omp parallel do"))
        self.assertFalse(result["regions"])
        self.assertTrue(any("directive" in row["reason"] for row in result["rejected"]))

    def test_optional_marker_is_a_selection_hint(self):
        source = one_loop("a(position)=b(position)").replace("subroutine update", "! kernel\nsubroutine update")
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        self.assertTrue(result["regions"][0]["annotated"])

    def test_conditional_liveout_persistent_storage_and_bound_effects_are_rejected(self):
        cases = [
            (one_loop("a(position)=b(position)", declaration="logical :: flag", suffix="if(flag)then\nposition=0\nendif\nprint *,position"), "live after"),
            (one_loop("a(position)=b(position)").replace("integer :: position", "integer :: position=10"), "persistent induction"),
            (one_loop("temporary=b(position)\na(position)=temporary").replace("real :: temporary", "real,save :: temporary"), "persistent local"),
            (one_loop("a(position)=b(position)", declaration="data position/10/"), "persistent induction"),
            (one_loop("temporary=b(position)\na(position)=temporary", declaration="data temporary/1.0/"), "persistent local"),
            (one_loop("a(position)=b(position)").replace("real, intent", "real, target, intent"), "unsafe storage"),
            (one_loop("a(position)=b(position)", declaration="target :: a,b"), "unsafe storage"),
            (one_loop("a(position)=b(position)").replace("do position=2,n-1", "do position=2,limit()"), "opaque calls"),
            (one_loop("a(position)=b(position)").replace("do position=2,n-1", "do position=2,4/n"), "trapping loop-bound"),
            (one_loop("a(position)=b(position)").replace("do position=2,n-1", "#ifdef SELECTED\ndo position=1,n\n#else\ndo position=2,n-1\n#endif"), "preprocessor directive"),
            (one_loop("a(position)=b(position)").replace("do position=2,n-1", "temporary=7; do position=2,n-1"), "multiple statements"),
        ]
        for source, reason in cases:
            with self.subTest(reason=reason):
                result = self.discover(source)
                self.assertFalse(result["regions"])
                self.assertTrue(any(reason in row["reason"] for row in result["rejected"]), result["rejected"])

    def test_aliases_and_explicit_attributes_cannot_hide_dependencies(self):
        base = """module storage
integer :: values(8), other(8)
{storage}
end module
module m
use storage, only: a=>values,b=>{second}
contains
subroutine run
integer::i
do i=2,7
b(i)=a(i-1)+1
enddo
end subroutine
end module
"""
        for source, reason in (
            (base.format(storage="", second="values"), "visible aliases"),
            (base.format(storage="equivalence(values(2),other(1))", second="other"), "EQUIVALENCE/COMMON"),
            (base.format(storage="volatile :: values", second="other"), "unsafe storage"),
        ):
            with self.subTest(reason=reason):
                result = self.discover(source)
                self.assertFalse(result["regions"])
                self.assertTrue(any(reason in row["reason"] for row in result["rejected"]), result["rejected"])

    def assert_native_outlined_output_equal(self, source: str, driver: str):
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        native = self.root / "source.f90"
        program = self.root / "driver.f90"
        program.write_text("program test\nuse m\nimplicit none\n" + driver + "\nend program\n")

        def execute(files, name):
            result = subprocess.run(["gfortran", "-O0", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                                     *map(str, files), "-o", name], cwd=self.root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([str(self.root / name)], cwd=self.root, capture_output=True, text=True,
                                    env={**os.environ, "OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result.stdout

        expected = execute([native, program], "original")
        lines = source.splitlines(keepends=True)
        lines[region["start_line"] - 1:region["end_line"]] = [region["call"] + "\n"]
        lines.insert(region["insertion_line"] - 1, region["use"] + "\n")
        native.write_text("".join(lines))
        actual = execute([region["input_path"], native, program], "outlined")
        self.assertEqual(expected, actual)

    def test_mutated_declaration_lower_bound_uses_array_descriptor(self):
        self.assert_native_outlined_output_equal("""module m
contains
subroutine entry(a,lb)
integer,intent(inout)::lb
integer,intent(inout)::a(lb:)
integer::i
lb=lb+1
do i=lb,lb+1
a(i)=i
enddo
end subroutine
end module
""", "integer::a(4),lb\na=0\nlb=0\ncall entry(a,lb)\nprint *,a")

    def test_renamed_module_array_uses_visible_name_in_generated_interface(self):
        self.assert_native_outlined_output_equal("""module storage
integer::field(3)
end module
module m
use storage, only: renamed=>field
contains
subroutine entry
integer::i
do i=1,3
renamed(i)=i
enddo
end subroutine
end module
""", "call entry()\nprint *,renamed")

    def test_kind_of_nondefault_integer_literal_is_resolved(self):
        self.assert_native_outlined_output_equal("""module m
integer,parameter::wp=kind(1_8)
contains
subroutine entry(a)
real(wp),intent(inout)::a(:)
integer::i
do i=1,3
a(i)=a(i)+0.25_wp
enddo
end subroutine
end module
""", "real(wp)::a(3)\na=0\ncall entry(a)\nprint *,a")

    def test_conversion_kind_parameter_remains_a_compile_time_constant(self):
        self.assert_native_outlined_output_equal("""module m
integer,parameter::wp=8
contains
subroutine entry(a)
real(wp),intent(inout)::a(:)
integer::i
do i=1,3
a(i)=real(i,wp)
enddo
end subroutine
end module
""", "real(wp)::a(3)\na=0\ncall entry(a)\nprint *,a")

    def test_nested_scopes_and_concurrent_contexts_require_explicit_support(self):
        loop = "do position=2,n-1\na(position)=b(position)\nend do"
        base = one_loop("a(position)=b(position)")
        cases = [
            (base.replace(loop, "block\nreal(kind=8)::temporary\n" + loop + "\nend block"), "Block_Construct"),
            (base.replace(loop, "associate(a=>b)\n" + loop + "\nend associate"), "Associate_Construct"),
            (base.replace("integer :: position", "integer :: position, concurrent_index").replace(
                loop, "do concurrent (concurrent_index=1:n)\n" + loop + "\nend do"), "DO CONCURRENT"),
        ]
        for source, reason in cases:
            with self.subTest(reason=reason):
                result = self.discover(source)
                self.assertFalse(result["regions"])
                self.assertTrue(any(reason in row["reason"] for row in result["rejected"]), result["rejected"])

    def test_inline_comment_on_outer_omp_loop_does_not_expand_inner_span(self):
        source = """module m
contains
subroutine entry(a,b,n)
real,intent(inout)::a(:,:)
real,intent(in)::b(:,:)
integer,intent(in)::n
integer::i,j
!$omp parallel do private(i,j)
do j=1,n ! the outer worksharing loop must remain in the original source
 do i=1,n
  a(i,j)=b(i,j)+i
 enddo
enddo
!$omp end parallel do
end subroutine
end module
"""
        result = self.discover(source)
        self.assertEqual(len(result["regions"]), 1, result["rejected"])
        region = result["regions"][0]
        self.assertEqual(region["loop_ranks"], [1])
        self.assertTrue(region["original_text"].lstrip().startswith("do i="))
        self.assertNotIn("do j=", region["original_text"])
        self.assert_native_outlined_output_equal(source, "real::a(4,4),b(4,4)\na=0\nb=3\ncall entry(a,b,4)\nprint *,a")

    def test_later_worksharing_loop_is_inside_an_existing_team(self):
        source = """module m
contains
subroutine entry(a,b,n)
real,intent(inout)::a(:,:)
real,intent(in)::b(:,:)
integer,intent(in)::n
integer::i,j
!$omp parallel private(i,j)
!$omp do
do j=1,n
 do i=1,n
  a(i,j)=b(i,j)
 enddo
enddo
!$omp end do
!$omp do
do j=1,n
 do i=1,n
  a(i,j)=a(i,j)+b(i,j)
 enddo
enddo
!$omp end do
!$omp end parallel
end subroutine
end module
"""
        result = self.discover(source)
        self.assertTrue(result["regions"], result["rejected"])
        later = result["regions"][-1]
        self.assertGreater(later["start_line"], 17)
        self.assertTrue(later["context"]["nested_parallel"])

    def test_ambiguous_imported_module_is_not_resolved_by_scan_order(self):
        for file_name, value in (("first.f90", "real64"), ("second.f90", "real32")):
            (self.root / file_name).write_text(f"module alternative\nuse iso_fortran_env\ninteger,parameter::wp={value}\nend module\n")
        source = one_loop("a(position)=b(position)").replace("implicit none", "use alternative\nimplicit none").replace("real,", "real(wp),")
        result = self.discover(source)
        self.assertFalse(result["regions"])
        paths = [str(self.root / name) for name in ("first.f90", "second.f90", "source.f90")]
        reordered = discover_sources(paths, str(self.root), str(self.root / "all"))
        self.assertFalse(reordered["regions"])

    def test_optional_arguments_cannot_be_evaluated_outside_an_empty_loop(self):
        cases = [
            one_loop("a(position)=b(position)").replace("real, intent(in) :: b(:)", "real, optional, intent(in) :: b(:)"),
            one_loop("a(position)=b(position)", declaration="optional :: b"),
            one_loop("a(position)=coefficient*b(position)", declaration="real, optional :: coefficient").replace(
                "subroutine update(a,b,n)", "subroutine update(a,b,n,coefficient)"),
        ]
        for source in cases:
            with self.subTest(source=source):
                result = self.discover(source)
                self.assertFalse(result["regions"])
                self.assertTrue(any("unsafe storage attributes" in row["reason"] for row in result["rejected"]), result["rejected"])

    def test_alternate_entry_is_rejected_before_argument_assumptions(self):
        source = one_loop("a(position)=b(position)", prefix="entry alternate(a,b,n)")
        result = self.discover(source)
        self.assertFalse(result["regions"])
        self.assertTrue(any("ENTRY procedure" in row["reason"] for row in result["rejected"]), result["rejected"])

    def test_contiguity_metadata_distinguishes_assumed_shape_and_proven_storage(self):
        cases = [
            (one_loop("a(position)=b(position)"), False),
            (one_loop("a(position)=b(position)").replace("real, intent", "real, contiguous, intent"), True),
            (one_loop("a(position)=b(position)").replace("a(:)", "a(n)").replace("b(:)", "b(n)"), True),
        ]
        for source, expected in cases:
            with self.subTest(expected=expected, source=source):
                result = self.discover(source)
                self.assertEqual(len(result["regions"]), 1, result["rejected"])
                arrays = [symbol for symbol in result["regions"][0]["symbols"] if symbol["rank"]]
                self.assertEqual([symbol["contiguous_proven"] for symbol in arrays], [expected, expected])

    def test_namelist_and_contained_procedures_cannot_hide_local_reads(self):
        cases = [
            (one_loop("temporary=b(position)\na(position)=temporary", declaration="namelist/state/temporary", suffix="write(*,nml=state)"), "NAMELIST"),
            (one_loop("temporary=b(position)\na(position)=temporary", suffix="call observe()\ncontains\nsubroutine observe\nprint *,temporary\nend subroutine observe"), "contained procedures"),
        ]
        for source, reason in cases:
            with self.subTest(reason=reason):
                result = self.discover(source)
                self.assertFalse(result["regions"])
                self.assertTrue(any(reason in row["reason"] for row in result["rejected"]), result["rejected"])

    def test_openmp_clause_after_region_reads_the_previous_scalar_value(self):
        source = one_loop("temporary=b(position)\na(position)=temporary",
                          suffix="!$omp parallel firstprivate(temporary)\n!$omp end parallel")
        result = self.discover(source)
        self.assertFalse(result["regions"])
        self.assertTrue(any("live after" in row["reason"] for row in result["rejected"]), result["rejected"])


if __name__ == "__main__":
    unittest.main()
