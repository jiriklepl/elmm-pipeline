"""Region publication and complete generic backend integration."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from elmm_pipeline.adaptation import adapt_project, patch_regions, selected


SOURCE = """module field_steps
  implicit none
  integer, parameter :: working = kind(1.0d0)
contains
  subroutine advance(state, forcing, coefficient)
    real(working), contiguous, intent(inout) :: state(-3:,2:)
    real(working), contiguous, intent(in) :: forcing(0:,-4:)
    real(working), intent(in) :: coefficient
    integer :: east, north
    state = -77.0_working
    do north = 3, 7
      do east = 0, 5
        state(east,north) = coefficient * forcing(east+3,north-6) + east
      end do
    end do
    do north = 3, 7
      do east = 0, 5
        state(east,north) = state(east,north) + 2.0_working
      end do
    end do
  end subroutine advance
end module field_steps
"""

DRIVER = """program test
use field_steps, only: advance
use reference_steps, only: reference_advance
implicit none
real(8):: state(-3:8,2:10), expected(-3:8,2:10), forcing(0:11,-4:4)
integer::i,j
do j=-4,4
do i=0,11
forcing(i,j)=i*0.1d0+j*0.25d0
enddo
enddo
call reference_advance(expected,forcing,0.37d0)
call advance(state,forcing,0.37d0)
if(maxval(abs(state-expected))>1d-12) stop 1
print *, 'GENERIC_REGION_MATCH'
end program
"""


class PublicationTests(unittest.TestCase):
    def test_selection_hints_are_optional_and_name_independent(self):
        region = {"module": "Fluid", "routine": "Update", "source": "physics.f90"}
        self.assertTrue(selected(region))
        self.assertTrue(selected(region, include=["fluid::*"]))
        self.assertFalse(selected(region, exclude=["*::update"]))
        self.assertFalse(selected(region, marked_only=True))
        self.assertTrue(selected({**region, "annotated": True}, marked_only=True))

    def test_source_span_mismatch_does_not_publish_partial_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "first.f90").write_text("subroutine first\ninteger::i\ni=1\nend\n")
            (root / "second.f90").write_text("subroutine second\ninteger::i\ni=2\nend\n")
            before = {p: p.read_bytes() for p in root.iterdir()}
            regions = []
            for name, value in (("first", "i=1\n"), ("second", "i=WRONG\n")):
                regions.append({"id": name, "source": name + ".f90", "start_line": 3,
                                "end_line": 3, "insertion_line": 2, "original_text": value,
                                "generated_module": "generated_" + name, "entry": "kernel_" + name,
                                "call": f"call kernel_{name}(i)"})
            with self.assertRaisesRegex(ValueError, "changed"):
                patch_regions(root, regions)
            self.assertEqual(before, {p: p.read_bytes() for p in root.iterdir()})

    def test_unmarked_patch_preserves_native_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = "subroutine native\ninteger::i\ni=1\nend\n"
            (root / "custom.f90").write_text(source)
            region = {"id": "unique", "source": "custom.f90", "start_line": 3, "end_line": 3,
                      "insertion_line": 2, "original_sha256": hashlib.sha256(b"i=1\n").hexdigest(),
                      "generated_module": "outlined", "entry": "run_region", "call": "call run_region(i)"}
            self.assertEqual(patch_regions(root, [region], coverage=False), ["custom.f90"])
            text = (root / "custom.f90").read_text()
            self.assertIn("use outlined, only: run_region", text)
            self.assertNotIn("ELMM_PIPELINE", text)
            self.assertTrue(text.endswith("end\n"))


class NativeGenericIntegrationTests(unittest.TestCase):
    def test_local_parallel_capture_requires_proved_contiguity(self):
        python = os.environ.get("ELMM_TEST_PYTHON")
        compiler_root = os.environ.get("ELMM_TEST_COMPILER")
        if not python or not compiler_root:
            self.skipTest("Set explicit local compiler paths")
        template = """module arbitrary_parallel
contains
subroutine advance(field,n)
real(8), ATTR intent(inout)::field(:,:)
integer,intent(in)::n
integer::row,column
!$omp parallel do private(column)
do row=1,n
do column=1,n
field(column,row)=field(column,row)+1.0_8
enddo
enddo
!$omp end parallel do
end subroutine
end module
"""
        for contiguous in (False, True):
            with self.subTest(contiguous=contiguous), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "src"
                source.mkdir()
                original = template.replace("ATTR", "contiguous," if contiguous else "")
                path = source / "unrelated.f90"
                path.write_text(original)
                result = adapt_project("local", source, Path(python), root / "generated",
                                       compiler_root=Path(compiler_root), defines=())
                self.assertEqual(result["discovered_regions"], 1, result["rejected"])
                self.assertEqual(result["outlined_regions"], int(contiguous), result["rejected"])
                if not contiguous:
                    self.assertEqual(path.read_text(), original)
                    self.assertTrue(any("copyback race" in item["reason"] for item in result["rejected"]))

    def test_discovery_marker_hints_restrict_only_when_requested(self):
        from elmm_pipeline.discovery import discover_project

        python = os.environ.get("ELMM_TEST_PYTHON")
        compiler_root = os.environ.get("ELMM_TEST_COMPILER")
        if not python or not compiler_root or not shutil.which(os.environ.get("FC", "gfortran")):
            self.skipTest("Set explicit local compiler paths and provide GNU Fortran")
        source_text = """module unrelated_selection
implicit none
contains
! kernel
subroutine suggested(field,extent)
real(8),intent(inout)::field(:)
integer,intent(in)::extent
integer::point
do point=1,extent
field(point)=field(point)*2.0_8
enddo
end subroutine
subroutine ordinary(field,extent)
real(8),intent(inout)::field(:)
integer,intent(in)::extent
integer::point
do point=1,extent
field(point)=field(point)*3.0_8
enddo
end subroutine
end module
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            initial = root / "initial"
            initial.mkdir()
            (initial / "unrelated.f90").write_text(source_text)
            discovery = discover_project(initial, Path(python), root / "discovery", defines=())
            hints = {region["routine"].lower(): region["annotated"] for region in discovery["regions"]}
            self.assertEqual(hints, {"suggested": True, "ordinary": False}, discovery["rejected"])
            for marked_only in (False, True):
                with self.subTest(marked_only=marked_only):
                    source = root / ("marked" if marked_only else "default")
                    shutil.copytree(initial, source)
                    metadata = adapt_project("local", source, Path(python),
                                             root / (source.name + "-generated"),
                                             compiler_root=Path(compiler_root), defines=(),
                                             marked_only=marked_only, coverage=False)
                    self.assertEqual(metadata["discovered_regions"], 2)
                    expected = {"suggested"} if marked_only else {"suggested", "ordinary"}
                    self.assertEqual({region["routine"].lower() for region in metadata["regions"]}, expected,
                                     metadata["rejected"])
                    if marked_only:
                        self.assertIn("field(point)=field(point)*3.0_8", (source / "unrelated.f90").read_text())
                        self.assertTrue(any(item.get("routine", "").lower() == "ordinary"
                                            and item["reason"] == "excluded by selection hints"
                                            for item in metadata["rejected"]))

    def test_unannotated_unrelated_program_matches_all_configured_backends(self):
        compilers = (shutil.which(os.environ.get("FC", "gfortran")), shutil.which("g++"))
        if not all(compilers):
            self.skipTest("GNU Fortran and C++ required")
        configured = {name: os.environ.get(variable) for name, variable in (
            ("loki", "ELMM_LOKI_PYTHON"), ("psyclone", "ELMM_PSYCLONE_PYTHON"),
            ("local", "ELMM_TEST_PYTHON"))}
        compiler_root = os.environ.get("ELMM_TEST_COMPILER")
        if not all(configured.values()) or not compiler_root:
            self.skipTest("Set all three explicit backend Python paths and ELMM_TEST_COMPILER")
        for tool, python in configured.items():
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "src"
                source.mkdir()
                (source / "custom.f90").write_text(SOURCE)
                metadata = adapt_project(tool, source, Path(python), root / "generated",
                                         compiler_root=Path(compiler_root), defines=())
                self.assertEqual(len(metadata["regions"]), 1, metadata["rejected"])
                self.assertEqual(metadata["optimized_regions"], 1)
                # Native whole-array initialization remains outside the region.
                self.assertIn("state = -77.0_working", (source / "custom.f90").read_text())
                reference = SOURCE.replace("field_steps", "reference_steps").replace(
                    "subroutine advance", "subroutine reference_advance")
                (root / "reference.f90").write_text(reference)
                (root / "driver.f90").write_text(DRIVER)
                objects = []
                for index, filename in enumerate(metadata["extra_sources"]):
                    path, obj = Path(filename), root / f"extra{index}.o"
                    objects.append(obj)
                    if path.suffix == ".cpp":
                        command = [compilers[1], "-std=c++17", "-O2", "-fopenmp", "-c", path, "-o", obj]
                    else:
                        command = [compilers[0], "-O2", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                                   f"-J{root}", f"-I{root}", "-c", path, "-o", obj]
                    result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                command = [compilers[0], "-O2", "-fopenmp", "-fcheck=all", "-ffree-line-length-none",
                           f"-J{root}", f"-I{root}", source / "custom.f90", root / "reference.f90",
                           root / "driver.f90", *objects, "-lstdc++", "-o", root / "test"]
                result = subprocess.run(list(map(str, command)), capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                output = subprocess.check_output([str(root / "test")], text=True,
                                                 env=dict(os.environ, OMP_NUM_THREADS="2"))
                self.assertIn("GENERIC_REGION_MATCH", output)
                self.assertIn("ELMM_PIPELINE_REGION " + metadata["regions"][0]["id"], output)
