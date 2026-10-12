"""Opt-in public discovery/build regression for GPU loop grouping.

Set ELMM_GPU_NATIVE_TESTS=1 after pipeline setup. ELMM_LOCAL_PYTHON,
ELMM_BUILD_PYTHON and ELMM_COMPILER_ROOT can override the private environments.
ELMM_GPU_TEST_ARTIFACTS retains generated sources, manifests and execution logs.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from elmm_pipeline.adaptation import adapt_project
from elmm_pipeline.build import build
from elmm_pipeline.common import clean_environment
from elmm_pipeline.gpu_validation import gpu_activity, trace_environment
from elmm_pipeline.instrumentation import execution_coverage

SOURCE = """module unrelated_fields
use omp_lib, only: omp_get_num_threads, omp_get_thread_num
implicit none
contains
subroutine advance(field, east, north, edge_a, edge_b, top, team_size)
real(kind=8), intent(in) :: field(-3:,-2:)
real(kind=8), intent(inout) :: east(-3:,-2:), north(-3:,-2:)
integer, intent(in) :: edge_a, edge_b, top
integer, intent(out) :: team_size
integer :: column, row
!$omp parallel private(column,row)
if (omp_get_thread_num()==0) team_size = omp_get_num_threads()
!$omp do
do row=-1,top
  do column=-2,edge_a
    east(column,row) = east(column,row) + field(column,row)*2.0_8
  end do
end do
!$omp end do
! A comment does not separate these two adjacent worksharing loops.
!$omp do
do row=0,top-1
  do column=0,edge_b
    north(column,row) = field(column+1,row-1) - 3.0_8
  end do
end do
!$omp end do
!$omp end parallel
end subroutine
end module
"""

DRIVER = """program compare
use unrelated_fields, only: candidate=>advance
use reference_fields, only: reference=>advance
implicit none
real(kind=8) :: field(-3:19,-2:8), east(-3:19,-2:8), north(-3:22,-2:10)
real(kind=8) :: expected_east(-3:19,-2:8), expected_north(-3:22,-2:10)
real(kind=8) :: original_field(-3:19,-2:8)
integer :: column,row,repetition,edge_a,edge_b,reference_team,candidate_team
character(len=32) :: scenario
call get_command_argument(1,scenario)
edge_a=17
edge_b=15
select case(trim(scenario))
case('regular')
case('empty_first')
  edge_a=-3
case('empty_both')
  edge_a=-3
  edge_b=-1
case default
  error stop 'unknown scenario'
end select
do repetition=1,2
  do row=-2,8
    do column=-3,19
      field(column,row)=real(column+100*row+10000*repetition,8)
    end do
  end do
  original_field=field
  east=-700.0_8-real(repetition,8)
  north=-900.0_8-real(repetition,8)
  expected_east=east
  expected_north=north
  call reference(field,expected_east,expected_north,edge_a,edge_b,6,reference_team)
  call candidate(field,east,north,edge_a,edge_b,6,candidate_team)
  if(reference_team/=4.or.candidate_team/=4) error stop 'expected four host threads'
  if(any(east/=expected_east)) error stop 'east mismatch or changed halo'
  if(any(north/=expected_north)) error stop 'north mismatch or changed halo'
  if(any(field/=original_field)) error stop 'read-only input changed'
end do
print *, 'GROUPED_GPU_PASS ',trim(scenario)
end program
"""

SCONS = """import os
env = Environment(ENV=os.environ, FORTRAN=os.environ['FC'], F90=os.environ['FC'], LINK=os.environ['FC'])
target_dir = ARGUMENTS['dir']
env.Append(FORTRANMODDIR=target_dir, F90PATH=[target_dir],
           F90FLAGS=['-O2', '-fopenmp', '-ffree-line-length-none', '-fcheck=all'],
           LIBS=['stdc++'], LINKFLAGS=['-fopenmp'])
objs = []
for name in ['physics', 'reference', 'main']:
    objs += [node for node in env.Object(target=target_dir + '/' + name, source=name + '.f90')
             if str(node).endswith('.o')]
t = env.Program(target=target_dir + '/ELMM', source=objs)
Default(t)
"""


@unittest.skipUnless(os.environ.get("ELMM_GPU_NATIVE_TESTS") == "1",
                     "Set ELMM_GPU_NATIVE_TESTS=1 for native grouped GPU execution")
class GroupedGPUBackendTests(unittest.TestCase):
    """Use the same discovery, adaptation and GNU/NVIDIA build path as ELMM."""

    def setUp(self):
        self.pipeline = Path(__file__).resolve().parents[1]
        self.build_python = Path(os.environ.get(
            "ELMM_BUILD_PYTHON", self.pipeline / ".cache/envs/build/bin/python"))
        self.compiler_root = Path(os.environ.get(
            "ELMM_COMPILER_ROOT", self.pipeline / ".cache/repos/local-compiler"))
        self.host_compiler = shutil.which(os.environ.get("FC", "gfortran-15")) or shutil.which("gfortran")
        self.assertTrue(self.host_compiler, "Install GNU Fortran before enabling native tests")
        self.assertTrue(self.build_python.is_file(), "Run pipeline setup before enabling native tests")
        self.gpu_options = {key: os.environ[variable] for key, variable in (
            ("nvcc", "ELMM_NVCC"),
            ("cuda_host_cxx", "ELMM_CUDA_HOST_CXX"), ("architecture", "ELMM_GPU_ARCH"))
            if os.environ.get(variable)}
        if os.environ.get("ELMM_GPU_TEST_ARTIFACTS"):
            self.root = Path(os.environ["ELMM_GPU_TEST_ARTIFACTS"]).resolve()
            self.root.mkdir(parents=True, exist_ok=True)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="gpu-grouping-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)

    def generate(self, tool):
        root = self.root / tool
        elmm, poisfft = root / "ELMM", root / "PoisFFT"
        source_dir = elmm / "src"
        source_dir.mkdir(parents=True, exist_ok=True)
        (poisfft / "src").mkdir(parents=True, exist_ok=True)
        (poisfft / "src/poisfft.f90").touch()
        (source_dir / "SConstruct").write_text(SCONS)
        (source_dir / "physics.f90").write_text(SOURCE)
        (source_dir / "reference.f90").write_text(SOURCE.replace("module unrelated_fields", "module reference_fields"))
        (source_dir / "main.f90").write_text(DRIVER)
        python = Path(os.environ.get(
            f"ELMM_{tool.upper()}_PYTHON", self.pipeline / f".cache/envs/{tool}/bin/python"))
        self.assertTrue(python.is_file(), f"Missing {tool} environment: run pipeline setup")
        with patch.dict(os.environ, {"FC": self.host_compiler}):
            metadata = adapt_project(
                tool, source_dir, python, root / "generated", compiler_root=self.compiler_root,
                source_patterns=("physics.f90",), target="gpu", gpu_options=self.gpu_options)
            self.assertEqual(metadata["rejected"], [], json.dumps(metadata["rejected"], indent=2))
            self.assertEqual(metadata["discovered_regions"], 1)
            self.assertEqual(metadata["outlined_regions"], 1)
            region, = metadata["regions"]
            self.assertEqual(region["loop_nests"], 2)
            self.assertEqual(region["loop_ranks"], [2, 2])
            self.assertEqual(region["gpu_scope"]["dispatch"], "collective_single")
            self.assertEqual(region["gpu_scope"]["expected_omp_level"], 1)
            self.assertEqual(region["transformation"]["gpu"]["data_regions"], 1)
            # Unequal loop bounds cannot be fused by the current legal fusion pass.
            self.assertEqual(region["transformation"]["gpu"]["kernel_regions"], 2)
            binary = build(
                elmm, poisfft, root / "build", self.build_python, root / "lib", 4, 4, [],
                extra_sources=[Path(path) for path in metadata["extra_sources"]],
                patched_sources=[source_dir / path for path in metadata["changed_files"]],
                gpu_sources=metadata["gpu_sources"], gpu_options=self.gpu_options)
        return root, binary, metadata

    def assert_transfers(self, log, tool, metadata):
        uploads = [int(value) for value in re.findall(r"(?m)^FORT_RUNTIME upload bytes=(\d+)", log)]
        downloads = [int(value) for value in re.findall(r"(?m)^FORT_RUNTIME download bytes=(\d+)", log)]
        field_bytes = east_bytes = 23 * 11 * 8
        north_bytes = 26 * 13 * 8
        self.assertCountEqual(uploads, [field_bytes, east_bytes, north_bytes] * 2)
        self.assertCountEqual(downloads, [east_bytes, north_bytes] * 2)
        self.assertEqual(sum(uploads), 2 * (field_bytes + east_bytes + north_bytes))
        self.assertEqual(sum(downloads), 2 * (east_bytes + north_bytes))

    def test_adjacent_unequal_domains_share_transfers(self):
        for tool in ("local",):
            with self.subTest(tool=tool):
                root, binary, metadata = self.generate(tool)
                for scenario in ("regular", "empty_first", "empty_both"):
                    with self.subTest(tool=tool, scenario=scenario):
                        environment = trace_environment(clean_environment(4), tool, enabled=True)
                        run = subprocess.run([str(binary), scenario], cwd=root, env=environment,
                                             capture_output=True, text=True, timeout=90, check=False)
                        log = run.stdout + run.stderr
                        (root / f"{scenario}.log").write_text(log)
                        self.assertEqual(run.returncode, 0, log)
                        self.assertIn("GROUPED_GPU_PASS " + scenario, log)
                        self.assertNotIn("Another OpenMP runtime", log)
                        self.assertNotIn("libgomp: TODO", log)
                        coverage = execution_coverage(log, metadata["regions"])
                        self.assertEqual(coverage["unobserved_regions"], [], log)
                        activity = gpu_activity(log, tool, metadata["regions"])
                        (root / f"{scenario}-activity.json").write_text(json.dumps(activity, indent=2) + "\n")
                        if scenario == "regular":
                            self.assertTrue(activity["verified"], log)
                            self.assertEqual(activity["kernel_launches"], 4, log)
                            self.assert_transfers(log, tool, metadata)


if __name__ == "__main__":
    unittest.main()
