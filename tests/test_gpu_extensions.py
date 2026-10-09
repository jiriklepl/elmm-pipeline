"""Opt-in native coverage for the richer local GPU entry contract.

Set ELMM_GPU_NATIVE_TESTS=1 after setup. ELMM_COMPILER_ROOT and
ELMM_{LOCAL,BUILD}_PYTHON select explicit, independent installations.
ELMM_GPU_TEST_ARTIFACTS retains generated sources and execution evidence.
These are correctness runs, with runtime tracing deliberately enabled.
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
from elmm_pipeline.common import clean_environment, digest_tree
from elmm_pipeline.gpu_validation import gpu_activity, trace_environment
from elmm_pipeline.instrumentation import execution_coverage


COMBINED_SOURCE = """module extension_fields
use omp_lib, only: omp_get_num_threads, omp_get_thread_num
implicit none
real(kind=8), allocatable :: field(:,:), scratch(:,:), result(:,:)
contains
subroutine advance(last_row, mode, team_size)
integer, intent(in) :: last_row, mode
integer, intent(out) :: team_size
integer :: column, row
real(kind=8) :: weight
team_size=0
if (mode /= 0) then
!$omp parallel private(column,row,weight)
if (omp_get_thread_num()==0) team_size=omp_get_num_threads()
!$omp do
do row=0,last_row
  weight=real(row,8)+0.25_8
  do column=0,size(field,1)-5
    if (field(column,row)>0.0_8) then
      scratch(column,row)=field(column,row)+weight
    else
      scratch(column,row)=field(column,row)-weight
    end if
  end do
end do
!$omp end do
!$omp do
do row=0,last_row
  do column=0,size(field,1)-5
    if (scratch(column-1,row)>0.0_8) then
      result(column,row)=scratch(column,row)+scratch(column-1,row)
    else
      result(column,row)=scratch(column,row)-scratch(column-1,row)
    end if
  end do
end do
!$omp end do
!$omp end parallel
end if
end subroutine
end module
"""

COMBINED_DRIVER = """program compare
use extension_fields, only: candidate=>advance, cf=>field, cs=>scratch, cr=>result
use reference_fields, only: reference=>advance, rf=>field, rs=>scratch, rr=>result
implicit none
integer :: column,row,repetition,shape,upper_x,upper_y,mode,cteam,rteam,last_row
character(len=32) :: scenario
call get_command_argument(1,scenario)
select case(trim(scenario))
case('unallocated_empty','unallocated_skipped')
  mode=1
  last_row=-1
  if (scenario=='unallocated_skipped') then
    mode=0
    last_row=5
  end if
  call reference(last_row,mode,rteam)
  call candidate(last_row,mode,cteam)
  if(cteam/=rteam) error stop 'native fallback team mismatch'
  if(allocated(cf).or.allocated(cs).or.allocated(cr)) error stop 'unexpected allocation'
case('regular','allocated_empty','allocated_skipped')
  do shape=1,2
    upper_x=5+shape*4
    upper_y=4+shape*2
    allocate(cf(-2:upper_x,-2:upper_y),cs(-2:upper_x,-2:upper_y),cr(-2:upper_x,-2:upper_y))
    allocate(rf(-2:upper_x,-2:upper_y),rs(-2:upper_x,-2:upper_y),rr(-2:upper_x,-2:upper_y))
    do repetition=1,2
      do row=-2,upper_y
        do column=-2,upper_x
          cf(column,row)=real(100+column+20*row+1000*repetition,8)
          if(modulo(column+row+repetition,2)==0) cf(column,row)=-cf(column,row)
        end do
      end do
      rf=cf
      cs=-700.0_8-real(repetition,8)
      cr=-900.0_8-real(repetition,8)
      rs=cs
      rr=cr
      last_row=upper_y-1
      mode=1
      if(scenario=='allocated_empty') last_row=-1
      if(scenario=='allocated_skipped') mode=0
      call reference(last_row,mode,rteam)
      call candidate(last_row,mode,cteam)
      if(cteam/=rteam) error stop 'host team mismatch'
      if(mode==1.and.cteam/=4) error stop 'expected four host threads'
      if(any(cf/=rf)) error stop 'read-only input changed'
      if(any(cs/=rs)) error stop 'scratch mismatch or changed halo'
      if(any(cr/=rr)) error stop 'result mismatch or changed halo'
    end do
    deallocate(cf,cs,cr)
    deallocate(rf,rs,rr)
  end do
case default
  error stop 'unknown scenario'
end select
print *, 'GPU_EXTENSIONS_PASS ',trim(scenario)
end program
"""

DEPENDENCE_SOURCE = """module extension_fields
use omp_lib, only: omp_get_num_threads, omp_get_thread_num
implicit none
contains
subroutine advance(a,b,n,team_size)
real(kind=8), allocatable, intent(inout) :: a(:),b(:)
integer, intent(in) :: n
integer, intent(out) :: team_size
integer :: cell
!$omp parallel private(cell)
if (omp_get_thread_num()==0) team_size=omp_get_num_threads()
!$omp do
do cell=2,n
  FIRST_ASSIGNMENT
end do
!$omp end do
!$omp do
do cell=2,n
  SECOND_ASSIGNMENT
end do
!$omp end do
!$omp end parallel
end subroutine
end module
"""

DEPENDENCE_DRIVER = """program compare
use extension_fields, only: candidate=>advance
use reference_fields, only: reference=>advance
implicit none
real(kind=8), allocatable :: a(:),b(:),expected_a(:),expected_b(:)
integer :: cell,repetition,cteam,rteam
allocate(a(-2:19),b(-2:19),expected_a(-2:19),expected_b(-2:19))
do repetition=1,2
  do cell=-2,19
    a(cell)=real(100*repetition+cell,8)
    b(cell)=real(-100*repetition+cell,8)
  end do
  expected_a=a
  expected_b=b
  call reference(expected_a,expected_b,15,rteam)
  call candidate(a,b,15,cteam)
  if(cteam/=4.or.rteam/=4) error stop 'expected four host threads'
  if(any(a/=expected_a)) error stop 'a mismatch or changed halo'
  if(any(b/=expected_b)) error stop 'b mismatch or changed halo'
end do
print *, 'GPU_EXTENSIONS_PASS ordered'
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

# This runs beside the relocated package, from an unrelated working directory.
# All inputs are explicit; no checkout-relative compiler/env lookup is available.
ISOLATED_WORKER = """import json
import os
import subprocess
import sys
from pathlib import Path
import elmm_pipeline
from elmm_pipeline.adaptation import adapt_project
from elmm_pipeline.build import build

request = json.loads(Path(sys.argv[1]).read_text())
root = Path(request['root'])
snapshot = Path(request['compiler_root'])
python = Path(request['python'])
source_dir = root / 'ELMM/src'
os.environ['FC'] = request['host_compiler']
metadata = adapt_project('local', source_dir, python, root / 'generated',
                         compiler_root=snapshot, source_patterns=('physics.f90',),
                         target='gpu', gpu_options=request['gpu_options'])
if metadata['outlined_regions'] != 1 or metadata['rejected']:
    raise RuntimeError(json.dumps(metadata['rejected'], indent=2))
binary = build(root / 'ELMM', root / 'PoisFFT', root / 'build',
               Path(request['build_python']), root / 'lib', 4, 4, [],
               extra_sources=[Path(p) for p in metadata['extra_sources']],
               patched_sources=[source_dir / p for p in metadata['changed_files']],
               gpu_sources=metadata['gpu_sources'], gpu_options=request['gpu_options'])
env = dict(os.environ, PYTHONPATH=str(snapshot))
origin = subprocess.check_output([str(python), '-c',
    'import compiler; print(compiler.__file__)'], cwd=snapshot, env=env, text=True).strip()
(root / 'isolated-result.json').write_text(json.dumps({
    'package': elmm_pipeline.__file__, 'compiler': origin,
    'binary': str(binary), 'metadata': metadata}, indent=2))
"""


@unittest.skipUnless(os.environ.get("ELMM_GPU_NATIVE_TESTS") == "1",
                     "Set ELMM_GPU_NATIVE_TESTS=1 for native local GPU extension tests")
class LocalGPUExtensionTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = Path(__file__).resolve().parents[1]
        self.python = Path(os.environ.get(
            "ELMM_LOCAL_PYTHON", self.pipeline / ".cache/envs/local/bin/python")).absolute()
        self.build_python = Path(os.environ.get(
            "ELMM_BUILD_PYTHON", self.pipeline / ".cache/envs/build/bin/python")).absolute()
        self.compiler_root = Path(os.environ.get(
            "ELMM_COMPILER_ROOT", self.pipeline / ".cache/repos/local-compiler")).resolve()
        self.host_compiler = shutil.which(os.environ.get("FC", "gfortran-15")) or shutil.which("gfortran")
        self.assertTrue(self.host_compiler, "Install GNU Fortran before enabling native tests")
        for path in (self.python, self.build_python, self.compiler_root / "compiler/__main__.py"):
            self.assertTrue(path.is_file(), f"Missing explicit installation: {path}; run pipeline setup")
        self.gpu_options = {key: os.environ[variable] for key, variable in (
            ("nvcc", "ELMM_NVCC"), ("cuda_host_cxx", "ELMM_CUDA_HOST_CXX"),
            ("architecture", "ELMM_GPU_ARCH")) if os.environ.get(variable)}
        if os.environ.get("ELMM_GPU_TEST_ARTIFACTS"):
            artifacts = Path(os.environ["ELMM_GPU_TEST_ARTIFACTS"]).resolve() / self._testMethodName
            artifacts.mkdir(parents=True, exist_ok=True)
            self.root = Path(tempfile.mkdtemp(prefix="run-", dir=artifacts))
        else:
            temporary = tempfile.TemporaryDirectory(prefix="gpu-extensions-")
            self.addCleanup(temporary.cleanup)
            self.root = Path(temporary.name)

    def fixture(self, root, source, driver):
        source_dir = root / "ELMM/src"
        source_dir.mkdir(parents=True, exist_ok=True)
        (root / "PoisFFT/src").mkdir(parents=True, exist_ok=True)
        (root / "PoisFFT/src/poisfft.f90").touch()
        (source_dir / "SConstruct").write_text(SCONS)
        (source_dir / "physics.f90").write_text(source)
        (source_dir / "reference.f90").write_text(
            source.replace("module extension_fields", "module reference_fields"))
        (source_dir / "main.f90").write_text(driver)
        return source_dir

    def generate(self, name, source, driver):
        root = self.root / name
        source_dir = self.fixture(root, source, driver)
        with patch.dict(os.environ, {"FC": self.host_compiler}):
            metadata = adapt_project(
                "local", source_dir, self.python, root / "generated", compiler_root=self.compiler_root,
                source_patterns=("physics.f90",), target="gpu", gpu_options=self.gpu_options)
            self.assert_entry(metadata)
            binary = build(
                root / "ELMM", root / "PoisFFT", root / "build", self.build_python,
                root / "lib", 4, 4, [],
                extra_sources=[Path(p) for p in metadata["extra_sources"]],
                patched_sources=[source_dir / p for p in metadata["changed_files"]],
                gpu_sources=metadata["gpu_sources"], gpu_options=self.gpu_options)
        return root, binary, metadata

    def assert_entry(self, metadata, *, kernels=None, allocated=None):
        self.assertEqual(metadata["rejected"], [], json.dumps(metadata["rejected"], indent=2))
        self.assertEqual(metadata["discovered_regions"], 1)
        self.assertEqual(metadata["outlined_regions"], 1)
        region, = metadata["regions"]
        self.assertEqual(region["loop_nests"], 2)
        self.assertEqual(region["gpu_scope"]["dispatch"], "collective_single")
        self.assertEqual(region["gpu_scope"]["expected_omp_level"], 1)
        if allocated is not None:
            self.assertCountEqual(region["gpu_scope"]["allocated_arrays"], allocated)
        transformation = region["transformation"]
        self.assertEqual(transformation["gpu"]["data_regions"], 1)
        report = transformation["compiler_report"]
        self.assertTrue(report["supported"])
        self.assertEqual(transformation["gpu"]["kernel_regions"], report["parallel_regions"])
        self.assertGreaterEqual(report["parallel_regions"], 2)
        if kernels is not None:
            self.assertEqual(report["parallel_regions"], kernels)
        return region

    def execute(self, root, binary, metadata, scenario):
        run = subprocess.run(
            [str(binary), scenario], cwd=root,
            env=trace_environment(clean_environment(4), "local", enabled=True),
            capture_output=True, text=True, timeout=90, check=False)
        log = run.stdout + run.stderr
        (root / f"{scenario}.log").write_text(log)
        self.assertEqual(run.returncode, 0, log)
        self.assertIn("GPU_EXTENSIONS_PASS " + scenario, log)
        self.assertNotIn("Another OpenMP runtime", log)
        self.assertNotIn("libgomp: TODO", log)
        activity = gpu_activity(log, "local", metadata["regions"])
        (root / f"{scenario}-activity.json").write_text(json.dumps(activity, indent=2) + "\n")
        return log, activity

    def assert_gpu_calls(self, log, activity, metadata, calls):
        self.assertTrue(activity["verified"], log)
        self.assertEqual(activity["kernel_launches"],
                         calls * metadata["regions"][0]["transformation"]["gpu"]["kernel_regions"], log)
        self.assertEqual(execution_coverage(log, metadata["regions"])["unobserved_regions"], [], log)

    def assert_transfers(self, log, uploads, downloads):
        self.assertCountEqual([int(n) for n in re.findall(r"(?m)^FORT_RUNTIME upload bytes=(\d+)", log)],
                              uploads)
        self.assertCountEqual([int(n) for n in re.findall(r"(?m)^FORT_RUNTIME download bytes=(\d+)", log)],
                              downloads)

    def test_allocated_conditional_imperfect_entry_matches_native_across_invocations(self):
        root, binary, metadata = self.generate("combined", COMBINED_SOURCE, COMBINED_DRIVER)
        self.assert_entry(metadata, allocated=["field", "scratch", "result"])
        for scenario in ("regular", "allocated_empty", "allocated_skipped",
                         "unallocated_empty", "unallocated_skipped"):
            with self.subTest(scenario=scenario):
                log, activity = self.execute(root, binary, metadata, scenario)
                if scenario == "regular":
                    self.assert_gpu_calls(log, activity, metadata, calls=4)
                    # Every call owns one shared allocation per captured array,
                    # including both in/out arrays' unchanged halo elements.
                    sizes = [12 * 9 * 8, 16 * 11 * 8]
                    self.assert_transfers(log, [size for size in sizes for _ in range(6)],
                                          [size for size in sizes for _ in range(4)])
                if scenario in {"allocated_skipped", "unallocated_empty", "unallocated_skipped"}:
                    self.assertEqual(activity["kernel_launches"], 0, log)
                    self.assertEqual(activity["allocations"], 0, log)
                    self.assert_transfers(log, [], [])

    def test_cross_point_raw_war_waw_keep_two_ordered_kernels(self):
        cases = {
            "raw": ("a(cell)=real(cell,8)", "b(cell)=a(cell-1)", ["a", "b"]),
            "war": ("b(cell)=a(cell+1)", "a(cell)=real(cell,8)", ["a", "b"]),
            "waw": ("a(cell)=real(cell,8)", "a(cell+1)=2.0_8*real(cell,8)", ["a"]),
        }
        for name, (first, second, captured) in cases.items():
            with self.subTest(dependence=name):
                source = DEPENDENCE_SOURCE.replace("FIRST_ASSIGNMENT", first).replace("SECOND_ASSIGNMENT", second)
                root, binary, metadata = self.generate(name, source, DEPENDENCE_DRIVER)
                self.assert_entry(metadata, kernels=2, allocated=captured)
                log, activity = self.execute(root, binary, metadata, "ordered")
                self.assert_gpu_calls(log, activity, metadata, calls=2)
                self.assert_transfers(log, [22 * 8] * (2 * len(captured)),
                                      [22 * 8] * (2 * len(captured)))

    def test_relocated_pipeline_uses_only_the_explicit_compiler_snapshot(self):
        isolated_pipeline = self.root / "independent-pipeline"
        isolated_compiler = self.root / "independent-compiler"
        ignore = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache")
        shutil.copytree(self.pipeline / "elmm_pipeline", isolated_pipeline / "elmm_pipeline", ignore=ignore)
        shutil.copytree(self.compiler_root / "compiler", isolated_compiler / "compiler", ignore=ignore)
        self.assertEqual(digest_tree(isolated_compiler / "compiler"),
                         digest_tree(self.compiler_root / "compiler"))
        root = self.root / "application"
        self.fixture(root, COMBINED_SOURCE, COMBINED_DRIVER)
        request_path = root / "request.json"
        request_path.write_text(json.dumps({
            "root": str(root), "compiler_root": str(isolated_compiler),
            "python": str(self.python), "build_python": str(self.build_python),
            "host_compiler": self.host_compiler, "gpu_options": self.gpu_options}, indent=2))
        runner = isolated_pipeline / "run.py"
        runner.write_text(ISOLATED_WORKER)
        unrelated_cwd = self.root / "unrelated-working-directory"
        unrelated_cwd.mkdir()
        environment = clean_environment(4)
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        environment["PYTHONNOUSERSITE"] = "1"
        run = subprocess.run([str(self.python), str(runner), str(request_path)], cwd=unrelated_cwd,
                             env=environment, capture_output=True, text=True, timeout=240, check=False)
        (root / "isolation-build.log").write_text(run.stdout + run.stderr)
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        result = json.loads((root / "isolated-result.json").read_text())
        self.assertTrue(Path(result["package"]).is_relative_to(isolated_pipeline))
        self.assertTrue(Path(result["compiler"]).is_relative_to(isolated_compiler))
        self.assert_entry(result["metadata"], allocated=["field", "scratch", "result"])
        log, activity = self.execute(root, Path(result["binary"]), result["metadata"], "regular")
        self.assert_gpu_calls(log, activity, result["metadata"], calls=4)


if __name__ == "__main__":
    unittest.main()
