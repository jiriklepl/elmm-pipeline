"""Opt-in end-to-end CUDA validation of generic call groups.

Enable with ELMM_GPU_NATIVE_TESTS=1; compiler and environment paths are explicit
through the same ELMM_* variables as the other native pipeline tests.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from elmm_pipeline.adaptation import adapt_project
from elmm_pipeline.build import build
from elmm_pipeline.common import clean_environment
from elmm_pipeline.gpu_validation import gpu_activity, trace_environment

from test_gpu_calls import SOURCE
from test_gpu_extensions import SCONS

DRIVER = """program check
use subject, only: candidate=>advance
use reference, only: native=>advance
implicit none
real(8),allocatable :: a(:),b(:),x(:),y(:)
integer :: extent,i,repetition
do extent=3,17,7
 allocate(a(-3:extent+3),b(-3:extent+3),x(-3:extent+3),y(-3:extent+3))
 do repetition=1,2
  do i=-3,extent+3
   a(i)=real(i*i+repetition,8)
   b(i)=-123-repetition
  end do
  x=a
  y=b
  call native(x,y,extent)
  call candidate(a,b,extent)
  if(any(a/=x).or.any(b/=y)) error stop 'call-group mismatch'
 end do
 deallocate(a,b,x,y)
end do
print *, 'CALL_GROUP_GPU_PASS'
end program
"""


@unittest.skipUnless(os.environ.get("ELMM_GPU_NATIVE_TESTS") == "1", "requires opt-in native GPU execution")
class GPUCallNativeTests(unittest.TestCase):
    def test_shared_entry_matches_native_with_real_cuda(self):
        self.run_case(SOURCE, DRIVER, 12)

    def test_section_policy_shares_prepared_helpers_through_public_interface(self):
        self.run_case(SOURCE, DRIVER, 12, policy="sections")

    def test_tiled_setup_and_empty_negative_stride_branch_on_cuda(self):
        source = SOURCE.replace("real(8) :: coefficient = 2", "real(8) :: coefficient = 2\ninteger :: stride=3")
        source = source.replace("integer :: i", "integer :: i,tile,step")
        source = source.replace("scale = coefficient", "scale = coefficient\nstep=stride")
        source = source.replace("!$omp parallel do private(i)", "!$omp parallel do private(i,tile)")
        source = source.replace("do i=-1,n", "do tile=-1,n,step\ndo i=tile,min(tile+step-1,n)")
        source = source.replace("!$omp end parallel do", "end do\n!$omp end parallel do")
        driver = DRIVER.replace("candidate=>advance", "candidate=>advance,stride")
        driver = driver.replace("native=>advance", "native=>advance,other_stride=>stride")
        driver = driver.replace("  call native(x,y,extent)", "  stride=9-6*repetition\n  other_stride=stride\n  call native(x,y,extent)")
        self.run_case(source, driver, 9)

    def run_case(self, source, driver, launches, *, policy="always"):
        pipeline = Path(__file__).resolve().parents[1]
        compiler = Path(os.environ.get("ELMM_COMPILER_ROOT", pipeline/".cache/repos/local-compiler"))
        python = Path(os.environ.get("ELMM_LOCAL_PYTHON", pipeline/".cache/envs/local/bin/python"))
        build_python = Path(os.environ.get("ELMM_BUILD_PYTHON", pipeline/".cache/envs/build/bin/python"))
        if os.environ.get("ELMM_GPU_TEST_ARTIFACTS"):
            root = Path(os.environ["ELMM_GPU_TEST_ARTIFACTS"])/self._testMethodName
            root.mkdir(parents=True, exist_ok=True)
        else:
            temporary = tempfile.TemporaryDirectory(prefix="gpu-calls-native-")
            self.addCleanup(temporary.cleanup)
            root = Path(temporary.name)
        options = {key: os.environ[variable] for key, variable in (
            ("nvcc", "ELMM_NVCC"), ("cuda_host_cxx", "ELMM_CUDA_HOST_CXX"),
            ("architecture", "ELMM_GPU_ARCH")) if os.environ.get(variable)}
        options.update(gpu_policy=policy, host_threads=4)
        sources = root/"ELMM/src"
        sources.mkdir(parents=True, exist_ok=True)
        (root/"PoisFFT/src").mkdir(parents=True, exist_ok=True)
        (root/"PoisFFT/src/poisfft.f90").touch()
        (sources/"SConstruct").write_text(SCONS)
        (sources/"physics.f90").write_text(source)
        (sources/"reference.f90").write_text(source.replace("module subject", "module reference"))
        (sources/"main.f90").write_text(driver)
        with patch.dict(os.environ, {"FC": os.environ.get("FC", "gfortran-15")}):
            metadata = adapt_project("local", sources, python, root/"generated", compiler_root=compiler,
                                     source_patterns=("physics.f90",), include=("advance",),
                                     target="gpu", gpu_options=options)
            self.assertEqual(metadata["outlined_regions"], 1, json.dumps(metadata["rejected"], indent=2))
            region, = metadata["regions"]
            self.assertEqual(region["gpu_scope"]["grouped_calls"], 2)
            binary = build(root/"ELMM", root/"PoisFFT", root/"build", build_python, root/"lib", 2, 4, [],
                           extra_sources=[Path(path) for path in metadata["extra_sources"]],
                           patched_sources=[sources/path for path in metadata["changed_files"]],
                           gpu_sources=metadata["gpu_sources"], gpu_options=options)
        run = subprocess.run([str(binary)], cwd=root, text=True, capture_output=True, timeout=90,
                             env=trace_environment(clean_environment(4), "local", enabled=True))
        log = run.stdout+run.stderr
        (root/"run.log").write_text(log)
        self.assertEqual(run.returncode, 0, log)
        self.assertIn("CALL_GROUP_GPU_PASS", log)
        activity = gpu_activity(log, "local", metadata["regions"])
        (root/"activity.json").write_text(json.dumps(activity, indent=2)+"\n")
        self.assertTrue(activity["verified"], log)
        self.assertEqual(activity["kernel_launches"], launches, log)
        # Both outputs are compared in full by the driver, including halos.
        # Whole-array CUDA uploads each shared array once per grouped call.
        # Separate helper entries would upload both arrays twice.
        uploaded = sum(map(int, re.findall(r"(?m)^FORT_RUNTIME upload bytes=(\d+)", log)))
        expected = (4*sum((extent+7)*8 for extent in (3,10,17)) if policy == "always"
                    else 2*sum((2*extent+3)*8 for extent in (3,10,17)))
        self.assertEqual(uploaded, expected, log)


if __name__ == "__main__":
    unittest.main()
