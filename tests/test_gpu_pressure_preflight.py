"""Inject CUDA visibility statuses into the real adapter without a device."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CONTROL = r'''
#include "gpu_pressure.h"
#include <cuda_runtime_api.h>
#include <cassert>
#include <cstdlib>
static int mode, visibility_calls, selections, claims, providers;
extern "C" cudaError_t cudaGetDeviceCount(int *count) {
  ++visibility_calls;*count=0;
  return mode==0?cudaErrorNoDevice:mode==1?cudaSuccess:cudaErrorLaunchFailure;
}
extern "C" cudaError_t cudaGetDevice(int *device) { ++selections;*device=0;return cudaSuccess; }
extern "C" cudaError_t cudaSetDevice(int) { ++selections;return cudaSuccess; }
extern "C" const char *cudaGetErrorString(cudaError_t) { return "injected CUDA status"; }
extern "C" int fort_scope_domain_device_get_v1(fort_scope_domain_t,int *device) {
  *device=0;return mode==3?FORT_SCOPE_STALE:FORT_SCOPE_OK;
}
extern "C" int fort_scope_domain_workspace_reserve_v1(fort_scope_domain_t,size_t,uint64_t*) {
  ++claims;return FORT_SCOPE_EXECUTION;
}
extern "C" int fort_scope_domain_workspace_resize_v1(fort_scope_domain_t,uint64_t,size_t) { return FORT_SCOPE_EXECUTION; }
extern "C" int fort_scope_domain_workspace_release_v1(fort_scope_domain_t,uint64_t) { return FORT_SCOPE_EXECUTION; }
extern "C" int fort_scope_domain_get_v1(fort_scope_t,fort_scope_domain_t*) { return FORT_SCOPE_EXECUTION; }
extern "C" int fort_scope_external_execute_v1(fort_scope_t,const fort_scope_external_call_v1*,
 fort_scope_external_worker_v1,void*,fort_scope_external_report_v1*) { return FORT_SCOPE_EXECUTION; }
extern "C" int poisson_solver_create_v1(const poisson_solver_config_v1*,poisson_solver_handle_v1**) {
  ++providers;return POISSON_SOLVER_EXECUTION;
}
extern "C" int poisson_solver_destroy_v1(poisson_solver_handle_v1**) { return POISSON_SOLVER_EXECUTION; }
extern "C" int poisson_solver_workspace_query_v1(poisson_solver_handle_v1*,poisson_solver_workspace_v1*) { return POISSON_SOLVER_EXECUTION; }
extern "C" int poisson_solver_execute_device_v1(poisson_solver_handle_v1*,void*,size_t,const void*,size_t,
 const int32_t*,const int32_t*,void*,int*) { return POISSON_SOLVER_EXECUTION; }
extern "C" const char *poisson_solver_last_error_v1(void) { return "unexpected provider execution"; }
extern "C" int elmm_gpu_pressure_launch_halo_v1(double*,const int32_t*,const int32_t*,void*,int*) { return FORT_SCOPE_EXECUTION; }
int main(int argc,char **argv) {
  assert(argc==2);mode=std::atoi(argv[1]);
  elmm_gpu_pressure_config_v1 config{};
  config.version=config.solver.version=1;config.solver.dimensions=3;
  config.solver.approximation=2;config.solver.device_budget_bytes=1048576;
  for(int i=0;i<3;++i) { config.solver.sizes[i]=2;config.solver.lengths[i]=1;config.phi_lower_bounds[i]=-1; }
  elmm_gpu_pressure_handle_v1 *handle=nullptr;
  const int status=elmm_gpu_pressure_create_v1(1,&config,&handle);
  assert(status==(mode<=1?FORT_SCOPE_RESOURCE:mode==2?FORT_SCOPE_EXECUTION:FORT_SCOPE_STALE));
  assert(!handle && !claims && !providers && !selections);
  assert(visibility_calls==(mode==3?0:1));
}
'''


def test_only_no_device_is_recoverable_before_provider_or_workspace_work(tmp_path):
    runtime_value = os.environ.get("ELMM_PRESSURE_RUNTIME")
    if not runtime_value:
        pytest.skip("Explicit current runtime header required: ELMM_PRESSURE_RUNTIME")
    runtime = Path(runtime_value).resolve(strict=True)
    provider_value = os.environ.get("ELMM_POISSON_SOURCE")
    if not provider_value:
        pytest.skip("Explicit local solver source required: ELMM_POISSON_SOURCE")
    root = Path(__file__).resolve().parents[1]
    adapter = root / "elmm_pipeline/gpu_pressure_sources"
    provider = Path(provider_value).resolve(strict=True) / "include"
    compiler = shutil.which("g++-14")
    assert compiler
    cuda = Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "include"
    assert (cuda / "cuda_runtime_api.h").is_file()
    source, binary = tmp_path / "control.cpp", tmp_path / "control"
    source.write_text(CONTROL)
    argv = [compiler, "-std=c++17", "-I", str(runtime), "-I", str(adapter), "-I", str(provider),
            "-I", str(cuda), str(source), str(adapter / "gpu_pressure.cpp"), "-o", str(binary)]
    compiled = subprocess.run(argv, capture_output=True, text=True, timeout=120, check=False)
    (tmp_path / "compile.stdout").write_text(compiled.stdout)
    (tmp_path / "compile.stderr").write_text(compiled.stderr)
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    for mode in range(4):
        result = subprocess.run([str(binary), str(mode)], capture_output=True, text=True, timeout=15, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
