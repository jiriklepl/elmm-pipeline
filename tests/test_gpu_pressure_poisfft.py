"""CPU qualification of actual pinned PoisFFT metadata and scratch reuse.

The explicit paths select authenticated original PoisFFT modules/objects. No
CUDA device, provider plan or manufactured library interface is used here.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

DRIVER = r'''
program qualification
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use PoisFFT, only: PoisFFT_Solver3D_DP, Execute, Finalize
  use elmm_gpu_pressure
  use elmm_gpu_pressure_poisfft
  use fort_scoped_memory, only: FORT_SCOPE_BOUNDARY, FORT_SCOPE_ARGUMENT
  implicit none
  type(PoisFFT_Solver3D_DP) :: actual, reference
  type(elmm_gpu_pressure_config_v1) :: config
  integer :: bcs(6), n(3), order, family, repeat, i,j,k,s
  real(c_double) :: lengths(3), rhs(9,7,11), before(9,7,11), phi(11,9,13), expected(11,9,13), length
  logical :: invalid
  character(32) :: arg
  call get_command_argument(1,arg);read(arg,*) order
  call get_command_argument(2,arg);read(arg,*) family
  n=[7,5,9];lengths=[2.75_c_double,1.25_c_double,3.5_c_double]
  bcs=0
  if(family==1) bcs=4
  if(family==2) bcs(5:6)=4
  actual=PoisFFT_Solver3D_DP(n,lengths,bcs,order)
  reference=PoisFFT_Solver3D_DP(n,lengths,bcs,order)
  config%solver%device_budget_bytes=4194304
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  if(s/=0) stop 1
  if(any(config%solver%sizes/=n) .or. any(config%solver%lengths/=lengths)) stop 2
  if(any(config%solver%boundary_conditions/=bcs) .or. config%solver%approximation/=order) stop 3
  if(config%solver%device_budget_bytes/=4194304) stop 4
  if(any(config%phi_lower_bounds/=-1) .or. any(config%rhs_lower_bounds/=0)) stop 5

  do repeat=1,2
    rhs=0
    do k=1,n(3)
      do j=1,n(2)
        do i=1,n(1)
          rhs(i+1,j+1,k+1)=sin(real(i+3*j+7*k,c_double))+real(repeat,c_double)/8
        enddo
      enddo
    enddo
    before=rhs;phi=-77;expected=-77
    ! Simulate arbitrary old CPU scratch after any intervening backend choice.
    ! Every supported original CPU path must reseed it before a payload read.
    if(associated(actual%cwork)) actual%cwork=cmplx(777._c_double,-444._c_double,c_double)
    if(associated(actual%rwork)) actual%rwork=333._c_double
    if(allocated(actual%Solvers1D)) then
      do i=1,size(actual%Solvers1D)
        if(associated(actual%Solvers1D(i)%rwork)) actual%Solvers1D(i)%rwork=555._c_double
        if(associated(actual%Solvers1D(i)%cwork)) actual%Solvers1D(i)%cwork=cmplx(666._c_double,-999._c_double,c_double)
      enddo
    endif
    if(allocated(actual%Solvers2D)) then
      do i=1,size(actual%Solvers2D)
        if(associated(actual%Solvers2D(i)%rwork)) actual%Solvers2D(i)%rwork=888._c_double
        if(associated(actual%Solvers2D(i)%cwork)) actual%Solvers2D(i)%cwork=cmplx(999._c_double,-666._c_double,c_double)
      enddo
    endif
    s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
    if(s/=0) stop 6
    if(associated(actual%rwork)) then
      if(any(actual%rwork/=333._c_double)) stop 7
    endif
    if(associated(actual%cwork)) then
      if(any(actual%cwork/=cmplx(777._c_double,-444._c_double,c_double))) stop 18
    endif
    if(allocated(actual%Solvers1D)) then
      do i=1,size(actual%Solvers1D)
        if(associated(actual%Solvers1D(i)%rwork)) then
          if(any(actual%Solvers1D(i)%rwork/=555._c_double)) stop 19
        endif
        if(associated(actual%Solvers1D(i)%cwork)) then
          if(any(actual%Solvers1D(i)%cwork/=cmplx(666._c_double,-999._c_double,c_double))) stop 20
        endif
      enddo
    endif
    if(allocated(actual%Solvers2D)) then
      do i=1,size(actual%Solvers2D)
        if(associated(actual%Solvers2D(i)%rwork)) then
          if(any(actual%Solvers2D(i)%rwork/=888._c_double)) stop 21
        endif
        if(associated(actual%Solvers2D(i)%cwork)) then
          if(any(actual%Solvers2D(i)%cwork/=cmplx(999._c_double,-666._c_double,c_double))) stop 22
        endif
      enddo
    endif
    call Execute(actual,phi,rhs)
    call Execute(reference,expected,rhs)
    if(any(rhs/=before)) stop 8
    if(any(abs(phi(3:9,3:7,3:11)-expected(3:9,3:7,3:11))>1e-14_c_double)) stop 9
  enddo

  actual%mpi%comm=0
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  if(s/=FORT_SCOPE_BOUNDARY) stop 10
  actual%mpi%comm=-1
  actual%offx=1
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  if(s/=FORT_SCOPE_BOUNDARY) stop 11
  actual%offx=0
  actual%gnx=actual%nx+1
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  if(s/=FORT_SCOPE_BOUNDARY) stop 12
  actual%gnx=actual%nx
  actual%approximation=0
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  if(s/=FORT_SCOPE_ARGUMENT) stop 13
  actual%approximation=order
  actual%BCs=2
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  if(s/=FORT_SCOPE_BOUNDARY) stop 14
  actual%BCs=bcs
  length=actual%Lx
  actual%Lx=transfer(int(z'7ff0000000000001',c_int64_t),0._c_double)
  call ieee_set_flag(ieee_invalid,.false.)
  s=elmm_gpu_pressure_configure_poisfft_v1(config,actual)
  call ieee_get_flag(ieee_invalid,invalid)
  if(s/=FORT_SCOPE_ARGUMENT .or. invalid) stop 15
  actual%Lx=length
  if(any(config%solver%sizes/=n) .or. any(config%solver%lengths/=lengths)) stop 16
  if(any(config%solver%boundary_conditions/=bcs) .or. config%solver%approximation/=order) stop 17
  call Finalize(actual)
  call Finalize(reference)
  print *, 'METADATA_AND_CPU_REUSE_OK'
end program
'''


@pytest.fixture(scope="module")
def metadata_executable(tmp_path_factory):
    values = {}
    for name in ("ELMM_PRESSURE_NATIVE_BUILD", "ELMM_PRESSURE_RUNTIME", "ELMM_PRESSURE_SOLVER_BUILD", "ELMM_PRESSURE_FFTW_DIR"):
        if name not in os.environ:
            pytest.skip("Explicit pinned pressure qualification inputs required: " + name)
        values[name] = Path(os.environ[name]).resolve(strict=True)
    fc = shutil.which("gfortran-15")
    assert fc
    native, runtime, solver, fftw = (values[name] for name in values)
    source = Path(__file__).resolve().parents[1] / "elmm_pipeline/gpu_pressure_sources"
    directory = tmp_path_factory.mktemp("poisfft_metadata")
    commands = []

    def run(argv, label):
        commands.append(argv)
        result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=120, check=False)
        (directory / (label + ".stdout")).write_text(result.stdout)
        (directory / (label + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr

    run([fc, "-c", str(runtime / "scoped_memory.f90"), "-o", str(directory / "memory.o")], "memory")
    includes = ["-I", str(directory), "-I", str(solver / "modules"), "-I", str(native)]
    run([fc, *includes, "-c", str(source / "gpu_pressure.f90"), "-o", str(directory / "api.o")], "api")
    run([fc, *includes, "-c", str(source / "gpu_pressure_poisfft.f90"), "-o", str(directory / "query.o")], "query")
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    references = [native / "P" / (name + ".o") for name in ("customfftw3", "fft", "main", "parameters", "poisfft", "precisions")]
    libraries = [fftw / ("lib" + name + ".so") for name in ("fftw3", "fftw3f", "fftw3_omp", "fftw3f_omp")]
    binary = directory / "run"
    run([fc, "-ffree-line-length-none", "-fopenmp", *includes, str(driver), str(directory / "query.o"),
         *(str(p) for p in references), *(str(p) for p in libraries), "-o", str(binary)], "link")
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return binary


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("family", [0, 1, 2])
def test_pinned_metadata_query_and_cpu_scratch_reinitialization(metadata_executable, order, family):
    result = subprocess.run([str(metadata_executable), str(order), str(family)], capture_output=True, text=True,
                            timeout=30, check=False, env={**os.environ, "OMP_NUM_THREADS": "4", "OMP_DYNAMIC": "FALSE"})
    (metadata_executable.parent / f"fd{order}-bc{family}.stdout").write_text(result.stdout)
    (metadata_executable.parent / f"fd{order}-bc{family}.stderr").write_text(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "METADATA_AND_CPU_REUSE_OK" in result.stdout
