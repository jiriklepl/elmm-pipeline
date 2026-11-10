"""Explicit source selection and pinned CPU lazy-plan behavior."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from elmm_pipeline.build_receipts import file_identity
from elmm_pipeline.gpu_pressure_build import (
    _json_identity,
    prepare_lazy_pressure_backend,
    prepare_poisfft_provider_manifest,
    require_pressure_backend,
)
from elmm_pipeline.gpu_pressure_lazy import (
    materialize_lazy_pressure_wrapper,
    prepare_lazy_poisfft_contract,
)


def test_lazy_source_switch_changes_only_one_use_and_leaves_original_unchanged(tmp_path):
    source = tmp_path / "poisson.f90"
    body = ("module PoissonSolvers\ncontains\n  subroutine Poiss_PoisFFT(Phi,RHS)\n#ifdef DPREC\n"
            "    use PoisFFT, PoisFFT_Solver => PoisFFT_Solver3D_DP\n#endif\n"
            "    Solver = PoisFFT_Solver(n,L,bcs,order)\n    call system_clock(count=t1)\n"
            "    call Execute(Solver,Phi,RHS)\n    call system_clock(count=t2)\n"
            "    if(debug) write(*,*) t2-t1\n  end subroutine Poiss_PoisFFT\nend module\n")
    source.write_text(body)
    before = file_identity(source)
    qualification = tmp_path / "qualification.json"
    # A byte-binding fixture, not a numerical/compiler qualification.
    qualification.write_text(json.dumps({"source_contract": {"caller": {
        "procedure": "poissonsolvers::poiss_poisfft", "sha256": before["sha256"]}}}))
    result = materialize_lazy_pressure_wrapper(source, tmp_path / "selected", original_qualification=file_identity(qualification))
    assert source.read_text() == body
    assert result["original"] == before
    changed = Path(result["replacement"]["path"]).read_text()
    assert changed == body.replace("use PoisFFT,", "use elmm_gpu_pressure_lazy_poisfft,", 1)
    assert result["source_authority"] is False
    assert result["placement_authority"] is False
    assert result["performance_claim"] is False
    assert result["configured_source_requirement"] == {"required_defines": ["DPREC"], "forbidden_defines": ["PAR"]}
    with pytest.raises(ValueError, match="qualified original"):
        source.write_text(body + "! unqualified source mutation\n")
        materialize_lazy_pressure_wrapper(source, tmp_path / "tampered", original_qualification=file_identity(qualification))
    assert not (tmp_path / "tampered").exists()


def test_lazy_provider_query_requires_exact_explicit_prepared_source(tmp_path):
    from test_gpu_pressure_build import backend_request

    compiler = Path(shutil.which("python3"))
    request = backend_request(tmp_path, compiler)
    path = Path(request["manifest"])
    backend = json.loads(path.read_text())
    for name in ("gpu_pressure_poisfft.f90", "gpu_pressure_lazy_poisfft.f90"):
        source = tmp_path / name
        source.write_text("! Source identity fixture; no execution qualification\n")
        backend["extra_sources"].append(str(source))
        backend["inputs"].append(file_identity(source))
    backend["requirements"]["cpu_fallback"] = {
        "implementation": "lazy_poisfft_v1", "source": file_identity(source)}

    def seal():
        backend["identity"] = _json_identity({key: value for key, value in backend.items() if key != "identity"})
        path.write_text(json.dumps(backend))
        request["sha256"] = file_identity(path)["sha256"]

    seal()
    qualification = tmp_path / "numeric.json"
    qualification.write_text('{"execution_evidence":false}')
    result = prepare_poisfft_provider_manifest(request, tmp_path / "provider.json", fortran=compiler,
        qualification=file_identity(qualification), original_body_sha256="a" * 64, task_contract_sha256="b" * 64)
    provider = json.loads(Path(result["path"]).read_text())
    assert provider["metadata_query"]["module"] == "elmm_gpu_pressure_lazy_poisfft"
    assert provider["metadata_query"]["procedure"] == "elmm_gpu_pressure_configure_lazy_v1"
    assert provider["version"] == 2
    assert provider["artifacts"][str(source)] == file_identity(source)["sha256"]
    assert provider["execution"]["fallback"] == "prework_only"
    backend["requirements"]["cpu_fallback"]["source"]["sha256"] = "0" * 64
    seal()
    with pytest.raises(ValueError, match="exact prepared implementation"):
        prepare_poisfft_provider_manifest(request, tmp_path / "wrong.json", fortran=compiler,
            qualification=file_identity(qualification), original_body_sha256="a" * 64, task_contract_sha256="b" * 64)


def test_lazy_backend_preserves_exact_compiled_origin_and_rejects_base_drift(tmp_path):
    from test_gpu_pressure_build import backend_request

    original = tmp_path / "original"
    original.mkdir()
    compiler = Path(shutil.which("python3"))
    request = backend_request(original, compiler)
    before = Path(request["manifest"]).read_bytes()
    base = require_pressure_backend(request, fortran=compiler)
    selected = prepare_lazy_pressure_backend(request, tmp_path / "lazy", fortran=compiler)
    value = require_pressure_backend(selected, fortran=compiler)
    assert Path(request["manifest"]).read_bytes() == before
    for key in ("objects", "provider_library", "runtime", "command_receipts", "fortran_compiler", "provider_source"):
        assert value[key] == base[key]
    assert value["extra_sources"][:-1] == base["extra_sources"]
    assert value["requirements"]["cpu_fallback"]["compiled_artifact_origin"] == request
    assert value["requirements"]["cpu_fallback"]["fresh_compilation"] is False
    Path(request["manifest"]).write_bytes(before + b"\n")
    with pytest.raises(ValueError, match="input changed"):
        require_pressure_backend(selected, fortran=compiler)


@pytest.mark.parametrize("mutation", ["original_receiver_qualification", "changed_analysis_source"])
def test_lazy_contract_cannot_reuse_old_receiver_or_different_source(tmp_path, mutation):
    from test_gpu_pressure_build import backend_request

    original = tmp_path / "original"
    original.mkdir()
    compiler = Path(shutil.which("python3"))
    request = backend_request(original, compiler)
    selected = prepare_lazy_pressure_backend(request, tmp_path / "lazy", fortran=compiler)
    implementation = Path(__file__).resolve().parents[1] / "elmm_pipeline/gpu_pressure_sources/gpu_pressure_lazy_poisfft.f90"
    analysis = {"source_inputs": {str(implementation): file_identity(implementation)["sha256"]}}
    opaque = {"kind": "explicit_lazy_cpu_poisfft_receiver_qualification", "implementation": file_identity(implementation)}
    if mutation == "original_receiver_qualification":
        opaque["kind"] = "unchanged_original_PoisFFT_receiver"
    else:
        analysis["source_inputs"][str(implementation)] = "0" * 64
    qualification = tmp_path / "opaque.json"
    qualification.write_text(json.dumps(opaque))
    with pytest.raises(ValueError, match="exact separately qualified source"):
        prepare_lazy_poisfft_contract([], analysis, tmp_path / "rejected", backend=selected, fortran=compiler,
            numerical_qualification=file_identity(qualification), opaque_qualification=file_identity(qualification),
            workspace_cap_bytes=4194304)
    assert not (tmp_path / "rejected").exists()


DRIVER = r'''
program lazy_qualification
  use iso_c_binding
  use, intrinsic :: ieee_arithmetic
  use, intrinsic :: ieee_exceptions
  use PoisFFT, only: native_solver => PoisFFT_Solver3D_DP, native_execute => Execute, native_finalize => Finalize
  use elmm_gpu_pressure
  use elmm_gpu_pressure_lazy_poisfft
  use fort_scoped_memory, only: FORT_SCOPE_ARGUMENT, FORT_SCOPE_BOUNDARY
  implicit none
  type(PoisFFT_Solver3D_DP) :: lazy, invalid
  type(native_solver) :: reference
  type(elmm_gpu_pressure_config_v1) :: config, before_config
  integer :: n(3), bcs(6), order, family, repeat, i,j,k,s
  real(c_double) :: lengths(3), bad_lengths(3)
  real(c_double) :: rhs(9,7,11), before(9,7,11), phi(11,9,13), expected(11,9,13)
  logical :: flag
  character(32) :: arg
  call get_command_argument(1,arg);read(arg,*) order
  call get_command_argument(2,arg);read(arg,*) family
  n=[7,5,9]; lengths=[2.75_c_double,1.25_c_double,3.5_c_double]
  bcs=0
  if(family==1) bcs=4
  if(family==2) bcs(5:6)=4
  lazy=PoisFFT_Solver3D_DP(n,lengths,bcs,order)
  if(elmm_lazy_cpu_initializations(lazy)/=0) stop 1
  config%solver%device_budget_bytes=4194304
  do repeat=1,3
    s=elmm_gpu_pressure_configure_lazy_v1(config,lazy)
    if(s/=0) stop 2
    if(elmm_lazy_cpu_initializations(lazy)/=0) stop 3
  enddo
  if(any(config%solver%sizes/=n) .or. any(config%solver%lengths/=lengths)) stop 4
  if(any(config%solver%boundary_conditions/=bcs) .or. config%solver%approximation/=order) stop 5
  if(config%solver%device_budget_bytes/=4194304) stop 6
  ! Direct pinned CPU reference. Native fallback creates exactly one lazy
  ! receiver plan across two changed RHS values; metadata never creates it.
  reference=native_solver(n,lengths,bcs,order)
  do repeat=1,2
    rhs=-99
    do k=1,n(3)
      do j=1,n(2)
        do i=1,n(1)
          rhs(i+1,j+1,k+1)=sin(real(i+3*j+7*k,c_double))+real(repeat,c_double)/8
        enddo
      enddo
    enddo
    before=rhs; phi=-77; expected=-77
    call Execute(lazy,phi,rhs)
    if(elmm_lazy_cpu_initializations(lazy)/=1) stop 7
    call native_execute(reference,expected,rhs)
    if(any(rhs/=before)) stop 8
    if(any(abs(phi(3:9,3:7,3:11)-expected(3:9,3:7,3:11))>1e-14_c_double)) stop 9
    s=elmm_gpu_pressure_configure_lazy_v1(config,lazy)
    if(s/=0 .or. elmm_lazy_cpu_initializations(lazy)/=1) stop 10
  enddo
  before_config=config
  bad_lengths=lengths
  bad_lengths(1)=transfer(int(z'7ff0000000000001',c_int64_t),0._c_double)
  call ieee_set_flag(ieee_invalid,.false.)
  invalid=PoisFFT_Solver3D_DP(n,bad_lengths,bcs,order)
  call ieee_get_flag(ieee_invalid,flag)
  if(flag) stop 15
  s=elmm_gpu_pressure_configure_lazy_v1(config,invalid)
  call ieee_get_flag(ieee_invalid,flag)
  if(s/=FORT_SCOPE_ARGUMENT .or. flag) stop 11
  if(elmm_lazy_cpu_initializations(invalid)/=0) stop 12
  if(any(config%solver%lengths/=before_config%solver%lengths)) stop 13
  invalid=PoisFFT_Solver3D_DP(n,lengths,[2,2,2,2,2,2],order)
  s=elmm_gpu_pressure_configure_lazy_v1(config,invalid)
  if(s/=FORT_SCOPE_BOUNDARY .or. elmm_lazy_cpu_initializations(invalid)/=0) stop 14
  call Finalize(lazy)
  call Finalize(lazy)
  call native_finalize(reference)
  print *, 'LAZY_PINNED_CPU_OK', order, family
end program
'''


@pytest.fixture(scope="module")
def lazy_executable(tmp_path_factory):
    names = ("ELMM_PRESSURE_NATIVE_BUILD", "ELMM_PRESSURE_RUNTIME", "ELMM_PRESSURE_SOLVER_BUILD", "ELMM_PRESSURE_FFTW_DIR")
    for name in names:
        if name not in os.environ:
            pytest.skip("Explicit pinned lazy pressure inputs required: " + name)
    native, runtime, solver, fftw = (Path(os.environ[name]).resolve(strict=True) for name in names)
    fc = shutil.which("gfortran-15")
    assert fc
    directory = tmp_path_factory.mktemp("lazy_poisfft")
    source = Path(__file__).resolve().parents[1] / "elmm_pipeline/gpu_pressure_sources"
    commands = []

    def run(argv, label):
        commands.append(argv)
        result = subprocess.run(argv, cwd=directory, capture_output=True, text=True, timeout=120, check=False)
        (directory / (label + ".stdout")).write_text(result.stdout)
        (directory / (label + ".stderr")).write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr

    includes = ["-I", str(directory), "-I", str(solver / "modules"), "-I", str(native)]
    run([fc, "-c", str(runtime / "scoped_memory.f90"), "-o", str(directory / "memory.o")], "memory")
    run([fc, *includes, "-c", str(source / "gpu_pressure.f90"), "-o", str(directory / "api.o")], "api")
    run([fc, *includes, "-c", str(source / "gpu_pressure_lazy_poisfft.f90"), "-o", str(directory / "lazy.o")], "lazy")
    driver = directory / "driver.f90"
    driver.write_text(DRIVER)
    references = [native / "P" / (name + ".o") for name in ("customfftw3", "fft", "main", "parameters", "poisfft", "precisions")]
    libraries = [fftw / ("lib" + name + ".so") for name in ("fftw3", "fftw3f", "fftw3_omp", "fftw3f_omp")]
    binary = directory / "run"
    run([fc, "-ffree-line-length-none", "-fopenmp", *includes, str(driver), str(directory / "lazy.o"),
         *(str(p) for p in references), *(str(p) for p in libraries), "-o", str(binary)], "link")
    (directory / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    return binary


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("family", [0, 1, 2])
def test_lazy_constructor_and_query_create_no_cpu_plan_then_fallback_reuses_one(lazy_executable, order, family):
    result = subprocess.run([str(lazy_executable), str(order), str(family)], capture_output=True,
                            text=True, timeout=30, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "LAZY_PINNED_CPU_OK" in result.stdout
