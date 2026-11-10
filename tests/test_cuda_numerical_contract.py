"""Compiler-owned numerical requirements survive public pipeline builds."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from elmm_pipeline.gpu_backend import transform
from elmm_pipeline.gpu_toolchain import compile_gpu_sources, validate_gpu_sources
from elmm_pipeline.local_adapter import LocalAdapterError, cuda_build_fields


def contract():
    value = {"schema_version": 1, "id": "separate-arithmetic-v1",
             "required_cuda_options": ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
             "required_host_options": ["-ffp-contract=off"]}
    value["identity"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "opaque.cu"
    path.write_text("// Adapter never interprets numerical source.\n")
    return path


def toolchain():
    return {"architecture": "86", "nvcc": "/cuda/nvcc", "cuda_host_cxx": "/usr/bin/g++-14",
            "cuda_library_dir": "/cuda/lib64"}


def compiled(command, **kwargs):
    Path(command[command.index("-o") + 1]).touch()
    return subprocess.CompletedProcess(command, 0)


def test_production_build_applies_and_records_the_complete_contract(source, tmp_path):
    approved = contract()
    descriptor = {"source": source, "language": "cuda", "numerical_contract": approved}
    with patch.dict(os.environ, {"NVCC_PREPEND_FLAGS": "", "NVCC_APPEND_FLAGS": ""}), \
            patch("elmm_pipeline.gpu_toolchain.subprocess.run", side_effect=compiled):
        # The application build normalizes descriptors before calling this API.
        result = compile_gpu_sources(validate_gpu_sources([descriptor]), tmp_path / "build", toolchain=toolchain())
    command = result["commands"][0]
    assert all(option in command for option in approved["required_cuda_options"])
    assert "-Xcompiler=-ffp-contract=off" in command
    unit = result["sources"][0]
    assert unit["numerical_contract"] == approved
    assert unit["resolved_numerical_options"] == {"cuda": approved["required_cuda_options"],
                                                  "host": approved["required_host_options"]}
    assert unit["build_status"] == "compiled"
    saved = json.loads((tmp_path / "build/gpu/gpu-build-command.json").read_text())
    assert saved == result
    assert descriptor["numerical_contract"] == approved


def test_legacy_descriptor_is_explicit_and_keeps_its_original_command(source, tmp_path):
    with patch("elmm_pipeline.gpu_toolchain.subprocess.run", side_effect=compiled):
        result = compile_gpu_sources([{"source": source, "language": "cuda"}], tmp_path / "build",
                                     toolchain=toolchain())
    unit = result["sources"][0]
    assert unit["numerical_contract_status"] == "legacy_unspecified"
    assert "numerical_contract" not in unit
    assert "--fmad=false" not in result["commands"][0]
    assert "-Xcompiler=-ffp-contract=off" not in result["commands"][0]


@pytest.mark.parametrize("change", [
    {"schema_version": True}, {"schema_version": 2}, {"id": "unknown-v1"},
    {"identity": "0" * 64}, {"required_cuda_options": ["--fmad=true"]},
    {"required_host_options": []}, {"required_cuda_options": ["--use_fast_math"]},
    {"required_device_rounding": "toward_zero"},
])
def test_unknown_or_conflicting_requirements_reject_before_any_build(source, tmp_path, change):
    bad = {**contract(), **change}
    with patch("elmm_pipeline.gpu_toolchain.subprocess.run") as run, pytest.raises(ValueError):
        compile_gpu_sources([{"source": source, "language": "cuda"},
                             {"source": _second_source(source), "language": "cuda", "numerical_contract": bad}],
                            tmp_path / "build", toolchain=toolchain())
    run.assert_not_called()
    assert not (tmp_path / "build").exists()


def _second_source(source):
    other = source.with_name("second.cu")
    other.write_bytes(source.read_bytes())
    return other


@pytest.mark.parametrize("value", [None, [], {}, {"schema_version": 1}])
def test_explicit_malformed_contract_is_not_legacy(source, value):
    with pytest.raises(ValueError):
        validate_gpu_sources([{"source": source, "language": "cuda", "numerical_contract": value}])


@pytest.mark.parametrize("name", ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"])
def test_hidden_compiler_options_cannot_override_public_requirements(source, tmp_path, name):
    with patch.dict(os.environ, {name: "--use_fast_math"}), \
            patch("elmm_pipeline.gpu_toolchain.subprocess.run") as run, \
            pytest.raises(ValueError, match="hidden NVCC"):
        compile_gpu_sources([{"source": source, "language": "cuda", "numerical_contract": contract()}],
                            tmp_path / "build", toolchain=toolchain())
    run.assert_not_called()
    assert not (tmp_path / "build").exists()


def test_contract_does_not_admit_an_unsupported_gpu_language(tmp_path):
    source = tmp_path / "kernel.f90"
    source.touch()
    with pytest.raises(ValueError, match="unsupported GPU source"):
        validate_gpu_sources([{"source": source, "language": "openacc", "numerical_contract": contract()}])


def test_removed_ordinary_adapter_cannot_publish_a_numerical_artifact(tmp_path):
    source = tmp_path / "input.f90"
    source.write_text("module original\nend module\n")
    generated = tmp_path / "generated"
    generated.mkdir()
    with patch("elmm_pipeline.local_adapter.transform") as generate, pytest.raises(ValueError, match="removed"):
        transform("local", source, Path(sys.executable), generated, compiler_root=tmp_path, entry="advance")
    generate.assert_not_called()
    assert list(generated.iterdir()) == []
    assert source.read_text() == "module original\nend module\n"


def test_public_build_report_missing_or_foreign_cuda_source_cannot_drop_requirements():
    assert cuda_build_fields({}, "region.cu") == {}
    descriptor = {"path": "other.cu", "language": "cuda", "numerical_contract": contract()}
    with pytest.raises(LocalAdapterError, match="exactly once"):
        cuda_build_fields({"build_sources": [descriptor]}, "region.cu")
    bad = copy.deepcopy(descriptor)
    bad["path"] = "region.cu"
    bad["numerical_contract"]["schema_version"] = True
    with pytest.raises(LocalAdapterError, match="numerical build contract"):
        cuda_build_fields({"build_sources": [bad]}, "region.cu")
    cpp = {**bad, "path": "region.cpp", "language": "c++"}
    with pytest.raises(LocalAdapterError, match="numerical build contract"):
        cuda_build_fields({"build_sources": [descriptor, cpp]}, "other.cu")


@pytest.mark.parametrize("requirement", [None, {}, {"schema_version": 1, "cuda_options": ["-O3"]}])
def test_exact_window_contract_cannot_be_dropped_or_treated_as_legacy(source, tmp_path, requirement):
    descriptor = {"source": source, "language": "cuda", "numerical_contract": contract(),
                  "exact_window_build_contract": requirement}
    with patch("elmm_pipeline.gpu_toolchain.subprocess.run") as run, \
            pytest.raises(ValueError, match="Exact-window build contracts"):
        compile_gpu_sources([descriptor], tmp_path / "build", toolchain=toolchain())
    run.assert_not_called()
    assert not (tmp_path / "build").exists()
    reported = {"path": "region.cu", **{key: value for key, value in descriptor.items() if key != "source"}}
    with pytest.raises(LocalAdapterError, match="Exact-window build contracts"):
        cuda_build_fields({"build_sources": [reported]}, "region.cu")


@pytest.mark.skipif(os.environ.get("ELMM_GPU_NATIVE_TESTS") != "1", reason="Opt-in production CUDA execution")
@pytest.mark.parametrize("bits,power", [(32, 13), (64, 28)])
def test_public_compiler_and_production_build_preserve_separate_arithmetic(tmp_path, bits, power):
    """A fused multiply/add is nonzero where the separate native result is zero."""
    compiler_root = os.environ.get("ELMM_COMPILER_ROOT")
    python = os.environ.get("ELMM_LOCAL_PYTHON")
    fortran = shutil.which(os.environ.get("FC", "gfortran"))
    if not compiler_root or not python or not fortran:
        pytest.skip("Requires explicit compiler snapshot/Python and a native Fortran compiler")
    kind = bits // 8
    source = tmp_path / "input.f90"
    original = f"""module original
implicit none
contains
subroutine advance(n,a,b,c,forward,reverse,grouped)
integer,intent(in)::n
real(kind={kind}),contiguous,intent(in)::a(:),b(:),c(:)
real(kind={kind}),contiguous,intent(inout)::forward(:),reverse(:),grouped(:)
integer::i
do i=1,n
forward(i)=a(i)*b(i)+c(i)
reverse(i)=c(i)+a(i)*b(i)
grouped(i)=a(i)*(b(i)+c(i))
end do
end subroutine
end module
"""
    source.write_text(original)
    metadata = transform("local", source, Path(python), tmp_path / "generated",
                         compiler_root=Path(compiler_root), entry="advance")
    assert metadata["gpu_sources"][0]["numerical_contract"] == contract()
    options = {key: os.environ[name] for key, name in (("architecture", "ELMM_GPU_ARCH"),
               ("nvcc", "ELMM_NVCC"), ("cuda_host_cxx", "ELMM_CUDA_HOST_CXX")) if name in os.environ}
    built = compile_gpu_sources(metadata["gpu_sources"], tmp_path / "build", options=options)
    assert "--fmad=false" in built["commands"][0]
    native = tmp_path / "native.f90"
    native.write_text(original.replace("module original", "module native_reference", 1))
    driver = tmp_path / "driver.f90"
    driver.write_text(f"""program compare
use iso_c_binding
use original,only:advance
use native_reference,only:native_advance=>advance
implicit none
integer,parameter::n=33
real(kind={kind})::a(n),b(n),c(n),forward(n),reverse(n),grouped(n),nf(n),nr(n),ng(n)
open(20,file='inputs.bin',access='stream',form='unformatted',status='old')
read(20)a,b,c
close(20)
forward=-999;reverse=-999;grouped=-999
nf=forward;nr=reverse;ng=grouped
call native_advance(n,a,b,c,nf,nr,ng)
if(any(nf/=0).or.any(nr/=0)) error stop 'native separate arithmetic fixture failed'
call advance(n,a,b,c,forward,reverse,grouped)
if(any(transfer(forward,[0_c_int{bits}_t],n)/=transfer(nf,[0_c_int{bits}_t],n))) error stop 'forward mismatch'
if(any(transfer(reverse,[0_c_int{bits}_t],n)/=transfer(nr,[0_c_int{bits}_t],n))) error stop 'reverse mismatch'
if(any(transfer(grouped,[0_c_int{bits}_t],n)/=transfer(ng,[0_c_int{bits}_t],n))) error stop 'grouped mismatch'
print *,'STRICT_NUMERICAL_CONTRACT_PASS'
end program
""")
    value = 2.0 ** -power
    packing = "<" + ("f" if bits == 32 else "d") * 99
    (tmp_path / "inputs.bin").write_bytes(struct.pack(packing, *([1 + value] * 33 + [1 - value] * 33 + [-1.] * 33)))
    command = [fortran, "-O3", "-fopenmp", "-ffp-contract=off", "-ffree-line-length-none",
               str(native), *metadata["extra_sources"], str(driver), *built["objects"],
               *["-L" + path for path in built["library_dirs"]],
               *["-Wl,-rpath," + path for path in built["rpaths"]],
               *built["link_flags"], *["-l" + name for name in built["libraries"]], "-o", str(tmp_path / "check")]
    (tmp_path / "native-build-command.json").write_text(json.dumps({
        "command": command, "cwd": str(tmp_path), "role": "native_reference_and_gpu_driver"}, indent=2) + "\n")
    result = subprocess.run(command, cwd=tmp_path, text=True, capture_output=True, check=False)
    (tmp_path / "native-build.stdout").write_text(result.stdout)
    (tmp_path / "native-build.stderr").write_text(result.stderr)
    (tmp_path / "native-build-result.json").write_text(json.dumps({"exit_code": result.returncode}) + "\n")
    assert result.returncode == 0, result.stdout + result.stderr
    execution_command = [str(tmp_path / "check")]
    (tmp_path / "execution-command.json").write_text(json.dumps({
        "command": execution_command, "cwd": str(tmp_path),
        "environment_overrides": {"FORT_RUNTIME_TRACE": "1"}}, indent=2) + "\n")
    executed = subprocess.run(execution_command, cwd=tmp_path, text=True, capture_output=True,
                              env=dict(os.environ, FORT_RUNTIME_TRACE="1"), check=False)
    (tmp_path / "execution.stdout").write_text(executed.stdout)
    (tmp_path / "execution.stderr").write_text(executed.stderr)
    (tmp_path / "execution-result.json").write_text(json.dumps({"exit_code": executed.returncode}) + "\n")
    assert executed.returncode == 0, executed.stdout + executed.stderr
    assert "STRICT_NUMERICAL_CONTRACT_PASS" in executed.stdout
    assert "FORT_RUNTIME kernel" in executed.stderr.splitlines()
