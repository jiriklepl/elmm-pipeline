"""Compiler-owned numerical requirements survive public pipeline builds."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from elmm_pipeline.gpu_toolchain import compile_gpu_sources, validate_gpu_sources


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
