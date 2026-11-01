"""Pinned optional provider delivery and unchanged native build transport."""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from elmm_pipeline.build import build
from elmm_pipeline.build_receipts import file_identity
from elmm_pipeline.gpu_pressure_build import (
    ROOT,
    _json_identity,
    _runtime_inputs,
    combine_pressure_backends,
    prepare_poisfft_provider_manifest,
    pressure_backend_from_manifest,
    pressure_runtime_cuda_binding,
    require_pressure_backend,
    verify_source_package,
)
from test_scope_composition import sources as _shared_sources

sources = _shared_sources


@pytest.mark.parametrize("mode", ["raw", "banner", "body_tamper", "banner_tamper", "missing_contract", "changed_raw"])
def test_runtime_source_matches_only_exact_code_and_authenticated_numerical_banner(tmp_path, mode):
    raw, selected = tmp_path / "original.cu", tmp_path / "selected.cu"
    body = b'#include "scoped_runtime.h"\nint executed_code=1;\n'
    raw.write_bytes(body)
    original = file_identity(raw)
    contract = {"schema_version": 1, "id": "separate-arithmetic-v1",
                "required_cuda_options": ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
                "required_host_options": ["-ffp-contract=off"]}
    contract["identity"] = _json_identity(contract)
    banner = ("// Compiler numerical build contract: separate-arithmetic-v1 " + contract["identity"] + "\n"
              "// Required NVCC options: --fmad=false --ftz=false --prec-div=true --prec-sqrt=true -Xcompiler=-ffp-contract=off\n"
              "// Consumers must enforce the public contract in the actual compile command.\n").encode()
    selected.write_bytes(body if mode == "raw" else banner + body)
    unit = {"source": str(selected), "numerical_contract": contract}
    if mode == "body_tamper":
        selected.write_bytes(banner + body.replace(b"code=1", b"code=2"))
    if mode == "banner_tamper":
        selected.write_bytes(banner.replace(b"--fmad=false", b"--fmad=true") + body)
    if mode == "missing_contract":
        unit.pop("numerical_contract")
    if mode == "changed_raw":
        raw.write_bytes(body + b"// changed after prepare\n")
        with pytest.raises(ValueError, match="raw runtime changed"):
            pressure_runtime_cuda_binding(unit, original)
    elif mode in {"raw", "banner"}:
        binding = pressure_runtime_cuda_binding(unit, original)
        assert binding["source"] == file_identity(selected)
        assert binding["raw_source"] == original
        if mode == "banner":
            assert binding["numerical_contract"] == contract
            assert binding["source"]["sha256"] != binding["raw_source"]["sha256"]
    else:
        assert pressure_runtime_cuda_binding(unit, original) is None


def test_pressure_request_requires_one_exact_prepared_identity():
    request = {"manifest": "/prepared/pressure-backend.json", "sha256": "a" * 64}
    assert combine_pressure_backends([None, request, dict(request)]) == request
    assert combine_pressure_backends([None]) is None
    for other in ({**request, "sha256": "b" * 64}, {**request, "manifest": "/other/backend.json"}):
        with pytest.raises(ValueError, match="one identical pressure backend"):
            combine_pressure_backends([request, other])
    for invalid in ({**request, "manifest": "relative.json"}, {**request, "sha256": "unbound"},
                    {**request, "source_authority": True}):
        with pytest.raises(ValueError, match="SHA-bound prepared manifest"):
            combine_pressure_backends([invalid])


def test_unchecked_external_task_duplicate_cannot_request_pressure_build():
    request = {"manifest": "/prepared/pressure-backend.json", "sha256": "a" * 64}
    manifest = {"scopes": [{"external_device_tasks": [{"provider": {"pressure_backend": request}}]}]}
    with pytest.raises(ValueError, match="external device tasks require their checked plan"):
        pressure_backend_from_manifest(manifest)
    manifest["scopes"][0]["checked_execution_plan"] = {"schema_version": 2}
    with pytest.raises(ValueError, match="complete v2 schema"):
        pressure_backend_from_manifest(manifest)


def test_checked_projection_runs_validator_before_inspecting_epoch_records():
    request = {"manifest": "/prepared/pressure-backend.json", "sha256": "a" * 64}
    manifest = {"scopes": [{"checked_execution_plan": {"storage_epoch": {"members": [
        {"device_tasks": [{"provider": {"pressure_backend": request}},
                          {"provider": {"pressure_backend": dict(request)}}]}]}}}]}
    # Test only projection after validation. This fixture is not an issued
    # compiler proof and cannot authorize source replacement or execution.
    with patch("elmm_pipeline.checked_execution_plan.validate_checked_execution_plans") as validate:
        assert pressure_backend_from_manifest(manifest) == request
        validate.assert_called_once_with(manifest)
    with patch("elmm_pipeline.checked_execution_plan.validate_checked_execution_plans",
               side_effect=ValueError("invalid issued epoch")), \
            pytest.raises(ValueError, match="invalid issued epoch"):
        pressure_backend_from_manifest(manifest)


def test_issued_epoch_request_is_transported_and_metadata_cannot_substitute_it(tmp_path, sources):
    from elmm_pipeline.local_adapter import (
        LocalAdapterError,
        apply_scope_sources,
        compose_scope_packages,
    )
    from test_checked_execution_plan import with_device_task
    from test_scope_composition import package, unchanged

    prepared = tmp_path / "prepared.json"
    prepared.write_text('{"protocol_fixture_only":true}\n')
    request = {"manifest": str(prepared), "sha256": file_identity(prepared)["sha256"]}
    metadata = package(tmp_path, sources, "one")
    with_device_task(metadata, pressure_backend=request)
    assert pressure_backend_from_manifest(metadata["scope_manifest"]) == request
    metadata["pressure_backend"] = {**request, "sha256": "0" * 64}
    with pytest.raises(LocalAdapterError, match="differs from its checked epoch"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)
    metadata["pressure_backend"] = request
    bundle = compose_scope_packages([metadata], sources[1])
    assert bundle["pressure_backend"] == request
    assert bundle["scope_packages"][0]["scope_manifest"] == metadata["scope_manifest"]


def test_conflicting_issued_backend_packages_reject_before_source_changes(tmp_path, sources):
    from elmm_pipeline.local_adapter import LocalAdapterError, compose_scope_packages
    from test_checked_execution_plan import with_device_task
    from test_scope_composition import package, unchanged

    packages = []
    for name, source in (("one", "producer.f90"), ("two", "consumer.f90")):
        prepared = tmp_path / (name + "-prepared.json")
        prepared.write_text('{"protocol_fixture_only":true}\n')
        request = {"manifest": str(prepared), "sha256": file_identity(prepared)["sha256"]}
        metadata = package(tmp_path, sources, name, source=source)
        with_device_task(metadata, pressure_backend=request)
        metadata["pressure_backend"] = request
        packages.append(metadata)
    with pytest.raises(LocalAdapterError, match="one identical pressure backend"):
        compose_scope_packages(packages, sources[1])
    unchanged(sources)


def test_selected_runtime_header_must_match_before_any_compilation(tmp_path):
    elmm, poisfft = tmp_path / "elmm", tmp_path / "PoisFFT"
    (elmm / "src").mkdir(parents=True)
    (poisfft / "src").mkdir(parents=True)
    (elmm / "src/SConstruct").write_text("t = env.Program()\n")
    (poisfft / "src/poisfft.f90").write_text("! fixture\n")
    expected, selected = tmp_path / "prepared-runtime", tmp_path / "selected-runtime"
    expected.mkdir()
    selected.mkdir()
    for directory in (expected, selected):
        (directory / "scoped_runtime.cu").write_text('#include "scoped_runtime.h"\n')
        (directory / "scoped_memory.f90").write_text("! identical Fortran runtime\n")
    (expected / "scoped_runtime.h").write_text("// prepared ABI\n")
    (selected / "scoped_runtime.h").write_text("// different ABI beside identical runtime CU\n")
    pressure = {"extra_sources": [], "runtime": {name: file_identity(expected / name)
                for name in ("scoped_runtime.cu", "scoped_runtime.h", "scoped_memory.f90")}}
    with patch("elmm_pipeline.gpu_pressure_build.require_pressure_backend", return_value=pressure), \
            patch("elmm_pipeline.build.compile_gpu_sources") as compile_gpu, \
            pytest.raises(ValueError, match="matching header beside its selected runtime"):
        build(elmm, poisfft, tmp_path / "output", Path("python"), tmp_path / "lib", 1, 1, [],
              extra_sources=[selected / "scoped_memory.f90"],
              gpu_sources=[{"source": str(selected / "scoped_runtime.cu"), "language": "cuda"}],
              pressure_backend={"manifest": "/explicit/backend.json", "sha256": "a" * 64})
    compile_gpu.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_selected_runtime_quote_header_must_match_before_any_compilation(tmp_path):
    elmm, poisfft = tmp_path / "elmm", tmp_path / "PoisFFT"
    (elmm / "src").mkdir(parents=True)
    (poisfft / "src").mkdir(parents=True)
    (elmm / "src/SConstruct").write_text("t = env.Program()\n")
    (poisfft / "src/poisfft.f90").write_text("! fixture\n")
    expected, selected = tmp_path / "prepared-runtime", tmp_path / "selected-runtime"
    expected.mkdir()
    selected.mkdir()
    for directory in (expected, selected):
        (directory / "scoped_runtime.cu").write_text('#include "scoped_runtime.h"\n#include "scoped_source_quote.hpp"\n')
        (directory / "scoped_memory.f90").write_text("! identical Fortran runtime\n")
        (directory / "scoped_runtime.h").write_text("// identical ABI\n")
    (expected / "scoped_source_quote.hpp").write_text("// prepared quote implementation\n")
    (selected / "scoped_source_quote.hpp").write_text("// different body beside identical CU\n")
    pressure = {"extra_sources": [], "runtime": {name: file_identity(expected / name) for name in
                ("scoped_runtime.cu", "scoped_runtime.h", "scoped_memory.f90", "scoped_source_quote.hpp")}}
    with patch("elmm_pipeline.gpu_pressure_build.require_pressure_backend", return_value=pressure), \
            patch("elmm_pipeline.build.compile_gpu_sources") as compile_gpu, \
            pytest.raises(ValueError, match="matching quote header beside its selected runtime"):
        build(elmm, poisfft, tmp_path / "output", Path("python"), tmp_path / "lib", 1, 1, [],
              extra_sources=[selected / "scoped_memory.f90"],
              gpu_sources=[{"source": str(selected / "scoped_runtime.cu"), "language": "cuda"}],
              pressure_backend={"manifest": "/explicit/backend.json", "sha256": "a" * 64})
    compile_gpu.assert_not_called()
    assert not (tmp_path / "output").exists()


def test_vendored_package_is_pinned_and_contains_one_implementation():
    directory, record = verify_source_package()
    assert record["single_implementation"]
    assert record["upstream"]["commit"] == "4a1f1bb7e6f9dab6a2934b62d9b8655cf3a9f99a"
    assert not (directory / ".git").exists()
    assert not (directory / "benchmark-results").exists()
    assert (directory / "include/poisson-solver/device_api.h").is_file()


@pytest.mark.parametrize("change", ["bytes", "additional", "symlink"])
def test_vendored_source_tampering_is_rejected(tmp_path, change):
    source, _ = verify_source_package()
    copied = tmp_path / "package"
    shutil.copytree(source, copied)
    header = copied / "include/poisson-solver/device_api.h"
    if change == "bytes":
        header.write_text(header.read_text() + "\n/* changed */\n")
    elif change == "additional":
        (copied / "injected.cpp").write_text("// must never compile\n")
    else:
        alternate = copied / "include/poisson-solver/alternate.h"
        header.rename(alternate)
        header.symlink_to(alternate.name)
    with pytest.raises(ValueError):
        verify_source_package(copied)


def backend_request(directory, compiler):
    library = directory / "provider.so"
    library.write_bytes(b"qualification fixture, not an executable provider")
    runtime = directory / "runtime.h"
    runtime.write_text("fixture byte identity only\n")
    package, _ = verify_source_package()
    payload = {"schema_version": 1, "kind": "prepared_gpu_pressure_backend", "placement_authority": False,
               "fortran_compiler": file_identity(compiler), "inputs": [file_identity(library)],
               "runtime": {name: file_identity(runtime) for name in ("scoped_runtime.h", "scoped_runtime.cu", "scoped_memory.f90")},
               "requirements": {"source_authority": False, "placement_authority": False},
               "package": file_identity(package / "source-manifest.json"), "objects": [str(library)],
               "extra_sources": [], "libraries": ["poisson-solver"], "library_dirs": [], "rpaths": [],
               "include_dirs": [], "command_receipts": str(directory / "receipts"),
               "provider_library": file_identity(library)}
    payload["identity"] = _json_identity(payload)
    path = directory / "pressure-backend.json"
    path.write_text(json.dumps(payload))
    return {"manifest": str(path), "sha256": file_identity(path)["sha256"]}


def test_runtime_quote_dependency_is_bound_before_backend_compilation(tmp_path):
    for name in ("scoped_runtime.h", "scoped_memory.f90"):
        (tmp_path / name).write_text("! exact fixture dependency\n")
    source = tmp_path / "scoped_runtime.cu"
    source.write_text('#include "scoped_runtime.h"\n')
    assert set(_runtime_inputs(tmp_path)) == {"scoped_runtime.h", "scoped_runtime.cu", "scoped_memory.f90"}
    source.write_text(source.read_text() + '#include "scoped_source_quote.hpp"\n')
    with pytest.raises(FileNotFoundError):
        _runtime_inputs(tmp_path)
    header = tmp_path / "scoped_source_quote.hpp"
    header.write_text("// exact source quote implementation\n")
    inputs = _runtime_inputs(tmp_path)
    assert set(inputs) == {"scoped_runtime.h", "scoped_runtime.cu", "scoped_memory.f90", "scoped_source_quote.hpp"}
    assert inputs["scoped_source_quote.hpp"] == file_identity(header)


@pytest.mark.parametrize("change", ["none", "missing", "unknown", "tamper", "phantom"])
def test_prepared_quote_dependency_is_exact_and_cannot_be_omitted_or_retagged(tmp_path, change):
    compiler = Path(shutil.which("python3"))
    request = backend_request(tmp_path, compiler)
    manifest = Path(request["manifest"])
    value = json.loads(manifest.read_text())
    source = tmp_path / "runtime.cu"
    source.write_text('#include "scoped_source_quote.hpp"\n' if change != "phantom" else "// old core\n")
    header = tmp_path / "scoped_source_quote.hpp"
    header.write_text("// qualified quote source\n")
    value["runtime"]["scoped_runtime.cu"] = file_identity(source)
    value["runtime"]["scoped_source_quote.hpp"] = file_identity(header)
    if change == "missing":
        del value["runtime"]["scoped_source_quote.hpp"]
    elif change == "unknown":
        value["runtime"]["unchecked.hpp"] = value["runtime"].pop("scoped_source_quote.hpp")
    value["identity"] = _json_identity({key: item for key, item in value.items() if key != "identity"})
    manifest.write_text(json.dumps(value))
    request["sha256"] = file_identity(manifest)["sha256"]
    if change == "tamper":
        header.write_text("// changed after qualification\n")
    if change == "none":
        assert require_pressure_backend(request, fortran=compiler)["runtime"]["scoped_source_quote.hpp"] == file_identity(header)
    else:
        with pytest.raises(ValueError, match="quote header dependency|schema is incomplete|runtime differs"):
            require_pressure_backend(request, fortran=compiler)


def test_prepared_manifest_binds_source_bytes_and_exact_fortran_compiler(tmp_path):
    compiler = Path(shutil.which("python3"))
    request = backend_request(tmp_path, compiler)
    assert require_pressure_backend(request, fortran=compiler)["placement_authority"] is False
    other = tmp_path / "different-compiler"
    other.write_bytes(b"different compiler identity")
    with pytest.raises(ValueError, match="exact selected host compiler"):
        require_pressure_backend(request, fortran=other)
    (tmp_path / "provider.so").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="link/interface input is not bound"):
        require_pressure_backend(request, fortran=compiler)


def test_prepared_manifest_and_runtime_cannot_be_retagged(tmp_path):
    compiler = Path(shutil.which("python3"))
    request = backend_request(tmp_path, compiler)
    path = Path(request["manifest"])
    original = path.read_bytes()
    path.write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="manifest changed"):
        require_pressure_backend(request, fortran=compiler)
    path.write_bytes(original)
    (tmp_path / "runtime.h").write_text("changed runtime\n")
    with pytest.raises(ValueError, match="runtime differs"):
        require_pressure_backend(request, fortran=compiler)


def test_poisfft_provider_requires_explicit_body_scope_and_qualification(tmp_path):
    compiler = Path(shutil.which("python3"))
    request = backend_request(tmp_path, compiler)
    qualification = tmp_path / "qualification.json"
    qualification.write_text('{"protocol_fixture_only":true,"execution_evidence":false}\n')
    reference = {"path": str(qualification), "sha256": file_identity(qualification)["sha256"]}
    with pytest.raises(ValueError, match="does not bind the typed original PoisFFT query"):
        prepare_poisfft_provider_manifest(request, tmp_path / "missing-query.json", fortran=compiler,
                                         qualification=reference, original_body_sha256="a" * 64,
                                         task_contract_sha256="b" * 64)
    query = tmp_path / "gpu_pressure_poisfft.f90"
    query.write_text("! protocol fixture only; not an executable interface\n")
    manifest_path = Path(request["manifest"])
    backend = json.loads(manifest_path.read_text())
    backend["extra_sources"] = [str(query)]
    backend["inputs"].append(file_identity(query))
    backend["identity"] = _json_identity({key: value for key, value in backend.items() if key != "identity"})
    manifest_path.write_text(json.dumps(backend))
    request["sha256"] = file_identity(manifest_path)["sha256"]
    result = prepare_poisfft_provider_manifest(request, tmp_path / "provider.json", fortran=compiler,
                                             qualification=reference, original_body_sha256="a" * 64,
                                             task_contract_sha256="b" * 64)
    provider = json.loads(Path(result["path"]).read_text())
    assert provider["version"] == 2
    assert file_identity(result["path"])["sha256"] == result["sha256"]
    assert provider["pressure_backend"] == request
    assert provider["numerical_contract"]["original_bodies"] == ["a" * 64]
    assert provider["numerical_contract"]["task_contracts"] == ["b" * 64]
    assert provider["numerical_contract"]["qualification"] == reference
    assert provider["metadata_query"] == {"module": "elmm_gpu_pressure_poisfft",
        "procedure": "elmm_gpu_pressure_configure_poisfft_v1", "opaque_formals": ["d"],
        "configuration_argument": "config", "read_only": True,
        "configuration_components": [
            {"component": "solver%sizes", "kind": "int32", "count": 3},
            {"component": "solver%lengths", "kind": "real64", "count": 3},
            {"component": "solver%boundary_conditions", "kind": "int32", "count": 6},
            {"component": "solver%approximation", "kind": "int32", "count": 1}]}
    assert provider["artifacts"][str(query)] == file_identity(query)["sha256"]
    qualification.write_text('{"changed_after_qualification":true}\n')
    with pytest.raises(ValueError, match="exact explicit qualification"):
        prepare_poisfft_provider_manifest(request, tmp_path / "changed.json", fortran=compiler,
                                         qualification=reference, original_body_sha256="a" * 64,
                                         task_contract_sha256="b" * 64)


def test_without_pressure_native_script_and_command_bytes_are_unchanged(tmp_path):
    # Compare the pre-change committed build function under the same supplied
    # native paths. This is a transport oracle; it does not pretend to compile
    # or qualify an ELMM binary.
    baseline = tmp_path / "baseline_build.py"
    old = subprocess.run(["git", "show", "HEAD:elmm_pipeline/build.py"], cwd=ROOT,
                         capture_output=True, check=True).stdout
    baseline.write_bytes(old)
    shutil.copyfile(ROOT / "elmm_pipeline/build_receipts.py", tmp_path / "build_receipts.py")
    spec = importlib.util.spec_from_file_location("elmm_pipeline._baseline_native_build", baseline)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    elmm, poisfft, output = (tmp_path / name for name in ("ELMM", "PoisFFT", "build"))
    (elmm / "src").mkdir(parents=True)
    (poisfft / "src").mkdir(parents=True)
    (poisfft / "src/poisfft.f90").write_text("! fixture\n")
    (elmm / "src/SConstruct").write_text("t = env.Program()\n")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    observed = []

    def no_compilation(argv, **kwargs):
        observed.append({"argv": argv, "cwd": str(kwargs["cwd"]), "env": kwargs["env"]})
        (output / "ELMM").write_bytes(b"transport fixture")
        return subprocess.CompletedProcess(argv, 0)

    def capture(function):
        observed.clear()
        with patch("tempfile.mkdtemp", return_value=str(receipts)), \
                patch("elmm_pipeline.build.subprocess.run", side_effect=no_compilation):
            function(elmm, poisfft, output, Path("python"), tmp_path / "native-lib", 2, 4, [])
        return ((elmm / "src/SConstruct.pipeline").read_bytes(),
                (output / "build-command.json").read_bytes(), list(observed))

    before = capture(module.build)
    after = capture(build)
    assert before == after
