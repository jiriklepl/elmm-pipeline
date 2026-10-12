"""Compose public artifacts without choosing numerical implementations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from elmm_pipeline.local_adapter import (
    LocalAdapterError,
    apply_scope_sources,
    compose_scope_packages,
)


def numerical_contract():
    value = {"schema_version": 1, "id": "separate-arithmetic-v1",
             "required_cuda_options": ["--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"],
             "required_host_options": ["-ffp-contract=off"]}
    value["identity"] = digest(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())
    return value


def digest(content):
    return hashlib.sha256(content).hexdigest()


@pytest.fixture
def sources(tmp_path):
    original, application = tmp_path / "original", tmp_path / "application"
    original.mkdir()
    application.mkdir()
    inputs, destinations = {}, {}
    for name in ("producer.f90", "consumer.f90", "shared.f90"):
        content = f"! original {name}\n".encode()
        source, destination = original / name, application / name
        source.write_bytes(content)
        destination.write_bytes(content)
        inputs[str(source)] = digest(content)
        destinations[str(source)] = destination
    return inputs, destinations


def package(tmp_path, sources, name, *, source="producer.f90", replacement=None,
            runtime_header="// common runtime\n", entry=None, entry_content=None):
    inputs, _ = sources
    root = tmp_path / name
    root.mkdir()
    selected = next(path for path in inputs if Path(path).name == source)
    entry = entry or name
    content = {
        "runtime.h": runtime_header.encode(),
        "runtime.cu": b'// common runtime implementation\n#include "runtime.h"\n',
        "runtime.f90": b"! common runtime interface\n",
        f"entries/{entry}.cu": (entry_content or f"// approved {entry}\n").encode(),
        f"entries/{entry}.f90": f"! approved interface {entry}\n".encode(),
        f"sources/{name}.f90": (replacement or f"! approved replacement {name}\n").encode(),
    }
    for path, value in content.items():
        artifact = root / path
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_bytes(value)
    runtime_hashes = {path: digest(content[path]) for path in ("runtime.h", "runtime.cu", "runtime.f90")}
    runtime = {"schema_version": 1, "abi_version": 1, "link_once": True,
               "runtime_id": digest("\n".join(f"{p}:{h}" for p, h in sorted(runtime_hashes.items())).encode()),
               "source_sha256": runtime_hashes,
               "sources": [{"path": "runtime.cu", "language": "cuda"},
                           {"path": "runtime.f90", "language": "fortran"}]}
    manifest = {"schema_version": 1, "entry": entry, "scope_count": 1,
                "source_inputs": inputs, "runtime": runtime,
                "artifacts_sha256": {path: digest(value) for path, value in content.items()},
                "sources": {selected: {"sha256": inputs[selected], "replacement": f"sources/{name}.f90"}},
                "build_sources": [
                    {"path": f"entries/{entry}.cu", "language": "cuda", "role": "shared_entry"},
                    {"path": f"entries/{entry}.f90", "language": "fortran", "role": "shared_entry"},
                    {"path": f"sources/{name}.f90", "language": "fortran", "role": "original_source"},
                    {"path": "runtime.cu", "language": "cuda", "role": "common_runtime"},
                    {"path": "runtime.f90", "language": "fortran", "role": "common_runtime"},
                ]}
    metadata = {"scope_manifest": manifest, "artifact_root": str(root)}
    save(metadata)
    return metadata


def save(metadata):
    (Path(metadata["artifact_root"]) / "scope-manifest.json").write_text(json.dumps(metadata["scope_manifest"]))


def unchanged(sources):
    inputs, destinations = sources
    assert all(digest(destinations[path].read_bytes()) == expected for path, expected in inputs.items())


def test_distinct_roots_with_shared_pristine_inputs_apply_together(tmp_path, sources):
    first = package(tmp_path, sources, "renamed_first")
    second = package(tmp_path, sources, "renamed_second", source="consumer.f90")
    bundle = compose_scope_packages([first, second], sources[1])
    changed = {Path(path).name for path in bundle["patched_sources"]}
    assert changed == {"producer.f90", "consumer.f90"}
    assert bundle["scope_packages"] == [first, second]
    assert "scope_manifest" not in bundle  # Never forge a merged compiler decision.
    assert len(bundle["gpu_sources"]) == 3
    assert len(bundle["extra_sources"]) == 3
    runtime = [item for item in bundle["build_artifacts"] if item["role"] == "common_runtime"]
    assert len(runtime) == 2 and all(item["packages"] == [0, 1] for item in runtime)
    assert {item["role"] for item in bundle["build_artifacts"]} == {"common_runtime", "shared_entry"}
    for source, expected in sources[0].items():
        assert digest(Path(source).read_bytes()) == expected


def test_exact_shared_entry_and_replacement_are_reused(tmp_path, sources):
    first = package(tmp_path, sources, "one", entry="same", replacement="! identical\n")
    second = package(tmp_path, sources, "two", entry="same", replacement="! identical\n")
    bundle = compose_scope_packages([first, second], sources[1])
    assert len(bundle["patched_sources"]) == 1
    assert len(bundle["gpu_sources"]) == len(bundle["extra_sources"]) == 2
    assert all(item["packages"] == [0, 1] for item in bundle["build_artifacts"])


def test_conflicting_replacements_reject_before_any_source_changes(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two")
    with pytest.raises(LocalAdapterError, match="Conflicting approved scope replacements"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


@pytest.mark.parametrize("path", ["sources/two.f90", "entries/two.f90", "runtime.h"])
def test_changed_second_package_artifact_cannot_partially_apply(tmp_path, sources, path):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90")
    (Path(second["artifact_root"]) / path).write_text("! stale\n")
    with pytest.raises(LocalAdapterError, match="Approved scope artifact changed"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_stale_second_package_dependency_cannot_partially_apply(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90")
    include = tmp_path / "configured.inc"
    include.write_bytes(b"! original include\n")
    second["scope_manifest"]["analysis_sources"] = {"dependencies": {str(include): digest(include.read_bytes())}}
    save(second)
    include.write_bytes(b"! changed include\n")
    with pytest.raises(LocalAdapterError, match="Configured source dependency changed"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_mapped_include_must_match_even_when_original_is_unchanged(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90")
    include, copied_include = tmp_path / "original.inc", tmp_path / "copied.inc"
    include.write_bytes(b"! original include\n")
    copied_include.write_bytes(b"! changed include\n")
    second["scope_manifest"]["analysis_sources"] = {"dependencies": {str(include): digest(include.read_bytes())}}
    save(second)
    destinations = {**sources[1], str(include): copied_include}
    with pytest.raises(LocalAdapterError, match="Configured source dependency changed"):
        compose_scope_packages([first, second], destinations)
    unchanged(sources)


def test_different_valid_runtime_identity_is_rejected(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90", runtime_header="// different runtime\n")
    with pytest.raises(LocalAdapterError, match="exactly matching runtime identities"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_forged_runtime_hash_is_rejected(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    first["scope_manifest"]["runtime"]["runtime_id"] = "0" * 64
    save(first)
    with pytest.raises(LocalAdapterError, match="runtime identity differs"):
        compose_scope_packages([first], sources[1])
    unchanged(sources)


def test_conflicting_shared_artifact_identity_is_rejected(tmp_path, sources):
    first = package(tmp_path, sources, "one", entry="repeated")
    second = package(tmp_path, sources, "two", entry="repeated", source="consumer.f90",
                     entry_content="// incompatible implementation\n")
    with pytest.raises(LocalAdapterError, match="Conflicting approved scope artifact"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_matching_entry_bytes_cannot_hide_conflicting_helper_headers(tmp_path, sources):
    entry = '#include "helper.cuh"\n// same approved entry\n'
    first = package(tmp_path, sources, "one", entry="repeated", entry_content=entry)
    second = package(tmp_path, sources, "two", entry="repeated", source="consumer.f90", entry_content=entry)
    for metadata, content in ((first, b"// first helper\n"), (second, b"// different helper\n")):
        path = "entries/helper.cuh"
        (Path(metadata["artifact_root"]) / path).write_bytes(content)
        metadata["scope_manifest"]["artifacts_sha256"][path] = digest(content)
        save(metadata)
    with pytest.raises(LocalAdapterError, match="Conflicting approved scope artifact: entries/helper.cuh"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_replacement_artifact_paths_do_not_override_destination_identity(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90")
    manifest = second["scope_manifest"]
    old, new = "sources/two.f90", "sources/one.f90"
    (Path(second["artifact_root"]) / old).rename(Path(second["artifact_root"]) / new)
    manifest["artifacts_sha256"][new] = manifest["artifacts_sha256"].pop(old)
    next(iter(manifest["sources"].values()))["replacement"] = new
    next(item for item in manifest["build_sources"] if item["role"] == "original_source")["path"] = new
    save(second)
    bundle = compose_scope_packages([first, second], sources[1])
    assert len(bundle["patched_sources"]) == 2


def test_runtime_roles_must_match_public_runtime_sources(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    first["scope_manifest"]["build_sources"][-1]["role"] = "shared_entry"
    save(first)
    with pytest.raises(LocalAdapterError, match="runtime build roles differ"):
        compose_scope_packages([first], sources[1])
    unchanged(sources)


@pytest.mark.parametrize("declaration", [{}, {"path": "runtime.cu"}, {"path": ["runtime.cu"], "language": "cuda"}])
def test_malformed_runtime_source_is_a_package_error(tmp_path, sources, declaration):
    first = package(tmp_path, sources, "one")
    first["scope_manifest"]["runtime"]["sources"] = [declaration]
    save(first)
    with pytest.raises(LocalAdapterError, match="runtime sources require public paths"):
        compose_scope_packages([first], sources[1])
    unchanged(sources)


def test_saved_manifest_must_match_package(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    first["scope_manifest"]["entry"] = "changed"
    with pytest.raises(LocalAdapterError, match="Saved compiler scope manifest differs"):
        compose_scope_packages([first], sources[1])
    unchanged(sources)


def test_native_package_and_empty_batch_need_no_runtime(tmp_path, sources):
    root = tmp_path / "native"
    root.mkdir()
    native = {"artifact_root": str(root), "scope_manifest": {
        "schema_version": 1, "scope_count": 0, "source_inputs": sources[0], "sources": {},
        "runtime": None, "artifacts_sha256": {}, "build_sources": []}}
    save(native)
    for packages in ([], [native]):
        bundle = compose_scope_packages(packages, sources[1])
        assert bundle["runtime"] is None
        assert bundle["extra_sources"] == bundle["gpu_sources"] == bundle["patched_sources"] == []
    unchanged(sources)


def test_compiler_numerical_contract_survives_shared_runtime_composition(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90")
    for metadata in (first, second):
        for descriptor in [*metadata["scope_manifest"]["build_sources"],
                           *metadata["scope_manifest"]["runtime"]["sources"]]:
            if descriptor["language"] == "cuda":
                descriptor["numerical_contract"] = numerical_contract()
        save(metadata)
    bundle = compose_scope_packages([first, second], sources[1])
    assert len(bundle["gpu_sources"]) == 3
    assert all(unit["numerical_contract"] == numerical_contract() for unit in bundle["gpu_sources"])
    cuda = [unit for unit in bundle["build_artifacts"] if unit["language"] == "cuda"]
    assert all(unit["numerical_contract"] == numerical_contract() for unit in cuda)


def test_equal_artifact_bytes_cannot_hide_different_numerical_requirements(tmp_path, sources):
    first = package(tmp_path, sources, "one", entry="same")
    second = package(tmp_path, sources, "two", source="consumer.f90", entry="same")
    second["scope_manifest"]["build_sources"][0]["numerical_contract"] = numerical_contract()
    save(second)
    with pytest.raises(LocalAdapterError, match="Conflicting approved scope numerical build contract"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_runtime_source_and_build_contract_must_agree_before_application(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    first["scope_manifest"]["build_sources"][-2]["numerical_contract"] = numerical_contract()
    save(first)
    with pytest.raises(LocalAdapterError, match="runtime numerical build contract differs"):
        compose_scope_packages([first], sources[1])
    unchanged(sources)


def test_bad_second_numerical_contract_cannot_partially_apply(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    second = package(tmp_path, sources, "two", source="consumer.f90")
    second["scope_manifest"]["build_sources"][0]["numerical_contract"] = {**numerical_contract(), "schema_version": True}
    save(second)
    with pytest.raises(LocalAdapterError, match="numerical build contract"):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)


def test_duplicate_runtime_declaration_cannot_hide_additional_requirements(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    runtime = first["scope_manifest"]["runtime"]
    runtime["sources"].append({**runtime["sources"][0], "numerical_contract": numerical_contract()})
    save(first)
    with pytest.raises(LocalAdapterError, match="must not repeat"):
        compose_scope_packages([first], sources[1])
    unchanged(sources)


def test_single_package_apply_checks_runtime_required_metadata_first(tmp_path, sources):
    first = package(tmp_path, sources, "one")
    first["scope_manifest"]["runtime"]["sources"][0]["numerical_contract"] = numerical_contract()
    save(first)
    with pytest.raises(LocalAdapterError, match="runtime numerical build contract differs"):
        apply_scope_sources(first, sources[1])
    unchanged(sources)


def test_single_package_applies_approved_replacement_and_keeps_other_sources(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    original = next(path for path in sources[0] if Path(path).name == "producer.f90")
    destination = sources[1][original]
    assert apply_scope_sources(metadata, sources[1]) == [destination]
    assert destination.read_text() == "! approved replacement one\n"
    assert digest(Path(original).read_bytes()) == sources[0][original]
    assert all(digest(path.read_bytes()) == sources[0][source]
               for source, path in sources[1].items() if source != original)


@pytest.mark.parametrize("path", ["runtime.h", "runtime.cu", "entries/one.cu", "sources/one.f90"])
def test_single_package_stale_artifact_rejects_before_source_changes(tmp_path, sources, path):
    metadata = package(tmp_path, sources, "one")
    (Path(metadata["artifact_root"]) / path).write_text("! changed artifact\n")
    with pytest.raises(LocalAdapterError, match="Approved scope artifact changed"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def test_single_package_saved_manifest_mismatch_rejects_before_source_changes(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    metadata["scope_manifest"]["entry"] = "different"
    with pytest.raises(LocalAdapterError, match="Saved compiler scope manifest differs"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


@pytest.mark.parametrize("failure", ["replacement", "destination"])
def test_single_package_later_replacement_failure_never_partially_applies(tmp_path, sources, failure):
    metadata = package(tmp_path, sources, "one")
    manifest = metadata["scope_manifest"]
    consumer = next(path for path in sources[0] if Path(path).name == "consumer.f90")
    replacement = "sources/consumer.f90"
    replacement_path = Path(metadata["artifact_root"]) / replacement
    replacement_path.write_bytes(b"! approved consumer replacement\n")
    manifest["artifacts_sha256"][replacement] = digest(replacement_path.read_bytes())
    manifest["sources"][consumer] = {"sha256": sources[0][consumer], "replacement": replacement}
    manifest["build_sources"].append({"path": replacement, "language": "fortran", "role": "original_source"})
    save(metadata)
    if failure == "replacement":
        replacement_path.write_text("! stale second replacement\n")
    else:
        sources[1][consumer].write_text("! independently changed consumer\n")
    before = {source: path.read_bytes() for source, path in sources[1].items()}
    with pytest.raises(LocalAdapterError, match="changed"):
        apply_scope_sources(metadata, sources[1])
    assert all(path.read_bytes() == before[source] for source, path in sources[1].items())
