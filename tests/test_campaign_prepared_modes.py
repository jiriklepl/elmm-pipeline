"""Four-mode imports retain independent authorities and exact public proof sidecars."""

import pytest
from elmm_pipeline.campaigns.identity import digest, write
from elmm_pipeline.campaigns.prepared_modes import load_adapter


def test_adapter_loads_exact_sidecars_and_preserves_embedded_contract(tmp_path):
    root = tmp_path / "proof"
    root.mkdir()
    manifest = root / "scope-manifest.json"
    contents = {"scopes": [{"id": "owner", "authority": "checked"}]}
    write(manifest, contents)
    adapter = tmp_path / "adapter.json"
    value = {"artifact_root": str(root), "scope_manifest_reference": {"path": str(manifest), "sha256": digest(manifest)},
             "scope_manifest": contents, "scope_packages": []}
    write(adapter, value)
    assert load_adapter(adapter)["scope_manifest"] == contents
    write(manifest, {"scopes": []})
    with pytest.raises(ValueError, match="path or hash"):
        load_adapter(adapter)
    write(manifest, contents)
    value["scope_manifest"] = {"scopes": []}
    write(adapter, value)
    with pytest.raises(ValueError, match="contents differ"):
        load_adapter(adapter)


def test_adapter_rejects_wrong_root_missing_and_recursive_reference(tmp_path):
    root = tmp_path / "proof"
    root.mkdir()
    manifest = root / "scope-manifest.json"
    write(manifest, {"scopes": []})
    ref = {"artifact_root": str(root), "scope_manifest_reference": {"path": str(manifest), "sha256": digest(manifest)}}
    adapter = tmp_path / "adapter.json"
    write(adapter, {**ref, "artifact_root": str(tmp_path)})
    with pytest.raises(ValueError, match="path or hash"):
        load_adapter(adapter)
    write(adapter, {**ref, "scope_packages": [ref]})
    with pytest.raises(ValueError, match="cyclic"):
        load_adapter(adapter)
    manifest.unlink()
    with pytest.raises(FileNotFoundError):
        load_adapter(adapter)


def prepared(tmp_path, label):
    profile = tmp_path / (label + "-profile.json")
    write(profile, {"runtime": label})
    builds, diagnostics = {}, {}
    for mode in ("native", "committed", "reworked"):
        binary, receipt = (tmp_path / (label + "-" + mode + suffix) for suffix in (".bin", ".json"))
        binary.write_bytes((label + mode).encode())
        write(receipt, {"binary": str(binary)})
        builds[mode] = {"binary": str(binary), "sha256": digest(binary), "build_receipt": str(receipt),
                        "build_receipt_sha256": digest(receipt)}
        diagnostics[mode] = {"binary": str(binary), "sha256": digest(binary)}
    return {"schema_version": 1, "original_application": {"revision": "approved-common-basis"},
            "contract_sha256": "common-contract", "field_policy": {"dtype": "<f8", "field_count": 20},
            "threads": 4, "build_args": [], "builds": builds, "diagnostic_builds": diagnostics,
            "compilers": {"committed": {"base_commit": "be58"}, "reworked": {"tree": label}},
            "calibration_profile": {"path": str(profile), "sha256": digest(profile)},
            "generation": {"committed": {"regions": []}, "reworked": {"regions": []}}}
