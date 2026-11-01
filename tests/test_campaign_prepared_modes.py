"""Four-mode imports retain independent authorities and exact public proof sidecars."""
from copy import deepcopy

import pytest
from elmm_pipeline.campaigns.identity import digest, read, write
from elmm_pipeline.campaigns.prepared_modes import BASELINE_REVISION, import_modes, load_adapter
from elmm_pipeline.campaigns.run import gates, summaries


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


def test_four_modes_keep_controls_and_candidate_profiles_and_complete_rotations(tmp_path):
    baseline, candidate = (prepared(tmp_path, name) for name in ("baseline", "candidate"))
    a, b, identity = (tmp_path / name for name in ("baseline.json", "candidate.json", "identity.json"))
    write(a, baseline)
    write(b, candidate)
    write(identity, {"revisions": {"main": BASELINE_REVISION}, "prepared_sha256": digest(a)})
    calls = []

    def authenticated(value, expected, *, threads, build_args):
        assert value == expected
        assert threads == 4
        assert build_args == []
        calls.append(value)
        return value, {row["binary"]: row["sha256"] for row in value["builds"].values()}

    result = import_modes(a, b, identity, tmp_path / "catalog", validate_prepared=authenticated)
    modes = result["modes"]
    assert len(calls) == 2
    assert [mode["name"] for mode in modes] == ["native", "be58", "baseline-7a612", "candidate"]
    assert [mode["binary"] for mode in modes[:3]] == [baseline["builds"][key]["binary"] for key in ("native", "committed", "reworked")]
    assert modes[-1]["binary"] == candidate["builds"]["reworked"]["binary"]
    assert modes[2]["calibration_profile"] != modes[3]["calibration_profile"]
    assert all(digest(path) == sha for path, sha in result["external_files"].items())
    assert read(tmp_path / "catalog/modes.json")["timings_reused"] is False
    case = {"performance_target": {"schema_version": 2, "metric": "ratio_to_control", "limit": .9,
            "comparison": "le", "control_mode": "baseline-7a612", "control_comparison": "le"}}
    runs = []
    for round_ in range(8):
        order = modes[round_ % 4:] + modes[:round_ % 4]
        for mode in order:
            runs.append({"mode": mode["name"], "round": round_, "warmup": round_ == 0, "traced": False,
                         "wall_seconds": 9 if mode["name"] == "candidate" else 10,
                         "peak_host_rss_bytes": 1024, "validation": {"passed": True}})
    summary = summaries(runs, modes)
    assert all(value["repetitions"] == 7 for value in summary.values())
    assert gates(result, summary, case, runs)["accepted"]


def test_four_mode_common_basis_mismatch_rejects_before_any_authentication(tmp_path):
    baseline, candidate = (prepared(tmp_path, name) for name in ("baseline", "candidate"))
    a, b, identity = (tmp_path / name for name in ("baseline.json", "candidate.json", "identity.json"))
    write(a, baseline)
    write(identity, {"revisions": {"main": BASELINE_REVISION}, "prepared_sha256": digest(a)})
    for key in ("original_application", "contract_sha256", "field_policy", "threads", "build_args"):
        value = deepcopy(candidate)
        value[key] = "different"
        write(b, value)
        with pytest.raises(ValueError, match="common basis"):
            import_modes(a, b, identity, tmp_path / key, validate_prepared=lambda *_args, **_kwargs: pytest.fail("not authenticated"))
