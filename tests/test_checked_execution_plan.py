"""Public plan integrity is checked before any compiler-approved source edit."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from elmm_pipeline.local_adapter import LocalAdapterError, apply_scope_sources, compose_scope_packages, transform_scopes
from test_scope_composition import package, save, sources, unchanged  # noqa: F401 - shared fixture


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def seal(plan):
    plan["identity"] = digest({key: value for key, value in plan.items() if key != "identity"})


def with_plan(metadata, *, facts=None, analysis=None):
    manifest = metadata["scope_manifest"]
    source = next(iter(manifest["sources"]))
    facts = facts or {"schema_version": 1, "sources": manifest["source_inputs"]}
    summary = "a" * 64
    plan = {
        "schema_version": 1, "execution_identity": "d" * 64,
        "entry": manifest["entry"], "sources": dict(manifest["source_inputs"]),
        "source_inputs_identity": digest({"schema_version": 1, "sources": manifest["source_inputs"],
            "dependencies": analysis["dependencies"] if analysis else {},
            "configuration": analysis["configuration"] if analysis else None,
            "entries": sorted(analysis["entries"], key=lambda item: item["source"]) if analysis else []}),
        "configuration_identity": "b" * 64, "capture_facts_identity": digest(facts),
        "procedures": [{"procedure": manifest["entry"], "source": source,
            "source_sha256": manifest["source_inputs"][source], "structured_summary_identity": summary,
            "emission_identity": "c" * 64,
            "units": [{"ordinal": 0, "selected_node_ids": ["original-statement-1"], "first_line": 1, "last_line": 1,
                "resources": {"argument::local": "argument::a"}, "ordered_definitions": {"required_whole": []},
                "operations": {"calls": [], "native": [], "order": []}}], "boundaries": [], "calls": []}],
        "resources": [{"resource": "argument::a", "registration_identity": 1, "descriptor": {"rank": 1}}],
        "numerical_invocations": 0, "gpu_leaves": [],
        "policy": {"public_record": "diagnostics only; cannot be imported as execution authority"},
        "cost_requirements": [{"kind": "native_environment", "reason": "compatible estimate unavailable"}],
    }
    seal(plan)
    manifest.update({"analysis_sources": analysis, "capture_facts_sha256":
                     hashlib.sha256(json.dumps(facts, sort_keys=True).encode()).hexdigest(),
                     "scopes": [{"path": source, "source_authority": {"version": 1,
                                 "structured_summary_identity": summary}, "checked_execution_plan": plan}]})
    save(metadata)
    return plan


def test_checked_plan_preserved_in_composition_without_placement_decisions(tmp_path, sources):
    first = package(tmp_path, sources, "first")
    second = package(tmp_path, sources, "second", source="consumer.f90")
    expected = [copy.deepcopy(with_plan(item)) for item in (first, second)]
    bundle = compose_scope_packages([first, second], sources[1])
    assert [item["scope_manifest"]["scopes"][0]["checked_execution_plan"]
            for item in bundle["scope_packages"]] == expected
    assert "checked_execution_plan" not in bundle
    assert len(bundle["patched_sources"]) == 2
    # An unavailable estimate in the explanatory record cannot alter approved
    # build roles or select a new native/GPU implementation in the pipeline.
    assert len(bundle["gpu_sources"]) == 3


@pytest.mark.parametrize("change,reason", [
    (lambda p: p.update(schema_version=True), "schema version"),
    (lambda p: p.update(schema_version=2), "schema version"),
    (lambda p: p.update(unknown_field="unsupported"), "complete v1 schema"),
    (lambda p: p.pop("policy"), "complete v1 schema"),
    (lambda p: p.update(configuration_identity="bad"), "SHA256"),
    (lambda p: p.pop("execution_identity"), "complete v1 schema"),
    (lambda p: p.update(execution_identity="D" * 64), "SHA256"),
    (lambda p: p.update(capture_facts_identity="C" * 64), "SHA256"),
    (lambda p: p["sources"].update({"relative.f90": "0" * 64}), "original source"),
    (lambda p: p["sources"].update({next(iter(p["sources"])): "0" * 64}), "original source"),
    (lambda p: p.update(source_inputs_identity="0" * 64), "configured source identity"),
    (lambda p: p.update(entry="different"), "entry differs"),
    (lambda p: p.update(procedures=[]), "original procedure"),
    (lambda p: p["procedures"].append(copy.deepcopy(p["procedures"][0])), "original procedure"),
    (lambda p: p["procedures"][0].update(source="/unavailable.f90"), "procedure source linkage"),
    (lambda p: p["procedures"][0].update(source_sha256="0" * 64), "procedure source linkage"),
    (lambda p: p["procedures"][0].update(structured_summary_identity="bad"), "procedure source linkage"),
    (lambda p: p["procedures"][0].pop("emission_identity"), "complete procedure records"),
    (lambda p: p["procedures"][0].update(emission_identity="C" * 64), "emission SHA256"),
    (lambda p: p["procedures"][0].update(procedure="different"), "root procedure"),
    (lambda p: p["procedures"][0].update(boundaries=["opaque-string"]), "boundary and call records"),
    (lambda p: p["procedures"][0].update(calls={}), "boundary and call records"),
    (lambda p: p["procedures"][0]["units"][0].update(ordinal=1), "ordinals"),
    (lambda p: p["procedures"][0]["units"][0].update(ordinal=False), "ordinals"),
    (lambda p: p["procedures"][0]["units"][0].update(selected_node_ids=[1]), "node identities"),
    (lambda p: p["procedures"][0]["units"][0].update(first_line=0), "line spans"),
    (lambda p: p["procedures"][0]["units"][0].update(last_line=False), "line spans"),
    (lambda p: p["procedures"][0]["units"][0].update(resources={"local": "missing"}), "resource linkage"),
    (lambda p: p["procedures"][0]["units"][0].update(ordered_definitions=[]), "definition and operation"),
    (lambda p: p["procedures"][0]["units"][0].update(operations=[]), "definition and operation"),
    (lambda p: p["resources"].append({**p["resources"][0], "resource": "argument::b"}), "registrations"),
    (lambda p: p["resources"].append({**p["resources"][0], "registration_identity": 2}), "registrations"),
    (lambda p: p["resources"][0].update(registration_identity=True), "registrations"),
    (lambda p: p["resources"][0].update(registration_identity=0), "registrations"),
    (lambda p: p["resources"][0].update(descriptor=[]), "resource descriptors"),
    (lambda p: p.update(numerical_invocations=-1), "invocation count"),
    (lambda p: p.update(numerical_invocations=True), "invocation count"),
    (lambda p: p.update(gpu_leaves=["one", "one"]), "leaf identities"),
    (lambda p: p.update(policy=[]), "policy facts"),
    (lambda p: p.update(cost_requirements=[{"kind": "unknown"}]), "cost requirements"),
])
def test_invalid_second_plan_rejects_before_any_source_edit(tmp_path, sources, change, reason):
    first = package(tmp_path, sources, "first")
    second = package(tmp_path, sources, "second", source="consumer.f90")
    plan = with_plan(second)
    change(plan)
    seal(plan)  # Exercise semantic links, rather than merely a stale digest.
    save(second)
    with pytest.raises(LocalAdapterError, match="checked execution plan .*" + reason):
        compose_scope_packages([first, second], sources[1])
    unchanged(sources)
    with pytest.raises(LocalAdapterError, match="checked execution plan .*" + reason):
        apply_scope_sources(second, sources[1])
    unchanged(sources)


@pytest.mark.parametrize("change,reason", [
    (lambda s: s["checked_execution_plan"].update(numerical_invocations=4), "identity differs"),
    (lambda s: s["checked_execution_plan"].update(execution_identity="0" * 64), "identity differs"),
    (lambda s: s.update(checked_execution_plan=None), "complete v1 schema"),
    (lambda s: s.update(path="/different.f90"), "root procedure"),
    (lambda s: s["source_authority"].update(structured_summary_identity="0" * 64), "root summary"),
    (lambda s: s["checked_execution_plan"]["policy"].update(value=float("nan")), "finite JSON"),
])
def test_corrupt_or_detached_plan_rejects(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    with_plan(metadata)
    change(metadata["scope_manifest"]["scopes"][0])
    save(metadata)
    with pytest.raises(LocalAdapterError, match="checked execution plan .*" + reason):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def test_configured_source_identity_includes_dependencies_configuration_and_sorted_entries(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    analysis = {"schema_version": 1, "source_inputs": sources[0], "preserves_source_order": True,
                "configuration": {"macro": 4}, "dependencies": {},
                "entries": [{"source": source, "path": source, "sha256": digest, "line_map": [1]}
                            for source, digest in reversed(list(sources[0].items()))]}
    with_plan(metadata, analysis=analysis)
    apply_scope_sources(metadata, sources[1])
    # The identity binds configuration even when all original source hashes match.
    analysis["configuration"]["macro"] = 2
    save(metadata)
    with pytest.raises(LocalAdapterError, match="configured source identity differs"):
        apply_scope_sources(metadata, sources[1])


@pytest.mark.parametrize("tamper", [False, "capture", "identity"])
def test_fresh_compiler_response_validates_requested_capture_facts(tmp_path, sources, tamper):
    metadata = package(tmp_path, sources, "fresh")
    facts = {"schema_version": 1, "participation": "serial", "captures": {}, "sources": sources[0]}
    plan = with_plan(metadata, facts=facts)
    expected = copy.deepcopy(plan)
    if tamper == "capture":
        plan["capture_facts_identity"] = "0" * 64
        seal(plan)
    elif tamper == "identity":
        plan["identity"] = "0" * 64
    save(metadata)
    manifest = metadata["scope_manifest"]
    response = {"supported": True, "scopes": manifest,
                "outputs": [*manifest["artifacts_sha256"], "scope-manifest.json"]}
    paths = list(sources[0])
    with patch("elmm_pipeline.local_adapter._invoke", return_value=(response, ["compiler"])):
        def transform():
            return transform_scopes(Path(paths[0]), tmp_path, Path(sys.executable), Path(metadata["artifact_root"]),
                                    entry="fresh", source_files=paths[1:], capture_facts=facts)
        if tamper:
            with pytest.raises(LocalAdapterError, match="checked execution plan .*identity differs"):
                transform()
            unchanged(sources)
        else:
            result = transform()
            assert result["scope_manifest"]["scopes"][0]["checked_execution_plan"] == expected
            apply_scope_sources(result, sources[1])


@pytest.mark.parametrize("exact_window_version", [3, 4])
def test_checked_plan_cannot_authorize_exact_window_build_contract(tmp_path, sources, exact_window_version):
    metadata = package(tmp_path, sources, "one")
    with_plan(metadata)
    metadata["scope_manifest"]["scopes"][0]["exact_window"] = {"schema_version": exact_window_version}
    save(metadata)
    with pytest.raises(LocalAdapterError, match="Exact-window"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)
