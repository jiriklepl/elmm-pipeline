"""Public plan integrity is checked before any compiler-approved source edit."""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.local_adapter import (
    LocalAdapterError,
    apply_scope_sources,
    compose_scope_packages,
    transform_scopes,
)
from test_scope_composition import package, save, unchanged
from test_scope_composition import sources as _shared_sources

sources = _shared_sources


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


def with_applicability(metadata):
    plan = with_plan(metadata)
    record = {
        "schema_version": 1, "kind": "checked-interval-cost-applicability", "plan_identity": plan["identity"],
        **{key: copy.deepcopy(plan[key]) for key in ("execution_identity", "entry", "sources",
            "source_inputs_identity", "configuration_identity", "capture_facts_identity")},
        "canonical_resources": copy.deepcopy(plan["resources"]),
        "key_candidates": [{"resource": "argument::a", "kind": "array_descriptor",
                            "safe_evaluation_authorized": False, "properties": [{"property": "size", "axis": 1}],
                            "missing_proofs": ["interval descriptor lifetime"]}],
        "unclassified_dependencies": [],
        "original_control_trees": [{"procedure": plan["entry"], "unit": 0, "tree": [],
                                    "entry_key_authorized": False}],
        "native_operations": [], "numerical_operations": [],
        "requirements": [{"kind": "runtime_workload_key", "status": "not_established", "reason": "coverage missing"}],
        "existing_exact_window": None, "placement_authorized": False, "execution_authorized": False,
        "existing_placement_unchanged": True, "identity_policy": "Exact configuration retained",
        "coverage": "Diagnostic candidates; no authority to evaluate or select execution",
    }
    seal(record)
    metadata["scope_manifest"]["scopes"][0]["interval_applicability"] = record
    save(metadata)
    return record


def test_interval_applicability_transports_without_changing_build_or_placement(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    expected = copy.deepcopy(with_applicability(metadata))
    bundle = compose_scope_packages([metadata], sources[1])
    scope = bundle["scope_packages"][0]["scope_manifest"]["scopes"][0]
    assert scope["interval_applicability"] == expected
    assert len(bundle["gpu_sources"]) == 2
    assert len(bundle["patched_sources"]) == 1
    assert "exact_window" not in scope


@pytest.mark.parametrize("change,reason", [
    (lambda r: r.update(schema_version=2), "schema version"),
    (lambda r: r.update(schema_version=True), "schema version"),
    (lambda r: r.update(plan_identity="0" * 64), "plan identity"),
    (lambda r: r.update(execution_identity="0" * 64), "execution_identity"),
    (lambda r: r.update(configuration_identity="0" * 64), "configuration_identity"),
    (lambda r: r.update(source_inputs_identity="0" * 64), "source_inputs_identity"),
    (lambda r: r.update(canonical_resources=[]), "canonical resources"),
    (lambda r: r.update(placement_authorized=True), "cannot authorize"),
    (lambda r: r.update(execution_authorized=0), "cannot authorize"),
    (lambda r: r.update(existing_exact_window={"available": True}), "cannot add or replace"),
    (lambda r: r["key_candidates"][0].update(safe_evaluation_authorized=True), "cannot authorize runtime"),
    (lambda r: r["requirements"][0].update(status="complete"), "unresolved explanatory"),
    (lambda r: r["original_control_trees"][0].update(procedure="foreign::scope"), "outside its checked interval"),
    (lambda r: r["original_control_trees"][0].update(unit=True), "outside its checked interval"),
])
def test_interval_applicability_rejects_detachment_or_authority_before_edits(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    record = with_applicability(metadata)
    change(record)
    seal(record)
    save(metadata)
    with pytest.raises(LocalAdapterError, match="interval applicability .*" + reason):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def test_interval_applicability_requires_its_plan_and_content_digest(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    record = with_applicability(metadata)
    record["requirements"][0]["reason"] = "changed after issuance"
    save(metadata)
    with pytest.raises(LocalAdapterError, match="interval applicability identity differs"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)
    seal(record)
    del metadata["scope_manifest"]["scopes"][0]["checked_execution_plan"]
    save(metadata)
    with pytest.raises(LocalAdapterError, match="interval applicability requires its checked plan"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def test_interval_applicability_projects_existing_attachment_without_authorizing_build(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    record = with_applicability(metadata)
    scope = metadata["scope_manifest"]["scopes"][0]
    scope["exact_window"] = {"schema_version": 4, "available": True, "selection": "native",
                             "reason": "measured native winner", "inventory": {"preserved": True}}
    record["existing_exact_window"] = {"available": True, "selection": "native",
                                        "reason": "measured native winner"}
    seal(record)
    validate_checked_execution_plans(metadata["scope_manifest"])
    save(metadata)
    with pytest.raises(LocalAdapterError, match="Exact-window"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def with_measured_applicability(metadata, selection="native"):
    record = with_applicability(metadata)
    scope = metadata["scope_manifest"]["scopes"][0]
    contract = {"protocol": "checked-interval-process-v4"}
    exact = {"available": True, "selection": selection, "reason": "verified complete-process decision",
             "capture": False, "contract": contract, "contract_sha256": digest(contract),
             "evidence_sha256": "e" * 64}
    scope["exact_window"] = exact
    record["requirements"][0].update(status="covered_by_checked_interval_attachment",
                                      attachment=exact["contract_sha256"])
    record["existing_exact_window"] = {key: value for key, value in exact.items() if key != "contract"}
    seal(record)
    save(metadata)
    return record, exact


@pytest.mark.parametrize("selection", ["native", "resident"])
def test_measured_interval_coverage_is_transported_but_does_not_authorize_build(tmp_path, sources, selection):
    metadata = package(tmp_path, sources, "one")
    record, _ = with_measured_applicability(metadata, selection)
    expected = copy.deepcopy(record)
    validate_checked_execution_plans(metadata["scope_manifest"])
    assert record == expected
    # A linked coverage diagnostic cannot substitute for an executed build
    # contract. The deliberately incomplete contract still blocks all edits.
    with pytest.raises(LocalAdapterError, match="Exact-window"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


@pytest.mark.parametrize("change", [
    lambda r, e: e.update(available=False),
    lambda r, e: e.update(available=1),
    lambda r, e: e.update(capture=True),
    lambda r, e: e.update(selection="unknown"),
    lambda r, e: e.update(selection=[]),
    lambda r, e: e.update(contract={"protocol": "checked-interval-process-v3"}),
    lambda r, e: e.update(contract=None),
    lambda r, e: e.update(contract_sha256="0" * 64),
    lambda r, e: e.update(evidence_sha256=None),
    lambda r, e: r["requirements"][0].update(attachment="0" * 64),
    lambda r, e: r["requirements"][0].pop("attachment"),
])
def test_measured_interval_coverage_rejects_detached_or_unavailable_evidence(tmp_path, sources, change):
    metadata = package(tmp_path, sources, "one")
    record, exact = with_measured_applicability(metadata)
    change(record, exact)
    record["existing_exact_window"] = {key: value for key, value in exact.items() if key != "contract"}
    seal(record)
    save(metadata)
    with pytest.raises(ValueError, match="covered requirement differs"):
        validate_checked_execution_plans(metadata["scope_manifest"])
    unchanged(sources)


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
    (lambda p: p.update(schema_version=6), "schema version"),
    (lambda p: p.update(schema_version=4), "complete v4 schema"),
    (lambda p: p.update(schema_version=5), "complete v5 schema"),
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
    metadata["scope_manifest"]["scope_execution"] = "graph"
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


def seal_epoch_plan(plan):
    epoch = plan["storage_epoch"]
    for member in epoch["members"]:
        for transfer in member["storage_transfers"]:
            seal(transfer)
        for task in member.get("device_tasks", []):
            seal(task)
    seal(epoch)
    plan["execution_identity"] = digest({
        "execution_identity_version": 2,
        **{key: copy.deepcopy(plan[key]) for key in ("entry", "sources", "source_inputs_identity", "configuration_identity",
                                                     "capture_facts_identity", "resources", "numerical_invocations", "gpu_leaves")},
        "procedures": [{"procedure": row["procedure"], "emission_identity": row["emission_identity"]}
                       for row in plan["procedures"]], "storage_epoch_identity": epoch["identity"],
    })
    seal(plan)


def with_epoch(metadata, *, opaque=False, transfer=False):
    plan = with_plan(metadata)
    plan["schema_version"] = 2
    if opaque:
        plan["procedures"][0]["units"][0]["resources"] = {}
    if transfer:
        plan["resources"].append({"resource": "argument::b", "registration_identity": 2, "descriptor": {"rank": 1}})
        plan["procedures"][0]["units"][0]["resources"]["argument::b"] = "argument::b"
    procedure = plan["procedures"][0]
    procedure["boundaries"] = [{"first_line": 2, "last_line": 2, "reason": "original opaque mutation",
                                "execution": "publish and close when reached; original continuation once", "reopen": True}]
    resources = []
    for original in plan["resources"]:
        facts = ({"storage": "opaque host reservation", "lifetime": "original checked native access",
                  "device_replica_authorized": False, "initialization_authorized": False} if opaque else
                 {"storage": "stable", "lifetime": "invocation", "allocation_changes": False, "escapes": False,
                  "device_replica_authorized": True})
        resources.append({**copy.deepcopy(original), **facts})
    epoch = {
        "version": 1, "entry": plan["entry"],
        "source_inputs": {"schema_version": 1, "sources": copy.deepcopy(plan["sources"]),
                          "dependencies": {}, "configuration": None, "entries": []},
        "resources": resources,
        "members": [{"procedure": procedure["procedure"], "source_graph": procedure["structured_summary_identity"],
                     "units": [{"ordinal": unit["ordinal"], "selected_node_ids": copy.deepcopy(unit["selected_node_ids"]),
                                "canonical_resources": copy.deepcopy(unit["resources"])} for unit in procedure["units"]],
                     "retirement_cuts": copy.deepcopy(procedure["boundaries"]), "storage_transfers": []}],
        "activation": "first reached managed access; no descriptor evaluation on zero-trip paths",
        "lifetime": "checked original invocation between mandatory retirement cuts",
        "generation": "runtime-checked allocation generation; never inferred from address reuse",
        "readonly": "add a valid replica; preserve other current replicas",
        "writes": "invalidate only overlapping regions after ordered completion",
        "return": "checked children borrow; unmanaged return publishes observable storage",
        "definitions_authorized": False, "placement_authorized": False, "cross_cut_retention_authorized": False,
    }
    plan["storage_epoch"] = epoch
    if transfer:
        associations = []
        for formal, resource in (("left", "argument::a"), ("right", "argument::b")):
            associations.append({"formal": formal, "actual": resource.split("::")[-1], "entry_definition_kill": False,
                "formal_descriptor": {"name": formal, "resource": "argument::" + formal, "type": "real", "kind": 8,
                    "rank": 1, "intent": "inout", "attributes": ["allocatable", "intent"], "lower_bounds": ["1"],
                    "shape": [{"kind": "Deferred_Shape_Spec", "source": ":", "bounds": [None, None]}]},
                "actual_type": {"type": "real", "kind": 8, "rank": 1, "storage_resource": resource, "declaring_type": None}})
        call = {"schema_version": 1, "procedure": "source::exchange", "target": "exchange", "source_kind": "module",
            "formal_arguments": ["left", "right"], "actual_arguments": ["a", "b"], "associations": associations,
            "original_arguments": [{"position": index, "formal": formal, "keyword": None, "actual": actual}
                                   for index, (formal, actual) in enumerate((("left", "a"), ("right", "b")))],
            "actual_evaluation": "unchanged original call position and guards", "effect_authority": False,
            "capture_authority": False, "alias_authority": False, "completion_authority": False, "execution_authority": False}
        epoch["members"][0]["storage_transfers"].append({"version": 1, "procedure": plan["entry"],
            "source_graph": procedure["structured_summary_identity"], "selected_node_ids": [plan["entry"] + "/body/1"],
            "callee": "source::exchange", "source_inputs": copy.deepcopy(epoch["source_inputs"]), "call": call,
            "body": ["CALL move_alloc(left, temporary)", "CALL move_alloc(right, left)", "CALL move_alloc(temporary, right)"],
            "resources": ["argument::a", "argument::b"],
            "transition": "exchange allocation associations; physical generations and replicas survive",
            "payload_reads": False, "payload_writes": False, "initialization_authorized": False,
            "runtime_guard": "both current layouts match opposite live allocations after the original call",
            "on_unproved_mutation": "publish and retire before original source operation"})
    seal_epoch_plan(plan)
    save(metadata)
    return plan


@pytest.mark.parametrize("variant", ["managed", "opaque", "transfer"])
def test_v2_storage_epoch_transports_exact_source_lifetime_without_execution_authority(tmp_path, sources, variant):
    metadata = package(tmp_path, sources, "one")
    plan = with_epoch(metadata, opaque=variant == "opaque", transfer=variant == "transfer")
    expected = copy.deepcopy(plan)
    validate_checked_execution_plans(metadata["scope_manifest"])
    bundle = compose_scope_packages([metadata], sources[1])
    assert bundle["scope_packages"][0]["scope_manifest"]["scopes"][0]["checked_execution_plan"] == expected
    assert expected["storage_epoch"]["definitions_authorized"] is False
    assert expected["storage_epoch"]["placement_authorized"] is False
    assert expected["storage_epoch"]["cross_cut_retention_authorized"] is False


@pytest.mark.parametrize(("change", "reason"), [
    (lambda e: e.update(extra=True), "complete v1 schema"),
    (lambda e: e.update(version=True), "unsupported version"),
    (lambda e: e.update(entry="foreign"), "entry differs"),
    (lambda e: e["source_inputs"].update(extra=True), "source inputs"),
    (lambda e: e["source_inputs"]["sources"].clear(), "source inputs"),
    (lambda e: e.update(definitions_authorized=True), "cannot authorize"),
    (lambda e: e.update(placement_authorized=0), "cannot authorize"),
    (lambda e: e.update(cross_cut_retention_authorized=True), "cannot authorize"),
    (lambda e: e.update(readonly="invalidate other replicas"), "lifetime policy"),
    (lambda e: e.update(resources=[]), "canonical resources"),
    (lambda e: e["resources"][0].update(registration_identity=True), "canonical descriptor"),
    (lambda e: e["resources"][0]["descriptor"].update(rank=True), "canonical descriptor"),
    (lambda e: e["resources"][0].update(allocation_changes=True), "capture facts"),
    (lambda e: e["resources"][0].update(escapes=0), "capture facts"),
    (lambda e: e["resources"][0].update(initialization_authorized=True), "capture facts"),
    (lambda e: e.update(members=[]), "members differ"),
    (lambda e: e["members"][0].update(extra=True), "complete member"),
    (lambda e: e["members"][0].update(source_graph="0" * 64), "source graph"),
    (lambda e: e["members"][0]["units"][0].update(ordinal=False), "selected units"),
    (lambda e: e["members"][0]["units"][0].update(selected_node_ids=["foreign"]), "selected units"),
    (lambda e: e["members"][0]["units"][0]["canonical_resources"].clear(), "canonical mappings"),
    (lambda e: e["members"][0].update(retirement_cuts=[]), "mandatory retirement cuts"),
    (lambda e: e["members"][0].update(storage_transfers={}), "storage transfer records"),
])
def test_v2_epoch_mutation_rejected_before_edits_even_with_resealed_identities(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    plan = with_epoch(metadata)
    change(plan["storage_epoch"])
    # A malformed transfer container is intentionally not normalized by the test's seal helper.
    if isinstance(plan["storage_epoch"]["members"], list) and all(isinstance(m.get("storage_transfers"), list) for m in plan["storage_epoch"]["members"]):
        seal_epoch_plan(plan)
    else:
        seal(plan["storage_epoch"])
        seal(plan)
    save(metadata)
    with pytest.raises(LocalAdapterError, match="storage epoch .*" + reason):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


@pytest.mark.parametrize(("change", "reason"), [
    (lambda t: t.update(extra=True), "complete v1 schema"),
    (lambda t: t.update(version=True), "unsupported version"),
    (lambda t: t.update(source_graph="0" * 64), "source linkage"),
    (lambda t: t.update(selected_node_ids=["foreign/body/1"]), "original call selection"),
    (lambda t: t.update(resources=["argument::a", "argument::a"]), "distinct original storage"),
    (lambda t: t.update(payload_reads=True), "payload access"),
    (lambda t: t.update(payload_writes=0), "payload access"),
    (lambda t: t.update(initialization_authorized=True), "payload access"),
    (lambda t: t.update(runtime_guard="unchecked"), "transition policy"),
    (lambda t: t["call"].update(extra=True), "exact original call schema"),
    (lambda t: t["call"].update(execution_authority=True), "cannot grant"),
    (lambda t: t["call"].update(capture_authority=0), "cannot grant"),
    (lambda t: t["call"]["original_arguments"][0].update(position=False), "argument order"),
    (lambda t: t["call"]["associations"][0].update(entry_definition_kill=True), "association differs"),
    (lambda t: t["call"]["associations"][0]["formal_descriptor"].update(intent="out"), "allocatable INOUT"),
    (lambda t: t["call"]["associations"][0]["actual_type"].update(storage_resource="foreign"), "actual storage"),
    (lambda t: t["call"]["associations"][0]["actual_type"].update(rank=True), "numeric descriptor"),
    (lambda t: t["body"].__setitem__(1, "CALL move_alloc(left,right)"), "MOVE_ALLOC order"),
])
def test_v2_storage_transfer_cannot_expand_payload_or_source_authority(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    plan = with_epoch(metadata, transfer=True)
    change(plan["storage_epoch"]["members"][0]["storage_transfers"][0])
    seal_epoch_plan(plan)
    save(metadata)
    with pytest.raises(LocalAdapterError, match="storage transfer .*" + reason):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def test_v2_epoch_and_execution_digests_are_both_required_and_v1_cannot_gain_epoch(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    plan = with_epoch(metadata)
    plan["storage_epoch"]["identity"] = "0" * 64
    seal(plan)
    with pytest.raises(ValueError, match="storage epoch identity differs"):
        validate_checked_execution_plans(metadata["scope_manifest"])
    seal_epoch_plan(plan)
    plan["execution_identity"] = "0" * 64
    seal(plan)
    with pytest.raises(ValueError, match="storage epoch execution identity differs"):
        validate_checked_execution_plans(metadata["scope_manifest"])
    plan["schema_version"] = 1
    seal(plan)
    with pytest.raises(ValueError, match="complete v1 schema"):
        validate_checked_execution_plans(metadata["scope_manifest"])


def with_device_task(metadata, *, pressure_backend=None):
    """An explicit protocol fixture with temporary provider artifacts, never execution evidence."""
    plan = with_epoch(metadata)
    epoch, procedure = plan["storage_epoch"], plan["procedures"][0]
    root = Path(metadata["artifact_root"]).parent / (Path(metadata["artifact_root"]).name + "-provider-fixture")
    root.mkdir()
    implementation, qualification, provider_path = (root / name for name in ("implementation.source", "qualification.json", "provider.json"))
    implementation.write_text("explicit synthetic provider fixture; not an executable\n")
    qualification.write_text('{"fixture_only":true,"execution_evidence":false}\n')
    body = "e" * 64
    descriptor = {"name": "a", "resource": "argument::a", "type": "real", "kind": 8, "rank": 1, "intent": "inout",
                  "attributes": ["intent"], "lower_bounds": ["1"],
                  "shape": [{"kind": "Assumed_Shape_Spec", "source": ":", "bounds": [None, None]}]}
    call = {"schema_version": 1, "procedure": "source::original", "target": "original", "source_kind": "module",
        "formal_arguments": ["a"], "actual_arguments": ["a"],
        "original_arguments": [{"position": 0, "formal": "a", "keyword": None, "actual": "a"}],
        "associations": [{"formal": "a", "actual": "a", "formal_descriptor": descriptor,
            "actual_type": {"type": "real", "kind": 8, "rank": 1, "storage_resource": "argument::a", "declaring_type": None},
            "entry_definition_kill": False}],
        "actual_evaluation": "unchanged original call position and guards", "effect_authority": False,
        "capture_authority": False, "alias_authority": False, "completion_authority": False, "execution_authority": False}
    upper = {"descriptor": "size", "formal": "a", "axis": 1, "offset": 0}
    contract = {"version": 1, "source": {"sha256": procedure["source_sha256"], "span": [1, 1],
        "call_sha256": "f" * 64, "callee": "source::original", "callee_body_sha256": body},
        "resources": [{"formal": "a", "effects": [{"kind": "read", "lower": [0], "upper": [copy.deepcopy(upper)]},
                                                    {"kind": "overwrite", "lower": [0], "upper": [copy.deepcopy(upper)]}]}],
        "configuration": [{"component": "extent", "kind": "size", "values": [copy.deepcopy(upper)]}]}
    provider = {"version": 1, "kind": "external_device_provider",
        "fortran": {"module": "provider", "config_type": "config", "create": "create", "execute": "execute", "destroy": "destroy"},
        "artifacts": {str(implementation): hashlib.sha256(implementation.read_bytes()).hexdigest()},
        "numerical_contract": {"identity": "c" * 64,
            "qualification": {"path": str(qualification), "sha256": hashlib.sha256(qualification.read_bytes()).hexdigest()},
            "original_bodies": [body], "task_contracts": [digest({"original_body": body, "resources": contract["resources"],
                                                                 "configuration": contract["configuration"]})]},
        "execution": {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim", "fallback": "prework_only"}}
    if pressure_backend is not None:
        provider["pressure_backend"] = copy.deepcopy(pressure_backend)
    provider_path.write_text(json.dumps(provider, sort_keys=True) + "\n")
    contract["provider_manifest"] = {"path": str(provider_path), "sha256": hashlib.sha256(provider_path.read_bytes()).hexdigest()}
    task = {"version": 1, "procedure": plan["entry"], "source_graph": procedure["structured_summary_identity"],
        "selected_node_ids": [plan["entry"] + "/body/1"], "source_inputs": copy.deepcopy(epoch["source_inputs"]),
        "call": call, "contract": contract, "provider": provider, "canonical_resources": {"a": "argument::a"},
        "source_effects": {"summary_identity": "d" * 64, "summary_version": 22, "complete": True,
                           "effects": {"a": ["overwrite", "read"]}, "host_effects": False, "persistent_state": False},
        "source_association_authorized": True, "definitions_authorized": False, "placement_authorized": False,
        "numerical_equivalence": "explicit supplied adapter qualification",
        "lifetime": "existing checked capture and current caller-owned domain only",
        "position": "exact original reached CALL; no motion or eager descriptor reads"}
    epoch["members"][0]["device_tasks"] = [task]
    seal_epoch_plan(plan)
    metadata["scope_manifest"]["scopes"][0]["external_device_tasks"] = copy.deepcopy([task])
    save(metadata)
    return plan


def test_v2_explicit_device_task_transports_without_granting_placement_or_definitions(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    plan = with_device_task(metadata)
    expected = copy.deepcopy(plan)
    validate_checked_execution_plans(metadata["scope_manifest"])
    bundle = compose_scope_packages([metadata], sources[1])
    assert bundle["scope_packages"][0]["scope_manifest"]["scopes"][0]["checked_execution_plan"] == expected
    task, = expected["storage_epoch"]["members"][0]["device_tasks"]
    assert task["definitions_authorized"] is False
    assert task["placement_authorized"] is False
    assert task["source_association_authorized"] is True


@pytest.mark.parametrize(("change", "reason"), [
    (lambda t: t.update(extra=True), "exact record fields"),
    (lambda t: t.update(version=True), "version or identity"),
    (lambda t: t.update(source_graph="0" * 64), "source linkage"),
    (lambda t: t.update(selected_node_ids=["foreign/body/1"]), "call selection"),
    (lambda t: t.update(definitions_authorized=True), "cannot authorize"),
    (lambda t: t.update(placement_authorized=0), "cannot authorize"),
    (lambda t: t["call"].update(execution_authority=True), "cannot grant"),
    (lambda t: t["call"]["associations"][0].update(entry_definition_kill=True), "array descriptors"),
    (lambda t: t["call"]["associations"][0]["formal_descriptor"].update(intent="out"), "array descriptors"),
    (lambda t: t["call"]["associations"][0]["formal_descriptor"].update(lower_bounds=[]), "descriptor shape"),
    (lambda t: t["call"]["associations"][0]["formal_descriptor"]["shape"][0].update(bounds=[True]), "descriptor axis"),
    (lambda t: t.update(canonical_resources={"a": "foreign"}), "canonical storage"),
    (lambda t: t["contract"]["source"].update(sha256="0" * 64), "call/body identity"),
    (lambda t: t["contract"]["resources"][0]["effects"][0].update(lower=[]), "half-open boxes"),
    (lambda t: t["contract"]["resources"][0]["effects"][0]["upper"][0].update(axis=True), "descriptor expression"),
    (lambda t: t["contract"]["resources"][0]["effects"][0]["upper"][0].update(formal="foreign"), "descriptor expression"),
    (lambda t: t["source_effects"].update(complete=False), "cannot erase"),
    (lambda t: t["source_effects"].update(persistent_state=True), "cannot erase"),
    (lambda t: t["source_effects"].update(host_effects=0), "cannot erase"),
    (lambda t: t["source_effects"].update(summary_version=True), "cannot erase"),
    (lambda t: t["source_effects"]["effects"].update(a=["read"]), "omits or invents"),
    (lambda t: t["contract"]["configuration"][0].update(kind="unknown"), "configuration component"),
    (lambda t: t["contract"]["configuration"][0].update(values=[-1]), "INTEGER configuration"),
    (lambda t: t["contract"]["resources"][0]["effects"][0].update(lower=[1]), "lost its exact boxes"),
    (lambda t: t["contract"]["configuration"][0].update(values=[1]), "lost its exact boxes"),
])
def test_v2_device_task_rejects_detachment_or_widened_authority_before_edits(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    plan = with_device_task(metadata)
    task = plan["storage_epoch"]["members"][0]["device_tasks"][0]
    change(task)
    seal_epoch_plan(plan)
    metadata["scope_manifest"]["scopes"][0]["external_device_tasks"] = copy.deepcopy([task])
    save(metadata)
    with pytest.raises(LocalAdapterError, match="device task .*" + reason):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


@pytest.mark.parametrize("kind", ["provider", "implementation", "qualification", "duplicate", "scope_copy", "empty"])
def test_v2_device_task_artifacts_and_duplicate_transports_are_authenticated(tmp_path, sources, kind):
    metadata = package(tmp_path, sources, "one")
    plan = with_device_task(metadata)
    member = plan["storage_epoch"]["members"][0]
    task = member["device_tasks"][0]
    if kind == "provider":
        Path(task["contract"]["provider_manifest"]["path"]).write_text("{}\n")
    elif kind == "implementation":
        Path(next(iter(task["provider"]["artifacts"]))).write_text("changed\n")
    elif kind == "qualification":
        Path(task["provider"]["numerical_contract"]["qualification"]["path"]).unlink()
    elif kind == "duplicate":
        member["device_tasks"].append(copy.deepcopy(task))
        seal_epoch_plan(plan)
    elif kind == "scope_copy":
        metadata["scope_manifest"]["scopes"][0]["external_device_tasks"][0]["placement_authorized"] = True
    else:
        member["device_tasks"] = []
        seal_epoch_plan(plan)
    save(metadata)
    with pytest.raises(LocalAdapterError):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)
