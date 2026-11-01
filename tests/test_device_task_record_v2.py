"""Strict transport of opaque-state device tasks; synthetic data is not execution evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.local_adapter import LocalAdapterError, apply_scope_sources
from test_checked_execution_plan import digest, seal_epoch_plan, with_device_task
from test_checked_execution_plan import sources as source_fixture
from test_scope_composition import package, save, unchanged

sources = source_fixture


def seal_v2(metadata, *, qualify=False):
    scope = metadata["scope_manifest"]["scopes"][0]
    plan = scope["checked_execution_plan"]
    task = plan["storage_epoch"]["members"][0]["device_tasks"][0]
    contract, provider = task["contract"], task["provider"]
    if qualify:
        provider["numerical_contract"]["task_contracts"] = [digest({
            "original_body": contract["source"]["callee_body_sha256"], "resources": contract["resources"],
            "configuration": contract["configuration"], "opaque_arguments": contract["opaque_arguments"],
            "completion": contract["completion"]})]
    path = Path(contract["provider_manifest"]["path"])
    path.write_text(json.dumps(provider, sort_keys=True) + "\n")
    contract["provider_manifest"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for state in task["source_effects"]["opaque_state"]:
        state["identity"] = digest({key: value for key, value in state.items() if key != "identity"})
    task["source_proof_identity"] = digest({key: value for key, value in task.items()
        if key not in ("identity", "source_proof_identity", "execution_binding")})
    seal_epoch_plan(plan)
    scope["external_device_tasks"] = copy.deepcopy([task])
    save(metadata)
    return task


def opaque_package(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    plan = with_device_task(metadata)
    task = plan["storage_epoch"]["members"][0]["device_tasks"][0]
    contract, call, provider = task["contract"], task["call"], task["provider"]
    descriptor = {"name": "d", "resource": "argument::d", "type": "type(state)", "kind": None, "rank": 0,
                  "intent": "inout", "attributes": ["intent"], "lower_bounds": [], "shape": []}
    actual_type = {"type": "type(state)", "kind": None, "rank": 0,
                   "storage_resource": plan["entry"] + "::cache", "declaring_type": {"module": "library", "name": "state"}}
    call["formal_arguments"].insert(0, "d")
    call["actual_arguments"].insert(0, "cache")
    for item in call["original_arguments"]:
        item["position"] += 1
    call["original_arguments"].insert(0, {"position": 0, "formal": "d", "actual": "cache", "keyword": None})
    call["associations"].insert(0, {"formal": "d", "actual": "cache", "formal_descriptor": descriptor,
                                   "actual_type": actual_type, "entry_definition_kill": False})
    call["associations"][1]["formal_descriptor"]["intent"] = "out"
    call["associations"][1]["entry_definition_kill"] = True
    contract.update(version=2, completion="wait_without_export", opaque_arguments=[{
        "formal": "d", "constructor": {"span": [1, 1], "statement_sha256": "a" * 64},
        "qualification": copy.deepcopy(provider["numerical_contract"]["qualification"])}])
    contract["resources"][0]["effects"] = [contract["resources"][0]["effects"][1]]
    provider.update(version=2, metadata_query={"module": "library_metadata", "procedure": "query",
        "opaque_formals": ["d"], "configuration_argument": "config", "read_only": True,
        "configuration_components": [{"component": "coefficient", "kind": "real64", "count": 1}]})
    state = {"version": 1, "procedure": task["procedure"], "source_graph": task["source_graph"],
        "source_inputs": copy.deepcopy(task["source_inputs"]), "selected_node_ids": copy.deepcopy(task["selected_node_ids"]),
        "formal": "d", "resource": actual_type["storage_resource"], "actual_type": copy.deepcopy(actual_type),
        "descriptor": {**descriptor, "name": "cache", "resource": actual_type["storage_resource"],
                       "intent": None, "attributes": ["save"]},
        "constructor": {**contract["opaque_arguments"][0]["constructor"],
                        "selected_node_ids": [task["procedure"] + "/body/0"], "execution": "unchanged original source and guards"},
        "qualification": copy.deepcopy(contract["opaque_arguments"][0]["qualification"]),
        "observers": "only original constructor LHS and exact contracted receiver argument",
        "workspace_effects": "explicit supplied private-scratch qualification; never a readonly object claim",
        "capture_authorized": False, "definitions_authorized": False, "gpu_access_authorized": False,
        "metadata_reads_authorized": False, "storage": "original SAVE declaration; no copy or C address export"}
    task.update(version=2, source_effects={"version": 2, "original_body_sha256": contract["source"]["callee_body_sha256"],
        "resource_effects": {"a": ["overwrite"]}, "entry_definition_kills": ["a"], "opaque_state": [state],
        "qualification": copy.deepcopy(provider["numerical_contract"]["qualification"]),
        "wrapper_effects": "unchanged original source; only exact inner CALL replaced"},
        execution_binding={"owner": plan["entry"], "canonical_resources": {"a": "argument::a"},
            "registration_slots": {"a": 1}, "provider_slot": 1, "configuration_words": 2,
            "ownership": "owning invocation controls forwarded by reference through every checked ENTRY"})
    seal_v2(metadata, qualify=True)
    return metadata


def test_opaque_v2_keeps_source_state_definition_kill_and_owner_binding_distinct(tmp_path, sources):
    metadata = opaque_package(tmp_path, sources)
    validate_checked_execution_plans(metadata["scope_manifest"])
    task = metadata["scope_manifest"]["scopes"][0]["external_device_tasks"][0]
    assert task["definitions_authorized"] is False
    assert task["source_effects"]["entry_definition_kills"] == ["a"]
    assert task["source_effects"]["opaque_state"][0]["gpu_access_authorized"] is False
    assert task["execution_binding"]["registration_slots"] == {"a": 1}


def test_opaque_v2_preserves_original_use_alias_and_resolved_declaring_type(tmp_path, sources):
    metadata = opaque_package(tmp_path, sources)
    task = metadata["scope_manifest"]["scopes"][0]["checked_execution_plan"]["storage_epoch"]["members"][0]["device_tasks"][0]
    association = task["call"]["associations"][0]
    association["actual_type"]["type"] = "type(local_alias)"
    state = task["source_effects"]["opaque_state"][0]
    state["actual_type"]["type"] = "type(local_alias)"
    state["descriptor"]["type"] = "type(local_alias)"
    seal_v2(metadata)
    validate_checked_execution_plans(metadata["scope_manifest"])
    assert association["formal_descriptor"]["type"] == "type(state)"
    assert association["actual_type"]["declaring_type"] == {"module": "library", "name": "state"}
    assert state["descriptor"]["type"] == "type(local_alias)"


@pytest.mark.parametrize("change", [
    lambda t: t["source_effects"]["entry_definition_kills"].clear(),
    lambda t: t["call"]["associations"][1].update(entry_definition_kill=False),
    lambda t: t["source_effects"]["opaque_state"][0].update(gpu_access_authorized=True),
    lambda t: t["source_effects"]["opaque_state"][0]["descriptor"].update(attributes=[]),
    lambda t: t["source_effects"]["opaque_state"][0]["descriptor"].update(attributes=["save", "target"]),
    lambda t: t["source_effects"]["opaque_state"][0]["constructor"].update(span=[2, 2]),
    lambda t: t["source_effects"]["opaque_state"][0].update(observers="additional component observers"),
    lambda t: t["source_effects"].update(wrapper_effects="replace entire wrapper"),
    lambda t: t["execution_binding"].update(owner="foreign::owner"),
    lambda t: t["execution_binding"]["registration_slots"].update(a=2),
    lambda t: t["execution_binding"].update(provider_slot=2),
    lambda t: t["execution_binding"].update(configuration_words=1),
    lambda t: t["execution_binding"].update(ownership="process global cache"),
    lambda t: t["execution_binding"]["canonical_resources"].update(a="foreign::a"),
    lambda t: t["provider"]["metadata_query"].update(read_only=False),
    lambda t: t["provider"]["metadata_query"].update(opaque_formals=["a"]),
    lambda t: t["provider"]["metadata_query"]["configuration_components"][0].update(count=True),
    lambda t: t["provider"]["metadata_query"]["configuration_components"][0].update(count=65),
    lambda t: t["provider"]["metadata_query"]["configuration_components"][0].update(component="extent"),
    lambda t: t["contract"].update(completion="unobserved_async"),
    lambda t: t["contract"]["opaque_arguments"][0]["constructor"].update(span=[True, 1]),
])
def test_v2_resealed_tampering_rejects_before_source_edits(tmp_path, sources, change):
    metadata = opaque_package(tmp_path, sources)
    task = metadata["scope_manifest"]["scopes"][0]["checked_execution_plan"]["storage_epoch"]["members"][0]["device_tasks"][0]
    change(task)
    seal_v2(metadata)
    with pytest.raises(LocalAdapterError, match="device task"):
        apply_scope_sources(metadata, sources[1])
    unchanged(sources)


def test_v2_qualification_binds_opaque_constructor_and_completion(tmp_path, sources):
    metadata = opaque_package(tmp_path, sources)
    task = metadata["scope_manifest"]["scopes"][0]["checked_execution_plan"]["storage_epoch"]["members"][0]["device_tasks"][0]
    task["contract"]["opaque_arguments"][0]["constructor"]["span"] = [2, 2]
    task["source_effects"]["opaque_state"][0]["constructor"]["span"] = [2, 2]
    seal_v2(metadata)
    with pytest.raises(ValueError, match="lost its exact boxes or configuration"):
        validate_checked_execution_plans(metadata["scope_manifest"])


def test_v2_detached_source_proof_digest_is_not_replaced_by_outer_resealing(tmp_path, sources):
    metadata = opaque_package(tmp_path, sources)
    scope = metadata["scope_manifest"]["scopes"][0]
    task = scope["checked_execution_plan"]["storage_epoch"]["members"][0]["device_tasks"][0]
    task["source_proof_identity"] = "0" * 64
    seal_epoch_plan(scope["checked_execution_plan"])
    scope["external_device_tasks"] = copy.deepcopy([task])
    with pytest.raises(ValueError, match="original source proof identity"):
        validate_checked_execution_plans(metadata["scope_manifest"])
