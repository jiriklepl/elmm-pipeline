"""Original CALL projections cannot forge another root, slot or graph effect."""
import json
from copy import deepcopy

import pytest
from compiler.tests.test_source_task_call_bindings import multi_field_builder
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.source_task_record import validate_source_task


@pytest.fixture(scope="module", params=[(False, False), (True, False), (False, True)])
def authentic(request, tmp_path_factory):
    keyword, repeat = request.param
    builder = multi_field_builder(tmp_path_factory.mktemp("call-reader"), keyword=keyword, repeat=repeat)
    outputs, _ = builder.run()
    return json.loads(outputs["scope-manifest.json"]), builder.facts


def values(manifest, index=0):
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    member = next(m for m in plan["storage_epoch"]["members"] if m.get("source_device_tasks"))
    procedure = next(p for p in plan["procedures"] if p["procedure"] == member["procedure"])
    return (member["source_device_tasks"][index], member, procedure, plan["storage_epoch"]["source_inputs"],
            {r["resource"]: r for r in plan["storage_epoch"]["resources"]}, plan["entry"], manifest)


def check(args):
    task, member, procedure, inputs, resources, owner, manifest = args
    validate_source_task(task, member, procedure, inputs, resources, owner=owner, manifest=manifest)


def seal(args):
    task = args[0]
    projection = task["execution_binding"].get("call_projection")
    if projection is not None:
        projection["identity"] = _digest({k: v for k, v in projection.items() if k != "identity"})
    task["identity"] = _digest({k: v for k, v in task.items() if k != "identity"})


def graph_tasks(value):
    if isinstance(value, list):
        for item in value:
            yield from graph_tasks(item)
    elif isinstance(value, dict):
        if "id" in value:
            yield value
        for child in value.values():
            yield from graph_tasks(child)


def test_original_keyword_repeated_nested_calls_round_trip(authentic):
    manifest, facts = authentic
    before = deepcopy(manifest)
    validate_checked_execution_plans(manifest, capture_facts=facts)
    for index in range(3):
        check(values(manifest, index))
    assert manifest == before


@pytest.mark.parametrize("fault", [
    "missing", "version", "kind", "extra", "owner", "procedure", "source_proof", "definition", "lifetime",
    "alias", "specialization", "semantics", "runtime", "members_empty", "members_duplicate", "members_order",
    "member_source", "member_graph", "member_hash", "member_namespace", "member_extra", "member_bool_slot",
    "member_alias_slot", "terminal_root", "terminal_slot", "field_slot", "field_extra", "calls_empty",
    "call_missing", "call_duplicate", "call_order", "call_caller", "call_callee", "call_graph", "call_span_bool",
    "call_span", "call_node", "call_node_duplicate", "call_extra", "call_handle", "call_handle_bool",
    "call_handle_alias", "storage_kind", "storage_slot", "storage_missing", "source_version", "source_actual",
    "source_order", "source_section", "source_type", "source_resource", "source_mapping", "canonical_actual",
    "resolved_missing", "emitted_missing", "emitted_mapping", "field_roots", "field_roots_order",
    "registration", "registration_bool", "effects_read", "effects_write", "effects_metadata", "effects_overwrite",
    "graph_read", "graph_write", "graph_metadata", "graph_missing", "graph_duplicate", "ancestor_effect",
    "graph_must_write", "graph_overwrite", "ancestor_must_write", "ancestor_overwrite", "namespace_abi",
])
def test_resealed_projection_preserves_every_original_binding(authentic, fault):
    args = values(deepcopy(authentic[0]))
    check(args)
    task, _member, procedure, _inputs, _resources, _owner, manifest = args
    binding = task["execution_binding"]
    p = binding["call_projection"]
    node = p["members"][0]
    terminal = next(n for n in p["members"] if not n["parameterized"])
    edge = p["calls"][0]
    handle = next(iter(edge["handle_bindings"]))
    original = edge["source_call"]
    array = next(r for r in original["resource_mappings"] if r["formal_descriptor"]["rank"])
    if fault == "missing": binding.pop("call_projection")
    elif fault == "version": p["version"] = True
    elif fault == "kind": p["kind"] = "caller claimed projection"
    elif fault == "extra": p["motion_authorized"] = True
    elif fault == "owner": p["owner"] = "foreign::step"
    elif fault == "procedure": p["procedure"] = "foreign::advance"
    elif fault == "source_proof": p["source_proof_identity"] = "0" * 64
    elif fault in {"definition", "lifetime", "alias", "specialization"}:
        p[{"specialization": "value_specialization_authorized"}.get(fault, fault + "_authorized")] = True
    elif fault == "semantics": p["projection_semantics"] = "joint paths guessed"
    elif fault == "runtime": p["runtime_validation"] = "none"
    elif fault == "members_empty": p["members"] = []
    elif fault == "members_duplicate": p["members"].append(deepcopy(node))
    elif fault == "members_order": p["members"].reverse()
    elif fault == "member_source": node["source"] = "foreign.f90"
    elif fault == "member_graph": node["source_graph"] = "0" * 64
    elif fault == "member_hash": node["source_sha256"] = "0" * 64
    elif fault == "member_namespace": node["parameterized"] = 1
    elif fault == "member_extra": node["lifetime"] = "forever"
    elif fault == "member_bool_slot": node["resource_slots"][next(iter(node["resource_slots"]))] = True
    elif fault == "member_alias_slot":
        a, b = node["resource_slots"]
        node["resource_slots"][b] = node["resource_slots"][a]
    elif fault == "terminal_root": terminal["owner_resources"]["argument::a"] = "argument::b"
    elif fault == "terminal_slot": terminal["resource_slots"]["argument::a"] = 55
    elif fault == "field_slot": p["field_slots"]["argument::a"] += 1
    elif fault == "field_extra": p["field_slots"]["argument::foreign"] = 55
    elif fault == "calls_empty": p["calls"] = []
    elif fault == "call_missing": p["calls"].pop()
    elif fault == "call_duplicate": p["calls"].append(deepcopy(edge))
    elif fault == "call_order": p["calls"].reverse()
    elif fault == "call_caller": edge["caller"] = "foreign::caller"
    elif fault == "call_callee": edge["callee"] = "foreign::callee"
    elif fault == "call_graph": edge["caller_graph"] = "0" * 64
    elif fault == "call_span_bool": edge["first_line"] = True
    elif fault == "call_span": edge["last_line"] += 1
    elif fault == "call_node": edge["selected_node_ids"] = ["foreign::caller/body/0"]
    elif fault == "call_node_duplicate": edge["selected_node_ids"] *= 2
    elif fault == "call_extra": edge["alias_authorized"] = True
    elif fault == "call_handle": edge["handle_bindings"][handle] += 1
    elif fault == "call_handle_bool": edge["handle_bindings"][handle] = True
    elif fault == "call_handle_alias":
        a, b = edge["handle_bindings"]
        edge["handle_bindings"][b] = edge["handle_bindings"][a]
    elif fault == "storage_kind": edge["storage_identities"][handle]["kind"] = "owner_registration"
    elif fault == "storage_slot": edge["storage_identities"][handle]["slot"] += 1
    elif fault == "storage_missing": edge["storage_identities"].pop(handle)
    elif fault == "source_version": original["schema_version"] = True
    elif fault == "source_actual": original["actual_arguments"][0] = "other"
    elif fault == "source_order": original["original_arguments"].reverse()
    elif fault == "source_section": array["section"] = {"lower": 2}
    elif fault == "source_type": array["actual_descriptor"]["kind"] = 4
    elif fault == "source_resource": array["resource"] = "argument::foreign"
    elif fault == "source_mapping": original["resource_mapping"]["argument::a"] = "argument::foreign"
    elif fault == "canonical_actual": edge["resource_mappings"]["argument::a"] = "argument::ma"
    elif fault == "resolved_missing": manifest["resolved_calls"] = []
    elif fault == "emitted_missing": procedure["calls"].pop()
    elif fault == "emitted_mapping": procedure["calls"][0]["handle_bindings"][handle] += 1
    elif fault == "field_roots": p["field_roots"]["argument::a"].pop()
    elif fault == "field_roots_order": p["field_roots"]["argument::a"].reverse()
    elif fault == "registration": p["root_registrations"]["argument::a"] = 55
    elif fault == "registration_bool": p["root_registrations"]["argument::a"] = True
    elif fault == "effects_read": p["conservative_effects"]["reads"].pop()
    elif fault == "effects_write": p["conservative_effects"]["may_writes"].pop()
    elif fault == "effects_metadata": p["conservative_effects"]["metadata_reads"].pop()
    elif fault == "effects_overwrite": p["conservative_effects"]["overwrites"] = ["argument::a"]
    elif fault == "namespace_abi":
        manifest["scopes"][0]["module_coordinators"][0]["parameterized_handles"] = False
    else:
        graph = manifest["scopes"][0]["graph_execution"]
        row = next(r for r in graph_tasks(graph) if r["id"] == "lines::advance:source_device:0")
        if fault == "graph_read": row["reads"].pop()
        elif fault == "graph_write": row["may_writes"].pop()
        elif fault == "graph_metadata": row["metadata_reads"].pop()
        elif fault == "graph_missing": row["id"] = "missing"
        elif fault == "graph_duplicate": row["nested"] = deepcopy(row)
        elif fault == "graph_must_write": row["must_writes"] = ["argument::a"]
        elif fault == "graph_overwrite": row["overwrites"] = ["argument::a"]
        elif fault in {"ancestor_must_write", "ancestor_overwrite"}:
            ancestor = next(r for r in graph_tasks(graph) if r["id"].endswith(":alternative:0"))
            ancestor["must_writes" if fault == "ancestor_must_write" else "overwrites"] = ["argument::a"]
        elif fault == "ancestor_effect":
            ancestor = next(r for r in graph_tasks(graph) if r["id"].endswith(":alternative:0"))
            ancestor["may_writes"].pop()
        else: raise AssertionError(fault)
    seal(args)
    with pytest.raises(ValueError):
        check(args)


def test_resealed_matching_transport_cannot_redirect_original_array_slot(authentic):
    args = values(deepcopy(authentic[0]))
    check(args)
    p = args[0]["execution_binding"]["call_projection"]
    edge = p["calls"][0]
    keys = list(edge["handle_bindings"])
    edge["handle_bindings"][keys[0]], edge["handle_bindings"][keys[1]] = (
        edge["handle_bindings"][keys[1]], edge["handle_bindings"][keys[0]])
    for key in keys:
        edge["storage_identities"][key]["slot"] = edge["handle_bindings"][key]
    args[2]["calls"][0]["handle_bindings"] = deepcopy(edge["handle_bindings"])
    seal(args)
    with pytest.raises(ValueError, match="original array association"):
        check(args)


@pytest.mark.parametrize("owner", [True, False])
def test_singleton_check_stops_at_existing_owner_namespace(owner):
    from elmm_pipeline.source_task_call_record import require_single_association

    name = "m::owner" if owner else "m::owner::internal"
    procedure = {"procedure": name, "calls": [
        {"caller": "m::external", "resource_mappings": {"argument::unused": "argument::a"}},
        {"caller": "m::external", "resource_mappings": {"argument::unused": "argument::b"}},
    ]}
    scope = {"checked_execution_plan": {"entry": "m::owner", "procedures": [procedure]},
             "module_coordinators": [], "internal_coordinators": []}
    require_single_association(procedure, "m::owner", {"scopes": [scope]})
