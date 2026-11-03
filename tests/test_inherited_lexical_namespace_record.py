"""Lexical namespace transport preserves original CALLs, slots and native roles."""
from copy import deepcopy
from hashlib import sha256

import pytest
from compiler.tests.test_inherited_lexical_namespace import (
    emit_package,
    scalar_element_source,
    source_fixture,
)
from test_conditional_return_record import seal

from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.conditional_return_record import _projected_implicit_uses
from elmm_pipeline.source_task_call_record import require_single_association


@pytest.fixture(scope="module", params=["flat", "nested", "without_native", "scalar"])
def issued(request, tmp_path_factory):
    source = scalar_element_source() if request.param == "scalar" else source_fixture(
        nested=request.param != "flat", native_call=request.param != "without_native")
    return emit_package(tmp_path_factory.mktemp("lexical-reader"), source)[-1]


@pytest.fixture(scope="module")
def nested(tmp_path_factory):
    return emit_package(tmp_path_factory.mktemp("lexical-forgeries"))[-1]


def parts(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    helper = "hierarchical::middle::internal_helper"
    coordinator = next(row for row in scope["internal_coordinators"] if row["procedure"] == helper)
    projection = next(row["call_projection"] for row in epoch["native_opaque_accesses"]
                      if row["procedure"] == "hierarchical::leaf")
    node = next(row for row in projection["members"] if row["procedure"] == helper)
    route = next(row for member in epoch["return_contract"]["members"] for row in member["routes"]
                 if row["kind"] == "checked_lexical_child")
    return scope, plan, epoch, coordinator, projection, node, route


def test_serialized_selected_owner_keeps_native_and_managed_descendant_uses(issued):
    report = deepcopy(issued)
    before = deepcopy(report)
    validate_checked_execution_plans(report)
    scope = report["scopes"][0]
    used = _projected_implicit_uses(scope, "hierarchical::middle")
    assert {"hierarchical::weights", "hierarchical::left", "hierarchical::right"} <= used
    assert _projected_implicit_uses(scope, "hierarchical::producer") == set()
    assert scope["checked_execution_plan"]["storage_epoch"]["return_contract"]["root_selection"]
    assert report == before


@pytest.mark.parametrize("fault", [
    "version_bool", "kind", "extra", "parent", "namespace", "missing_calls", "repeated_calls", "call_graph",
    "call_selection", "call_actual", "call_formal", "call_kind", "call_span", "call_execution", "terminal_credit",
    "parameterized", "slot", "missing_node", "route_kind", "route_namespace", "route_call", "policy_version",
    "policy_missing", "return_old_version", "coherent_foreign_namespace", "coherent_wrong_parent",
])
def test_resealed_lexical_transport_cannot_change_its_original_contract(nested, fault):
    report = deepcopy(nested)
    _scope, _plan, epoch, coordinator, projection, node, route = parts(report)
    relation = node["inherited_namespace"]
    row = relation["source_calls"][0]
    call = row["source_call"]
    if fault == "version_bool": relation["version"] = True
    elif fault == "kind": relation["kind"] = "guessed_host_association"
    elif fault == "extra": relation["lifetime_authorized"] = True
    elif fault == "parent": relation["parent"] = "hierarchical::advance"
    elif fault == "namespace": relation["namespace_owner"] = "hierarchical::leaf"
    elif fault == "missing_calls": relation["source_calls"] = []
    elif fault == "repeated_calls": relation["source_calls"].append(deepcopy(row))
    elif fault == "call_graph": row["caller_graph"] = "0" * 64
    elif fault == "call_selection": row["selected_node_ids"] = ["hierarchical::middle/body/0"]
    elif fault == "call_actual": call["actual_arguments"] = ["n"]
    elif fault == "call_formal": call["formal_arguments"] = ["n"]
    elif fault == "call_kind": call["source_kind"] = "module"
    elif fault == "call_span": call["first_line"] = True
    elif fault == "call_execution": call["execution"] = "invented borrowed execution"
    elif fault == "terminal_credit": node["owner_resources"] = {root: root for root in node["resource_slots"]}
    elif fault == "parameterized": node["parameterized"] = True
    elif fault == "slot": node["resource_slots"]["hierarchical::weights"] += 1
    elif fault == "missing_node": projection["members"].remove(node)
    elif fault == "route_kind": route["kind"] = "checked_child"
    elif fault == "route_namespace": route["namespace_owner"] = "hierarchical::leaf"
    elif fault == "route_call": route["source_call"]["target"] = "other"
    elif fault == "policy_version": epoch["return_contract"]["lexical_child_policy_version"] = True
    elif fault == "policy_missing": epoch["return_contract"].pop("lexical_child_policy_version")
    elif fault == "return_old_version": epoch["return_contract"]["version"] = 3
    elif fault == "coherent_foreign_namespace":
        relation["namespace_owner"] = coordinator["lexical_namespace"]["namespace_owner"] = route["namespace_owner"] = "hierarchical::leaf"
    elif fault == "coherent_wrong_parent":
        relation["parent"] = coordinator["lexical_namespace"]["parent"] = "hierarchical::advance"
    else: raise AssertionError(fault)
    seal(report)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(report)


@pytest.mark.parametrize("root", ["hierarchical::weights", "hierarchical::left"])
def test_unrelated_native_or_managed_sibling_cannot_justify_implicit_capture(nested, root):
    report = deepcopy(nested)
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    producer = "hierarchical::producer"
    number = next(row["registration_identity"] for row in plan["resources"] if row["resource"] == root)
    slot = str(int(sha256((producer + "|" + root).encode()).hexdigest()[:8], 16) + 1)
    route = next(row for member in epoch["return_contract"]["members"] for row in member["routes"]
                 if row.get("kind") == "checked_child" and row["procedure"] == producer)
    route["canonical_resources"][root] = root
    route["canonical_handles"][slot] = number
    route["handle_forwarding"][slot] = number
    procedure = next(row for row in plan["procedures"] if row["procedure"] == producer)
    coordinator = next(row for row in scope["module_coordinators"] if row["procedure"] == producer)
    procedure["calls"][0]["handle_bindings"][slot] = number
    coordinator["calls"][0]["handle_bindings"][slot] = number
    seal(report)
    with pytest.raises(ValueError, match="conditional return child formal-to-canonical association differs"):
        validate_checked_execution_plans(report)


def test_single_association_cannot_stop_at_inherited_nonparameterized_helper(nested):
    report = deepcopy(nested)
    _scope, plan, _epoch, _coordinator, _projection, _node, _route = parts(report)
    helper = next(row for row in plan["procedures"] if row["procedure"].endswith("::internal_helper"))
    middle = next(row for row in plan["procedures"] if row["procedure"] == "hierarchical::middle")
    extra = deepcopy(next(row for row in middle["calls"] if "handle_bindings" in row))
    extra["resource_mappings"] = {"argument::unused": "hierarchical::left"}
    middle["calls"].append(extra)
    with pytest.raises(ValueError, match="varying CALL projection"):
        require_single_association(helper, plan["entry"], report)


def test_raw_and_serialized_scalar_projection_share_exact_json_semantics(tmp_path):
    _path, _builder, _outputs, raw, serialized = emit_package(tmp_path, scalar_element_source())
    validate_checked_execution_plans(raw)
    validate_checked_execution_plans(serialized)


@pytest.mark.parametrize("fault", ["bool_key", "float_key", "bool_value", "duplicate_json_key"])
def test_original_slot_types_remain_exact_despite_source_call_json_transport(nested, fault):
    report = deepcopy(nested)
    _scope, _plan, _epoch, _coordinator, projection, _node, _route = parts(report)
    mapping = projection["calls"][0]["handle_bindings"]
    key = next(iter(mapping))
    value = mapping[key]
    if fault == "bool_key":
        mapping[True] = mapping.pop(key)
    elif fault == "float_key":
        mapping[float(key)] = mapping.pop(key)
    elif fault == "bool_value":
        mapping[key] = True
    else:
        mapping[int(key)] = value
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        # Mixed non-string keys cannot even form a canonical JSON record;
        # valid JSON carrying a boolean slot value reaches typed validation.
        seal(report)
        validate_checked_execution_plans(report)


@pytest.mark.parametrize("fault", ["duplicate_module", "fake_parent", "fake_call"])
def test_single_association_does_not_trust_unchecked_lexical_metadata(nested, fault):
    report = deepcopy(nested)
    scope, plan, _epoch, coordinator, _projection, _node, _route = parts(report)
    helper = next(row for row in plan["procedures"] if row["procedure"].endswith("::internal_helper"))
    if fault == "duplicate_module":
        fake = deepcopy(coordinator)
        fake["procedure"] = "hierarchical::middle"
        scope["internal_coordinators"].append(fake)
        target = next(row for row in plan["procedures"] if row["procedure"] == fake["procedure"])
    else:
        target = helper
        if fault == "fake_parent":
            coordinator["lexical_namespace"]["parent"] = "hierarchical::advance"
        else:
            coordinator["lexical_namespace"]["source_calls"] = []
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        require_single_association(target, plan["entry"], report)
