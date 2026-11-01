"""Conditional lifetime records cannot stand in for source or payload authority."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)

from compiler.tests.test_conditional_return import compile_fixture


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _outputs, report, _scope, _proof = compile_fixture(tmp_path_factory.mktemp("conditional-reader"))
    return report


@pytest.fixture
def issued(original):
    return deepcopy(original)


def seal(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    if "return_contract" in epoch:
        contract = epoch["return_contract"]
        contract["identity"] = _digest({k: v for k, v in contract.items() if k != "identity"})
    epoch["identity"] = _digest({k: v for k, v in epoch.items() if k != "identity"})
    plan["execution_identity"] = _digest({"execution_identity_version": 2,
        **{key: plan[key] for key in ("entry", "sources", "source_inputs_identity", "configuration_identity",
                "capture_facts_identity", "resources", "numerical_invocations", "gpu_leaves")},
        "procedures": [{"procedure": p["procedure"], "emission_identity": p["emission_identity"]} for p in plan["procedures"]],
        "storage_epoch_identity": epoch["identity"]})
    plan["identity"] = _digest({k: v for k, v in plan.items() if k != "identity"})
    if "interval_applicability" in scope:
        record = scope["interval_applicability"]
        record.update(plan_identity=plan["identity"], execution_identity=plan["execution_identity"])
        record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})


def test_real_compiler_conditional_plan3_preserves_exact_old_execution_identity_recipe(issued):
    validate_checked_execution_plans(issued)
    plan = issued["scopes"][0]["checked_execution_plan"]
    assert plan["schema_version"] == 3
    assert len(plan["storage_epoch"]["return_contract"]["members"]) == 4


@pytest.mark.parametrize("fault", ["v2_extra", "v3_missing", "version", "authority", "escape", "policy",
    "entry", "source", "control", "control_bool", "symbol", "selector", "selector_authority", "member_missing",
    "member_graph", "specification", "exit", "entry_authority", "unit_missing", "native_ordinal", "child_missing",
    "retirement_missing", "unproved_lifetime", "canonical", "canonical_bool", "forwarding", "local_slot",
    "coordinator_version", "root_return", "route_unknown", "route_extra", "unknown_key"])
def test_resealed_return_contract_cannot_change_source_controls_or_routes(issued, fault):
    scope = issued["scopes"][0]
    plan = scope["checked_execution_plan"]
    contract = plan["storage_epoch"]["return_contract"]
    middle = next(m for m in contract["members"] if m["procedure"].endswith("::middle"))
    child = next(r for r in middle["routes"] if r["kind"] == "checked_child")
    cut = next(r for r in middle["routes"] if r["kind"] == "reached_retirement")
    if fault == "v2_extra":
        plan["schema_version"] = 2
    elif fault == "v3_missing":
        del plan["storage_epoch"]["return_contract"]
    elif fault == "version":
        contract["version"] = True
    elif fault == "authority":
        contract["entry_definition_authority"] = True
    elif fault == "escape":
        contract["escape_completion_assumed"] = True
    elif fault == "policy":
        contract["unproved_boundary_admission"] = "rearm unchecked alias on a later call"
    elif fault == "entry":
        contract["entry"] = middle["procedure"]
    elif fault == "source":
        contract["source_inputs"] = {}
    elif fault == "control":
        contract["controls"]["module_control_version"] = 4
    elif fault == "control_bool":
        contract["controls"]["module_control_version"] = True
    elif fault == "symbol":
        contract["controls"]["retention_allowed"] = "unbound_flag"
    elif fault == "selector":
        contract["selection_identity"] = "e" * 64
    elif fault == "selector_authority":
        issued["native_ancestor_selection"]["alias_authority"] = True
        selection = issued["native_ancestor_selection"]
        selection["identity"] = _digest({k: v for k, v in selection.items() if k != "identity"})
        contract["selection_identity"] = selection["identity"]
    elif fault == "member_missing":
        contract["members"].pop()
    elif fault == "member_graph":
        middle["source_graph"] = "f" * 64
    elif fault == "specification":
        middle["entry_exit"]["specification"] = "unchecked entry side effects"
    elif fault == "exit":
        middle["entry_exit"]["exit"] = "retain child local allocation after return"
    elif fault == "entry_authority":
        middle["entry_exit"]["definition_authority"] = True
    elif fault == "unit_missing":
        middle["routes"] = [r for r in middle["routes"] if r["kind"] != "checked_unit"]
    elif fault == "native_ordinal":
        next(r for r in middle["routes"] if r["kind"] == "checked_native")["ordinal"] = 99
    elif fault == "child_missing":
        middle["routes"].remove(child)
    elif fault == "retirement_missing":
        middle["routes"].remove(cut)
    elif fault == "unproved_lifetime":
        cut.update(lifetime="unproved_lifetime_or_escape", rearm_allowed=False)
    elif fault == "canonical":
        child["canonical_resources"]["argument::a"] = "argument::b"
    elif fault == "canonical_bool":
        key = next(k for k, v in child["canonical_handles"].items() if v == 1)
        child["canonical_handles"][key] = True
    elif fault == "forwarding":
        child["handle_forwarding"][next(iter(child["handle_forwarding"]))] += 1
    elif fault == "local_slot":
        child["canonical_handles"]["999"] = child["canonical_handles"].pop(next(iter(child["canonical_handles"])))
    elif fault == "coordinator_version":
        scope["module_coordinators"][0]["control_version"] = 4
    elif fault == "root_return":
        root = next(m for m in contract["members"] if m["procedure"] == plan["entry"])
        root["routes"].append({"kind": "checked_return", "selected_node_ids": [plan["entry"] + "/999999"]})
    elif fault == "route_unknown":
        cut["kind"] = "unchecked_completed_escape"
    elif fault == "route_extra":
        cut["payload_authorized"] = True
    else:
        contract["new_authority"] = True
    seal(issued)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(issued)
