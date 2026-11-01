"""Selected original lifetimes retain strict cuts and independent NONE facts."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from test_checked_execution_plan import seal_epoch_plan

from compiler.tests.test_graph_storage_epochs import SOURCE, emit


@pytest.fixture(scope="module", params=[False, True], ids=["calls", "if"])
def issued(request, tmp_path_factory):
    source = SOURCE
    if request.param:
        source = source.replace("call producer(right,n)\ncall consumer(left,right,n)",
                                "if(n>0)then\ncall producer(right,n)\ncall consumer(left,right,n)\nendif")
    _path, _outputs, manifest = emit(tmp_path_factory.mktemp("selected-epoch-reader"), source)
    return manifest


def parts(manifest):
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    return scope, plan, epoch, epoch["selected_position_epoch"]


def reseal(manifest):
    scope, plan, _epoch, selected = parts(manifest)
    selected["identity"] = _digest({k: v for k, v in selected.items() if k != "identity"})
    seal_epoch_plan(plan)
    if "interval_applicability" in scope:
        record = scope["interval_applicability"]
        record["plan_identity"] = plan["identity"]
        record["execution_identity"] = plan["execution_identity"]
        record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})


def test_actual_issued_selected_lifetime_carries_no_entry_or_execution_authority(issued):
    validate_checked_execution_plans(issued)
    _scope, plan, epoch, selected = parts(issued)
    assert plan["schema_version"] == 2
    assert epoch["version"] == selected["version"] == 1
    assert selected["entry_definition_authority"] is selected["execution_authorized"] is False
    assert selected["cross_frontier_retention_authorized"] is False
    assert [call["procedure"] for call in selected["calls"]] == ["epochs::producer", "epochs::consumer"]


@pytest.mark.parametrize("fault", ["version_bool", "owner", "descriptor", "extra", "definitions", "execution",
                                  "retention_bool", "presence", "activation", "selection_duplicate", "selection_gap",
                                  "cut_inside", "cut_gap", "foreign_selection", "unknown_root", "duplicate_root",
                                  "not_selected_window", "missing_retirement", "conditional_retirement", "missing_call",
                                  "outside_call", "duplicate_call", "unknown_mapping", "aliased_mapping",
                                  "child_mapping", "missing_graph"])
def test_resealed_selected_epoch_cannot_change_original_associations_frontier_or_authority(issued, fault):
    data = deepcopy(issued)
    scope, plan, epoch, selected = parts(data)
    if fault in {"version_bool", "owner", "descriptor", "extra", "definitions", "execution", "retention_bool",
                 "presence", "activation"}:
        key, value = {"version_bool": ("version", True), "owner": ("procedure", "foreign"),
                      "descriptor": ("descriptor_identity", "not-a-sha"), "extra": ("extra", False),
                      "definitions": ("entry_definition_authority", True), "execution": ("execution_authorized", True),
                      "retention_bool": ("cross_frontier_retention_authorized", 0),
                      "presence": ("allocation_presence", "entry assumed present"),
                      "activation": ("repeated_activation", "whole on reentry")}[fault]
        selected[key] = value
    elif fault == "selection_duplicate":
        selected["selected_node_ids"].append(selected["selected_node_ids"][0])
    elif fault == "selection_gap":
        selected["selected_node_ids"].append(selected["cut_before_node_id"] + "0")
    elif fault == "cut_inside":
        selected["cut_before_node_id"] = selected["selected_node_ids"][0]
    elif fault == "cut_gap":
        selected["cut_before_node_id"] += "0"
    elif fault == "foreign_selection":
        selected["selected_node_ids"][0] = "foreign/body/0"
    elif fault == "unknown_root":
        selected["resources"][0] = "foreign::root"
    elif fault == "duplicate_root":
        selected["resources"].append(selected["resources"][0])
    elif fault == "not_selected_window":
        epoch["resources"][0]["lifetime"] = "invocation"
    elif fault in {"missing_retirement", "conditional_retirement"}:
        for procedure, member in zip(plan["procedures"], epoch["members"], strict=True):
            if procedure["procedure"] != plan["entry"]:
                continue
            for boundary in procedure["boundaries"]:
                if boundary["reason"] == "checked selected-position allocation epoch retirement":
                    boundary["conditional_close" if fault == "conditional_retirement" else "reason"] = "forged"
            member["retirement_cuts"] = deepcopy(procedure["boundaries"])
    elif fault == "missing_call":
        selected["calls"].clear()
    elif fault == "outside_call":
        selected["calls"][0]["node_id"] = selected["cut_before_node_id"]
    elif fault == "duplicate_call":
        selected["calls"].append(deepcopy(selected["calls"][0]))
    elif fault == "unknown_mapping":
        selected["calls"][0]["canonical_resources"]["argument::b"] = "foreign::root"
    elif fault == "aliased_mapping":
        mappings = selected["calls"][1]["canonical_resources"]
        mappings["argument::a"] = mappings["argument::b"]
    elif fault == "child_mapping":
        mappings = selected["calls"][0]["canonical_resources"]
        mappings["argument::b"] = next(root for root in selected["resources"] if root != mappings["argument::b"])
    else:
        scope.pop("graph_execution")
    reseal(data)
    with pytest.raises(ValueError, match="selected position epoch"):
        validate_checked_execution_plans(data)
