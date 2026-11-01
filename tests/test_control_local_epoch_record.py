"""Public region transport keeps compiler proofs opaque and cuts mandatory."""
from copy import deepcopy

import pytest

from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from compiler.tests.test_control_local_graph_epochs import CONDITIONAL_SOURCE, emit
from test_selected_epoch_record import parts, reseal


@pytest.fixture(scope="module", params=["scalar", "descriptor", "three_arms"])
def issued(request, tmp_path_factory):
    source = CONDITIONAL_SOURCE
    if request.param == "descriptor":
        source = source.replace("if(change)then", "if(allocated(left))then")
    elif request.param == "three_arms":
        source = source.replace("call relocate(left,right)\nendif",
            "call relocate(left,right)\nelse if(stage>1)then\ncall relocate(right,left)\n"
            "else\ncall relocate(left,right)\nendif")
    _path, _outputs, manifest = emit(tmp_path_factory.mktemp("control-local-reader"), source)
    return manifest


def nested(manifest):
    return parts(manifest)[3]["control_local_regions"]


def seal(manifest, *, link=True):
    _scope, plan, epoch, selected = parts(manifest)
    record = selected["control_local_regions"]
    record["proof_identity"] = _digest({key: value for key, value in record.items() if key != "proof_identity"})
    if link:
        selected["descriptor_identity"] = record["proof_identity"]
    # Keep ordinary plan/epoch retirement linkage valid so the focused reader
    # must detect changes to the original control-local cut itself.
    epoch["members"][0]["retirement_cuts"] = deepcopy(plan["procedures"][0]["boundaries"])
    reseal(manifest)


def test_actual_source_issued_partition_keeps_original_calls_and_no_new_authority(issued):
    validate_checked_execution_plans(issued)
    _scope, plan, epoch, selected = parts(issued)
    record = nested(issued)
    assert plan["schema_version"] == 2
    assert epoch["version"] == 1 and selected["version"] == 2
    assert [call["procedure"] for call in selected["calls"]] == ["epochs::producer", "epochs::consumer"]
    assert record["proof_identity"] == selected["descriptor_identity"]
    assert record["retirement_subtrees"]
    assert all(record[key] is False for key in ("entry_definition_authority", "payload_access_authorized",
                                               "execution_authorized", "cross_cut_retention_authorized"))
    # These identities are compiler-owned proof references. The public reader
    # checks schema and linkage, not original-source or no-escape derivations.
    assert all(len(cut["storage_projection_identity"]) == 64 for cut in record["retirement_subtrees"])


@pytest.mark.parametrize("fault", [
    "extra", "version_bool", "kind", "owner", "graph", "selection", "parent", "descriptor_link",
    "definitions", "payload", "execution", "retention_bool", "root", "module_root", "root_order",
    "roots_nonlist", "root_nontext", "outer_version_bool",
    "missing_state", "state_extra", "state_kind", "state_binding", "state_authority", "state_guard",
    "closure", "missing_call_closure", "missing_regions", "missing_cuts", "missing_one_cut", "duplicate_cut",
    "foreign_region", "region_hash", "duplicate_region", "overlap_region", "call_under_cut",
    "call_site_under_cut", "foreign_cut", "unknown_cut", "cut_hash", "cut_source_hash",
    "cut_bool_line", "cut_bool_last_line", "cut_reopening", "cut_span", "missing_boundary", "duplicate_boundary",
    "conditional_boundary", "closed_boundary", "boundary_execution", "missing_tail",
])
def test_resealed_partition_cannot_change_owner_storage_roles_or_retirement(issued, fault):
    data = deepcopy(issued)
    _scope, plan, _epoch, selected = parts(data)
    record = nested(data)
    region, cut, state = record["regions"][0], record["retirement_subtrees"][0], record["resources"][0]
    boundary = next(row for row in plan["procedures"][0]["boundaries"]
                    if row["reason"] == "checked control-local allocation retirement")
    replacements = {
        "extra": ("unexpected", False), "version_bool": ("version", True), "kind": ("kind", "unissued"),
        "owner": ("procedure", "foreign"), "graph": ("structured_identity", "0" * 64),
        "selection": ("selected_node_ids", []), "parent": ("parent_sequence", "foreign/body"),
        "definitions": ("entry_definition_authority", True), "payload": ("payload_access_authorized", True),
        "execution": ("execution_authorized", True), "retention_bool": ("cross_cut_retention_authorized", 0),
    }
    if fault in replacements:
        key, value = replacements[fault]
        record[key] = value
    elif fault == "root":
        record["roots"][0] = "foreign::root"
    elif fault == "module_root":
        record["module_roots"].append(record["roots"].pop())
    elif fault == "root_order":
        record["roots"].reverse()
    elif fault == "roots_nonlist":
        record["roots"] = None
    elif fault == "root_nontext":
        record["roots"][0] = []
    elif fault == "outer_version_bool":
        selected["version"] = True
    elif fault == "missing_state":
        record["resources"].pop()
    elif fault == "state_extra":
        state["execution_authorized"] = True
    elif fault == "state_kind":
        state["resource_kind"] = []
    elif fault == "state_binding":
        state["original_binding"]["rank"] = 4
    elif fault == "state_authority":
        state["entry_definition_authority"] = True
    elif fault == "state_guard":
        state["execution_requires_runtime_guard"] = 1
    elif fault == "closure":
        record["source_call_closure"][selected["calls"][0]["procedure"]] = "not-a-sha"
    elif fault == "missing_call_closure":
        record["source_call_closure"].pop(selected["calls"][0]["procedure"])
    elif fault == "missing_regions":
        record["regions"].clear()
    elif fault == "missing_cuts":
        record["retirement_subtrees"].clear()
    elif fault == "missing_one_cut":
        record["retirement_subtrees"].pop()
    elif fault == "duplicate_cut":
        record["retirement_subtrees"].append(deepcopy(cut))
    elif fault == "foreign_region":
        region["node_id"] = "foreign/body/1"
    elif fault == "region_hash":
        region["descriptor_identity"] = "not-a-sha"
    elif fault == "duplicate_region":
        record["regions"].append(deepcopy(region))
    elif fault == "overlap_region":
        record["regions"].append({**region, "node_id": region["node_id"] + "/body/0"})
    elif fault == "call_under_cut":
        region["node_id"] = cut["node_id"]
    elif fault == "call_site_under_cut":
        selected["calls"][0]["node_id"] = cut["node_id"]
    elif fault == "foreign_cut":
        cut["node_id"] = selected["cut_before_node_id"]
    elif fault == "unknown_cut":
        cut["node_id"] += "/invented/0"
    elif fault == "cut_hash":
        cut["storage_projection_identity"] = "not-a-sha"
    elif fault == "cut_source_hash":
        cut["original_source_sha256"] = "not-a-sha"
    elif fault == "cut_bool_line":
        cut["first_line"] = True
    elif fault == "cut_bool_last_line":
        cut["last_line"] = True
    elif fault == "cut_reopening":
        cut["reopening"] = "retain prior generation"
    elif fault == "cut_span":
        cut["last_line"] += 1
    elif fault == "missing_boundary":
        plan["procedures"][0]["boundaries"].remove(boundary)
    elif fault == "duplicate_boundary":
        plan["procedures"][0]["boundaries"].append(deepcopy(boundary))
    elif fault == "conditional_boundary":
        boundary["conditional_close"] = "change"
    elif fault == "closed_boundary":
        boundary["reopen"] = False
    elif fault == "boundary_execution":
        boundary["execution"] = "skip original continuation"
    elif fault == "missing_tail":
        plan["procedures"][0]["boundaries"] = [row for row in plan["procedures"][0]["boundaries"]
            if row["reason"] != "checked selected-position allocation epoch retirement"]
    seal(data, link=fault != "descriptor_link")
    if fault == "descriptor_link":
        selected["descriptor_identity"] = "0" * 64
        reseal(data)
    with pytest.raises(ValueError, match="selected position epoch"):
        validate_checked_execution_plans(data)
