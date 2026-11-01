"""Hierarchical root spans retain existing child routes and exact emitted retirement."""
from copy import deepcopy
from hashlib import sha256

import pytest

from compiler.tests.test_hierarchical_child_regions import fixture, SOURCE, READ_FIRST_SOURCE
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from test_conditional_return_record import seal
from elmm_pipeline.conditional_return_record import _slot


@pytest.fixture(scope="module", params=[SOURCE, READ_FIRST_SOURCE], ids=["overwrite", "read-first"])
def authentic(request, tmp_path_factory):
    _path, builder = fixture(tmp_path_factory.mktemp("hierarchical-reader"), request.param)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    return report


def parts(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    contract = plan["storage_epoch"]["return_contract"]
    return scope, plan, contract, contract["root_selection"]


def reseal(report):
    _scope, _plan, contract, selection = parts(report)
    selection["identity"] = _digest({k: v for k, v in selection.items() if k != "identity"})
    contract["selection_identity"] = selection["identity"]
    seal(report)


def test_authentic_hierarchical_regions_round_trip_without_mutating_records(authentic):
    before = deepcopy(authentic)
    validate_checked_execution_plans(authentic)
    assert authentic == before
    scope, plan, contract, selection = parts(authentic)
    assert plan["schema_version"] == 3 and contract["version"] == 2
    assert selection["source_span"] == [scope["first_line"], scope["last_line"]]
    assert scope["selected_end_retirement"]["context"] == scope["owner"]


@pytest.mark.parametrize("fault", [
    "contract_version", "selection_version", "selection_kind", "native_selector", "parent", "cut_gap", "cut_inside",
    "selected_reverse", "selected_gap", "selected_repeat", "span", "span_bool", "source_graph", "source_hash",
    "activation", "cleanup", "outside", "definition", "descriptor", "execution", "cross_exit", "alias",
    "root_route_cut", "root_route_prefix_collision", "root_route_ancestor", "root_return", "missing_child",
    "unproved_retirement", "child_return_storage", "forwarding", "end_missing", "end_version", "end_context",
    "end_span", "end_source", "end_hash", "edit_missing", "edit_duplicate", "edit_changed", "edit_source_hash",
    "implicit_canonical", "implicit_foreign", "implicit_argument", "implicit_descriptor", "implicit_rank",
    "implicit_parent_slot", "missing_formal", "unknown_root_key", "unknown_end_key", "end_context_both", "edit_no_close_resealed", "edit_reset_before_close",
])
def test_resealed_v2_cannot_change_selection_child_lifetime_or_cleanup(authentic, fault):
    report = deepcopy(authentic)
    validate_checked_execution_plans(report)  # Every mutation begins with accepted genuine source output.
    scope, plan, contract, selection = parts(report)
    root = next(m for m in contract["members"] if m["procedure"] == plan["entry"])
    middle = next(m for m in contract["members"] if m["procedure"].endswith("::middle"))
    end = scope["selected_end_retirement"]
    edit = next(e for e in report["source_edits"] if [e["first_line"], e["last_line"]] == end["insertion_span"])
    if fault == "contract_version": contract["version"] = 1
    elif fault == "selection_version": selection["version"] = True
    elif fault == "selection_kind": selection["kind"] = "original_native_ancestor_selection"
    elif fault == "native_selector": report["native_ancestor_selection"] = {}
    elif fault == "parent": selection["parent_sequence"] += "/other"
    elif fault == "cut_gap": selection["cut_before"] = selection["parent_sequence"] + "/9"
    elif fault == "cut_inside": selection["cut_before"] = selection["selected_node_ids"][0]
    elif fault == "selected_reverse": selection["selected_node_ids"].reverse()
    elif fault == "selected_gap": selection["selected_node_ids"].pop(1)
    elif fault == "selected_repeat": selection["selected_node_ids"].append(selection["selected_node_ids"][-1])
    elif fault == "span": selection["source_span"][1] += 1
    elif fault == "span_bool": selection["source_span"][0] = True
    elif fault == "source_graph": selection["source_graph"] = "f" * 64
    elif fault == "source_hash": selection["original_source_sha256"] = "not a hash"
    elif fault == "activation": selection["activation"] = "reuse arbitrary earlier domain"
    elif fault == "cleanup": selection["cleanup"] = "retain after selected span"
    elif fault == "outside": selection["outside_selection"] = "retain handles in unchecked suffix"
    elif fault in {"definition", "descriptor", "execution", "cross_exit", "alias"}:
        key = {"definition":"entry_definition_authority", "descriptor":"descriptor_stability_authorized",
               "execution":"execution_authorized", "cross_exit":"cross_exit_retention_authorized", "alias":"alias_authority"}[fault]
        selection[key] = True
    elif fault.startswith("root_route_"):
        node = selection["cut_before"] if fault == "root_route_cut" else selection["selected_node_ids"][0] + "0" if fault == "root_route_prefix_collision" else selection["parent_sequence"]
        root["routes"].append({"kind":"checked_condition", "selected_node_ids":[node], "summary_identity":"e"*64})
    elif fault == "root_return": root["routes"].append({"kind":"checked_return", "selected_node_ids":[selection["selected_node_ids"][0]+"/return"]})
    elif fault == "missing_child": root["routes"] = [r for r in root["routes"] if r["kind"] != "checked_child"]
    elif fault == "unproved_retirement":
        route = next(r for r in middle["routes"] if r["kind"] == "reached_retirement")
        route.update(lifetime="unproved_lifetime_or_escape", rearm_allowed=False)
    elif fault == "child_return_storage": middle["entry_exit"]["exit"] = "retain child-local allocation"
    elif fault == "forwarding":
        route = next(r for r in middle["routes"] if r["kind"] == "checked_child")
        route["handle_forwarding"][next(iter(route["handle_forwarding"]))] += 1
    elif fault.startswith("implicit_"):
        route = next(r for r in root["routes"] if r.get("procedure", "").endswith("::middle"))
        name = next(iter(route["canonical_resources"]))
        if fault == "implicit_canonical":
            route["canonical_resources"][name] = next(n for n in route["canonical_resources"] if n != name)
        elif fault in {"implicit_foreign", "implicit_argument"}:
            value = route["canonical_resources"].pop(name)
            replacement = ("foreign::" if fault == "implicit_foreign" else "argument::") + name.split("::")[-1]
            route["canonical_resources"][replacement] = value
        if fault in {"implicit_canonical", "implicit_foreign", "implicit_argument"}:
            numbers = {r["resource"]: r["registration_identity"] for r in plan["resources"]}
            old_forward = dict(route["handle_forwarding"])
            route["canonical_handles"] = {str(_slot(route["procedure"], key)): numbers[value]
                                            for key, value in route["canonical_resources"].items()}
            route["handle_forwarding"] = dict(route["canonical_handles"])
            called = next(p for p in plan["procedures"] if p["procedure"] == route["procedure"])
            call = next(c for c in called["calls"] if c["caller"] == plan["entry"] and c["handle_bindings"] == old_forward)
            call["handle_bindings"] = dict(route["handle_forwarding"])
        elif fault == "implicit_parent_slot":
            key = next(iter(route["handle_forwarding"]))
            route["handle_forwarding"][key] += 1
        else:
            resource = next(r for r in plan["resources"] if r["resource"] == name)
            resource["descriptor"] = deepcopy(resource["descriptor"])
            resource["descriptor"]["resource" if fault == "implicit_descriptor" else "rank"] = "foreign::"+name.split("::")[-1] if fault == "implicit_descriptor" else 2
            epoch_resource = next(r for r in plan["storage_epoch"]["resources"] if r["resource"] == name)
            epoch_resource["descriptor"] = deepcopy(resource["descriptor"])
    elif fault == "missing_formal":
        consumer = next(p for p in plan["procedures"] if p["procedure"].endswith("::consumer"))
        call = next(c for c in consumer["calls"] if c["caller"].endswith("::middle"))
        del call["resource_mappings"][next(iter(call["resource_mappings"]))]
    elif fault == "end_missing": del scope["selected_end_retirement"]
    elif fault == "end_version": end["version"] = True
    elif fault == "end_context": end["context"] = "unbound_context"
    elif fault == "end_context_both": scope["owner"] = end["context"] = "unbound_context"
    elif fault == "edit_no_close_resealed":
        edit["replacement"] = edit["replacement"].replace("fort_close_status = fort_scope_resident_close_v1(", "fort_close_status = ignored_close(")
        end["replacement_sha256"] = sha256(edit["replacement"].encode()).hexdigest()
    elif fault == "edit_reset_before_close":
        lines = edit["replacement"].splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("fort_close_status = fort_scope_resident_close_v1("))
        second = lines.index(end["context"] + " = 0")
        lines[first], lines[second] = lines[second], lines[first]
        edit["replacement"] = "\n".join(lines) + "\n"
        end["replacement_sha256"] = sha256(edit["replacement"].encode()).hexdigest()
    elif fault == "end_span": end["insertion_span"][0] += 1
    elif fault == "end_source": end["source"] = "foreign.f90"
    elif fault == "end_hash": end["replacement_sha256"] = "e" * 64
    elif fault == "edit_missing": report["source_edits"].remove(edit)
    elif fault == "edit_duplicate": report["source_edits"].append(deepcopy(edit))
    elif fault == "edit_changed": edit["replacement"] += "! changed cleanup\n"
    elif fault == "edit_source_hash": edit["source_sha256"] = "e" * 64
    elif fault == "unknown_root_key": selection["placement_authorized"] = True
    else: end["retention_authorized"] = True
    reseal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)
