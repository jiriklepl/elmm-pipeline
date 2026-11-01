"""A real emitted source path keeps its original gate, sites and storage links."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)

from compiler.tests.test_supported_path_scope import (
    clock_source,
    emit_supported_path_fixture,
)


def parts(report):
    scope, = report["scopes"]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    return scope, plan, epoch, epoch["supported_path_epoch"]


def seal(report, *, duplicate=True):
    scope, plan, epoch, path = parts(report)
    path["identity"] = _digest({key: value for key, value in path.items() if key != "identity"})
    if duplicate:
        scope["supported_path_epoch"] = deepcopy(path)
    epoch["identity"] = _digest({key: value for key, value in epoch.items() if key != "identity"})
    plan["execution_identity"] = _digest({"execution_identity_version": 2,
        **{key: plan[key] for key in ("entry", "sources", "source_inputs_identity", "configuration_identity",
            "capture_facts_identity", "resources", "numerical_invocations", "gpu_leaves")},
        "procedures": [{"procedure": row["procedure"], "emission_identity": row["emission_identity"]}
                       for row in plan["procedures"]], "storage_epoch_identity": epoch["identity"]})
    plan["identity"] = _digest({key: value for key, value in plan.items() if key != "identity"})
    applicability = scope["interval_applicability"]
    applicability.update(plan_identity=plan["identity"], execution_identity=plan["execution_identity"])
    applicability["identity"] = _digest({key: value for key, value in applicability.items() if key != "identity"})


@pytest.fixture(scope="module")
def plain_original(tmp_path_factory):
    _path, builder, _outputs, report = emit_supported_path_fixture(tmp_path_factory.mktemp("supported-path-reader"))
    return report, deepcopy(builder.facts)


@pytest.fixture(scope="module")
def clock_original(tmp_path_factory):
    _path, builder, _outputs, report = emit_supported_path_fixture(
        tmp_path_factory.mktemp("supported-path-clock-reader"), clock_source())
    return report, deepcopy(builder.facts)


@pytest.fixture(scope="module", params=["plain_original", "clock_original"])
def original(request):
    return request.getfixturevalue(request.param)


def test_real_emitted_plan5_preserves_source_gate_and_checked_call_links(original):
    report, facts = original
    validate_checked_execution_plans(report, capture_facts=facts)
    scope, plan, _epoch, path = parts(report)
    assert plan["schema_version"] == 5
    assert path == scope["supported_path_epoch"]
    assert path["eligibility"]["initial"] is False
    assert len(path["calls"]) == 4
    assert path["entry_definition_authority"] is False
    assert all(row["control_version"] == 6 for row in scope["module_coordinators"])


@pytest.mark.parametrize("fault", [
    "version_bool", "version_unknown", "extra", "missing", "kind", "procedure", "graph", "selector",
    "identity", "definitions", "payload", "aliases", "execution", "retention", "activation", "ordering",
    "presence", "edge_policy", "selection_omitted", "selection_reversed", "selection_foreign",
    "resources_empty", "resources_duplicate", "guard_overlap", "guard_invalid", "call_missing", "call_order",
    "call_alias", "call_span", "call_node", "call_barred", "exit_early", "exit_last", "exit_cleanup",
    "control_initial", "control_rearm", "control_future", "control_order", "control_child", "control_descriptor",
    "control_attrs", "control_resource", "control_kind", "control_kind8", "control_arm", "clear_after", "clear_hash", "exposure_resource", "exposure_hash",
    "barred_missing", "retained_missing", "retained_barred", "outside_missing", "outside_foreign", "scope_copy",
    "coordinator_version", "coordinator_veto", "coordinator_site", "child_boundary", "root_boundary",
    "site_span", "site_mapping", "handle_bool", "handle_wrong", "native_only_child", "graph_reordering",
    "request_source", "request_first", "request_barred", "request_resources",
])
def test_resealed_path_cannot_detach_source_lifetime_or_checked_control(original, fault):
    report, facts = deepcopy(original)
    scope, plan, epoch, path = parts(report)
    gate = path["eligibility"]
    call = path["calls"][0]
    child = next(row for row in plan["procedures"] if row["procedure"] == call["procedure"])
    coordinator = next(row for row in scope["module_coordinators"] if row["procedure"] == call["procedure"])
    simple = {
        "version_bool": ("version", True), "version_unknown": ("version", 3), "kind": ("kind", "guessed_path"),
        "procedure": ("procedure", "foreign::run"), "graph": ("source_graph", "a"*64),
        "selector": ("selection_identity", "a"*64), "identity": ("descriptor_identity", "invalid"),
        "definitions": ("entry_definition_authority", True), "payload": ("payload_access_authorized", True),
        "aliases": ("alias_authority", True), "execution": ("execution_authorized", True),
        "retention": ("cross_frontier_retention_authorized", 0), "activation": ("activation", "eager"),
        "ordering": ("ordering", "move source work"), "presence": ("allocation_presence", "eager"),
        "edge_policy": ("additional_unproved_edge", "ignore"), "resources_empty": ("resources", []),
        "guard_invalid": ("descriptor_guard_resources", [False]), "outside_missing": ("outside_interval_edges", []),
    }
    if fault in simple:
        key, replacement = simple[fault]
        path[key] = replacement
    elif fault == "extra":
        path["inferred_entry_definitions"] = True
    elif fault == "missing":
        path.pop("gate_identity")
    elif fault.startswith("selection_"):
        if fault == "selection_omitted":
            path["selected_node_ids"].pop(1)
        elif fault == "selection_reversed":
            path["selected_node_ids"].reverse()
        else:
            path["selected_node_ids"][1] = plan["entry"]+"/body/100"
    elif fault == "resources_duplicate":
        path["resources"].append(path["resources"][0])
    elif fault == "guard_overlap":
        path["descriptor_guard_resources"] = path["resources"][:1]
    elif fault == "call_missing":
        path["calls"].pop(0)
    elif fault == "call_order":
        path["calls"].reverse()
    elif fault == "call_alias":
        mappings = call["canonical_resources"]
        mappings[next(iter(mappings))] = list(mappings.values())[-1]
    elif fault == "call_span":
        call["source_span"][0] -= 1
    elif fault == "call_node":
        call["node_id"] += "/foreign"
    elif fault == "call_barred":
        call["barred"] = True
    elif fault.startswith("exit_"):
        key, replacement = {"exit_early": ("after_line", 1), "exit_last": ("last_call_node_id", call["node_id"]),
                            "exit_cleanup": ("cleanup", "retain until program exit")}[fault]
        path["sequence_exit"][key] = replacement
    elif fault.startswith("control_"):
        if fault == "control_initial":
            gate["initial"] = True
        elif fault == "control_rearm":
            gate["rearm_after_excluded_edge"] = True
        elif fault == "control_future":
            gate["future_activation"] = "only returned contexts are vetoed"
        elif fault == "control_order":
            gate["controls"].reverse()
        elif fault == "control_child":
            gate["controls"][1]["argument"] = "fresh_true_flag"
        elif fault == "control_descriptor":
            gate["descriptor"]["rank"] = False
        elif fault == "control_attrs":
            gate["descriptor"]["attributes"] = ["pointer", "save"]
        elif fault == "control_resource":
            gate["descriptor"]["resource"] = "foreign::eligible"
        elif fault == "control_kind":
            gate["descriptor"]["kind"] = True
        elif fault == "control_kind8":
            gate["descriptor"]["kind"] = 8
        else:
            gate["arm_source_span"] = path["sequence_exit"]["source_span"][:]
    elif fault == "clear_after":
        gate["barred_calls"][0]["clear_source_span"] = gate["barred_calls"][0]["source_span"][:]
    elif fault == "clear_hash":
        gate["barred_calls"][0]["clear_source_sha256"] = "invalid"
    elif fault == "exposure_resource":
        gate["barred_calls"][0]["exposure_resources"] = path["resources"][:1]
    elif fault == "exposure_hash":
        gate["barred_calls"][0]["exposure_descriptor_identity"] = "a"*64
    elif fault == "barred_missing":
        gate["barred_calls"] = []
    elif fault == "retained_missing":
        path["retained_edges"].pop(0)
    elif fault == "retained_barred":
        path["retained_edges"][0]["node_id"] = gate["barred_calls"][0]["node_id"]
    elif fault == "outside_foreign":
        path["outside_interval_edges"][0]["node_id"] = plan["entry"]+"/body/999"
    elif fault == "scope_copy":
        scope["supported_path_epoch"]["execution_authorized"] = True
    elif fault == "coordinator_version":
        coordinator["control_version"] = 5
    elif fault == "coordinator_veto":
        coordinator["eligibility_control"]["argument"] = "replacement_flag"
    elif fault == "coordinator_site":
        coordinator["calls"] = []
    elif fault in {"child_boundary", "root_boundary"}:
        member = child if fault == "child_boundary" else plan["procedures"][0]
        member["boundaries"].append({"reason": "unproved edge"})
        next(row for row in epoch["members"] if row["procedure"] == member["procedure"])["retirement_cuts"] = deepcopy(member["boundaries"])
        if fault == "child_boundary":
            coordinator["boundaries"] = deepcopy(member["boundaries"])
        else:
            scope["boundaries"] = deepcopy(member["boundaries"])
    elif fault in {"site_span", "site_mapping", "handle_bool", "handle_wrong"}:
        site = child["calls"][0]
        if fault == "site_span":
            site["first_line"] -= 1
        elif fault == "site_mapping":
            site["resource_mappings"] = {}
        else:
            slot = next(iter(site["handle_bindings"]))
            site["handle_bindings"][slot] = True if fault == "handle_bool" else 1000
        coordinator["calls"] = deepcopy(child["calls"])
    elif fault == "native_only_child":
        call["procedure"] = plan["entry"]
    elif fault == "graph_reordering":
        scope["graph_execution"]["source_motion"] = [{"move": "unchecked"}]
    elif fault.startswith("request_"):
        selector = report["native_ancestor_selection"]
        request = selector["path_request"]
        if fault == "request_source":
            request["source"]["sha256"] = "a"*64
        elif fault == "request_first":
            request["first"]["callee"] = "foreign::callee"
        elif fault == "request_barred":
            request["eligibility"]["barred_calls"][0]["sha256"] = "a"*64
        else:
            request["eligibility"]["resources"] = ["unknown::allocation"]
        selector["identity"] = _digest({key: value for key, value in selector.items() if key != "identity"})
        path["selection_identity"] = selector["identity"]
    else:
        raise AssertionError(fault)
    seal(report, duplicate=fault != "scope_copy")
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report, capture_facts=facts)


@pytest.mark.parametrize("version,metadata", [(1, []), (2, []), (2, None)])
def test_gate_metadata_extension_requires_its_explicit_nonempty_version(original, version, metadata):
    report, facts = deepcopy(original)
    _scope, _plan, _epoch, path = parts(report)
    path.update(version=version, gate_metadata_edges=metadata)
    seal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report, capture_facts=facts)


@pytest.mark.parametrize("fault", ["old_version", "missing", "empty", "extra", "duplicate", "foreign",
    "selected_node", "outside_node", "nonexistent_node", "after_arm", "invalid_span", "source_hash",
    "clock_hash", "descriptor_hash", "identity_bool"])
def test_resealed_clock_metadata_requires_its_distinct_original_guard_receipts(clock_original, fault):
    report, facts = deepcopy(clock_original)
    _scope, _plan, _epoch, path = parts(report)
    row, = path["gate_metadata_edges"]
    if fault == "old_version":
        path["version"] = 1
    elif fault == "missing":
        path.pop("gate_metadata_edges")
    elif fault == "empty":
        path["gate_metadata_edges"] = []
    elif fault == "extra":
        row["definitions_authorized"] = True
    elif fault == "duplicate":
        path["gate_metadata_edges"].append(deepcopy(row))
    elif fault == "foreign":
        row["node_id"] = "foreign::run/body/0/body0/3"
    elif fault == "selected_node":
        row["node_id"] = path["selected_node_ids"][0]
    elif fault == "outside_node":
        row["node_id"] = path["outside_interval_edges"][0]["node_id"]
    elif fault == "nonexistent_node":
        row["node_id"] = row["node_id"].rsplit("/", 1)[0]+"/999"
    elif fault == "after_arm":
        row["source_span"] = path["eligibility"]["arm_source_span"][:]
    elif fault == "invalid_span":
        row["source_span"] = [True, True]
    elif fault in {"source_hash", "clock_hash", "descriptor_hash"}:
        key = {"source_hash": "original_source_sha256", "clock_hash": "clock_identity",
               "descriptor_hash": "descriptor_identity"}[fault]
        row[key] = "invalid"
    else:
        row["clock_identity"] = True
    seal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report, capture_facts=facts)
