"""Real issued local groups keep strict boundaries and per-child forwarding."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)

from compiler.tests.test_owner_local_call_groups import SOURCE, fixture


def issue(path, *, anchor="endif", lifetime="selected_window", unused=False):
    source = SOURCE
    if anchor in {"else", "elseif"}:
        header = "else" if anchor == "else" else "else if(.not.flag)then"
        source = source.replace("endif\ncall unknown_suffix()", header+"\ncall unknown_other()\nendif\ncall unknown_suffix()")
    elif anchor in {"last_else", "middle_elseif"}:
        header = "else" if anchor == "last_else" else "else if(flag)then"
        source = source.replace("if(flag)then\ncall first", "if(.not.flag)then\ncall unknown_other()\n"+header+"\ncall first")
        if anchor == "middle_elseif":
            source = source.replace("endif\ncall unknown_suffix()", "else\ncall unknown_last()\nendif\ncall unknown_suffix()")
    if unused:
        source = source.replace("subroutine run(a,b,n,flag)", "subroutine run(a,b,unused,n,flag)")
        source = source.replace("::a(:),b(:)", "::a(:),b(:),unused(:)")
    _path, builder = fixture(path, source, lifetime=lifetime)
    if unused:
        builder.facts["captures"]["argument::unused"] = dict(builder.facts["captures"]["argument::a"])
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report["boundaries"]
    return report


@pytest.fixture(scope="module", params=[(anchor, lifetime) for anchor in ("endif", "else", "elseif", "last_else", "middle_elseif")
                                       for lifetime in ("selected_window", "invocation")])
def original(request, tmp_path_factory):
    anchor, lifetime = request.param
    return issue(tmp_path_factory.mktemp("local-group-reader"), anchor=anchor, lifetime=lifetime)


def parts(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    return scope, plan, epoch, epoch["selected_position_epoch"]


def seal(report, *, preserve_scope_copy=False):
    scope, plan, epoch, group = parts(report)
    if isinstance(group, dict):
        if isinstance(group.get("sequence_exit"), dict):
            exit_record = group["sequence_exit"]
            exit_record["identity"] = _digest({k: v for k, v in exit_record.items() if k != "identity"})
        group["identity"] = _digest({k: v for k, v in group.items() if k != "identity"})
        if not preserve_scope_copy:
            scope["owner_local_call_group"] = deepcopy(group)
    epoch["identity"] = _digest({k: v for k, v in epoch.items() if k != "identity"})
    plan["execution_identity"] = _digest({"execution_identity_version": 2,
        **{key: plan[key] for key in ("entry", "sources", "source_inputs_identity", "configuration_identity",
            "capture_facts_identity", "resources", "numerical_invocations", "gpu_leaves")},
        "procedures": [{"procedure": p["procedure"], "emission_identity": p["emission_identity"]}
                       for p in plan["procedures"]], "storage_epoch_identity": epoch["identity"]})
    plan["identity"] = _digest({k: v for k, v in plan.items() if k != "identity"})
    if "interval_applicability" in scope:
        applicability = scope["interval_applicability"]
        applicability.update(plan_identity=plan["identity"], execution_identity=plan["execution_identity"])
        applicability["identity"] = _digest({k: v for k, v in applicability.items() if k != "identity"})


def test_real_compiler_local_plan4_keeps_exact_span_exit_and_no_definition_grant(original):
    validate_checked_execution_plans(original)
    scope, plan, epoch, group = parts(original)
    assert plan["schema_version"] == 4
    assert "return_contract" not in epoch
    assert group == scope["owner_local_call_group"]
    assert group["sequence_exit"]["cross_exit_retention_authorized"] is False
    assert group["entry_definition_authority"] is False
    assert [call["procedure"].rsplit("::", 1)[1] for call in group["calls"]] == ["first", "second", "third"]


def test_unused_capture_is_descriptor_only_without_new_managed_registration(tmp_path):
    report = issue(tmp_path, unused=True)
    validate_checked_execution_plans(report)
    _scope, _plan, epoch, group = parts(report)
    assert group["descriptor_guard_resources"] == ["argument::unused"]
    assert "argument::unused" not in group["resources"]
    assert "argument::unused" not in {record["resource"] for record in epoch["resources"]}


@pytest.mark.parametrize("fault", [
    "old_plan", "missing_epoch", "returned_contract", "group_version_bool", "group_version_old", "group_extra",
    "descriptor", "selector_identity", "definition", "execution", "retention_bool", "activation", "presence",
    "scope_copy", "scope_missing", "scope_kind", "policy", "selector_version", "selector_lifetime",
    "selector_authority", "selector_source", "selector_duplicate", "selector_omitted_child", "managed_missing", "managed_guarded",
    "managed_unknown", "managed_duplicate", "managed_lifetime", "guard_overlap", "guard_type",
    "exit_version", "exit_owner", "exit_parent", "exit_last", "exit_source", "exit_definition",
    "exit_anchor", "exit_early", "exit_call_order", "selection", "graph_missing", "graph_call_missing", "graph_call_transparent",
    "call_missing", "call_duplicate", "call_order", "call_callee", "call_span", "call_mapping", "call_alias",
    "site_span", "site_mapping", "handle_bool", "handle_wrong", "coordinator_version", "coordinator_missing",
])
def test_resealed_local_group_cannot_change_original_edges_or_borrowed_storage(original, fault):
    report = deepcopy(original)
    scope, plan, epoch, group = parts(report)
    first, second = group["calls"][:2]
    child = next(p for p in plan["procedures"] if p["procedure"] == first["procedure"])
    site = next(c for c in child["calls"] if c.get("caller") == plan["entry"])
    exit_record = group["sequence_exit"]
    selector = report["native_ancestor_selection"]
    simple = {
        "group_version_bool": ("version", True), "group_version_old": ("version", 1),
        "descriptor": ("descriptor_identity", "invalid"), "selector_identity": ("selection_identity", "e"*64),
        "definition": ("entry_definition_authority", True), "execution": ("execution_authorized", True),
        "retention_bool": ("cross_frontier_retention_authorized", 0),
        "activation": ("repeated_activation", "reuse earlier initialized handles"),
        "presence": ("allocation_presence", "eager at caller entry"), "managed_unknown": ("resources", ["unknown"]),
        "guard_type": ("descriptor_guard_resources", [False]),
    }
    if fault in simple:
        key, value = simple[fault]
        group[key] = value
    elif fault == "old_plan":
        plan["schema_version"] = 2
    elif fault == "missing_epoch":
        # Reseal the containing plan after removing the mandatory new field.
        epoch.pop("selected_position_epoch")
        group = None
    elif fault == "returned_contract":
        epoch["return_contract"] = {}
    elif fault == "group_extra":
        group["cut_before_node_id"] = "forged-sibling"
    elif fault == "scope_copy":
        scope["owner_local_call_group"]["execution_authorized"] = True
    elif fault == "scope_missing":
        scope.pop("owner_local_call_group")
    elif fault == "scope_kind":
        scope["graph_root_kind"] = "original native ancestor"
    elif fault == "policy":
        plan["policy"]["lifetime"] = "retain across unmanaged caller"
    elif fault.startswith("selector_"):
        if fault == "selector_version":
            selector["version"] = 1
        elif fault == "selector_lifetime":
            selector["lifetime"] = "returned_domains"
        elif fault == "selector_authority":
            selector["alias_authority"] = True
        elif fault == "selector_source":
            selector["device_roots"][0]["source_sha256"] = "0"*64
        elif fault == "selector_duplicate":
            selector["device_roots"].append(deepcopy(selector["device_roots"][0]))
        else:
            selector["device_roots"] = [r for r in selector["device_roots"] if r["procedure"] != first["procedure"]]
        selector["identity"] = _digest({k: v for k, v in selector.items() if k != "identity"})
        group["selection_identity"] = selector["identity"]
    elif fault == "managed_missing":
        group["resources"] = []
    elif fault == "managed_guarded":
        group["descriptor_guard_resources"].append(group["resources"].pop())
    elif fault == "managed_duplicate":
        group["resources"].append(group["resources"][0])
    elif fault == "managed_lifetime":
        epoch["resources"][0]["lifetime"] = "invocation"
    elif fault == "guard_overlap":
        group["descriptor_guard_resources"].append(group["resources"][0])
    elif fault.startswith("exit_"):
        if fault == "exit_version":
            exit_record["version"] = True
        elif fault == "exit_owner":
            exit_record["procedure"] = "foreign::owner"
        elif fault == "exit_parent":
            exit_record["parent_sequence"] += "/foreign"
        elif fault == "exit_last":
            exit_record["last_call_node_id"] = exit_record["selected_node_ids"][0]
        elif fault == "exit_source":
            exit_record["source_graph"] = "a"*64
        elif fault == "exit_definition":
            exit_record["entry_definition_authority"] = True
        elif fault == "exit_anchor":
            exit_record["exit_anchor"]["parser_kind"] = "End_Do_Stmt"
        elif fault == "exit_early":
            exit_record["exit_anchor"]["source_span"][0] = exit_record["source_span"][-1]
        else:
            exit_record["selected_node_ids"].reverse()
    elif fault == "selection":
        group["selected_node_ids"] = group["selected_node_ids"][:-1]
    elif fault == "graph_missing":
        scope.pop("graph_execution")
    elif fault in {"graph_call_missing", "graph_call_transparent"}:
        from elmm_pipeline.source_placement_record import _tasks

        graph = next(r for r in scope["graph_execution"]["procedures"] if r["procedure"] == plan["entry"])
        task = next(t for t in _tasks(graph["task_graph"]) if t["id"] == first["node_id"])
        if fault == "graph_call_missing":
            task["id"] = "foreign"
        else:
            task["opaque"] = False
    elif fault == "call_missing":
        group["calls"].pop()
    elif fault == "call_duplicate":
        group["calls"].append(deepcopy(first))
    elif fault == "call_order":
        group["calls"].reverse()
    elif fault == "call_callee":
        first["procedure"] = "foreign::callee"
    elif fault == "call_span":
        second["source_span"][0] = first["source_span"][1]
    elif fault == "call_mapping":
        first["canonical_resources"]["argument::b"] = "unknown"
    elif fault == "call_alias":
        second["canonical_resources"]["argument::a"] = second["canonical_resources"]["argument::b"]
    elif fault == "site_span":
        site["first_line"] += 1
    elif fault == "site_mapping":
        site["resource_mappings"] = {}
    elif fault == "handle_bool":
        site["handle_bindings"][next(iter(site["handle_bindings"]))] = True
    elif fault == "handle_wrong":
        key = next(iter(site["handle_bindings"]))
        site["handle_bindings"][key] = next(r["registration_identity"] for r in epoch["resources"]
                                                  if r["registration_identity"] != site["handle_bindings"][key])
    elif fault == "coordinator_version":
        scope["module_coordinators"][0]["control_version"] = 5
    elif fault == "coordinator_missing":
        scope["module_coordinators"].pop()
    if fault == "missing_epoch":
        epoch["selected_position_epoch"] = None
        seal(report, preserve_scope_copy=True)
        del epoch["selected_position_epoch"]
        epoch["identity"] = _digest({k: v for k, v in epoch.items() if k != "identity"})
        # The incomplete v4 schema rejects before execution identity validation.
        plan["identity"] = _digest({k: v for k, v in plan.items() if k != "identity"})
    else:
        seal(report, preserve_scope_copy=fault in {"scope_copy", "scope_missing"})
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)
