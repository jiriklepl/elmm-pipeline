"""Native READ role transport is closed, source linked and canonically disjoint."""
import json
from copy import deepcopy

import pytest
from compiler.tests.test_graph_source_owner import NATIVE_HELPER_SOURCE, emit
from compiler.tests.test_native_host_fixed_arrays import SOURCE
from test_conditional_return_record import seal

from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)


@pytest.fixture(scope="module", params=["fixed", "allocatable"])
def original(tmp_path_factory, request):
    path, outputs, _report = emit(tmp_path_factory.mktemp("opaque-" + request.param),
                                  SOURCE if request.param == "fixed" else NATIVE_HELPER_SOURCE)
    report = json.loads(outputs["scope-manifest.json"])
    assert len(report["scopes"]) == 1
    (path.parent / "typed-report.json").write_text(json.dumps(report, indent=2))
    validate_checked_execution_plans(report)
    return report


def parts(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    row = epoch["native_opaque_accesses"][0]
    procedure = next(item for item in plan["procedures"] if item["procedure"] == row["procedure"])
    unit = procedure["units"][row["unit_ordinal"]]
    operation = unit["operations"]["native_operations"][row["operation_id"]]
    return scope, plan, epoch, row, procedure, unit, operation


def test_real_epoch_v4_preserves_both_original_actuals_without_managed_credit(original):
    report = deepcopy(original)
    validate_checked_execution_plans(report)
    _scope, _plan, epoch, row, _procedure, unit, operation = parts(report)
    assert epoch["version"] == 4
    resource, = row["resources"]
    assert resource["canonical_resources"] == ["graph_fixture::other_types", "graph_fixture::types"]
    assert len(row["call_projection"]["calls"]) == 2
    local = resource["resource"]
    assert local in operation["resource_effects"]
    assert local not in operation["managed_resources"]
    assert local not in unit["resources"]
    assert report == original


@pytest.mark.parametrize("fault", ["mode", "permit_kind", "permit_authority", "entry_credit", "permit_source",
    "permit_selection", "permit_descriptor", "missing_inventory", "duplicate_operation", "operation_bool",
    "legacy_key", "epoch_downgrade", "required_whole", "whole_on_return", "possible_definition_changes",
    "missing_second_call", "missing_site_and_call", "wrong_actual", "wrong_call_graph", "missing_resolved_call", "managed_epoch"])
def test_resealed_native_opaque_roles_cannot_change_source_or_gain_managed_credit(original, fault):
    report = deepcopy(original)
    _scope, _plan, epoch, row, procedure, unit, operation = parts(report)
    value = operation["native_opaque_accesses"]
    access = value["accesses"][0]
    permit = access["permit"]
    local = access["resource"]
    if fault == "mode":
        access["mode"] = "RW"
    elif fault == "permit_kind":
        access["permit_kind"] = "numeric_write"
    elif fault == "permit_authority":
        permit["allocation_changes"] = True
    elif fault == "entry_credit":
        value["initialization_authorized"] = True
    elif fault == "permit_source":
        permit["source_graph"] = "a" * 64
    elif fault == "permit_selection":
        permit["selected_node_ids"] = []
    elif fault == "permit_descriptor":
        permit["binding"]["rank"] = 2
    elif fault == "missing_inventory":
        epoch["native_opaque_accesses"] = []
    elif fault == "duplicate_operation":
        epoch["native_opaque_accesses"].append(deepcopy(row))
    elif fault == "operation_bool":
        operation["operation_id"] = False
    elif fault == "legacy_key":
        operation["host_only_native_reads"] = {"device_capture": False}
    elif fault == "epoch_downgrade":
        epoch["version"] = 1
        epoch.pop("native_opaque_accesses")
    elif fault in {"required_whole", "whole_on_return", "possible_definition_changes"}:
        unit["ordered_definitions"][fault].append(local)
    elif fault == "missing_second_call":
        row["call_projection"]["calls"].pop()
    elif fault == "missing_site_and_call":
        row["call_projection"]["calls"].pop()
        procedure["calls"].pop()
    elif fault == "wrong_actual":
        edge = row["call_projection"]["calls"][1]
        edge["source_call"]["resource_mappings"][1]["resource"] = "graph_fixture::types"
    elif fault == "wrong_call_graph":
        row["call_projection"]["calls"][1]["caller_graph"] = "a" * 64
    elif fault == "missing_resolved_call":
        report["resolved_calls"].remove(row["call_projection"]["calls"][1]["source_call"])
    elif fault == "managed_epoch":
        item = next(item for item in epoch["resources"] if item["resource"] == "graph_fixture::types")
        item.update(device_replica_authorized=True, storage="stable", lifetime="invocation",
                    allocation_changes=False, escapes=False)
        item.pop("initialization_authorized")
    permit["identity"] = _digest({key: item for key, item in permit.items() if key != "identity"})
    value["identity"] = _digest({key: item for key, item in value.items() if key != "identity"})
    row["access_identity"] = value["identity"]
    row["resources"][0]["permit_identity"] = permit["identity"]
    seal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)


@pytest.fixture(scope="module", params=["ordinary", "guarded", "allocatable", "managed_write", "repeated", "integer8"])
def original_write(tmp_path_factory, request):
    from compiler.tests.test_native_opaque_write_calls import call_source
    from compiler.tests.test_native_opaque_write_calls import emit as emit_write

    text = call_source(guarded=request.param != "ordinary")
    if request.param == "allocatable":
        text = text.replace(
            "real(8)::values(-2:3,-2:3,-2:3), mean(-1:1), other(4:6), output(-2:3,-2:3,-2:3)",
            "real(8)::mean(-1:1),other(4:6)\nreal(8),allocatable::values(:,:,:),output(:,:,:)")
    elif request.param == "integer8":
        text = text.replace("real(8)::values(-2:3,-2:3,-2:3), mean(-1:1), other(4:6), output(-2:3,-2:3,-2:3)",
            "real(8)::values(-2:3,-2:3,-2:3), output(-2:3,-2:3,-2:3)\ninteger(8)::mean(-1:1),other(4:6)")
        text = text.replace("real(8),intent(out)::result(3)", "integer(8),intent(out)::result(3)")
    elif request.param == "managed_write":
        text = text.replace("contiguous,intent(in)::values", "contiguous,intent(inout)::values")
        text = text.replace("result=result/max(1,n)", "result=result/max(1,n)\nvalues(-2,-2,-2)=values(-2,-2,-2)+1")
    elif request.param == "repeated":
        call = "if(go) call reduce_mean(mean,values,n,masked)"
        text = text.replace(call, call + "\n" + call)
    _path, _builder, _outputs, report = emit_write(tmp_path_factory.mktemp("opaque-write-" + request.param), text=text)
    validate_checked_execution_plans(report)
    return report


def reseal_write(report):
    _scope, _plan, _epoch, row, _procedure, _unit, operation = parts(report)
    value = operation["native_opaque_accesses"]
    access = value["accesses"][0]
    permit = access["permit"]
    for receipt in access["association_receipts"]:
        prior = receipt["identity"]
        receipt["identity"] = _digest({k: v for k, v in receipt.items() if k != "identity"})
        permit["association_identities"] = [receipt["identity"] if i == prior else i for i in permit["association_identities"]]
        for item in permit["incoming_instances"]:
            if item["association_identity"] == prior:
                item["association_identity"] = receipt["identity"]
    permit["identity"] = _digest({k: v for k, v in permit.items() if k != "identity"})
    value["identity"] = _digest({k: v for k, v in value.items() if k != "identity"})
    row["access_identity"] = value["identity"]
    row["resources"][0]["permit_identity"] = permit["identity"]
    seal(report)


def test_original_native_write_json_preserves_all_hazards_without_managed_credit(original_write):
    report = deepcopy(original_write)
    validate_checked_execution_plans(report)
    _scope, _plan, epoch, _row, _procedure, unit, operation = parts(report)
    access, = operation["native_opaque_accesses"]["accesses"]
    resource = next(r for r in epoch["resources"] if r["resource"] == access["resource"])
    assert resource["device_replica_authorized"] is resource["initialization_authorized"] is False
    assert resource["storage"] == "opaque host reservation"
    assert access["resource"] not in unit["resources"]
    assert access["resource"] not in unit["ordered_definitions"]["possible_definition_changes"]
    assert report == original_write


@pytest.mark.parametrize("fault", ["guard", "action_node", "source_call", "managed_mapping", "complete_mapping",
    "actual_evaluation", "presence", "requirements", "formal_descriptor", "actual_descriptor", "missing_call",
    "missing_association", "association_bounds", "association_flag", "association_source", "association_call",
    "association_extra", "missing_completion", "missing_join", "missing_incoming", "duplicate_incoming", "instance_bool",
    "effect_omission", "target_actions", "effect_capture", "alias_authority", "entry_credit", "definition_credit",
    "device_credit", "output_definedness", "extra", "storage_bytes", "mode", "consumer_graph", "source_inputs",
    "canonical_managed", "managed_definition", "graph_guard", "graph_effect", "graph_definition"])
def test_resealed_write_transport_cannot_replace_original_source_or_roles(original_write, fault):
    report = deepcopy(original_write)
    scope, _plan, epoch, _row, procedure, unit, operation = parts(report)
    access = operation["native_opaque_accesses"]["accesses"][0]
    permit, control = access["permit"], access["execution_control"]
    receipt = access["association_receipts"][0]
    call = control["source_call"]
    if fault == "guard":
        control["original_guard"] = ".false."
    elif fault == "action_node":
        control["selected_node_ids"] = permit["selected_node_ids"] + ["foreign"]
    elif fault == "source_call":
        call["target"] = "other"
    elif fault == "managed_mapping":
        call["resource_mappings"][1]["resource"] = "formal_reduction::output"
    elif fault == "complete_mapping":
        call["resource_mapping"]["argument::values"] = "formal_reduction::output"
    elif fault == "actual_evaluation":
        call["actual_evaluation"] = "eager"
    elif fault == "presence":
        call["resource_mappings"][1]["presence"] = "omitted"
    elif fault == "requirements":
        call["resource_mappings"][1]["requirements"]["physical_section_mapping"] = True
    elif fault == "formal_descriptor":
        call["resource_mappings"][1]["formal_descriptor"]["intent"] = "out"
    elif fault == "actual_descriptor":
        call["resource_mappings"][1]["actual_descriptor"]["resource"] = "formal_reduction::output"
    elif fault == "missing_call":
        report["resolved_calls"].remove(call)
    elif fault == "missing_association":
        access["association_receipts"] = []
    elif fault == "association_bounds":
        receipt["actual_index_minus_formal_index"] += 1
    elif fault == "association_flag":
        receipt["native_write_authority"] = True
    elif fault == "association_source":
        receipt["selected_node_ids"] = ["foreign"]
    elif fault == "association_call":
        receipt["source_call"]["associations"][1]["actual_type"]["storage_resource"] = "formal_reduction::output"
    elif fault == "association_extra":
        receipt["new_permission"] = True
    elif fault == "missing_completion":
        permit["completion_identities"] = []
    elif fault == "missing_join":
        permit["joined_team_identities"] = []
    elif fault == "missing_incoming":
        permit["incoming_instances"] = []
    elif fault == "duplicate_incoming":
        permit["incoming_instances"].append(deepcopy(permit["incoming_instances"][0]))
    elif fault == "instance_bool":
        permit["incoming_instances"][0]["instance"] = True
    elif fault == "effect_omission":
        permit["original_effect_inventory"] = [r for r in permit["original_effect_inventory"]
            if not (r["external"] and r["storage"]["resource"] == "formal_reduction::values")]
    elif fault == "target_actions":
        permit["target_actions"] = ["write"]
    elif fault == "effect_capture":
        permit["original_effect_inventory"][0]["storage"]["numerical_capture_authority"] = True
    elif fault == "alias_authority":
        permit["storage_disjointness_obligations"][0]["discharged"] = True
    elif fault in {"entry_credit", "definition_credit", "device_credit", "output_definedness"}:
        permit[{"entry_credit": "entry_definition_authority", "definition_credit": "managed_definition_authorized",
                "device_credit": "device_replica_authorized", "output_definedness": "whole_output_definedness_authorized"}[fault]] = True
    elif fault == "extra":
        permit["new_authority"] = True
    elif fault == "storage_bytes":
        permit["storage_bytes"] += 1
    elif fault == "mode":
        access["mode"] = "W"
    elif fault == "consumer_graph":
        control["source_graph"] = "b" * 64
    elif fault == "source_inputs":
        permit["source_inputs"]["sources"] = {}
    elif fault == "canonical_managed":
        next(r for r in epoch["resources"] if r["resource"] == access["resource"])["device_replica_authorized"] = True
    elif fault == "managed_definition":
        unit["ordered_definitions"]["whole_on_return"].append(access["resource"])
    elif fault in {"graph_effect", "graph_definition"}:
        graph = next(p for p in scope["graph_execution"]["procedures"] if p["procedure"] == procedure["procedure"])
        target = graph["admitted_unit_tasks"][str(unit["ordinal"])]
        stack = [graph["task_graph"]]
        while stack:
            current = stack.pop()
            for task in current["tasks"]:
                if task["id"] == target:
                    if fault == "graph_effect":
                        task["may_writes"] = []
                    else:
                        task["must_writes"] = [access["resource"]]
                stack.extend(task.get("branch", []))
    elif fault == "graph_guard":
        if control["original_guard"] is None:
            control["original_guard"] = "go"
        else:
            graph = next(p for p in scope["graph_execution"]["procedures"] if p["procedure"] == procedure["procedure"])
            target = permit["selected_node_ids"][0] + ":alternative:0"
            stack = [graph["task_graph"]]
            while stack:
                current = stack.pop()
                for task in current["tasks"]:
                    if task["id"] == target:
                        task["predicate"] = [".false."]
                    stack.extend(task.get("branch", []))
    # For CALL mutations also update the retained source-call copy; the independent
    # exact original A association still has to reject the forged transport.
    if fault in {"source_call", "managed_mapping", "complete_mapping", "actual_evaluation", "presence", "requirements",
                 "formal_descriptor", "actual_descriptor"}:
        index = next(i for i, row in enumerate(report["resolved_calls"])
                     if row["caller"] == call["caller"] and row["first_line"] == call["first_line"])
        report["resolved_calls"][index] = deepcopy(call)
    reseal_write(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)
