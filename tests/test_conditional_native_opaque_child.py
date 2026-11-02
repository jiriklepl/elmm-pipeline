"""Implicit native reservations retain exact source and forwarding authority."""
import json
from copy import deepcopy

import pytest
from compiler.tests.test_same_family_resource_roles import emit
from test_conditional_return_record import seal

from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _path, builder, _owner, _middle, _proof, _outputs, report = emit(tmp_path_factory.mktemp("implicit-native"), opaque=True)
    assert builder.finish()
    report = json.loads(_outputs["scope-manifest.json"])
    validate_checked_execution_plans(report)
    return report


def parts(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    row, = [r for r in epoch["native_opaque_accesses"] if r["procedure"] == "hierarchical::middle"]
    procedure = next(p for p in plan["procedures"] if p["procedure"] == row["procedure"])
    unit = procedure["units"][row["unit_ordinal"]]
    operation = unit["operations"]["native_operations"][row["operation_id"]]
    access, = operation["native_opaque_accesses"]["accesses"]
    route, = [r for m in epoch["return_contract"]["members"] if m["procedure"] == "hierarchical::advance"
              for r in m["routes"] if r.get("kind") == "checked_child" and r["procedure"] == "hierarchical::middle"]
    return plan, epoch, row, unit, operation, access, route


def test_original_native_module_read_completes_implicit_child_inventory(original):
    report = deepcopy(original)
    before = deepcopy(report)
    validate_checked_execution_plans(report)
    _plan, epoch, _row, unit, _operation, access, route = parts(report)
    assert access["resource"] == "hierarchical::weights"
    assert access["mode"] == "R"
    assert access["permit_kind"] == "original_readonly"
    assert access["resource"] not in unit["resources"]
    assert route["canonical_resources"][access["resource"]] == access["resource"]
    resource = next(r for r in epoch["resources"] if r["resource"] == access["resource"])
    assert resource["device_replica_authorized"] is resource["initialization_authorized"] is False
    assert resource["storage"] == "opaque host reservation"
    assert report == before


@pytest.mark.parametrize("fault", ["missing_permit", "changed_permit", "wrong_procedure", "wrong_root", "wrong_slot",
                                   "managed_conflict", "missing_typed_operation"])
def test_implicit_native_use_cannot_invent_proofs_or_change_forwarding(original, fault):
    report = deepcopy(original)
    _plan, epoch, row, _unit, operation, access, route = parts(report)
    value, permit = operation["native_opaque_accesses"], access["permit"]
    if fault == "missing_permit":
        access.pop("permit")
    elif fault == "changed_permit":
        permit["calls_or_escapes"] = True
    elif fault == "wrong_procedure":
        permit["procedure"] = "hierarchical::consumer"
    elif fault == "wrong_root":
        access["resource"] = "hierarchical::ratios"
    elif fault == "wrong_slot":
        slot = str(row["call_projection"]["field_slots"]["hierarchical::weights"])
        route["handle_forwarding"][slot] += 1
    elif fault == "managed_conflict":
        resource = next(r for r in epoch["resources"] if r["resource"] == "hierarchical::weights")
        resource.update(device_replica_authorized=True, storage="stable", lifetime="invocation",
                        allocation_changes=False, escapes=False)
        resource.pop("initialization_authorized")
    else:
        operation.pop("native_opaque_accesses")
    if "permit" in access:
        permit["identity"] = _digest({k: v for k, v in permit.items() if k != "identity"})
        row["resources"][0]["permit_identity"] = permit["identity"]
    value["identity"] = _digest({k: v for k, v in value.items() if k != "identity"})
    row["access_identity"] = value["identity"]
    seal(report)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(report)
