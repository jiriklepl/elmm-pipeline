"""Ordinary numeric children retain exact uses without inventing source tasks."""
from copy import deepcopy
from hashlib import sha256

import pytest
from compiler.tests.test_empty_member_edit_boundaries import emit_package
from test_conditional_return_record import seal

from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.conditional_return_record import validate_conditional_return


@pytest.fixture(scope="module", params=[1, 6])
def original(request, tmp_path_factory):
    return emit_package(tmp_path_factory.mktemp("ordinary-lexical"), arity=request.param, nested=True)[-1]


def parts(report):
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    contract = epoch["return_contract"]
    return scope, plan, epoch, contract


def route_to(contract, callee):
    return next(route for member in contract["members"] for route in member["routes"]
                if route.get("kind") == "checked_child" and route["procedure"] == callee)


def slot(procedure, root):
    return str(int(sha256((procedure + "|" + root).encode()).hexdigest()[:8], 16) + 1)


def test_plain_nested_numeric_children_publish_without_opaque_or_source_task_projections(original):
    report = deepcopy(original)
    before = deepcopy(report)
    validate_checked_execution_plans(report)
    _scope, _plan, epoch, contract = parts(report)
    assert contract["version"] == 4
    assert not epoch.get("native_opaque_accesses")
    assert not any(member.get("source_device_tasks") for member in epoch["members"])
    assert not any("unused" in row["resource"] for row in epoch["resources"])
    assert route_to(contract, "hierarchical::empty_worker")["canonical_resources"] == {}
    middle = route_to(contract, "hierarchical::middle")
    assert middle["canonical_resources"] == {
        "hierarchical::left": "hierarchical::left", "hierarchical::right": "hierarchical::right"}
    assert report == before


@pytest.mark.parametrize("fault", ["missing_call", "missing_coordinator_call", "missing_lexical_call", "slot",
                                   "implicit_argument", "changed_canonical", "managed_role", "descriptor"])
def test_resealed_plain_lexical_transport_keeps_associations_slots_and_roles(original, fault):
    report = deepcopy(original)
    scope, plan, epoch, contract = parts(report)
    middle = route_to(contract, "hierarchical::middle")
    consumer = next(row for row in plan["procedures"] if row["procedure"] == "hierarchical::consumer")
    coordinator = next(row for row in scope["module_coordinators"] if row["procedure"] == consumer["procedure"])
    if fault == "missing_call": consumer["calls"] = []
    elif fault == "missing_coordinator_call": coordinator["calls"] = []
    elif fault == "missing_lexical_call": scope["internal_coordinators"][0]["lexical_namespace"]["source_calls"] = []
    elif fault == "slot": middle["handle_forwarding"][next(iter(middle["handle_forwarding"]))] += 1
    elif fault == "implicit_argument":
        value = middle["canonical_resources"].pop("hierarchical::left")
        old = slot("hierarchical::middle", "hierarchical::left")
        new = slot("hierarchical::middle", "argument::invented")
        middle["canonical_resources"]["argument::invented"] = value
        middle["canonical_handles"][new] = middle["canonical_handles"].pop(old)
        middle["handle_forwarding"][new] = middle["handle_forwarding"].pop(old)
    elif fault == "changed_canonical": middle["canonical_resources"]["hierarchical::left"] = "hierarchical::right"
    elif fault == "managed_role":
        resource = next(row for row in epoch["resources"] if row["resource"] == "hierarchical::left")
        resource.update(device_replica_authorized=False, initialization_authorized=False, storage="opaque host reservation")
    elif fault == "descriptor":
        next(row for row in plan["resources"] if row["resource"] == "hierarchical::left")["descriptor"]["resource"] = "hierarchical::right"
    else: raise AssertionError(fault)
    seal(report)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(report)


def test_sibling_use_cannot_authorize_a_plain_producer_implicit_capture(original):
    report = deepcopy(original)
    scope, plan, _epoch, contract = parts(report)
    name, root = "hierarchical::producer", "hierarchical::left"
    producer = route_to(contract, name)
    number = next(row["registration_identity"] for row in plan["resources"] if row["resource"] == root)
    local_slot = slot(name, root)
    producer["canonical_resources"][root] = root
    producer["canonical_handles"][local_slot] = number
    producer["handle_forwarding"][local_slot] = number
    for rows in (plan["procedures"], scope["module_coordinators"]):
        next(row for row in rows if row["procedure"] == name)["calls"][0]["handle_bindings"][local_slot] = number
    seal(report)
    with pytest.raises(ValueError, match="conditional return child formal-to-canonical association differs"):
        validate_checked_execution_plans(report)


def validate_transport(report):
    # Exercise the existing conditional-return transport boundary directly so
    # forged graph connectivity is rejected before any proposed capture can
    # rely on it. Full-plan and strict-package positives run separately above.
    scope, plan, epoch, contract = parts(report)
    seal(report)
    validate_conditional_return(contract, epoch, plan, report, scope)


def remove_numeric_uses(report):
    scope, plan, epoch, contract = parts(report)
    for procedure in plan["procedures"]:
        procedure["units"] = []
        for call in procedure["calls"]:
            if "handle_bindings" in call:
                call["handle_bindings"] = {}
                if procedure["procedure"] != "hierarchical::empty_worker":
                    call["resource_mappings"] = {}
    for coordinator in scope["module_coordinators"]:
        for call in coordinator["calls"]:
            call["handle_bindings"] = {}
            if coordinator["procedure"] != "hierarchical::empty_worker":
                call["resource_mappings"] = {}
    for member in epoch["members"]:
        member["units"] = []
    for member in contract["members"]:
        member["routes"] = [row for row in member["routes"] if row["kind"] != "checked_unit"]
        for route in member["routes"]:
            if route["kind"] == "checked_child":
                route["canonical_resources"] = {}
                route["canonical_handles"] = {}
                route["handle_forwarding"] = {}


def test_unused_empty_actual_cannot_seed_an_implicit_global_capture(original):
    report = deepcopy(original)
    remove_numeric_uses(report)
    scope, plan, _epoch, contract = parts(report)
    root, name = "hierarchical::right", "hierarchical::middle"
    route = route_to(contract, name)
    number = next(row["registration_identity"] for row in plan["resources"] if row["resource"] == root)
    route["canonical_resources"][root] = root
    route["canonical_handles"][slot(name, root)] = number
    route["handle_forwarding"][slot(name, root)] = number
    for rows in (plan["procedures"], scope["module_coordinators"]):
        next(row for row in rows if row["procedure"] == name)["calls"][-1]["handle_bindings"] = dict(route["handle_forwarding"])
    # The untouched original empty-worker CALL has array actuals, but no
    # retained capture or payload unit is an independent use of those actuals.
    with pytest.raises(ValueError, match="conditional return child formal-to-canonical association differs"):
        validate_transport(report)


def test_checked_ordinary_call_relation_rejects_a_disconnected_member(original):
    report = deepcopy(original)
    scope, plan, _epoch, contract = parts(report)
    empty = "hierarchical::empty_worker"
    for member in contract["members"]:
        member["routes"] = [route for route in member["routes"] if route.get("procedure") != empty]
    for rows in (plan["procedures"], scope["module_coordinators"]):
        next(row for row in rows if row["procedure"] == empty)["calls"] = []
    with pytest.raises(ValueError, match="checked CALL relation has a disconnected member"):
        validate_transport(report)


def test_unseeded_checked_ordinary_call_cycle_cannot_justify_captures(original):
    report = deepcopy(original)
    remove_numeric_uses(report)
    scope, plan, epoch, contract = parts(report)
    helper, middle, empty = "hierarchical::middle::internal_helper", "hierarchical::middle", "hierarchical::empty_worker"
    route = route_to(contract, empty)
    route["procedure"] = middle
    old = next(row for row in plan["procedures"] if row["procedure"] == empty)["calls"][0]
    for rows in (plan["procedures"], scope["module_coordinators"]):
        rows[:] = [row for row in rows if row["procedure"] != empty]
        target = next(row for row in rows if row["procedure"] == middle)
        target["calls"].append({**old, "caller": helper, "handle_bindings": {}, "resource_mappings": {}})
    epoch["members"] = [row for row in epoch["members"] if row["procedure"] != empty]
    contract["members"] = [row for row in contract["members"] if row["procedure"] != empty]
    scope["graph_execution"]["procedures"] = [row for row in scope["graph_execution"]["procedures"] if row["procedure"] != empty]
    with pytest.raises(ValueError, match="checked CALL relation contains a cycle"):
        validate_transport(report)
