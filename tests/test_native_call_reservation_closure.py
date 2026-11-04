"""Physical host reservations seed only their checked lexical CALL ancestors."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from test_ordinary_lexical_use_closure import parts, route_to, slot, validate_transport

from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_native_record_lifetime import hierarchical_source


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _path, builder = fixture(tmp_path_factory.mktemp("native-reservation"), hierarchical_source())
    outputs, report = builder.run()
    assert builder.finish()
    return report


def operation(plan):
    return next(op for procedure in plan["procedures"] for unit in procedure["units"]
                for op in unit["operations"]["native_operations"]
                if op["kind"] == "complete original native CALL")


def test_original_native_record_reservation_is_a_physical_lexical_use(original):
    report = deepcopy(original)
    before = deepcopy(report)
    validate_checked_execution_plans(report)
    _scope, plan, _epoch, contract = parts(report)
    root = "hierarchical::native_points"
    assert operation(plan)["host_metadata"][root]["device_capture"] is False
    assert route_to(contract, "hierarchical::middle")["canonical_resources"][root] == root
    assert root not in route_to(contract, "hierarchical::producer")["canonical_resources"]
    assert report == before


@pytest.mark.parametrize("fault", ["descriptor", "device_capture", "storage_role", "replica", "initialization"])
def test_resealed_reservation_must_keep_its_descriptor_and_opaque_role(original, fault):
    report = deepcopy(original)
    _scope, plan, epoch, _contract = parts(report)
    root = "hierarchical::native_points"
    metadata = operation(plan)["host_metadata"][root]
    resource = next(row for row in epoch["resources"] if row["resource"] == root)
    if fault == "descriptor":
        metadata["descriptor"]["resource"] = "hierarchical::right"
    elif fault == "device_capture":
        metadata["device_capture"] = True
    elif fault == "storage_role":
        resource["storage"] = "managed"
    elif fault == "replica":
        resource["device_replica_authorized"] = True
    else:
        resource["initialization_authorized"] = True
    with pytest.raises(ValueError, match="native reservation descriptor or storage role differs"):
        validate_transport(report)


def test_sibling_reservation_cannot_seed_an_unrelated_child_capture(original):
    report = deepcopy(original)
    scope, plan, _epoch, contract = parts(report)
    root, name = "hierarchical::native_points", "hierarchical::producer"
    producer = route_to(contract, name)
    number = next(row["registration_identity"] for row in plan["resources"] if row["resource"] == root)
    local_slot = slot(name, root)
    producer["canonical_resources"][root] = root
    producer["canonical_handles"][local_slot] = number
    producer["handle_forwarding"][local_slot] = number
    for rows in (plan["procedures"], scope["module_coordinators"]):
        next(row for row in rows if row["procedure"] == name)["calls"][0]["handle_bindings"][local_slot] = number
    with pytest.raises(ValueError, match="conditional return child formal-to-canonical association differs"):
        validate_transport(report)
