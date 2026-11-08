"""Original joined CPU metadata contributes to checked caller use accounting."""

from copy import deepcopy

import pytest

from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.conditional_return_record import validate_conditional_return

from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_joined_boundary_lifecycle import ordinary_source


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    source = ordinary_source().replace(
        "real(8)::observed",
        "type point\ninteger::xi\nend type\ntype(point)::points(1024)\n"
        "real(8),allocatable::weights(:)\nreal(8)::observed",
    ).replace(
        "right(modulo(i,n)+2)=left(i-4)",
        "right(modulo(i,n)+2)=left(i-4)+weights(points(i)%xi)",
    ).replace(
        "if(change)call relocate(left,right)",
        "if(change)deallocate(weights)\nif(change)call relocate(left,right)",
    )
    _path, builder = fixture(tmp_path_factory.mktemp("native-metadata-return"), source)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    return report


def operation(report):
    plan = report["scopes"][0]["checked_execution_plan"]
    member = next(p for p in plan["procedures"] if p["procedure"] == "hierarchical::middle")
    return next(op for unit in member["units"] for op in unit["operations"]["native_operations"]
                if "hierarchical::points" in op.get("host_metadata", {}))


def test_joined_native_metadata_is_used_without_becoming_a_device_capture(original):
    op = operation(original)
    assert op["kind"] == "joined native OpenMP"
    assert "hierarchical::points" not in op["managed_resources"]
    assert op["host_metadata"]["hierarchical::points"]["device_capture"] is False
    validate_checked_execution_plans(original)


@pytest.mark.parametrize("fault", ["descriptor", "device_capture", "removed_use", "managed_role"])
def test_metadata_use_keeps_exact_host_reservation_obligations(original, fault):
    report = deepcopy(original)
    scope = report["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    op = operation(report)
    root = "hierarchical::points"
    if fault == "descriptor":
        op["host_metadata"][root]["descriptor"]["lower_bounds"] = ["17"]
    elif fault == "device_capture":
        op["host_metadata"][root]["device_capture"] = True
    elif fault == "removed_use":
        del op["host_metadata"][root]
    else:
        next(row for row in epoch["resources"] if row["resource"] == root)["device_replica_authorized"] = True
    # Exercise the use-closure checks independently of the outer content hash.
    with pytest.raises(ValueError, match="native reservation|formal-to-canonical"):
        validate_conditional_return(epoch["return_contract"], epoch, plan, report, scope)
