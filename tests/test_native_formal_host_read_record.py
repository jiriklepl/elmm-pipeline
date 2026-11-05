"""Caller-owned native READ roles cannot gain write or GPU authority in transport."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from test_conditional_return_record import seal
from test_native_opaque_access_record import parts

from compiler.tests.test_native_call_graph import generate
from compiler.tests.test_native_formal_host_reads import direct_source


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _path, _builder, _outputs, report = generate(tmp_path_factory.mktemp("caller-read"), direct_source())
    validate_checked_execution_plans(report)
    return report


@pytest.mark.parametrize("fault", ["version", "legacy_inout", "nonallocatable", "target", "out", "write",
                                  "initialization", "device", "canonical"])
def test_resealed_caller_read_cannot_change_storage_or_effect_role(original, fault):
    report = deepcopy(original)
    _scope, _plan, epoch, row, _procedure, _unit, operation = parts(report)
    value = operation["native_opaque_accesses"]
    access, = value["accesses"]
    permit = access["permit"]
    assert permit["version"] == 2
    if fault == "version":
        permit["version"] = True
    elif fault == "legacy_inout":
        permit["version"] = 1
    elif fault == "nonallocatable":
        permit["binding"]["attributes"].remove("allocatable")
    elif fault == "target":
        permit["binding"]["attributes"].append("target")
    elif fault == "out":
        permit["binding"]["intent"] = "out"
    elif fault == "write":
        access["mode"] = "RW"
    elif fault in {"initialization", "device"}:
        resource, = [item for item in epoch["resources"] if item["resource"] == "argument::mesh"]
        resource["initialization_authorized" if fault == "initialization" else "device_replica_authorized"] = True
    else:
        row["resources"][0]["canonical_resources"] = ["native_graph::a"]
    permit["identity"] = _digest({key: item for key, item in permit.items() if key != "identity"})
    value["identity"] = _digest({key: item for key, item in value.items() if key != "identity"})
    row["access_identity"] = value["identity"]
    row["resources"][0]["permit_identity"] = permit["identity"]
    seal(report)
    with pytest.raises(ValueError, match="native opaque|storage epoch|canonical"):
        validate_checked_execution_plans(report)
