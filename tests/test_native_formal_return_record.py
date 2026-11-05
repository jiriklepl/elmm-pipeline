"""Explicit native formals keep local descriptors and canonical caller storage."""
from copy import deepcopy
from hashlib import sha256

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from test_conditional_return_record import seal
from test_ordinary_lexical_use_closure import parts, validate_transport

from compiler.driver.options import CompilerOptions
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_native_formal_host_reads import source


@pytest.fixture(scope="module", params=["coords", "mesh"])
def original(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("formal-return") / "source.f90"
    path.write_text(source().replace("coords", request.param))
    fact = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"native_graph::" + name: dict(fact) for name in ("a", "b", "c")}}
    builder = ScopeBuilder([path], "native_graph::advance", facts=facts, options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
        contracts={"native_graph::advance": {"native_ancestor": {
            "version": 1, "device_roots": ["native_graph::child"]}}})
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report.get("graph_owner_attempt")
    assert builder.finish()
    return report


def native_operation(report):
    _scope, plan, _epoch, _contract = parts(report)
    child, = [p for p in plan["procedures"] if p["procedure"] == "native_graph::child"]
    return next(op for unit in child["units"] for op in unit["operations"]["native_operations"]
                if op["kind"] == "complete original native CALL")


def test_returned_domain_keeps_borrowed_descriptor_distinct_from_original_allocation(original):
    before = deepcopy(original)
    validate_checked_execution_plans(original)
    validate_transport(deepcopy(original))
    _scope, plan, epoch, _contract = parts(original)
    access, = native_operation(original)["native_opaque_accesses"]["accesses"]
    canonical, = [row for row in plan["resources"] if row["resource"] == "argument::mesh"]
    assert access["permit"]["binding"] != canonical["descriptor"]
    resource, = [row for row in epoch["resources"] if row["resource"] == "argument::mesh"]
    assert resource["storage"] == "opaque host reservation"
    assert not resource["device_replica_authorized"]
    assert not resource["initialization_authorized"]
    assert original == before


@pytest.mark.parametrize("fault", ["local_descriptor", "missing_typed_access", "permit_descriptor",
                                  "canonical_mapping", "numeric_role", "device", "initialization"])
@pytest.mark.parametrize("reader", ["full", "return_contract"])
def test_explicit_formal_reservation_keeps_its_typed_call_and_storage_evidence(original, fault, reader):
    report = deepcopy(original)
    operation = native_operation(report)
    access, = operation["native_opaque_accesses"]["accesses"]
    root = access["resource"]
    _scope, _plan, epoch, _contract = parts(report)
    resource, = [row for row in epoch["resources"] if row["resource"] == "argument::mesh"]
    if fault == "local_descriptor":
        operation["host_metadata"][root]["descriptor"]["lower_bounds"] = ["17"]
    elif fault == "missing_typed_access":
        operation.pop("native_opaque_accesses")
    elif fault == "permit_descriptor":
        access["permit"]["binding"]["intent"] = "out"
    elif fault == "canonical_mapping":
        row, = [row for row in epoch["native_opaque_accesses"] if row["procedure"] == "native_graph::child"]
        row["resources"][0]["canonical_resources"] = ["native_graph::a"]
    elif fault == "numeric_role":
        resource["storage"] = "managed"
    elif fault == "device":
        resource["device_replica_authorized"] = True
    else:
        resource["initialization_authorized"] = True
    seal(report)
    with pytest.raises(ValueError, match="native opaque|native reservation|storage epoch|canonical"):
        (validate_checked_execution_plans if reader == "full" else validate_transport)(report)
