"""Resealed manifests cannot turn numerical accesses into original CPU CALLs."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from test_conditional_return_record import seal

from compiler.tests.test_native_call_graph import generate


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _path, builder, _outputs, report = generate(tmp_path_factory.mktemp("original-host"), initialized="none")
    assert builder.finish()
    return report


def operation(report):
    plan = report["scopes"][0]["checked_execution_plan"]
    return next(op for proc in plan["procedures"] for unit in proc["units"]
                for op in unit["operations"]["native_operations"] if op.get("original_host_access"))


def test_real_original_host_contract_preserves_unknown_entry_definitions(original):
    before = deepcopy(original)
    validate_checked_execution_plans(original)
    assert operation(original)["original_host_access"]["proof"]["definitions_authorized"] is False
    assert original == before


@pytest.mark.parametrize("fault", ["missing", "schema", "version", "bool_version", "begin", "plan_kind",
                                  "effects", "definitions", "runtime", "runtime_kind", "runtime_symbol", "kind",
                                  "rewritten", "async", "cuts", "exact_sections", "proof_schema", "identity",
                                  "procedure", "source", "selection", "alias", "capture", "definition_credit",
                                  "motion", "internal_cuts", "resource", "role", "write", "undefine"])
def test_resealed_original_host_contract_keeps_source_and_runtime_authority(original, fault):
    report = deepcopy(original)
    op = operation(report)
    contract, runtime = op["original_host_access"], report["runtime"]["task_acquisition"]
    proof = contract["proof"]
    if fault == "missing":
        op.pop("original_host_access")
    elif fault == "schema":
        contract["unchecked"] = True
    elif fault in {"version", "bool_version", "begin", "plan_kind", "effects", "definitions"}:
        contract[fault.removeprefix("bool_")] = True if fault == "bool_version" else 99
    elif fault.startswith("runtime"):
        runtime[{"runtime": "abi_version", "runtime_kind": "original_host_plan_kind",
                 "runtime_symbol": "original_host_begin"}[fault]] = 99
    elif fault == "kind":
        op["kind"] = "generated CPU kernel"
    elif fault == "rewritten":
        op["preserves_original_source"] = False
    elif fault in {"async", "cuts"}:
        op["completion"]["available" if fault == "async" else "internal_cuts_authorized"] = fault == "cuts"
    elif fault == "exact_sections":
        op["sections"]["available"] = True
    elif fault == "proof_schema":
        proof["unchecked"] = True
    elif fault in {"identity", "procedure"}:
        proof[fault] = "f" * 64
    elif fault == "source":
        op["structured_summary_identity"] = "f" * 64
    elif fault == "selection":
        op["selected_original_nodes"] = ["outside/body/999"]
    elif fault in {"alias", "capture", "definition_credit", "motion", "internal_cuts"}:
        proof[("definitions" if fault == "definition_credit" else fault) + "_authorized"] = True
    else:
        item = next(item for item in proof["resources"] if item["role"] == "managed")
        if fault in {"resource", "role"}:
            item[fault] = "unknown"
        else:
            item["actions"] = sorted(set(item["actions"]) | {fault})
    seal(report)
    with pytest.raises(ValueError, match="original host"):
        validate_checked_execution_plans(report)
