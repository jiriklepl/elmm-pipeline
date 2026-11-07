"""The reader requires both original bounds and their emitted entry phase."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from test_conditional_return_record import seal

from compiler.tests.test_conditional_return import compile_fixture
from compiler.tests.test_specification_entry import ENTRY_SOURCE


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _, report, _, _ = compile_fixture(tmp_path_factory.mktemp("entry-reader"), ENTRY_SOURCE)
    return report


def test_original_entry_contract_and_runtime_are_linked(original):
    validate_checked_execution_plans(original)


@pytest.mark.parametrize("mutation", ["missing_phase", "missing_proof", "proof_authority", "proof_body",
    "phase_proof", "phase_context", "phase_span", "publication_missing", "publication_after", "publication_error",
    "runtime_missing", "runtime_symbol", "policy", "old_version", "declaration_write", "declaration_call",
    "array_roots", "storage_alias"])
def test_resealed_entry_contract_cannot_lose_original_source_or_publication(original, mutation):
    report = deepcopy(original)
    returned = report["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]
    member, = [member for member in returned["members"] if "specification_reads" in member["entry_exit"]]
    route, = [route for member in returned["members"] for route in member["routes"] if "specification_entry" in route]
    phase = route["specification_entry"]
    proof = member["entry_exit"]["specification_reads"]
    if mutation == "missing_phase":
        del route["specification_entry"]
    elif mutation == "missing_proof":
        del member["entry_exit"]["specification_reads"]
    elif mutation in {"proof_authority", "proof_body"}:
        proof["execution_authorized" if mutation == "proof_authority" else "original_procedure_sha256"] = (
            True if mutation == "proof_authority" else "0" * 64)
        proof["identity"] = _digest({k: v for k, v in proof.items() if k != "identity"})
        phase["proof_identity"] = proof["identity"]
    elif mutation in {"declaration_write", "declaration_call", "array_roots", "storage_alias"}:
        access, = [access for node in proof["declarations"] for access in node["items"] if access["storage"]["rank"]]
        if mutation.startswith("declaration_"):
            access["kind"] = "write" if mutation == "declaration_write" else "call"
        elif mutation == "array_roots":
            proof["array_resources"] = ["unrelated::metadata"]
        else:
            access["storage"]["attributes"].append("pointer")
        proof["identity"] = _digest({k: v for k, v in proof.items() if k != "identity"})
        phase["proof_identity"] = proof["identity"]
    elif mutation.startswith("phase_"):
        phase[{"phase_proof": "proof_identity", "phase_context": "context", "phase_span": "source_span"}[mutation]] = (
            "0" * 64 if mutation == "phase_proof" else "wrong_context" if mutation == "phase_context" else [1, 1])
    elif mutation.startswith("publication_"):
        edit, = [edit for edit in report["source_edits"]
                 if [edit["first_line"], edit["last_line"]] == phase["source_span"]]
        from compiler.scopes.specification_entry import publication
        block = "\n".join(publication(phase["context"])) + "\n"
        if mutation == "publication_error":
            edit["replacement"] = edit["replacement"].replace("error stop 'original specification publication failed'", "continue")
        else:
            edit["replacement"] = edit["replacement"].replace(block, "")
            if mutation == "publication_after":
                edit["replacement"] += block
    elif mutation == "runtime_missing":
        del report["runtime"]["specification_entry"]
    elif mutation == "runtime_symbol":
        report["runtime"]["specification_entry"]["observe"] = "fort_scope_close"
    elif mutation == "policy":
        returned["specification_entry_policy_version"] = 2
    else:
        returned["version"] = 1
        del returned["specification_entry_policy_version"]
    seal(report)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(report)
