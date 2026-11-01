"""Explicit public selection is consumed only by a fresh checked compiler plan."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.provider_lifetime_record import _digest
from test_poisfft_entry_read_preparation import digest, fixture

from compiler.driver.options import CompilerOptions
from compiler.ir import CompilationError
from compiler.offload.config import OffloadConfig
from compiler.scopes.source import ScopeBuilder
from compiler.tests.test_provider_only_lifetime import fixture as parent_fixture


def test_strict_reader_binds_original_parent_private_helper_and_checked_external_slot(tmp_path):
    _, builder = parent_fixture(tmp_path)
    _, report = builder.run()
    validate_checked_execution_plans(report)


@pytest.mark.parametrize("change", ["missing", "helper", "helper_build", "provider", "slot", "configuration", "native_grant", "source_edit", "duplicate_call", "missing_edge", "budget_type", "bool_slot", "helper_name", "leaf_name"])
def test_strict_reader_rejects_changed_lifetime_even_with_outer_record_rehash(tmp_path, change):
    _, builder = parent_fixture(tmp_path)
    _, report = builder.run()
    record = report["provider_only_lifetime"]
    if change == "missing":
        report.pop("provider_only_lifetime")
    elif change == "helper":
        record["helper"]["sha256"] = "0" * 64
    elif change == "helper_build":
        report["build_sources"] = [item for item in report["build_sources"] if item["path"] != record["helper"]["path"]]
    elif change == "provider":
        record["provider_source_proof_identity"] = "0" * 64
    elif change == "slot":
        record["provider_slot"] = 2
    elif change == "bool_slot":
        record["provider_slot"] = True
    elif change == "helper_name":
        old = record["helper"]["path"]
        changed = old.replace(".f90", "_foreign.f90")
        record["helper"]["path"] = changed
        report["artifacts_sha256"][changed] = report["artifacts_sha256"].pop(old)
        for item in report["build_sources"]:
            if item["path"] == old:
                item["path"] = changed
    elif change == "leaf_name":
        record["checked_leaf_entry"] += "_foreign"
    elif change == "configuration":
        record["configuration_words"] += 1
    elif change == "native_grant":
        record["native_effect_authority"] = True
    elif change == "source_edit":
        report["source_edits"][0]["replacement"] += "! unbound source edit\n"
    elif change == "duplicate_call":
        record["edges"][0]["calls"].append(deepcopy(record["edges"][0]["calls"][0]))
    elif change == "missing_edge":
        record["edges"].pop()
    else:
        record["budgets"]["depth_limit"] = "8"
    if change != "missing":
        record["execution_identity"] = _digest({key: value for key, value in record.items() if key != "execution_identity"})
    with pytest.raises(ValueError, match="provider-only lifetime"):
        validate_checked_execution_plans(report)


PARENT = """module solver_parent
use poissonsolvers,only:Poiss_PoisFFT
implicit none
contains
subroutine owner(phi,rhs)
real(8),intent(inout)::phi(:,:,:),rhs(:,:,:)
call pressure(phi,rhs)
rhs=rhs+1.0_8
call pressure(phi,rhs)
end subroutine
subroutine pressure(phi,rhs)
real(8),intent(inout)::phi(:,:,:),rhs(:,:,:)
call Poiss_PoisFFT(phi,rhs)
end subroutine
end module
"""
REQUEST = {"version": 1, "call_path": ["solver_parent::owner", "solver_parent::pressure", "poissonsolvers::poiss_poisfft"]}


def public_preparation(tmp_path, monkeypatch, *, request=REQUEST, source=PARENT, retention_change=None):
    paths, document, kwargs, _, _ = fixture(tmp_path, monkeypatch)
    parent = tmp_path / "parent.f90"
    parent.write_text(source)
    paths.append(parent)
    document["source_inputs"][str(parent)] = digest(parent)
    document["entries"].append({"source": str(parent), "path": str(parent), "sha256": digest(parent),
                                "line_map": list(range(1, len(source.splitlines())+1))})
    from elmm_pipeline.gpu_pressure_contract import prepare_poisfft_provider_manifest
    from elmm_pipeline.gpu_pressure_lazy import prepare_lazy_poisfft_contract

    from compiler.frontend.external_entry_reads import provider_read_binding
    from compiler.frontend.provider_lifetime import PROVIDER_RETENTION_DUTIES
    stub = prepare_poisfft_provider_manifest(kwargs["backend"], tmp_path / "binding-only-provider.json",
        fortran=kwargs["fortran"], qualification=kwargs["numerical_qualification"],
        original_body_sha256="a" * 64, task_contract_sha256="b" * 64)
    evidence = tmp_path / "provider-retention-review.json"
    evidence.write_text(json.dumps({"test_only": True, "claim": "reviewed private-plan fixture state"}))
    prerequisite = tmp_path / "provider-retention-prerequisite.json"
    prerequisite.write_text(json.dumps({"version": 1, "kind": "reviewed_external_provider_retention_prerequisite",
        "provider_binding": provider_read_binding(json.loads(Path(stub["path"]).read_text())),
        "duties": dict(PROVIDER_RETENTION_DUTIES), "evidence": [{"path": str(evidence), "sha256": digest(evidence)}]}))
    premise = json.loads(prerequisite.read_text())
    if retention_change == "binding":
        premise["provider_binding"]["artifacts"][next(iter(premise["provider_binding"]["artifacts"]))] = "0" * 64
    elif retention_change == "field_pointer":
        premise["duties"]["host_payload_retention_authorized"] = True
    elif retention_change == "conditional_publication":
        premise["duties"]["publication"] = "may complete later after native work"
    elif retention_change == "evidence":
        premise["evidence"][0]["sha256"] = "0" * 64
    elif retention_change == "wrong_version":
        premise["version"] = True
    prerequisite.write_text(json.dumps(premise))
    if retention_change != "missing":
        kwargs["provider_retention_prerequisite"] = {"path": str(prerequisite), "sha256": digest(prerequisite)}
    result = prepare_lazy_poisfft_contract(paths, document, tmp_path / "prepared",
                                          provider_only_lifetime=request, **kwargs)
    facts = json.loads(Path(result["capture_facts"]["path"]).read_text())
    contracts = json.loads(Path(result["effect_contracts"]["path"]).read_text())["procedures"]
    builder = ScopeBuilder(paths, result["entry"], facts=facts, contracts=contracts, analysis_sources=document,
        options=CompilerOptions(), config=OffloadConfig(policy="sections", scope_execution="graph"))
    return result, facts, builder


def test_public_lazy_api_transports_explicit_parent_request_into_checked_generic_lifetime(tmp_path, monkeypatch):
    result, facts, builder = public_preparation(tmp_path, monkeypatch)
    assert result["provider_only_lifetime_request"] == facts["provider_only_lifetime"] == REQUEST
    assert all(item["initialized"] == "none" for item in facts["captures"].values())
    _, report = builder.run()
    assert report["provider_only_lifetime"]["request"] == REQUEST
    assert report["provider_only_lifetime"]["field_retention_authorized"] is False
    validate_checked_execution_plans(report)


def test_preparation_selection_alone_cannot_issue_a_missing_original_dispatch_hop(tmp_path, monkeypatch):
    request = {"version": 1, "call_path": ["solver_parent::owner", "poissonsolvers::poiss_poisfft"]}
    with pytest.raises(CompilationError, match="missing a selected original CALL"):
        public_preparation(tmp_path, monkeypatch, request=request)
    result = json.loads((tmp_path / "prepared/prepared-contract.json").read_text())
    assert "requires compiler-issued exact call spine" in result["provider_only_lifetime_qualification"]


def test_strict_reader_binds_metadata_program_owner_and_rejects_changed_source_kind(tmp_path):
    from compiler.tests.test_provider_only_lifetime import PARENT as NATIVE
    from compiler.tests.test_provider_only_lifetime import PROGRAM, PROGRAM_REQUEST

    _, builder = parent_fixture(tmp_path, request=PROGRAM_REQUEST, source=NATIVE + PROGRAM)
    _, report = builder.run()
    validate_checked_execution_plans(report)
    record = report["provider_only_lifetime"]
    record["members"][0]["source_kind"] = "module"
    from elmm_pipeline.provider_lifetime_record import _SOURCE_FIELDS

    record["identity"] = _digest({key: record[key] for key in _SOURCE_FIELDS})
    record["execution_identity"] = _digest({key: value for key, value in record.items() if key != "execution_identity"})
    with pytest.raises(ValueError, match="original member linkage"):
        validate_checked_execution_plans(report)


def test_public_preparation_transports_an_explicit_program_owner_without_program_execution(tmp_path, monkeypatch):
    source = PARENT + """program solver_main
use solver_parent
implicit none
real(8)::phi(6,6,6),rhs(4,4,4)
integer::step
do step=1,2
 call owner(phi,rhs)
 if(step==1)cycle
enddo
end program
"""
    request = {"version": 1, "call_path": ["$program::solver_main", *REQUEST["call_path"]]}
    result, facts, builder = public_preparation(tmp_path, monkeypatch, request=request, source=source)
    assert result["provider_only_lifetime_request"] == facts["provider_only_lifetime"] == request
    _, report = builder.run()
    assert report["provider_only_lifetime"]["members"][0]["source_kind"] == "program"
    validate_checked_execution_plans(report)


@pytest.mark.parametrize("change", ["missing", "binding", "field_pointer", "conditional_publication", "evidence", "wrong_version"])
def test_retained_plan_needs_reviewed_current_provider_state_and_complete_publication(tmp_path, monkeypatch, change):
    with pytest.raises((ValueError, CompilationError)):
        public_preparation(tmp_path, monkeypatch, retention_change=change)
