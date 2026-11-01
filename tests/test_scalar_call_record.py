"""Scalar call readability preserves original CALLs and never specializes constants."""
import json
from copy import deepcopy

import pytest
from compiler.tests.test_scalar_call_readability import call_builder
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from test_line_source_task_record import check, reseal


@pytest.fixture(scope="module", params=["positional", "keyword", "parameter"])
def authentic(request, tmp_path_factory):
    calls, declaration = None, ""
    if request.param == "keyword":
        calls = "\n".join("call advance(dir=" + str(n) + ",a=a,mask=mask,nz=nz,ny=ny,nx=nx)" for n in (1, 1, 2, 3))
    elif request.param == "parameter":
        declaration = "integer,parameter::one=1"
        calls = "call advance(a,mask,nx,ny,nz,one)\nif(nx>0)then\ncall advance(a,mask,nx,ny,nz,2)\nendif"
    builder = call_builder(tmp_path_factory.mktemp("scalar-call-reader"), calls, declaration)
    outputs, _report = builder.run()
    report = json.loads(outputs["scope-manifest.json"])
    assert report["scope_count"] == 1
    assert len(report["scopes"][0]["source_device_tasks"]) == 3
    return report, builder.facts


def values(manifest, index=0):
    plan = manifest["scopes"][0]["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    member = next(member for member in epoch["members"] if member.get("source_device_tasks"))
    procedure = next(p for p in plan["procedures"] if p["procedure"] == member["procedure"])
    return (member["source_device_tasks"][index], member, procedure, epoch["source_inputs"],
            {r["resource"]: r for r in epoch["resources"]}, plan["entry"], manifest)


def dependency(args):
    prefix = next(c["authority"]["proof"] for c in args[0]["scalar_captures"] if "proof" in c["authority"])
    row = next(d for d in prefix["dependencies"] if d["authority"]["kind"] == "original source-constant scalar call inputs")
    return prefix, row, row["authority"]


def seal(args):
    prefix, _row, _authority = dependency(args)
    prefix["identity"] = _digest({k: v for k, v in prefix.items() if k != "identity"})
    reseal(args)


def test_original_positional_keyword_parameter_calls_keep_unchanged_readability(authentic):
    report, facts = authentic
    before = deepcopy(report)
    validate_checked_execution_plans(report, capture_facts=facts)
    for index in range(3):
        args = values(report, index)
        check(args)
        _prefix, row, authority = dependency(args)
        assert row["resource"] not in facts["captures"]
        assert authority["value_specialization_authorized"] is False
        assert authority["entry_fact_authorized"] is False
    assert report == before


def test_original_outside_selection_calls_need_not_be_in_retained_manifest(authentic):
    args = values(deepcopy(authentic[0]))
    check(args)
    _prefix, _row, authority = dependency(args)
    # Restrict the retained companion calls, preserving the complete original
    # source inventory. This is transport consistency, not new source issuance.
    removed = authority["calls"][-1]
    args[2]["calls"] = [site for site in args[2]["calls"]
                         if (site["first_line"], site["last_line"]) != (removed["first_line"], removed["last_line"])]
    args[6]["resolved_calls"] = [call for call in args[6]["resolved_calls"] if call != removed["source_call"]]
    assert len(authority["calls"]) > len(args[2]["calls"]) >= 1
    check(args)


@pytest.mark.parametrize("fault", [
    "version", "kind", "extra", "caller", "procedure", "inputs", "graph", "formal", "intent", "parameter_formal",
    "specialization", "motion", "array", "lifetime", "entry", "semantics", "calls_empty", "missing_emitted",
    "duplicate", "reorder", "span_bool", "span_wrong", "nodes_foreign", "nodes_duplicate", "site_extra",
    "call_version", "call_procedure", "call_extra", "evaluation", "formals", "original_position", "original_keyword",
    "original_actual", "mapping_formal", "mapping_descriptor", "mapping_section", "mapping_storage",
    "mapping_requirement", "mapping_resource", "actual_descriptor", "resolved_missing", "resolved_changed",
    "emitted_caller", "emitted_mapping", "caller_context_graph", "manifest_missing",
])
def test_resealed_call_evidence_cannot_change_association_or_add_authority(authentic, fault):
    args = values(deepcopy(authentic[0]))
    check(args)
    prefix, row, authority = dependency(args)
    site = authority["calls"][0]
    call = site["source_call"]
    mapping = next(m for m in call["resource_mappings"] if m["formal"] == "dir")
    if fault == "version": authority["version"] = True
    elif fault == "kind": authority["kind"] = "constant entry fact"
    elif fault == "extra": authority["numeric_range_authorized"] = True
    elif fault == "caller": authority["caller"] = "foreign::step"
    elif fault == "procedure": authority["procedure"] = "foreign::advance"
    elif fault == "inputs": authority["source_inputs"] = {}
    elif fault == "graph": authority["caller_graph"] = "0" * 64
    elif fault == "formal": authority["formal"] = {**authority["formal"], "kind": 8}
    elif fault == "intent": row["descriptor"]["intent"] = authority["formal"]["intent"] = "inout"
    elif fault == "parameter_formal":
        row["descriptor"]["attributes"] = authority["formal"]["attributes"] = ["parameter"]
    elif fault in {"specialization", "motion", "array", "lifetime", "entry"}:
        key = {"specialization": "value_specialization_authorized", "motion": "evaluation_motion_authorized",
               "array": "array_payload_definition_authorized", "lifetime": "lifetime_authorized",
               "entry": "entry_fact_authorized"}[fault]
        authority[key] = True
    elif fault == "semantics": authority["semantics"] = "evaluate once at entry"
    elif fault == "calls_empty": authority["calls"] = []
    elif fault == "missing_emitted": authority["calls"].pop()
    elif fault == "duplicate": authority["calls"].append(deepcopy(site))
    elif fault == "reorder": authority["calls"].reverse()
    elif fault == "span_bool": site["first_line"] = True
    elif fault == "span_wrong": site["first_line"] = site["last_line"] = 1
    elif fault == "nodes_foreign": site["selected_node_ids"] = ["foreign::step/body/0"]
    elif fault == "nodes_duplicate": site["selected_node_ids"] *= 2
    elif fault == "site_extra": site["numeric_value"] = 1
    elif fault == "call_version": call["schema_version"] = True
    elif fault == "call_procedure": call["procedure"] = "foreign::advance"
    elif fault == "call_extra": call["constant_value"] = 1
    elif fault == "evaluation": call["actual_evaluation"] = "hoisted"
    elif fault == "formals": call["formal_arguments"][-1] = "foreign"
    elif fault == "original_position": call["original_arguments"][0]["position"] = True
    elif fault == "original_keyword": call["original_arguments"][0]["keyword"] = "foreign"
    elif fault == "original_actual": call["original_arguments"][0]["actual"] = "other"
    elif fault == "mapping_formal": mapping["formal_resource"] = "argument::other"
    elif fault == "mapping_descriptor": mapping["formal_descriptor"] = {**mapping["formal_descriptor"], "kind": 8}
    elif fault == "mapping_section": mapping["section"] = {}
    elif fault == "mapping_storage": mapping["storage"] = "scalar_element"
    elif fault == "mapping_requirement": mapping["requirements"]["original_actual_presence"] = True
    elif fault == "mapping_resource": mapping["resource"] = "argument::other"
    elif fault == "actual_descriptor":
        mapping["actual_descriptor"] = {**row["descriptor"], "attributes": [], "intent": None}
    elif fault == "resolved_missing": args[6]["resolved_calls"] = []
    elif fault == "resolved_changed":
        args[6]["resolved_calls"] = [c for c in args[6]["resolved_calls"] if c != call]
    elif fault == "emitted_caller": args[2]["calls"][0]["caller"] = "foreign::step"
    elif fault == "emitted_mapping": args[2]["calls"][0]["resource_mappings"]["argument::a"] = "argument::foreign"
    elif fault == "caller_context_graph":
        caller = next(p for p in args[6]["scopes"][0]["checked_execution_plan"]["procedures"]
                      if p["procedure"] == authority["caller"])
        caller["structured_summary_identity"] = "0" * 64
    else: args[6]["scopes"] = []
    # The malformed kind has no matching lookup after mutation.
    prefix["identity"] = _digest({k: v for k, v in prefix.items() if k != "identity"})
    reseal(args)
    with pytest.raises(ValueError):
        check(args)
