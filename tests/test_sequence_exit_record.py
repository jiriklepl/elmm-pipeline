"""A portable exit record cannot add source, alias or retention authority."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import _digest
from elmm_pipeline.sequence_exit_record import validate_sequence_exit

from compiler.frontend.source_span_exit import prove_original_sequence_exit
from compiler.tests.test_source_span_exit import SOURCE, fixture


@pytest.fixture(scope="module", params=["else", "endif", "elseif"])
def original(request, tmp_path_factory):
    source = SOURCE
    if request.param == "endif":
        source = source.replace("else\ncall native_other()\n", "")
    elif request.param == "elseif":
        source = source.replace("else\n", "else if(.not.flag)then\n")
    _path, analysis, calls = fixture(tmp_path_factory.mktemp("sequence-exit-reader"), source)
    value = prove_original_sequence_exit(analysis, "groups::run", calls).public()
    plan = {"entry": value["procedure"], "procedures": [
        {"structured_summary_identity": value["source_graph"]}]}
    epoch = {"entry": value["procedure"], "source_inputs": deepcopy(value["source_inputs"])}
    return value, epoch, plan


def test_exact_issued_branch_exit_transports_all_three_original_anchor_kinds(original):
    value, epoch, plan = deepcopy(original)
    assert validate_sequence_exit(value, epoch, plan) is value
    assert value["cross_exit_retention_authorized"] is False
    assert value["entry_definition_authority"] is False


@pytest.mark.parametrize("fault", [
    "missing", "extra", "version_bool", "version_float", "version_old", "kind", "identity",
    "owner", "inputs", "graph", "execution", "definitions", "retention_bool", "aliases", "cleanup",
    "activation", "ordinal_bool", "ordinal_negative", "branch", "parent", "empty", "single", "duplicate",
    "reverse", "gap", "foreign", "noncanonical_ordinal", "last", "span_bool", "span_reverse", "anchor_kind",
    "anchor_extra", "anchor_missing", "anchor_sha", "anchor_bool", "overlapping_anchor", "anchor_container",
])
def test_resealed_exit_cannot_change_original_control_or_grant_authority(original, fault):
    value, epoch, plan = deepcopy(original)
    simple = {
        "version_bool": ("version", True), "version_float": ("version", 1.0), "version_old": ("version", 0),
        "kind": ("kind", "synthetic_close"), "owner": ("procedure", "foreign::entry"),
        "inputs": ("source_inputs", {}), "graph": ("source_graph", "e"*64),
        "execution": ("execution_authorized", True), "definitions": ("entry_definition_authority", True),
        "retention_bool": ("cross_exit_retention_authorized", 0), "aliases": ("alias_authority", True),
        "cleanup": ("cleanup", "close on next invocation"), "activation": ("activation", "before predicate"),
        "ordinal_bool": ("branch_ordinal", True), "ordinal_negative": ("branch_ordinal", -1),
        "branch": ("branch_node_id", "foreign/body/1"), "parent": ("parent_sequence", "groups::run/body/1/body9"),
        "empty": ("selected_node_ids", []), "single": ("selected_node_ids", value["selected_node_ids"][:1]),
        "last": ("last_call_node_id", value["selected_node_ids"][0]),
        "span_bool": ("source_span", [True, 9]), "span_reverse": ("source_span", [9, 7]),
    }
    if fault in simple:
        key, changed = simple[fault]
        value[key] = changed
    elif fault == "missing":
        del value["alias_authority"]
    elif fault == "extra":
        value["payload_authority"] = False
    elif fault == "duplicate":
        value["selected_node_ids"].append(value["selected_node_ids"][-1])
    elif fault == "reverse":
        value["selected_node_ids"].reverse()
        value["last_call_node_id"] = value["selected_node_ids"][-1]
    elif fault == "gap":
        value["selected_node_ids"][1] = value["parent_sequence"]+"/7"
    elif fault == "foreign":
        value["selected_node_ids"][0] = "foreign/body/0"
    elif fault == "noncanonical_ordinal":
        value["selected_node_ids"][0] = value["parent_sequence"]+"/00"
    elif fault == "anchor_kind":
        value["exit_anchor"]["parser_kind"] = "End_Do_Stmt"
    elif fault == "anchor_container":
        value["exit_anchor"]["parser_kind"] = []
    elif fault == "anchor_extra":
        value["exit_anchor"]["execution_authorized"] = True
    elif fault == "anchor_missing":
        del value["exit_anchor"]["original_source_sha256"]
    elif fault == "anchor_sha":
        value["exit_anchor"]["original_source_sha256"] = "missing"
    elif fault == "anchor_bool":
        value["exit_anchor"]["source_span"] = [True, 10]
    elif fault == "overlapping_anchor":
        value["exit_anchor"]["source_span"][0] = value["source_span"][-1]
    value["identity"] = _digest({key: item for key, item in value.items() if key != "identity"})
    if fault == "identity":
        value["identity"] = "0"*64
    with pytest.raises(ValueError, match="local sequence exit"):
        validate_sequence_exit(value, epoch, plan)
