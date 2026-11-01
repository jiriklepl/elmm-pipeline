"""Check real issued entry records; these fixtures do not issue runtime authority."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import _digest
from elmm_pipeline.reached_entry_preservation_record import (
    validate_reached_entry_preservation,
)

from compiler.tests import test_original_reached_position as root_position
from compiler.tests import test_private_entry_prefix as private_prefix


def _parts(proof, analysis, facts):
    position = proof._position.public()
    record = proof.public()
    graph = analysis.structure(record["procedure"])
    # Original source IDs use the diagnostic graph's branch-container grammar.
    # This is a linkage fixture, not an emitted execution manifest.
    tasks = [{"id": identity if node.kind != "branch" else identity+":alternative:0"}
             for identity, node in graph.nodes.items() if node.kind not in {"entry", "sequence"}]
    roots = record["preserved_resources"]
    group = {"procedure": record["procedure"], "source_graph": graph.identity,
             "selected_node_ids": position["selected_node_ids"],
             "sequence_exit": {"source_span": position["source_span"]},
             "resources": roots, "descriptor_guard_resources": []}
    epoch = {"source_inputs": record["source_inputs"], "resources": [
        {"resource": root, "descriptor": {"type": "real", "rank": 1, "intent": "inout", "attributes": []}}
        for root in roots]}
    plan = {"entry": record["procedure"], "capture_facts_identity": _digest(facts)}
    keywords = {"original_tasks": tasks, "capture_facts": facts}
    if record["version"] == 4:
        keywords.update(gate_identity=proof._projection._gate.identity,
                        private_resources=record["prefix_projection"]["private_resources"])
    return record, position, group, epoch, plan, keywords


def _issued(kind, tmp_path_factory):
    path = tmp_path_factory.mktemp("reached-entry-reader")
    if kind == "root":
        parts = root_position.prepared(path)
        proof = root_position.preserved(parts)
        return _parts(proof, parts[1], parts[4])
    data = private_prefix.prepared(path, clock=kind == "clock_optional", optional=kind == "clock_optional")
    proof, _originals, _bindings, facts = private_prefix.entry(data)
    return _parts(proof, data[0][1], facts)


@pytest.fixture(scope="module", params=["root", "private", "clock_optional"])
def original(request, tmp_path_factory):
    return _issued(request.param, tmp_path_factory)


@pytest.fixture(scope="module", params=["private", "clock_optional"])
def private_original(request, tmp_path_factory):
    return _issued(request.param, tmp_path_factory)


def check(parts):
    record, position, group, epoch, plan, keywords = parts
    return validate_reached_entry_preservation(record, position, group, epoch, plan, **keywords)


def reseal(record):
    record["identity"] = _digest({key: value for key, value in record.items() if key != "identity"})


def test_real_issued_root_entry_records_preserve_only_independent_definitions(original):
    assert check(original) is original[0]
    assert original[0]["new_definition_authority"] is False


@pytest.mark.parametrize("fault", ["version_bool", "extra", "source", "position", "facts", "authority",
                                  "prefix_omission", "segment_proof", "unknown_root", "out", "pointer",
                                  "undefined", "changed_fact_identity"])
def test_resealed_entry_records_cannot_bypass_original_prefix_or_independent_facts(original, fault):
    parts = deepcopy(original)
    record, _position, group, epoch, plan, keywords = parts
    if fault == "version_bool":
        record["version"] = True
    elif fault == "extra":
        record["replica_is_current"] = True
    elif fault == "source":
        record["source_inputs"] = {}
    elif fault == "position":
        record["reached_position_identity"] = "a"*64
    elif fault == "facts":
        record["entry_capture_facts_identity"] = "a"*64
    elif fault == "authority":
        record["new_definition_authority"] = True
    elif fault == "prefix_omission":
        record["segments"].pop(0)
    elif fault == "segment_proof":
        record["segments"][0]["preservation_identity"] = "unknown"
    elif fault == "unknown_root":
        record["preserved_resources"] = ["unknown::array"]
    elif fault == "out":
        epoch["resources"][0]["descriptor"]["intent"] = "out"
    elif fault == "pointer":
        epoch["resources"][0]["descriptor"]["attributes"] = ["pointer"]
    else:
        root = group["resources"][0]
        keywords["capture_facts"]["captures"][root]["initialized"] = "none"
        if fault == "undefined":
            plan["capture_facts_identity"] = record["entry_capture_facts_identity"] = _digest(keywords["capture_facts"])
    reseal(record)
    with pytest.raises(ValueError):
        check(parts)


@pytest.mark.parametrize("fault", ["version", "flag", "gate", "private_promoted", "step_node", "step_duplicate",
                                  "step_omission", "step_reverse", "structural_node", "declaration", "segments",
                                  "kind_list", "kind_dict", "missing_end", "outside_reclassified", "arm_reclassified"])
def test_resealed_private_projection_cannot_drop_work_or_promote_private_storage(private_original, fault):
    parts = deepcopy(private_original)
    record, _position, _group, _epoch, _plan, _keywords = parts
    projection = record["prefix_projection"]
    if fault == "version":
        projection["version"] = 2
    elif fault == "flag":
        projection["execution_authorized"] = True
    elif fault == "gate":
        projection["allocation_gate_identity"] = "a"*64
    elif fault == "private_promoted":
        record["preserved_resources"].append(projection["private_resources"][0])
    elif fault == "step_node":
        projection["steps"][0]["source_node_id"] = "foreign::procedure/body/0"
    elif fault == "step_duplicate":
        projection["steps"].append(deepcopy(projection["steps"][0]))
    elif fault == "step_omission":
        projection["steps"].pop(0)
    elif fault == "step_reverse":
        projection["steps"].reverse()
    elif fault == "structural_node":
        next(step for step in projection["steps"] if step["kind"] == "original_end_if")["source_node_id"] = "invented"
    elif fault == "declaration":
        projection["local_declaration_identities"].reverse()
    elif fault in {"kind_list", "kind_dict"}:
        projection["steps"][0]["kind"] = [] if fault == "kind_list" else {}
    elif fault == "missing_end":
        projection["steps"] = [step for step in projection["steps"] if step["kind"] != "original_end_if"]
    elif fault in {"outside_reclassified", "arm_reclassified"}:
        native = [step for step in projection["steps"] if step["kind"] == "ordinary_native"]
        step = native[-1] if fault == "outside_reclassified" else native[1]
        step["kind"] = "fresh_private_literal_overwrite"
        record["segments"] = [segment for segment in record["segments"]
                              if segment["selected_node_ids"] != [step["source_node_id"]]]
    else:
        projection["original_segments"][0].pop(0)
    reseal(projection)
    reseal(record)
    with pytest.raises(ValueError):
        check(parts)
