"""Transport an actually emitted pressure-style epoch with independent input facts."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from test_supported_path_epoch_record import parts, seal

from compiler.tests.test_supported_path_entry_capture import (
    emit_incoming_supported_path_fixture,
)


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _path, builder, _outputs, report = emit_incoming_supported_path_fixture(
        tmp_path_factory.mktemp("supported-entry-reader"))
    return report, deepcopy(builder.facts)


def reseal(report):
    _scope, _plan, _epoch, path = parts(report)
    entry = path.get("entry_preservation")
    if isinstance(entry, dict):
        projection = entry.get("prefix_projection")
        if isinstance(projection, dict):
            projection["identity"] = _digest({key: value for key, value in projection.items() if key != "identity"})
        entry["identity"] = _digest({key: value for key, value in entry.items() if key != "identity"})
    position = path.get("reached_position")
    if isinstance(position, dict):
        position["identity"] = _digest({key: value for key, value in position.items() if key != "identity"})
    seal(report)


def test_actual_emitted_epoch3_binds_original_prefix_and_keeps_private_storage_out(original):
    report, facts = original
    validate_checked_execution_plans(report, capture_facts=facts)
    scope, _plan, _epoch, path = parts(report)
    assert path["version"] == 3
    assert path == scope["supported_path_epoch"]
    assert path["gate_metadata_edges"] == []
    assert path["entry_preservation"]["preserved_resources"] == ["argument::x"]
    assert path["entry_definition_authority"] is False


@pytest.mark.parametrize("fault", ["version", "missing_entry", "missing_position", "activation", "metadata_null",
                                  "entry_version", "entry_permission", "private_promoted", "position_graph",
                                  "projection_graph", "prefix_omitted", "input_undefined", "gate_link"])
def test_resealed_actual_epoch3_cannot_infer_inputs_or_omit_original_prefix(original, fault):
    report, facts = deepcopy(original)
    _scope, plan, _epoch, path = parts(report)
    entry = path["entry_preservation"]
    projection = entry["prefix_projection"]
    if fault == "version":
        path["version"] = 2
    elif fault == "missing_entry":
        path.pop("entry_preservation")
    elif fault == "missing_position":
        path.pop("reached_position")
    elif fault == "activation":
        path["activation"] = "original reached interval; source gate required before every activation; NONE"
    elif fault == "metadata_null":
        path["gate_metadata_edges"] = None
    elif fault == "entry_version":
        entry["version"] = 3
    elif fault == "entry_permission":
        entry["execution_authorized"] = True
    elif fault == "private_promoted":
        entry["preserved_resources"].append(projection["private_resources"][0])
    elif fault == "position_graph":
        path["reached_position"]["source_graph"] = "a"*64
    elif fault == "projection_graph":
        projection["source_graph"] = "a"*64
    elif fault == "prefix_omitted":
        step = next(step for step in reversed(projection["steps"]) if step["kind"] == "ordinary_native")
        projection["steps"].remove(step)
        entry["segments"] = [segment for segment in entry["segments"]
                             if segment["selected_node_ids"] != [step["source_node_id"]]]
    elif fault == "input_undefined":
        facts["captures"]["argument::x"]["initialized"] = "none"
        plan["capture_facts_identity"] = entry["entry_capture_facts_identity"] = _digest(facts)
    else:
        projection["allocation_gate_identity"] = "a"*64
    reseal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report, capture_facts=facts)
