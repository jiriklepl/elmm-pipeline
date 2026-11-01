"""Transport a real original position; no runtime manifest is manufactured."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import _digest
from elmm_pipeline.source_position_record import validate_source_position

from compiler.tests.test_original_reached_position import PROCEDURE, position, prepared


@pytest.fixture
def original(tmp_path):
    data = prepared(tmp_path, middle="if(flag)then\ncall second(a,b,n)\nelse\na(:)=b(:)\nendif")
    record = position(data).public()
    graph = data[1].structure(PROCEDURE)
    tasks = [{"id": identity} for identity in graph.nodes]
    group = {"procedure": PROCEDURE, "source_graph": graph.identity,
             "selected_node_ids": record["selected_node_ids"], "sequence_exit": {"source_span": record["source_span"]}}
    return record, group, {"source_inputs": record["source_inputs"]}, {"entry": PROCEDURE}, tasks


def test_real_source_position_transports_complete_root_units_without_exit_authority(original):
    record, group, epoch, plan, tasks = original
    assert validate_source_position(record, group, epoch, plan, original_tasks=tasks) is record
    assert record["exit_authority"] is False
    assert record["execution_authorized"] is False


@pytest.mark.parametrize("fault", ["version_bool", "extra", "flag", "source", "graph", "parent", "selected",
                                  "activation", "policy", "duplicate", "omit", "reverse", "unit", "unit_hash",
                                  "unit_span", "unit_overlap", "span", "empty", "missing_original_task"])
def test_resealed_source_position_cannot_detach_from_original_units_or_acquire_authority(original, fault):
    record, group, epoch, plan, tasks = deepcopy(original)
    if fault == "version_bool":
        record["version"] = True
    elif fault == "extra":
        record["mutable_aliases"] = True
    elif fault == "flag":
        record["exit_authority"] = True
    elif fault == "source":
        record["source_inputs"] = {}
    elif fault == "graph":
        record["source_graph"] = "a"*64
    elif fault == "parent":
        record["parent_sequence"] += "/0/body0"
    elif fault == "selected":
        record["selected_node_ids"] = [record["selected_node_ids"][0]+"/body0/0"]
    elif fault == "activation":
        record["activation_node_id"] = record["selected_node_ids"][-1]
    elif fault == "policy":
        record["prefix_scope"] = "arbitrary reached source"
    elif fault == "duplicate":
        record["selected_node_ids"] *= 2
    elif fault == "omit":
        record["selected_node_ids"].pop(1)
    elif fault == "reverse":
        record["selected_node_ids"].reverse()
    elif fault == "unit":
        record["original_units"].pop()
    elif fault == "unit_hash":
        record["original_units"][0]["original_source_sha256"] = "unqualified"
    elif fault == "unit_span":
        record["original_units"][0]["source_span"][0] = True
    elif fault == "unit_overlap":
        record["original_units"][1]["source_span"] = list(record["original_units"][0]["source_span"])
    elif fault == "span":
        record["source_span"][1] += 1
    elif fault == "empty":
        record["selected_node_ids"] = []
    else:
        selected = record["selected_node_ids"][1]
        tasks[:] = [task for task in tasks if task["id"] != selected and not task["id"].startswith(selected+"/")]
    record["identity"] = _digest({key: value for key, value in record.items() if key != "identity"})
    with pytest.raises(ValueError):
        validate_source_position(record, group, epoch, plan, original_tasks=tasks)
