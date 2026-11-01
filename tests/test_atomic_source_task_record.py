"""Atomic original-team records retain closed source/physical/provider linkage."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)

from compiler.tests.test_atomic_joined_team_source import emit


@pytest.fixture
def issued(tmp_path):
    _outputs, report, _text = emit(tmp_path)
    return report


def task(report):
    return report["scopes"][0]["source_device_tasks"][0]


def reseal(report):
    scope = report["scopes"][0]
    record = task(report)
    team = record["source_proof"]
    team["identity"] = _digest({k: v for k, v in team.items() if k != "identity"})
    for alternative in record["alternatives"]:
        source, effects, reads = (alternative[key] for key in ("source", "physical_effects", "source_reads"))
        source["team_identity"] = team["identity"]
        source["proof_identity"] = _digest({k: v for k, v in source.items() if k != "proof_identity"})
        alternative["preparation"]["completion"] = deepcopy(source)
        effects["proof_identity"] = _digest({k: v for k, v in effects.items() if k != "proof_identity"})
        reads["numerical_effect_identity"] = effects["proof_identity"]
        reads["proof_identity"] = _digest({k: v for k, v in reads.items() if k != "proof_identity"})
        metadata = alternative["provider"]["metadata"]
        metadata["team_identity"] = team["identity"]
        metadata["alternative_identity"] = source["proof_identity"]
        metadata["physical_effect_identity"] = effects["proof_identity"]
        metadata["source_read_identity"] = reads["proof_identity"]
    record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    epoch["members"][0]["source_device_tasks"] = [deepcopy(record)]
    epoch["identity"] = _digest({k: v for k, v in epoch.items() if k != "identity"})
    plan["identity"] = _digest({k: v for k, v in plan.items() if k != "identity"})


def test_real_compiler_atomic_record_round_trips_checked_plan(issued):
    validate_checked_execution_plans(issued)
    assert task(issued)["source_proof"]["alternatives"] == [[0, 1], [2, 3], [2, 4]]
    assert all(a["provider"]["metadata"]["version"] == 2 for a in task(issued)["alternatives"])


@pytest.mark.parametrize("fault", ["version", "authority", "parent_join", "per_unit", "source", "slot", "slot_bool",
    "canonical", "missing_path", "tree", "dependency", "preparation", "effect_authority", "read_authority",
    "unit_tail", "provider_version", "provider_fields", "provider_kind", "parameter_kind", "parameter_axis",
    "artifact", "provider_replay", "unknown", "graph_target", "graph_cpu", "graph_writes"])
def test_resealed_atomic_metadata_cannot_change_authority_abi_or_original_paths(issued, fault):
    record = task(issued)
    team = record["source_proof"]
    variant = record["alternatives"][0]
    provider = variant["provider"]
    metadata = provider["metadata"]
    binding = record["execution_binding"]
    graph = issued["scopes"][0]["graph_execution"]["procedures"][0]
    if fault == "version":
        record["version"] = True
    elif fault == "authority":
        team["definitions_authorized"] = True
    elif fault == "parent_join":
        team["parent_completion"]["join"] = "individual NOWAIT completion"
    elif fault == "per_unit":
        variant["source"]["individual_unit_completion_authorized"] = True
    elif fault == "source":
        team["source_graph"] = "a" * 64
    elif fault == "slot":
        binding["registration_slots"]["argument::a"] += 1
    elif fault == "slot_bool":
        binding["registration_slots"]["argument::a"] = True
    elif fault == "canonical":
        binding["canonical_resources"]["argument::a"] = "unissued::storage"
    elif fault == "missing_path":
        record["alternatives"].pop()
    elif fault == "tree":
        team["tree"] = []
    elif fault == "dependency":
        team["units"][1]["reads"].append("argument::a")
    elif fault == "preparation":
        variant["preparation"]["physical_backend"] = "by_value"
    elif fault == "effect_authority":
        variant["physical_effects"]["input_definitions_authorized"] = True
    elif fault == "read_authority":
        variant["source_reads"]["prefix_definitions_authorized"] = True
    elif fault == "unit_tail":
        variant["source_reads"]["units"].pop()
    elif fault == "provider_version":
        metadata["version"] = True
    elif fault == "provider_fields":
        metadata["fields"].pop()
    elif fault == "provider_kind":
        metadata["fields"][0]["kind"] = 4
    elif fault == "parameter_kind":
        metadata["parameters"][0]["kind"] = "c_double"
    elif fault == "parameter_axis":
        metadata["parameters"][0]["axis"] = True
    elif fault == "artifact":
        provider["cuda_source_sha256"] = metadata["cuda_sha256"] = "b" * 64
    elif fault == "provider_replay":
        metadata["fallback"] = "replay each failed unit"
    elif fault == "unknown":
        variant["preparation"]["new_authority"] = True
    elif fault == "graph_target":
        graph["schedule"]["placements"][0]["target"] = "cpu"
    elif fault == "graph_cpu":
        graph["task_graph"]["tasks"][0]["implementations"] = ["cpu", "gpu"]
    else:
        graph["task_graph"]["tasks"][0]["may_writes"] = []
    reseal(issued)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(issued)
