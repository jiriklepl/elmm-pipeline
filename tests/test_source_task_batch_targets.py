"""Provider8 binds a finite experiment target; historical batching stays exact."""
from copy import deepcopy
from functools import partial

import pytest
import test_source_task_record as original

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider


@pytest.fixture(params=[4194304, 16777216, 67108864])
def batched(request, tmp_path, monkeypatch):
    # The existing real-source fixture issues the proof and all canonical bindings.
    monkeypatch.setattr(original, "emit_tile_task_provider",
                        partial(emit_tile_task_provider, batch_target_bytes=request.param))
    return original._issued(tmp_path, envelopes=True)


def test_actual_compiler_provider8_round_trips_with_exact_target_namespace(batched):
    original.check(batched)
    metadata = batched[0]["execution_binding"]["provider"]["metadata"]
    assert metadata["version"] == 8 and metadata["batching"]["version"] == 2
    assert metadata["entry"].endswith(f"_batch_b{metadata['batching']['target_bytes'] // 1048576}m_boxes")
    assert metadata["definitions_authorized"] is metadata["placement_authorized"] is False
    assert metadata["access_envelopes"]["overwrites_authorized"] is False
    assert metadata["access_envelopes"]["initialization_authorized"] is False


@pytest.mark.parametrize(("key", "value"), [
    ("version", 1), ("version", 3), ("version", True), ("version", 2.0),
    ("target_bytes", True), ("target_bytes", 4194304.0), ("target_bytes", "4194304"),
    ("target_bytes", 0), ("target_bytes", 8388608), ("target_bytes", 134217728),
    ("minimum_tiles", 0), ("minimum_tiles", True), ("planned_launches", "one per unit"),
])
def test_resealed_nonfinite_or_changed_batch_protocol_is_rejected(batched, key, value):
    values = deepcopy(batched)
    values[0]["execution_binding"]["provider"]["metadata"]["batching"][key] = value
    original.reseal(values[0])
    with pytest.raises(ValueError):
        original.check(values)


@pytest.mark.parametrize("mutation", [
    "target_name", "module", "missing_envelopes", "version7", "version9", "extra_batch",
    "initialized", "overwrite", "priced", "artifact", "team", "unit",
])
def test_resealed_provider_remains_bound_to_target_source_team_and_false_authority(batched, mutation):
    values = deepcopy(batched)
    record = values[0]
    provider = record["execution_binding"]["provider"]
    metadata = provider["metadata"]
    if mutation == "target_name":
        old = metadata["batching"]["target_bytes"]
        metadata["batching"]["target_bytes"] = 16777216 if old != 16777216 else 67108864
    elif mutation == "module":
        provider["module"] = metadata["module"] = provider["module"] + "x"
    elif mutation == "missing_envelopes":
        metadata.pop("access_envelopes")
    elif mutation == "version7":
        metadata["version"] = 7
    elif mutation == "version9":
        metadata["version"] = 9
    elif mutation == "extra_batch":
        metadata["batching"]["max_tiles"] = 288
    elif mutation == "initialized":
        metadata["access_envelopes"]["initialization_authorized"] = True
    elif mutation == "overwrite":
        metadata["access_envelopes"]["overwrites_authorized"] = True
    elif mutation == "priced":
        metadata["pricing"] = "measured"
    elif mutation == "artifact":
        values[-1]["artifacts_sha256"][provider["entry"] + ".cu"] = "f" * 64
    elif mutation == "team":
        metadata["team_identity"] = "f" * 64
    else:
        metadata["unit_proof_identities"] = ["f" * 64]
    original.reseal(record)
    with pytest.raises(ValueError):
        original.check(values)


def test_legacy_provider7_cannot_adopt_batch2_or_another_target(tmp_path):
    values = original._issued(tmp_path, envelopes=True)
    original.check(values)
    for key, value in (("version", 2), ("target_bytes", 16777216)):
        changed = deepcopy(values)
        changed[0]["execution_binding"]["provider"]["metadata"]["batching"][key] = value
        original.reseal(changed[0])
        with pytest.raises(ValueError):
            original.check(changed)
