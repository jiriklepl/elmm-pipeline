"""Access envelopes are a closed unpriced provider contract, never authority."""

from copy import deepcopy

import pytest
from test_source_task_record import _issued, check, reseal


@pytest.fixture
def enveloped(tmp_path):
    return _issued(tmp_path, envelopes=True)


def test_actual_issued_provider_retains_conservative_graph_effects(enveloped):
    check(enveloped)
    record = enveloped[0]
    assert record["execution_binding"]["provider"]["metadata"]["version"] == 7
    assert all(effect["regions"] == "complete physical root; read before partial writes"
               and effect["initialization_authorized"] is False for effect in record["resources"])
    assert record["definitions_authorized"] is record["placement_authorized"] is False


@pytest.mark.parametrize(("key", "value"), [
    ("version", True), ("version", 1.0), ("version", 2),
    ("source", "caller-supplied boxes"), ("reads", "complete physical root"),
    ("writes", "exact writes; overwrite without preservation"),
    ("overwrites_authorized", True), ("overwrites_authorized", 0),
    ("initialization_authorized", True), ("initialization_authorized", 0),
    ("preflight", "after submission"), ("pricing", "measured"),
])
def test_resealed_envelopes_cannot_expand_authority_or_change_protocol(enveloped, key, value):
    values = deepcopy(enveloped)
    values[0]["execution_binding"]["provider"]["metadata"]["access_envelopes"][key] = value
    reseal(values[0])
    with pytest.raises(ValueError):
        check(values)


@pytest.mark.parametrize("mutation", ["missing", "extra", "old_version", "batch", "symbol", "foreign_team"])
def test_resealed_access_contract_remains_bound_to_provider_and_batch(enveloped, mutation):
    values = deepcopy(enveloped)
    provider = values[0]["execution_binding"]["provider"]
    metadata = provider["metadata"]
    if mutation == "missing":
        metadata.pop("access_envelopes")
    elif mutation == "extra":
        metadata["access_envelopes"]["finite_authorized"] = True
    elif mutation == "old_version":
        metadata["version"] = 2
    elif mutation == "batch":
        metadata["batching"]["target_bytes"] *= 2
    elif mutation == "symbol":
        provider["entry"] = metadata["entry"] = provider["entry"].removesuffix("_boxes")
    else:
        metadata["team_identity"] = "f" * 64
    reseal(values[0])
    with pytest.raises(ValueError):
        check(values)


def test_historical_provider_schema_cannot_smuggle_new_access_fields(tmp_path):
    values = _issued(tmp_path)
    values[0]["execution_binding"]["provider"]["metadata"]["access_envelopes"] = {}
    reseal(values[0])
    with pytest.raises(ValueError):
        check(values)
