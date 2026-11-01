"""A failed GPU protocol remains explicit, without losing native-live integrity."""
from copy import deepcopy

import pytest
from test_source_task_native_live_record import check, reseal, values


@pytest.fixture
def native_only(tmp_path):
    return values(tmp_path, native_only=True)


def test_real_issuer_retains_independent_native_costs_and_unqualified_gpu_flag(native_only):
    check(native_only)
    record = native_only[0]
    pricing = record["pricing"]
    assert record["version"] == 3
    assert pricing["version"] == 3
    assert pricing["gpu_protocol_available"] is False
    assert pricing["native_transaction_costs"]["prepare_start_complete_seconds"] > 0
    assert pricing["source_costs"]["costs"]["source_submit_seconds"] > 0
    assert record["execution_binding"]["provider"]["metadata"]["version"] == 4
    assert record["source_proof"]["pricing"] == "unpriced; explicit sections only"
    assert pricing["execution_authorized"] is False
    assert pricing["whole_owner_estimate_available"] is False


@pytest.mark.parametrize("flag", [True, 0, None, "false"])
def test_native_only_gpu_availability_is_exact_false_not_a_cost_authority(native_only, flag):
    data = deepcopy(native_only)
    data[0]["pricing"]["gpu_protocol_available"] = flag
    reseal(data)
    with pytest.raises(ValueError, match="native-only"):
        check(data)


@pytest.mark.parametrize("mutation", ["missing_flag", "old_live_pricing", "old_wrapper", "old_provider",
                                      "missing_native_costs", "zero_native_cost", "zero_gpu_cost", "extra_key"])
def test_native_only_extension_cannot_relax_historical_versions_costs_or_schema(native_only, mutation):
    data = deepcopy(native_only)
    record, pricing = data[0], data[0]["pricing"]
    if mutation == "missing_flag":
        del pricing["gpu_protocol_available"]
    elif mutation == "old_live_pricing":
        pricing["version"] = 2
    elif mutation == "old_wrapper":
        record.update(version=2, kind="quoted_original_tile_private_team")
    elif mutation == "old_provider":
        record["execution_binding"]["provider"]["metadata"]["version"] = 3
    elif mutation == "missing_native_costs":
        del pricing["native_transaction_costs"]
    elif mutation == "zero_native_cost":
        pricing["native_transaction_costs"]["prepare_start_complete_seconds"] = 0
    elif mutation == "zero_gpu_cost":
        pricing["source_costs"]["costs"]["source_submit_seconds"] = 0
    else:
        pricing["gpu_execution_authorized"] = False
    reseal(data)
    with pytest.raises(ValueError, match="schema|pricing|costs|provider"):
        check(data)
