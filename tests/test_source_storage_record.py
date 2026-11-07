"""Storage-only fields cannot acquire definition, motion or pricing authority."""
from copy import deepcopy

import pytest
from elmm_pipeline.source_task_record import _validate_source_authority
from test_source_task_record import _issued, check, reseal


@pytest.fixture
def issued(tmp_path):
    args = _issued(tmp_path, envelopes=True, storage_transport=True)
    check(args)
    return args


def test_storage_runtime_source_and_provider_are_linked(issued):
    record = issued[0]
    assert record["version"] == 3
    assert record["transaction"]["runtime"] == "fort_scope_source_execute_v2"
    metadata = record["execution_binding"]["provider"]["metadata"]
    assert metadata["version"] == 10
    assert metadata["scalar_transport"]["resources"] == []
    assert metadata["field_transport"]["definitions_authorized"] is False


@pytest.mark.parametrize("mutation", ["source_version", "provider_version", "field_definition", "effect_definition",
    "scalar_value", "field_regions", "transaction", "runtime_missing", "runtime_cost", "observation_version",
    "observation_regions", "provider_transaction", "field_source", "field_control", "pricing",
    "runtime_float_version", "observation_float_version"])
def test_resealed_storage_record_cannot_change_its_transport(issued, mutation):
    args = deepcopy(issued)
    record, _, _, _, _, _, manifest = args
    metadata = record["execution_binding"]["provider"]["metadata"]
    if mutation == "source_version":
        record["version"] = 1
    elif mutation == "provider_version":
        metadata["version"] = 7
    elif mutation == "field_definition":
        metadata["field_transport"]["definitions_authorized"] = True
    elif mutation == "effect_definition":
        record["resources"][0]["initialization_authorized"] = True
    elif mutation == "scalar_value":
        metadata["scalar_transport"]["host"] = "read value eagerly"
    elif mutation == "field_regions":
        record["resources"][0]["regions"] = "complete physical root; read before partial writes"
    elif mutation == "transaction":
        record["transaction"]["runtime"] = "fort_scope_source_execute_v1"
    elif mutation == "runtime_missing":
        del manifest["runtime"]["source_storage_tasks"]
    elif mutation == "runtime_cost":
        manifest["runtime"]["source_storage_tasks"]["cost_estimate_available"] = True
    elif mutation == "observation_version":
        manifest["runtime"]["specification_entry"]["abi_version"] = 2
    elif mutation == "observation_regions":
        manifest["runtime"]["specification_entry"]["regions"] = "only initialized cells"
    elif mutation == "runtime_float_version":
        manifest["runtime"]["source_storage_tasks"]["abi_version"] = 2.0
    elif mutation == "observation_float_version":
        manifest["runtime"]["specification_entry"]["abi_version"] = 3.0
    elif mutation == "provider_transaction":
        metadata["transaction"] = "fort_scope_source_execute_v1"
    elif mutation == "field_source":
        metadata["field_transport"]["source_proof_identity"] = "0" * 64
    elif mutation == "field_control":
        metadata["field_transport"]["conditional_reads"] = "unconditional reads"
    else:
        metadata["field_transport"]["pricing_authorized"] = True
    reseal(record)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        check(args)


def test_storage_cannot_enter_the_common_queued_or_priced_authority(issued):
    record, member, procedure, inputs, resources, owner, manifest = issued
    with pytest.raises(ValueError, match="source proof version differs"):
        _validate_source_authority(record, member, procedure, inputs, resources, owner=owner, manifest=manifest)
