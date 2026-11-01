"""Native-live cost transport remains distinct from historical full-close costs."""
from copy import deepcopy
from hashlib import sha256

import pytest
from elmm_pipeline.checked_execution_plan import _digest
from elmm_pipeline.source_task_record import validate_source_task

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.offload.source_task_pricing import source_task_pricing
from compiler.tests.test_source_native_pricing import live_issued
from compiler.tests.test_source_task_pricing import issued


def values(directory, *, live=True, native_only=False):
    # These compiler-issued synthetic observations test schema/integrity only;
    # they do not attest measured costs or runtime execution.
    proof, pricing = live_issued(directory) if live else issued(directory)
    if native_only:
        assert live
        profile = pricing.profile()
        profile["source_tasks"]["available"] = False
        pricing = source_task_pricing(proof.team, profile, pricing.public()["runtime_id"], native_live=True)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    metadata, source = provider.public(), proof.public()
    resources = {binding.root: {"resource": binding.root, "descriptor": binding.public(),
                 "device_replica_authorized": True, "registration_identity": index}
                 for index, (binding, _written) in enumerate(proof.fields, 1)}
    record = {"version": 3 if live else 2,
              "kind": "quoted_original_tile_private_team_live" if live else "quoted_original_tile_private_team",
              "source_proof": source, "pricing": pricing.public(), "execution_binding": {
                  "owner": source["procedure"], "canonical_resources": {root: root for root in resources},
                  "registration_slots": {root: item["registration_identity"] for root, item in resources.items()},
                  "provider": {"entry": provider.entry, "module": provider.module,
                      "cuda_source_sha256": metadata["cuda_sha256"],
                      "fortran_source_sha256": metadata["fortran_sha256"], "metadata": metadata}}}
    record["identity"] = _digest(record)
    manifest = {"runtime": {"runtime_id": pricing.public()["runtime_id"]}, "artifacts_sha256": {
        provider.entry + ".cu": metadata["cuda_sha256"], provider.module + ".f90": metadata["fortran_sha256"],
        pricing.public()["profile_artifact"]: pricing.public()["profile_identity"]}}
    member = {"procedure": source["procedure"], "source_graph": source["source_graph"]}
    procedure = {"procedure": source["procedure"], "calls": []}
    return record, member, procedure, proof.team._analysis.inputs.identity(), resources, source["procedure"], manifest


def check(data):
    record, member, procedure, inputs, resources, owner, manifest = data
    validate_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest)


@pytest.fixture
def live(tmp_path):
    return values(tmp_path)


def reseal(data):
    """Rehash public envelopes without obtaining source or measurement authority."""
    record = data[0]
    pricing = record["pricing"]
    pricing["work"]["identity"] = _digest({k: v for k, v in pricing["work"].items() if k != "identity"})
    pricing["identity"] = _digest({k: v for k, v in pricing.items() if k != "identity"})
    provider = record["execution_binding"]["provider"]
    old, metadata = provider["entry"], provider["metadata"]
    entry = old.partition("_batch_q_")[0] + "_batch_q_" + pricing["identity"][:10]
    if metadata.get("native_prepare_entry") == "fort_tile_native_" + sha256(old.encode()).hexdigest()[:20]:
        metadata["native_prepare_entry"] = "fort_tile_native_" + sha256(entry.encode()).hexdigest()[:20]
    provider["entry"] = metadata["entry"] = entry
    metadata["pricing"] = deepcopy(pricing)
    artifacts = data[-1]["artifacts_sha256"]
    if old + ".cu" in artifacts:
        artifacts[entry + ".cu"] = artifacts.pop(old + ".cu")
    record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})


def test_real_issued_live_wrapper_retains_pure_source_and_native_protocol_limits(live):
    check(live)
    record = live[0]
    assert record["version"] == 3
    assert record["pricing"]["version"] == 2
    assert record["execution_binding"]["provider"]["metadata"]["version"] == 4
    assert record["source_proof"]["version"] == 1
    assert record["source_proof"]["pricing"] == "unpriced; explicit sections only"
    assert record["pricing"]["endpoints"]["prefix_fence"] == (
        "GPU-only pre-scratch wait after choice; native acquisition waits exact conflicts")
    assert record["pricing"]["whole_owner_estimate_available"] is False
    assert record["pricing"]["execution_authorized"] is False


@pytest.mark.parametrize("fault", [
    "wrapper_version", "wrapper_kind", "pricing_version", "provider_version", "missing_native_costs", "extra_native_cost",
    "native_version_bool", "native_protocol_bool", "native_protocol", "native_bindings_bool", "native_bindings_zero",
    "native_bindings_outside_protocol", "native_seconds_bool", "native_seconds_negative", "native_seconds_zero",
    "native_endpoint", "gpu_endpoint", "prefix_fence", "native_entry", "native_entry_type", "api_missing",
    "api_unknown", "api_substitution", "whole_owner", "execution_authority",
])
def test_resealed_live_records_cannot_change_endpoints_typed_costs_or_native_abi(live, fault):
    data = deepcopy(live)
    record = data[0]
    pricing, metadata = record["pricing"], record["execution_binding"]["provider"]["metadata"]
    native = pricing["native_transaction_costs"]
    if fault == "wrapper_version":
        record["version"] = 2
    elif fault == "wrapper_kind":
        record["kind"] = "quoted_original_tile_private_team"
    elif fault == "pricing_version":
        pricing["version"] = 1
    elif fault == "provider_version":
        metadata["version"] = 3
    elif fault == "missing_native_costs":
        del pricing["native_transaction_costs"]
    elif fault == "extra_native_cost":
        native["ignored"] = 0
    elif fault == "native_version_bool":
        native["version"] = True
    elif fault == "native_protocol_bool":
        native["protocol_id"] = True
    elif fault == "native_protocol":
        native["protocol_id"] += 1
    elif fault == "native_bindings_bool":
        native["max_bindings"] = True
    elif fault == "native_bindings_zero":
        native["max_bindings"] = 0
    elif fault == "native_bindings_outside_protocol":
        native["max_bindings"] = 9
    elif fault.startswith("native_seconds"):
        native["prepare_start_complete_seconds"] = {
            "native_seconds_bool": True, "native_seconds_negative": -1.0, "native_seconds_zero": 0.0}[fault]
    elif fault == "native_endpoint":
        pricing["endpoints"]["native"] = "complete context/domain close then original team"
    elif fault == "gpu_endpoint":
        pricing["endpoints"]["gpu"] = "source transaction; domain remains live; complete close is only a ranking projection"
    elif fault == "prefix_fence":
        pricing["endpoints"]["prefix_fence"] = "existing pre-scratch wait before quote; common incurred term excluded"
    elif fault == "native_entry":
        metadata["native_prepare_entry"] = "unchecked_native_entry"
    elif fault == "native_entry_type":
        metadata["native_prepare_entry"] = []
    elif fault == "api_missing":
        del metadata["native_transaction_api"]["cancel"]
    elif fault == "api_unknown":
        metadata["native_transaction_api"]["ignore"] = "unchecked"
    elif fault == "api_substitution":
        metadata["native_transaction_api"]["start"] = "fort_scope_access_begin_v1"
    elif fault == "whole_owner":
        pricing["whole_owner_estimate_available"] = True
    else:
        pricing["execution_authorized"] = True
    reseal(data)
    with pytest.raises(ValueError, match="source"):
        check(data)


def test_historical_full_close_attachment_never_accepts_native_live_extension(tmp_path):
    data = values(tmp_path, live=False)
    check(data)
    data[0]["pricing"]["native_transaction_costs"] = {
        "version": 1, "protocol_id": 0x465453484f530001, "max_bindings": 8, "prepare_start_complete_seconds": 1e-6}
    reseal(data)
    with pytest.raises(ValueError, match="source"):
        check(data)
