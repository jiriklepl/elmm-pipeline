"""Quoted source tasks retain independent source and typed cost attachments."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import _digest
from elmm_pipeline.source_task_record import validate_source_task

from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.tests.test_source_task_pricing import issued


@pytest.fixture
def quoted(tmp_path):
    proof, pricing = issued(tmp_path)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, pricing=pricing)
    metadata = provider.public()
    source = proof.public()
    resources = {binding.root: {"resource": binding.root, "descriptor": binding.public(),
                 "device_replica_authorized": True, "registration_identity": index}
                 for index, (binding, _written) in enumerate(proof.fields, 1)}
    record = {"version": 2, "kind": "quoted_original_tile_private_team", "source_proof": source,
              "pricing": pricing.public(), "execution_binding": {
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


def check(values):
    record, member, procedure, inputs, resources, owner, manifest = values
    validate_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest)


def reseal(values):
    """Model an editor who can recalculate every public digest, not an issuer."""
    record = values[0]
    pricing = record["pricing"]
    work = pricing["work"]
    work["identity"] = _digest({k: v for k, v in work.items() if k != "identity"})
    pricing["identity"] = _digest({k: v for k, v in pricing.items() if k != "identity"})
    provider = record["execution_binding"]["provider"]
    old = provider["entry"]
    base = old.partition("_batch_q_")[0]
    provider["entry"] = provider["metadata"]["entry"] = base + "_batch_q_" + pricing["identity"][:10]
    provider["metadata"]["pricing"] = deepcopy(pricing)
    artifacts = values[-1]["artifacts_sha256"]
    if old + ".cu" in artifacts:
        artifacts[provider["entry"] + ".cu"] = artifacts.pop(old + ".cu")
    record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})


def test_real_issued_pricing_preserves_unpriced_source_and_model_limits(quoted):
    check(quoted)
    record = quoted[0]
    assert record["source_proof"]["pricing"] == "unpriced; explicit sections only"
    assert record["pricing"]["whole_owner_estimate_available"] is False
    assert record["pricing"]["full_wall_prediction"] is False
    assert record["pricing"]["execution_authorized"] is False
    assert record["execution_binding"]["provider"]["metadata"]["version"] == 3


@pytest.mark.parametrize("mutation", [
    "profile_artifact", "profile_missing", "profile_hash", "runtime", "runtime_shape", "protocol", "protocol_bool",
    "team", "work_team", "work_unit", "work_domain", "count_bool", "count_negative", "offset_bool",
    "work_mapping", "work_placement", "source_authority", "owner_estimate", "wall_prediction",
    "pricing_execution", "native_rank", "native_units", "native_collapse", "native_seconds",
    "schedule_modifier", "schedule_chunk", "cost_bool", "cost_negative", "rate_mismatch",
    "model", "exclusion", "unknown_field", "work_unknown", "provider_version", "provider_pricing",
])
def test_resealed_quotes_cannot_erase_model_or_source_linkage(quoted, mutation):
    values = deepcopy(quoted)
    record, manifest = values[0], values[-1]
    pricing = record["pricing"]
    work, native = pricing["work"], pricing["native_protocol"]
    provider = record["execution_binding"]["provider"]["metadata"]
    if mutation == "profile_artifact":
        pricing["profile_artifact"] = "/unchecked/hardware.json"
    elif mutation == "profile_missing":
        manifest["artifacts_sha256"].pop(pricing["profile_artifact"])
    elif mutation == "profile_hash":
        manifest["artifacts_sha256"][pricing["profile_artifact"]] = "a" * 64
    elif mutation == "runtime":
        pricing["runtime_id"] = "a" * 64
    elif mutation == "runtime_shape":
        manifest["runtime"] = []
    elif mutation == "protocol":
        pricing["protocol_id"] += 1
    elif mutation == "protocol_bool":
        pricing["protocol_id"] = True
    elif mutation == "team":
        pricing["team_identity"] = "a" * 64
    elif mutation == "work_team":
        work["team_identity"] = "a" * 64
    elif mutation == "work_unit":
        work["units"][0]["source_identity"] = "a" * 64
    elif mutation == "work_domain":
        work["units"][0]["axes"][0]["lower"] = "unchecked_lower"
    elif mutation.startswith("count_"):
        work["units"][0]["producer"]["external_read_bytes"] = True if mutation == "count_bool" else -1
    elif mutation == "offset_bool":
        work["units"][0]["axes"][0]["producer_clip_offset"] = True
    elif mutation == "work_mapping":
        work["mapping"] = "thread" if provider["mapping"] == "cooperative" else "cooperative"
    elif mutation == "work_placement":
        work["placement_authorized"] = True
    elif mutation == "source_authority":
        record["source_proof"]["definitions_authorized"] = True
        source = record["source_proof"]
        source["identity"] = _digest({k: v for k, v in source.items() if k != "identity"})
        provider["scalar_readability_identity"] = source["identity"]
    elif mutation == "owner_estimate":
        pricing["whole_owner_estimate_available"] = True
    elif mutation == "wall_prediction":
        pricing["full_wall_prediction"] = True
    elif mutation == "pricing_execution":
        pricing["execution_authorized"] = True
    elif mutation == "native_rank":
        native["rank"] = 0
    elif mutation == "native_units":
        native["units"] += 1
    elif mutation == "native_collapse":
        native["collapse"] = native["rank"] + 1
    elif mutation == "native_seconds":
        native["seconds"] = -1.0
    elif mutation == "schedule_modifier":
        native["c_schedule"]["monotonic_modifier"] = not native["c_schedule"]["monotonic_modifier"]
    elif mutation == "schedule_chunk":
        native["c_schedule"]["chunk"] += 1
    elif mutation.startswith("cost_"):
        pricing["source_costs"]["costs"]["source_submit_seconds"] = True if mutation == "cost_bool" else -1.0
    elif mutation == "rate_mismatch":
        pricing["rates"]["cpu_flops_per_second"] *= 2
    elif mutation == "model":
        pricing["model"] = "measured complete source-family speedup"
    elif mutation == "exclusion":
        pricing["exclusions"].pop()
    elif mutation == "unknown_field":
        pricing["ignore_errors"] = True
    elif mutation == "work_unknown":
        work["unchecked_ir"] = {}
    elif mutation == "provider_version":
        provider["version"] = 2
    reseal(values)
    if mutation == "provider_pricing":
        provider["pricing"]["profile_identity"] = "b" * 64
        record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})
    with pytest.raises(ValueError):
        check(values)
