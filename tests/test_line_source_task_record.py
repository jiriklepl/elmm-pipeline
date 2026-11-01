"""Line source records retain their distinct exact shape and direct provider ABI."""
from copy import deepcopy

import pytest

from compiler.tests.test_line_source_tasks import emit_chain
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from elmm_pipeline.source_task_record import _validate_source_authority, validate_source_task


@pytest.fixture(scope="module")
def authentic(tmp_path_factory):
    _path, _outputs, manifest = emit_chain(tmp_path_factory.mktemp("line-reader-source"))
    return manifest


def values(manifest, index=0):
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    member = plan["storage_epoch"]["members"][0]
    procedure = next(p for p in plan["procedures"] if p["procedure"] == member["procedure"])
    return (member["source_device_tasks"][index], member, procedure,
            plan["storage_epoch"]["source_inputs"], {r["resource"]: r for r in plan["storage_epoch"]["resources"]}, plan["entry"], manifest)


def check(args):
    record, member, procedure, inputs, resources, owner, manifest = args
    validate_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest)


def reseal(args):
    record, _member, _procedure, _inputs, _resources, _owner, manifest = args
    team = record["team"]
    team["identity"] = _digest({k: v for k, v in team.items() if k != "identity"})
    record["source_proof_identity"] = _digest({k: v for k, v in record.items()
                                              if k not in {"identity", "source_proof_identity", "execution_binding"}})
    provider = record["execution_binding"]["provider"]
    metadata = provider["metadata"]
    metadata["team_identity"] = team["identity"]
    metadata["unit_proof_identities"] = deepcopy(team["unit_proof_identities"])
    metadata["scalar_readability_identity"] = record["source_proof_identity"]
    suffix = team["identity"][:20] + "_thread_batch_boxes"
    for key, prefix, ext in (("entry", "fort_tile_task_", ".cu"), ("module", "fort_tile_module_", ".f90")):
        old = provider[key]
        provider[key] = metadata[key] = prefix + suffix
        manifest["artifacts_sha256"][provider[key] + ext] = manifest["artifacts_sha256"].pop(old + ext)
    record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})


def test_actual_three_axis_line_chain_round_trips_whole_checked_plan(authentic):
    before = deepcopy(authentic)
    validate_checked_execution_plans(authentic)
    for index in range(3):
        check(values(authentic, index))
    assert authentic == before


@pytest.mark.parametrize("fault", [
    "record_kind", "team_kind", "provider_kind", "team_version", "provider_version", "domain_alias", "width",
    "step", "step_bool", "repeat_axis", "missing_axis", "extra_axis", "iterator_resource", "units_empty",
    "units_many", "complete_numeric", "completion", "order", "team_pricing", "team_cooperative", "prework_missing",
    "prework_unknown", "prework_bool_version", "prework_numeric_false", "prework_integer", "cooperative_mapping",
    "field_descriptor_alias", "scratch_rank", "scratch_missing", "slot_bool", "parameter_order", "artifact_hash",
])
def test_resealed_line_records_cannot_borrow_tile_authority_or_change_exact_contract(authentic, fault):
    args = values(deepcopy(authentic))
    check(args)  # Every corruption starts from an independently accepted authentic record.
    record = args[0]; team = record["team"]
    provider = record["execution_binding"]["provider"]; metadata = provider["metadata"]
    if fault == "record_kind": record["kind"] = "original_tile_private_team"
    elif fault == "team_kind": team["kind"] = "original_tile_private_team"
    elif fault == "provider_kind": metadata["kind"] = "original_tile_private_team"
    elif fault == "team_version": team["version"] = True
    elif fault == "provider_version": metadata["version"] = 8
    elif fault == "domain_alias": team["domain"] = team.pop("transverse_domain")
    elif fault == "width": team["transverse_domain"][0]["width"] = "32"
    elif fault == "step": team["transverse_domain"][0]["step"] = 2
    elif fault == "step_bool": team["transverse_domain"][0]["step"] = True
    elif fault == "repeat_axis": team["transverse_domain"][1] = deepcopy(team["transverse_domain"][0])
    elif fault == "missing_axis": team["transverse_domain"] = []
    elif fault == "extra_axis": team["transverse_domain"].append(deepcopy(team["transverse_domain"][0]))
    elif fault == "iterator_resource": team["transverse_domain"][0]["resource"] = "argument::k"
    elif fault == "units_empty": team["unit_proof_identities"] = []
    elif fault == "units_many": team["unit_proof_identities"].append("1" * 64)
    elif fault == "complete_numeric": team["complete_combined_team"] = 1
    elif fault == "completion": team["completion"] = "caller asserted"
    elif fault == "order": team["source_order"] = "arbitrary fused work"
    elif fault == "team_pricing": team["pricing_authorized"] = True
    elif fault == "team_cooperative": team["cooperative_mapping_authorized"] = True
    elif fault == "prework_missing": metadata.pop("line_snapshot_prework")
    elif fault == "prework_unknown": metadata["line_snapshot_prework"]["unchecked"] = True
    elif fault == "prework_bool_version": metadata["line_snapshot_prework"]["version"] = True
    elif fault == "prework_numeric_false": metadata["line_snapshot_prework"]["initialization_authorized"] = 0
    elif fault == "prework_integer": metadata["line_snapshot_prework"]["integer"] = "caller ranges"
    elif fault == "cooperative_mapping": metadata["mapping"] = "cooperative"
    elif fault == "field_descriptor_alias": team["fields"][0]["binding"] = team["fields"][0].pop("descriptor")
    elif fault == "scratch_rank": team["scratch"][0]["rank"] = 2
    elif fault == "scratch_missing": team["scratch"] = record["scratch"] = []
    elif fault == "slot_bool": record["execution_binding"]["registration_slots"]["argument::a"] = True
    elif fault == "parameter_order": metadata["parameters"][0], metadata["parameters"][1] = metadata["parameters"][1], metadata["parameters"][0]
    else: provider["cuda_source_sha256"] = "0" * 64
    reseal(args)
    with pytest.raises(ValueError): check(args)


def test_line_authority_is_not_available_to_existing_queued_or_priced_common_readers(authentic):
    record, member, procedure, inputs, resources, owner, _ = values(authentic)
    with pytest.raises(ValueError, match="source proof version"):
        _validate_source_authority(record, member, procedure, inputs, resources, owner=owner)


@pytest.mark.parametrize("fault", ["source_kind", "provider_kind", "line_prework"])
def test_authentic_tile_records_cannot_be_retagged_as_line_work(tmp_path, fault):
    from test_source_task_record import _issued, reseal as reseal_tile

    args = _issued(tmp_path, envelopes=True)
    check(args)
    record = args[0]
    metadata = record["execution_binding"]["provider"]["metadata"]
    if fault == "source_kind":
        record["kind"] = "original_line_snapshot_team"
    elif fault == "provider_kind":
        metadata["kind"] = "original_line_snapshot_team"
    else:
        metadata["line_snapshot_prework"] = {"version": 1}
    reseal_tile(record)
    with pytest.raises(ValueError):
        check(args)
