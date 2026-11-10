"""Modeled choices retain checked attribution without implying execution."""
import json
from copy import deepcopy

import pytest
from elmm_pipeline.gpu_validation import gpu_activity, scoped_modeled_evidence
from elmm_pipeline.scoped_compact_trace import SOURCE_FIELDS
from elmm_pipeline.scoped_provenance import scoped_provenance


def manifest():
    return {"runtime_provenance": {"schema_version": 1, "records": [
        {"id": str(i + 1) * 64, "kind": "native_operation" if field == "operation" else field}
        for i, field in enumerate(SOURCE_FIELDS)]}}


def pricing(**overrides):
    return {"schema_version": 1, "context": 2, "event": "copy_pricing", "version": 1,
            "evidence": "MODELED", "phase": "before_submission", "confirms_execution": False,
            "buffer": 29, "identity": 3, "generation": 2, "device": 0, "upload": True,
            "decision": "credit_funded_cold", "capacity_bytes": 256 * 1024,
            "direct_seconds": 10., "selected_seconds": 6., "cold_seconds": 6., "reuse_seconds": 2.,
            "prior_credit_seconds": 4., "startup_premium_seconds": 4., "state_revision": 0,
            "state_generation": 1, "model_key": [0, 3, 1, 4602678819172646912] + [0] * 151,
            **{name: str(i + 1) * 64 for i, name in enumerate(SOURCE_FIELDS)}, **overrides}


def evidence(row):
    return "FORT_SCOPED evidence " + json.dumps(row) + "\n"


def test_previous_copy_model_requires_rebuilt_trace():
    row = pricing(model_key=[0, 2, 1, 4602678819172646912] + [0] * 151)
    with pytest.raises(ValueError, match="V3 model key; rebuild older traces"):
        scoped_modeled_evidence(evidence(row))


def lifecycle(event, context=0, last=0):
    row = {"schema_version": 1, "event": event, "domain": 1, "context": context,
           "physical_context": 1, "last_context": last, "device": 0, "failed": False,
           **{field: str(i + 1) * 64 for i, field in enumerate(SOURCE_FIELDS)}}
    return "FORT_SCOPED domain " + json.dumps(row) + "\n"


def test_selected_choices_do_not_create_execution_counts_or_verification():
    public = manifest()
    before = deepcopy(public)
    rows = [pricing(), pricing(decision="locally_profitable_cold", selected_seconds=3., cold_seconds=3.,
                               startup_premium_seconds=1., prior_credit_seconds=0.),
            pricing(decision="reuse", selected_seconds=2., prior_credit_seconds=None)]
    log = "".join(map(evidence, rows))
    activity = gpu_activity(log, "local", [], scope_manifest=public)
    assert activity["scoped_runtime"]["modeled_evidence"] == rows
    source = activity["scoped_runtime"]["source_provenance"]
    assert source["modeled_copy_pricing"] == rows
    assert not source["available"]
    assert source["public_mapping_available"]
    assert source["versioned_events"] == source["legacy_events"] == 0
    assert source["groups"] == []
    assert source["totals"] == {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}
    assert not activity["verified"]
    assert activity["kernel_launches"] == activity["upload_bytes"] == activity["download_bytes"] == 0
    assert activity["allocations"] == 0
    assert "copy_dma_calls" not in activity
    assert public == before


def test_oversized_reuse_has_no_invented_cold_price_or_credit():
    row = pricing(decision="reuse", capacity_bytes=4 * 1024**2, selected_seconds=2.,
                  cold_seconds=None, startup_premium_seconds=None, prior_credit_seconds=None,
                  **dict.fromkeys(SOURCE_FIELDS, "unknown"))
    activity = gpu_activity(evidence(row), "local", [])
    source = activity["scoped_runtime"]["source_provenance"]
    assert source["modeled_copy_pricing"] == [row]
    assert source["records"] == []
    assert source["totals"]["events"] == 0
    assert not activity["verified"]


def test_active_borrower_and_final_storage_attribution_are_checked_without_domain_counts():
    row = pricing(domain_version=1, domain=1, domain_role="execution")
    storage = {**row, "context": 1, "domain_role": "storage"}
    log = (lifecycle("domain_create") + lifecycle("domain_attach", context=2) + evidence(row)
           + lifecycle("domain_detach", context=2, last=2) + evidence(storage)
           + lifecycle("domain_retire", last=2))
    domain, = scoped_provenance(log, [manifest()])["residency_domains"]["domains"]
    zero = {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}
    assert domain["execution"] == domain["storage"] == zero
    assert domain["terminal"] == "domain_retire"
    detached = (lifecycle("domain_create") + lifecycle("domain_attach", context=2)
                + lifecycle("domain_detach", context=2, last=2) + evidence(row))
    with pytest.raises(ValueError, match="detached"):
        scoped_provenance(detached, [manifest()])
    with pytest.raises(ValueError, match="unknown|retired"):
        scoped_provenance(evidence(row), [manifest()])
    with pytest.raises(ValueError, match="device"):
        scoped_provenance(lifecycle("domain_create") + lifecycle("domain_attach", context=2)
                          + evidence({**row, "device": 1, "model_key": [1, *row["model_key"][1:]]}), [manifest()])


@pytest.mark.parametrize(("field", "value"), [
    ("confirms_execution", True), ("confirms_execution", 0), ("evidence", "OBSERVED"),
    ("phase", "after_submission"), ("schema_version", True), ("version", 2),
    ("context", 0), ("buffer", 0), ("identity", True), ("generation", 2**64),
    ("device", -1), ("device", 2**31), ("device", True), ("upload", 1),
    ("state_revision", -1), ("state_generation", 2**64), ("capacity_bytes", 512 * 1024),
    ("model_key", [0] * 155), ("model_key", [0, 3, 1, 4602678819172646912] + [0] * 150),
    ("model_key", [0, 3, 1, 4602678819172646912] + [False] * 151),
    ("direct_seconds", 0), ("direct_seconds", True), ("selected_seconds", float("inf")),
    ("cold_seconds", float("nan")), ("reuse_seconds", -1), ("prior_credit_seconds", 3.),
    ("startup_premium_seconds", 3.), ("decision", "direct"), ("selected_seconds", 2.),
    ("cold_seconds", 4.), ("reuse_seconds", 5.), ("prior_credit_seconds", None),
    ("owner", "invalid"), ("domain", 1), ("bytes", 0), ("operation_count", 1),
    ("direct_seconds", 10**1000),
])
def test_invalid_modeled_price_is_rejected_by_both_readers(field, value):
    log = evidence(pricing(**{field: value}))
    with pytest.raises(ValueError, match="copy pricing|Scoped evidence|finite"):
        scoped_modeled_evidence(log)
    with pytest.raises(ValueError, match="copy pricing|Scoped evidence|finite"):
        scoped_provenance(log, [manifest()])


@pytest.mark.parametrize("field", sorted(pricing()))
def test_closed_schema_requires_every_field(field):
    row = pricing()
    del row[field]
    with pytest.raises(ValueError, match="copy pricing|Scoped evidence|finite"):
        scoped_modeled_evidence(evidence(row))
    if field != "event":
        with pytest.raises(ValueError, match="copy pricing|Scoped evidence|finite"):
            scoped_provenance(evidence(row), [manifest()])


@pytest.mark.parametrize("change", [
    {"decision": "locally_profitable_cold"},
    {"decision": "reuse", "selected_seconds": 2., "prior_credit_seconds": 0.},
    {"decision": "reuse", "selected_seconds": 2., "reuse_seconds": None},
    {"cold_seconds": None, "startup_premium_seconds": None},
    {"reuse_seconds": None, "startup_premium_seconds": None},
    {"decision": "locally_profitable_cold", "selected_seconds": 5., "cold_seconds": 5.,
     "startup_premium_seconds": 3.},
    {"decision": "reuse", "selected_seconds": 5., "reuse_seconds": 5.,
     "startup_premium_seconds": 1., "prior_credit_seconds": None},
])
def test_only_strict_half_saving_or_funded_startup_choices_are_accepted(change):
    with pytest.raises(ValueError, match="copy pricing|Scoped evidence|finite"):
        scoped_modeled_evidence(evidence(pricing(**change)))


@pytest.mark.parametrize("field", SOURCE_FIELDS)
def test_public_source_membership_and_kind_are_required_for_named_attribution(field):
    with pytest.raises(ValueError, match="absent"):
        scoped_provenance(evidence(pricing(**{field: "a" * 64})), [manifest()])
    with pytest.raises(ValueError, match="kind"):
        scoped_provenance(evidence(pricing(**{field: "2" * 64 if field == "owner" else "1" * 64})), [manifest()])


def test_duplicate_fields_reject_and_default_log_report_remains_exact():
    log = evidence(pricing()).rstrip()[:-1] + ',"selected_seconds":6.0}\n'
    with pytest.raises(ValueError, match="duplicate"):
        scoped_modeled_evidence(log)
    with pytest.raises(ValueError, match="duplicate"):
        scoped_provenance(log, [manifest()])
    source = scoped_provenance("", [manifest()])
    assert "modeled_copy_pricing" not in source
    assert source == scoped_provenance("ordinary application text\n", [manifest()])
