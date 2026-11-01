"""Synthetic schema fixtures: copy kernels never verify numerical GPU work."""
import json
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import pytest
from elmm_pipeline.gpu_validation import gpu_activity, scoped_modeled_evidence
from elmm_pipeline.scoped_compact_trace import SOURCE_FIELDS
from elmm_pipeline.scoped_provenance import scoped_provenance


def manifest():
    return {"schema_version": 1, "scope_count": 1, "scopes": [{"mode": "sections"}],
            "runtime_provenance": {"schema_version": 1, "available": True,
                "records": [{"id": str(i + 1) * 64, "kind": "native_operation" if field == "operation" else field}
                            for i, field in enumerate(SOURCE_FIELDS)]}}


def compact(**overrides):
    row = {"schema_version": 1, "context": 17, "event": "compact_transfer", "version": 1,
           "buffer": 29, "identity": 3, "generation": 2, "upload": True, "bytes": 80,
           "logical_rectangles": 4, "physical_operations": 4, "dma_calls": 1,
           "copy_kernel_launches": 1, "word_bytes": 8, "workspace_bytes": 512 * 1024, "reused": False,
           **{name: str(i + 1) * 64 for i, name in enumerate(SOURCE_FIELDS)}}
    return {**row, **overrides}


def evidence(row):
    return "FORT_SCOPED evidence " + json.dumps(row) + "\n"


def ordinary(name, amount=0, **overrides):
    fields = {"buffer": "3", "generation": "2", "bytes": str(amount), "buffer_handle": "29",
              "context": "17", "provenance_version": "1",
              **{field: str(i + 1) * 64 for i, field in enumerate(SOURCE_FIELDS)}, **overrides}
    return "FORT_SCOPED " + name + " " + " ".join(k + "=" + v for k, v in fields.items()) + "\n"


def lifecycle(event, context=0, last=0):
    row = {"schema_version": 1, "event": event, "domain": 1, "context": context,
           "physical_context": 1, "last_context": last, "device": 0, "failed": False,
           **{field: str(i + 1) * 64 for i, field in enumerate(SOURCE_FIELDS)}}
    return "FORT_SCOPED domain " + json.dumps(row) + "\n"


def test_copy_dma_and_kernels_are_counted_once_apart_from_numerical_launches_and_payload():
    public = manifest()
    before = deepcopy(public)
    upload = compact()
    download = compact(upload=False, bytes=392, logical_rectangles=1, physical_operations=49,
                       copy_kernel_launches=2, reused=True)
    log = (evidence(upload) + ordinary("upload", 80) + ordinary("launch")
           + evidence(download) + ordinary("download", 392))
    activity = gpu_activity(log, "local", [], scope_manifest=public)
    assert activity["verified"] and activity["kernel_launches"] == 1
    assert activity["upload_bytes"] == 80 and activity["download_bytes"] == 392
    assert activity["compact_batches"] == 2 and activity["copy_dma_calls"] == 2
    assert activity["copy_kernel_launches"] == 3
    assert activity["compact_physical_operations"] == 53 and activity["compact_logical_rectangles"] == 5
    source = activity["scoped_runtime"]["source_provenance"]
    assert source["versioned_events"] == 5 and source["totals"]["events"] == 5
    group, = source["groups"]
    assert group["event_counts"] == {"compact_transfer": 2, "upload": 1, "launch": 1, "download": 1}
    assert group["copy_kernel_launches"] == 3 and group["kernel_launches"] == 1
    assert activity["scoped_runtime"]["modeled_evidence"] == [upload, download]
    assert public == before


def test_copy_only_log_does_not_verify_numerical_gpu_execution_or_invent_payload():
    row = compact(**dict.fromkeys(SOURCE_FIELDS, "unknown"))
    activity = gpu_activity(evidence(row), "local", [])
    assert not activity["verified"] and activity["kernel_launches"] == 0
    assert activity["upload_bytes"] == activity["download_bytes"] == 0
    assert activity["copy_kernel_launches"] == activity["copy_dma_calls"] == 1
    assert not activity["scoped_runtime"]["source_provenance"]["available"]
    assert activity["scoped_runtime"]["source_provenance"]["unattributed_fields"]["owner"]["copy_dma_calls"] == 1


def test_existing_domain_lifecycle_validates_borrower_and_final_storage_copy_attribution():
    execution = compact(context=2, domain_version=1, domain=1, domain_role="execution")
    storage = compact(context=1, domain_version=1, domain=1, domain_role="storage", upload=False)
    common = {"domain_version": "1", "domain": "1"}
    log = (lifecycle("domain_create") + lifecycle("domain_attach", context=2)
           + evidence(execution) + ordinary("upload", 80, context="2", domain_role="execution", **common)
           + lifecycle("domain_detach", context=2, last=2)
           + evidence(storage) + ordinary("download", 80, context="1", domain_role="storage", **common)
           + lifecycle("domain_retire", last=2))
    result = scoped_provenance(log, [manifest()])
    assert result["residency_domains"]["complete"]
    domain, = result["residency_domains"]["domains"]
    assert domain["execution"]["copy_dma_calls"] == domain["storage"]["copy_dma_calls"] == 1
    assert domain["execution"]["upload_bytes"] == domain["storage"]["download_bytes"] == 80
    assert domain["execution"]["kernel_launches"] == domain["storage"]["kernel_launches"] == 0
    bad = (lifecycle("domain_create") + lifecycle("domain_attach", context=2)
           + lifecycle("domain_detach", context=2, last=2) + evidence(execution))
    with pytest.raises(ValueError, match="detached"):
        scoped_provenance(bad, [manifest()])


@pytest.mark.parametrize("name,value", [
    ("version", True), ("version", 2), ("schema_version", True), ("context", 0), ("buffer", 0),
    ("identity", 0), ("generation", 0), ("bytes", True), ("bytes", -1), ("bytes", 2**64),
    ("logical_rectangles", 5), ("logical_rectangles", 0), ("physical_operations", 1),
    ("physical_operations", 257), ("dma_calls", 2), ("copy_kernel_launches", 2),
    ("word_bytes", 2), ("bytes", 81), ("workspace_bytes", 123), ("workspace_bytes", 2**64),
    ("upload", 1), ("reused", 0), ("owner", "bad"), ("domain", 1), ("unexpected", 0),
])
def test_invalid_physical_batch_is_rejected_by_both_public_readers(name, value):
    log = evidence(compact(**{name: value}))
    with pytest.raises(ValueError):
        scoped_modeled_evidence(log)
    with pytest.raises(ValueError):
        scoped_provenance(log, [manifest()])


@pytest.mark.parametrize("name", sorted(compact()))
def test_every_required_batch_field_is_required(name):
    row = compact()
    del row[name]
    # Missing the event discriminator cannot be interpreted as this event by
    # the standalone source reader; the enclosing evidence reader still rejects.
    with pytest.raises(ValueError):
        scoped_modeled_evidence(evidence(row))
    if name != "event":
        with pytest.raises(ValueError):
            scoped_provenance(evidence(row), [manifest()])


@pytest.mark.parametrize("field", ["owner", "operation"])
def test_compact_source_identities_require_public_catalog_membership_and_kinds(field):
    unknown = compact(**{field: "a" * 64})
    with pytest.raises(ValueError, match="absent"):
        scoped_provenance(evidence(unknown), [manifest()])
    wrong = compact(**{field: "2" * 64 if field == "owner" else "1" * 64})
    with pytest.raises(ValueError, match="kind"):
        scoped_provenance(evidence(wrong), [manifest()])


def test_duplicate_compact_json_fields_cannot_bypass_standalone_attribution_validation():
    log = evidence(compact()).rstrip()[:-1] + ',"physical_operations":4}\n'
    with pytest.raises(ValueError, match="duplicate"):
        scoped_provenance(log, [manifest()])
    with pytest.raises(ValueError, match="duplicate"):
        scoped_modeled_evidence(log)


def test_modeled_copy_transport_is_not_observed_execution():
    row = {"schema_version": 1, "context": 5, "event": "copy", "phase": "read", "unit": 9,
           "direction": "h2d", "bytes": 392, "copy_calls": 1, "transport": "compact",
           "physical_operations": 49, "copy_kernel_launches": 2}
    result = gpu_activity(evidence(row), "local", [])
    assert result["scoped_runtime"]["modeled_evidence"] == [row]
    assert not result["verified"] and result["upload_bytes"] == result["kernel_launches"] == 0
    assert "copy_kernel_launches" not in result
    for change in ({"copy_calls": 2}, {"transport": "other"}, {"physical_operations": True},
                   {"copy_kernel_launches": 1}, {"bytes": 0}):
        with pytest.raises(ValueError):
            scoped_modeled_evidence(evidence({**row, **change}))
    for key in ("transport", "physical_operations", "copy_kernel_launches"):
        missing = {k: v for k, v in row.items() if k != key}
        with pytest.raises(ValueError):
            scoped_modeled_evidence(evidence(missing))


def test_actual_qualified_runtime_trace_retains_physical_copies_without_double_counting():
    directory = Path(__file__).parent / "fixtures"
    raw = (directory / "compact_transfer_v1.log").read_bytes()
    metadata = json.loads((directory / "compact_transfer_v1.json").read_text())
    assert sha256(raw).hexdigest() == metadata["trace_sha256"]
    assert metadata["performance_or_application_claim"] is False
    activity = gpu_activity(raw.decode(), "local", [])
    assert activity["verified"] and activity["kernel_launches"] == 1
    assert activity["upload_bytes"] == 1440 and activity["download_bytes"] == 12240
    assert activity["copy_dma_calls"] == activity["copy_kernel_launches"] == activity["compact_batches"] == 3
    assert activity["compact_physical_operations"] == activity["compact_logical_rectangles"] == 13
    records = [r for r in activity["scoped_runtime"]["modeled_evidence"] if r["event"] == "compact_transfer"]
    assert [r["upload"] for r in records] == [False, True, False]
    assert not activity["scoped_runtime"]["source_provenance"]["available"]
