"""Observed source attribution consumes only the versioned public manifest."""
import copy

import pytest

from elmm_pipeline.gpu_validation import gpu_activity
from elmm_pipeline.scoped_provenance import FIELDS, scoped_provenance


def manifest():
    return {"schema_version": 1, "scope_count": 1, "scopes": [{"mode": "sections"}],
            "runtime_provenance": {"schema_version": 1, "available": True,
                "records": [{"id": str(index + 1) * 64, "kind": "native_operation" if field == "operation" else field,
                             "procedure": "renamed::original"} for index, field in enumerate(FIELDS)]}}


def event(event_name, amount=0, **overrides):
    fields = {"bytes": str(amount), "context": "17", "provenance_version": "1",
              **{field: str(index + 1) * 64 for index, field in enumerate(FIELDS)}}
    if event_name in {"upload", "download"}:
        fields.update(buffer="3", generation="2", buffer_handle="29")
    fields.update(overrides)
    return "FORT_SCOPED " + event_name + " " + " ".join(key + "=" + value for key, value in fields.items()) + "\n"


def test_public_grouping_preserves_actual_totals_and_distinct_contexts():
    public = manifest()
    before = copy.deepcopy(public)
    log = (event("upload", 80) + event("launch") + event("download", 40)
           + event("device_commit") + event("close", context="18")
           + event("position") + "FORT_SCOPED evidence {\"event\":\"operation\"}\n")
    result = scoped_provenance(log, [public])
    assert result["available"]
    assert result["totals"] == {"events": 5, "upload_bytes": 80, "download_bytes": 40, "kernel_launches": 1}
    group, = result["groups"]
    assert group["contexts"] == [17, 18]
    assert group["event_counts"] == {"upload": 1, "download": 1, "launch": 1, "device_commit": 1, "close": 1}
    assert all(value["events"] == 0 for value in result["unattributed_fields"].values())
    assert public == before
    assert result["timing_sample"] is False


def test_unknown_batch_implementation_and_legacy_events_stay_unattributed():
    log = (event("launch", implementation="unknown")
           + event("download", 32, boundary="unknown")
           + "FORT_SCOPED upload buffer=3 generation=2 bytes=64\n")
    result = scoped_provenance(log, [manifest()])
    assert result["versioned_events"] == 2 and result["legacy_events"] == 1
    assert result["totals"]["upload_bytes"] == 64
    assert result["unattributed_fields"]["implementation"]["kernel_launches"] == 1
    assert result["unattributed_fields"]["owner"]["upload_bytes"] == 64
    assert result["unattributed_fields"]["boundary"]["download_bytes"] == 32
    assert len(result["groups"]) == 3
    legacy = scoped_provenance("FORT_SCOPED launch bytes=0\n", [{}])
    assert legacy["available"] is False and legacy["legacy_events"] == 1
    assert legacy["totals"]["kernel_launches"] == 1


def test_identical_shared_records_across_independent_packages_are_reused():
    public = manifest()
    result = gpu_activity(event("upload", 80) + event("launch"), "local", [],
                          scope_packages=[{"scope_manifest": public}, {"scope_manifest": copy.deepcopy(public)}])
    assert result["verified"] and result["upload_bytes"] == 80
    assert result["scoped_runtime"]["source_provenance"]["totals"]["kernel_launches"] == 1
    assert len(result["scoped_runtime"]["source_provenance"]["records"]) == 6


@pytest.mark.parametrize("change,message", [
    (lambda data: data["runtime_provenance"].update(schema_version=2), "version"),
    (lambda data: data["runtime_provenance"]["records"].append(data["runtime_provenance"]["records"][0]), "duplicate"),
    (lambda data: data["runtime_provenance"]["records"][0].update(id="a" * 12), "identity"),
])
def test_invalid_public_authority_is_rejected(change, message):
    public = manifest()
    change(public)
    with pytest.raises(ValueError, match=message):
        scoped_provenance("", [public])


def test_conflicting_package_records_fail_instead_of_choosing_one():
    first, second = manifest(), manifest()
    second["runtime_provenance"]["records"][0]["procedure"] = "different::source"
    with pytest.raises(ValueError, match="conflicting"):
        scoped_provenance(event("launch"), [first, second])


@pytest.mark.parametrize("log,message", [
    (event("launch", owner="a" * 64), "absent"),
    (event("launch", segment="1" * 64), "incorrect kind"),
    (event("launch", operation="1" * 64), "incorrect kind"),
    (event("launch", provenance_version="2"), "version"),
    (event("launch", context="0"), "context"),
    (event("upload", 8, buffer_handle="0"), "buffer_handle"),
    (event("download", bytes="-8"), "bytes"),
    (event("launch").rstrip() + " owner=unknown\n", "malformed"),
    (event("launch").replace("provenance_version=1 ", ""), "version"),
])
def test_corrupt_observed_identity_or_fields_cannot_silently_misattribute(log, message):
    with pytest.raises(ValueError, match=message):
        scoped_provenance(log, [manifest()])


def test_context_creation_before_owner_tag_is_explicitly_unknown():
    log = event("create", **dict.fromkeys(FIELDS, "unknown"))
    result = scoped_provenance(log, [])
    assert result["available"] is False
    assert result["versioned_events"] == 1
    assert result["unattributed_fields"]["owner"]["events"] == 1
    assert result["records"] == []


def test_actual_nonbuffer_event_format_needs_no_buffer_handle():
    log = """FORT_SCOPED create bytes=0 provenance_version=1 context=17 owner=unknown procedure=unknown segment=unknown operation=unknown implementation=unknown boundary=unknown
FORT_SCOPED initialize bytes=0 provenance_version=1 context=17 owner=unknown procedure=unknown segment=unknown operation=unknown implementation=unknown boundary=unknown
FORT_SCOPED launch bytes=0 provenance_version=1 context=17 owner=unknown procedure=unknown segment=unknown operation=unknown implementation=unknown boundary=unknown
FORT_SCOPED wait bytes=0 provenance_version=1 context=17 owner=unknown procedure=unknown segment=unknown operation=unknown implementation=unknown boundary=unknown
FORT_SCOPED close bytes=0 provenance_version=1 context=17 owner=unknown procedure=unknown segment=unknown operation=unknown implementation=unknown boundary=unknown
"""
    result = scoped_provenance(log, [])
    assert result["totals"] == {"events": 5, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 1}
    assert result["unattributed_fields"]["owner"]["events"] == 5
