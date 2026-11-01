"""Observed source attribution consumes only the versioned public manifest."""
import copy
import json

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


def domain_event(name, *, domain=1, context=0, last=0, failed=False, **overrides):
    row = {"schema_version": 1, "event": name, "domain": domain, "context": context,
           "physical_context": domain, "last_context": last, "device": 0, "failed": failed,
           **{field: str(index + 1) * 64 for index, field in enumerate(FIELDS)}}
    row.update(overrides)
    return "FORT_SCOPED domain " + json.dumps(row) + "\n"


def domain_operation(name, amount=0, *, context="2", role="execution", **overrides):
    return event(name, amount, **{"context": context, "domain_version": "1", "domain": "1", "domain_role": role, **overrides})


def test_storage_publication_keeps_explicit_domain_and_two_borrower_identities():
    log = (domain_event("domain_create") + domain_operation("create")
           + domain_event("domain_attach", context=2) + domain_operation("upload", 64)
           + domain_operation("launch") + domain_event("domain_detach", context=2, last=2)
           + domain_operation("create", context="3") + domain_event("domain_attach", context=3, last=2)
           + domain_operation("download", 8, context="3") + domain_operation("launch", context="3")
           + domain_event("domain_detach", context=3, last=3)
           + domain_operation("download", 56, context="1", role="storage")
           + domain_operation("close", context="1", role="storage") + domain_event("domain_retire", last=3))
    result = scoped_provenance(log, [manifest()])
    assert result["totals"] == {"events": 8, "upload_bytes": 64, "download_bytes": 64, "kernel_launches": 2}
    ledger = result["residency_domains"]
    assert ledger["complete"] and ledger["timing_sample"] is False
    domain, = ledger["domains"]
    assert domain["contexts"] == [2, 3] and domain["physical_context"] == 1
    assert domain["execution"] == {"events": 6, "upload_bytes": 64, "download_bytes": 8, "kernel_launches": 2}
    assert domain["storage"] == {"events": 2, "upload_bytes": 0, "download_bytes": 56, "kernel_launches": 0}
    assert domain["terminal"] == "domain_retire"


def test_unfinished_domain_and_poisoned_abandonment_remain_distinct():
    prefix = domain_event("domain_create") + domain_event("domain_attach", context=2)
    assert scoped_provenance(prefix, [manifest()])["residency_domains"]["complete"] is False
    log = (prefix + domain_event("domain_detach", context=2, last=2, failed=True)
           + domain_event("domain_abandon", last=2, failed=True))
    domain, = scoped_provenance(log, [manifest()])["residency_domains"]["domains"]
    assert domain["failed"] and domain["terminal"] == "domain_abandon"
    assert domain["storage"]["download_bytes"] == 0


@pytest.mark.parametrize("tail", [
    domain_event("domain_create"),
    domain_event("domain_attach", context=3),
    domain_event("domain_detach", context=3, last=3),
    domain_event("domain_retire", last=0),
    domain_operation("download", 8, context="1", role="storage"),
    domain_operation("launch", context="3"),
    domain_operation("launch", domain_version="2"),
    domain_event("domain_detach", context=2, last=2, owner="a" * 64),
    domain_event("domain_detach", context=2, last=2, physical_context=5),
    domain_event("domain_detach", context=2, last=2, schema_version=True),
])
def test_invalid_domain_lifetime_or_cross_context_attribution_rejects(tail):
    prefix = domain_event("domain_create") + domain_event("domain_attach", context=2)
    with pytest.raises(ValueError):
        scoped_provenance(prefix + tail, [manifest()])


def test_detached_or_retired_context_cannot_submit_more_copies():
    prefix = (domain_event("domain_create") + domain_event("domain_attach", context=2)
              + domain_event("domain_detach", context=2, last=2))
    for tail in (domain_operation("download", 8),
                 domain_event("domain_retire", last=2) + domain_operation("download", 8, context="1", role="storage")):
        with pytest.raises(ValueError):
            scoped_provenance(prefix + tail, [manifest()])


def test_external_commit_uses_qualified_call_identity_without_inventing_a_kernel():
    public = manifest()
    operation = next(row for row in public["runtime_provenance"]["records"] if row["kind"] == "native_operation")
    operation["kind"] = "external_device_task"
    log = (domain_event("domain_create") + domain_event("domain_attach", context=2)
           + domain_operation("external_commit") + domain_event("domain_detach", context=2, last=2)
           + domain_event("domain_retire", last=2))
    result = scoped_provenance(log, [public])
    assert result["totals"] == {"events": 1, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}
    group, = result["groups"]
    assert group["event_counts"] == {"external_commit": 1}


def test_original_source_task_uses_exact_published_operation_and_provider_identity():
    public = manifest()
    records = public["runtime_provenance"]["records"]
    operation = next(row for row in records if row["kind"] == "native_operation")
    operation.update(kind="source_device_task", proof_identity="a" * 64)
    implementation = next(row for row in records if row["kind"] == "implementation")
    implementation["source_task_provider"] = {"entry": "fort_tile_task_example", "metadata": {"version": 2}}
    assert "backend" not in implementation
    log = event("launch") + event("external_commit")
    result = gpu_activity(log, "local", [], scope_packages=[{"scope_manifest": public}])
    assert result["verified"]
    observed = result["scoped_runtime"]["source_provenance"]
    assert observed["totals"]["kernel_launches"] == 1
    group, = observed["groups"]
    assert group["operation"] == operation["id"]
    assert group["implementation"] == implementation["id"]
    assert group["event_counts"] == {"launch": 1, "external_commit": 1}
    assert operation in observed["records"]
    assert implementation in observed["records"]


@pytest.mark.parametrize("mutation", ["unpublished", "unknown_kind", "wrong_field"])
def test_source_task_cannot_use_unpublished_unknown_or_wrong_field_identity(mutation):
    public = manifest()
    operation = next(row for row in public["runtime_provenance"]["records"] if row["kind"] == "native_operation")
    operation["kind"] = "source_device_task"
    overrides = {}
    if mutation == "unpublished":
        overrides["operation"] = "a" * 64
    elif mutation == "unknown_kind":
        operation["kind"] = "unchecked_source_device_task"
    else:
        overrides["implementation"] = operation["id"]
    with pytest.raises(ValueError, match="absent|incorrect kind"):
        scoped_provenance(event("launch", **overrides), [public])


@pytest.mark.parametrize("in_domain", [False, True])
def test_native_source_transactions_count_bindings_without_inventing_traffic(in_domain):
    public = manifest()
    operation = next(row for row in public["runtime_provenance"]["records"] if row["kind"] == "native_operation")
    operation["kind"] = "source_device_task"
    names = ("source_native_prepare", "source_native_start", "source_native_commit", "source_native_cancel")
    emit = domain_operation if in_domain else event
    log = "".join(emit(name, amount) for name, amount in zip(names, (8, 7, 6, 5), strict=True))
    log += emit("upload", 80) + emit("launch") + emit("download", 40)
    if in_domain:
        log = (domain_event("domain_create") + domain_event("domain_attach", context=2)
               + log + domain_event("domain_detach", context=2, last=2)
               + domain_event("domain_retire", last=2))
    result = scoped_provenance(log, [public])
    expected = {"events": 7, "upload_bytes": 80, "download_bytes": 40, "kernel_launches": 1}
    assert result["totals"] == expected
    group, = result["groups"]
    assert group["event_counts"] == dict.fromkeys((*names, "upload", "launch", "download"), 1)
    assert group["operation"] == operation["id"]
    assert group["contexts"] == ([2] if in_domain else [17])
    if in_domain:
        domain, = result["residency_domains"]["domains"]
        assert domain["execution"] == expected
        assert domain["storage"] == {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}
        assert result["residency_domains"]["complete"]


@pytest.mark.parametrize("name", ["source_native_prepare", "source_native_start", "source_native_commit", "source_native_cancel"])
def test_native_source_transaction_diagnostics_require_published_identity(name):
    with pytest.raises(ValueError, match="absent"):
        scoped_provenance(event(name, 8, operation="a" * 64), [manifest()])


def test_native_source_transaction_cannot_use_a_detached_domain():
    log = (domain_event("domain_create") + domain_event("domain_attach", context=2)
           + domain_event("domain_detach", context=2, last=2)
           + domain_operation("source_native_start", 8))
    with pytest.raises(ValueError, match="detached"):
        scoped_provenance(log, [manifest()])
