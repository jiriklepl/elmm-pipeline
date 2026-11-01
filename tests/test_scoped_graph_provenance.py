"""Graph API calls and successfully submitted members retain distinct origins."""
from pathlib import Path

import pytest
from elmm_pipeline.gpu_validation import gpu_activity
from elmm_pipeline.scoped_provenance import scoped_provenance
from test_scoped_provenance import domain_event, domain_operation, event, manifest


def public_origins():
    public = manifest()
    public["runtime_provenance"]["records"] += [
        {"id": "7" * 64, "kind": "owner"},
        {"id": "8" * 64, "kind": "implementation"},
        {"id": "9" * 64, "kind": "source_device_task"},
    ]
    return public


def test_queued_members_keep_source_ids_without_inflating_direct_launches():
    log = (event("launch") + event("graph_submit", 2)
           + event("graph_kernel", owner="7" * 64, implementation="8" * 64, operation="9" * 64)
           + event("graph_kernel", boundary="unknown"))
    result = scoped_provenance(log, [public_origins()])
    assert result["totals"] == {"events": 4, "upload_bytes": 0, "download_bytes": 0,
                                "kernel_launches": 1, "graph_submissions": 1,
                                "graph_kernel_submissions": 2}
    original = next(row for row in result["groups"] if row["owner"] == "7" * 64)
    assert original["operation"] == "9" * 64
    assert original["implementation"] == "8" * 64
    assert original["event_counts"] == {"graph_kernel": 1}
    assert original["kernel_launches"] == 0
    assert original["graph_kernel_submissions"] == 1
    assert result["unattributed_fields"]["boundary"]["graph_kernel_submissions"] == 1
    assert not result["timing_sample"]


def test_interleaved_execution_contexts_preserve_each_complete_member_block():
    log = (event("graph_submit", 2) + event("graph_submit", 1, context="18")
           + event("graph_kernel") + event("graph_kernel", context="18")
           + event("wait", context="18") + event("graph_kernel") + event("wait"))
    result = scoped_provenance(log, [manifest()])
    assert result["totals"]["graph_submissions"] == 2
    assert result["totals"]["graph_kernel_submissions"] == 3
    assert result["totals"]["kernel_launches"] == 0
    assert result["groups"][0]["contexts"] == [17, 18]


def test_historical_aggregates_do_not_invent_member_execution():
    log = (event("graph_submit", 3) + event("wait") + event("graph_submit", 7)
           + "FORT_SCOPED graph_submit bytes=2\n")
    result = scoped_provenance(log, [manifest()])
    assert result["totals"]["graph_submissions"] == 3
    assert "graph_kernel_submissions" not in result["totals"]
    assert result["totals"]["kernel_launches"] == 0
    assert result["legacy_events"] == 1


@pytest.mark.parametrize("log,message", [
    (event("graph_kernel"), "matching observed submission"),
    (event("graph_submit", 1) + event("graph_kernel") + event("graph_kernel"), "matching observed submission"),
    (event("graph_submit", 2) + event("graph_kernel"), "incomplete"),
    (event("graph_submit", 2) + event("graph_kernel") + event("wait"), "incomplete"),
    (event("graph_submit", 2) + event("graph_kernel") + event("position"), "incomplete"),
    (event("graph_submit", 1) + event("position") + event("graph_kernel"), "matching observed submission"),
    (event("graph_submit", 2) + event("graph_kernel") + event("graph_submit", 1), "incomplete"),
    (event("graph_submit", 1) + event("graph_kernel", context="18"), "matching observed submission"),
    (event("graph_submit", 1) + event("graph_kernel", 1), "bytes must be zero"),
    (event("graph_submit", 0), "positive member count"),
    ("FORT_SCOPED graph_kernel bytes=0\n", "requires runtime provenance"),
    (event("graph_submit", 1) + event("graph_kernel", provenance_version="2"), "version"),
    (event("graph_submit", 1) + event("graph_kernel", implementation="a" * 64), "absent"),
    (event("graph_submit", 1) + event("graph_kernel", implementation="1" * 64), "incorrect kind"),
    (event("graph_submit", 1) + event("graph_kernel", context="0"), "context"),
])
def test_invalid_or_partial_member_evidence_rejects(log, message):
    with pytest.raises(ValueError, match=message):
        scoped_provenance(log, [manifest()])


def test_graph_member_submissions_preserve_domain_and_borrower():
    log = (domain_event("domain_create") + domain_event("domain_attach", context=2)
           + domain_operation("graph_submit", 2) + domain_operation("graph_kernel")
           + domain_operation("graph_kernel") + domain_event("domain_detach", context=2, last=2)
           + domain_event("domain_retire", last=2))
    result = scoped_provenance(log, [manifest()])
    domain, = result["residency_domains"]["domains"]
    assert domain["execution"] == {"events": 3, "upload_bytes": 0, "download_bytes": 0,
                                    "kernel_launches": 0, "graph_submissions": 1,
                                    "graph_kernel_submissions": 2}
    assert domain["contexts"] == [2]
    assert result["residency_domains"]["complete"]
    assert "graph_submissions" not in domain["storage"]


def test_graph_member_cannot_use_retired_domain():
    log = (domain_event("domain_create") + domain_event("domain_attach", context=2)
           + domain_operation("graph_submit", 1)
           + domain_event("domain_detach", context=2, last=2)
           + domain_operation("graph_kernel"))
    with pytest.raises(ValueError, match="detached"):
        scoped_provenance(log, [manifest()])


def test_actual_selective_queue_trace_matches_all_nine_original_origins():
    # The C++ fixture owns these exact test identities before executing. This
    # public test registry is independent of values discovered in the trace.
    public = {"runtime_provenance": {"schema_version": 1, "records": [
        {"id": char * 64, "kind": kind}
        for char, kind in zip("abcde", ("owner", "procedure", "segment", "native_operation", "boundary"), strict=True)
    ] + [{"id": char * 64, "kind": "implementation"} for char in "f123456789"]}}
    trace = Path(__file__).with_name("fixtures") / "queued_provenance_v1.log"
    result = scoped_provenance(trace.read_text(), [public])
    assert result["totals"]["kernel_launches"] == 3
    assert result["totals"]["graph_submissions"] == 4
    assert result["totals"]["graph_kernel_submissions"] == 6
    for char in "123456789":
        group, = [row for row in result["groups"] if row["implementation"] == char * 64]
        assert group["kernel_launches"] + group.get("graph_kernel_submissions", 0) == 1
        assert group["owner"] == "a" * 64
    flush, = [row for row in result["groups"] if row["implementation"] == "f" * 64]
    assert flush.get("graph_kernel_submissions", 0) == flush["kernel_launches"] == 0
    assert flush["graph_submissions"] == 4


def test_enqueue_position_alone_does_not_count_execution():
    log = event("queue_enqueue") + event("position")
    result = scoped_provenance(log, [manifest()])
    assert result["totals"] == {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}
    assert not result["groups"]


def test_only_validated_graph_members_verify_gpu_activity():
    aggregate = event("graph_submit", 1)
    assert not gpu_activity(aggregate, "local", [], scope_manifest=manifest())["verified"]
    result = gpu_activity(aggregate + event("graph_kernel"), "local", [], scope_manifest=manifest())
    assert result["verified"]
    assert result["kernel_launches"] == 0
    assert result["graph_submissions"] == result["graph_kernel_submissions"] == 1
    assert result["scoped_runtime"]["graph_kernel_submissions"] == 1
    public = manifest()
    public.update(scope_count=0, scopes=[])
    with pytest.raises(ValueError, match="contradicts"):
        gpu_activity(aggregate + event("graph_kernel"), "local", [], scope_manifest=public)
