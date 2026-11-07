"""Generic schedule labels are aggregate decisions, never source execution."""
from copy import deepcopy

import pytest
from elmm_pipeline.gpu_validation import gpu_activity, offload_decisions
from test_inline_offload_fallback import ENTRY, fixture


def scheduled(gpu=0, cpu=0):
    manifest, lines = fixture()
    lines = lines[:2]
    lines.append(f"FORT_SCOPED decision available=1 gpu_units={gpu} cpu_units={cpu} candidates=1 "
                 "simulated_operations=4 upload_bytes=0 download_bytes=0 launches=0 waits=0 "
                 "peak_device_bytes=0 estimated_seconds=0.01 native_seconds=0.02 "
                 "reason=modeled context=2 domain_version=1 domain=1 domain_role=execution")
    lines.append("FORT_SCOPED planning_timing context=2 domain_version=1 domain=1 domain_role=execution "
                 "phase=selection seconds=0.001 cache_hit=1 query_generation=3 state_generation=4")
    mode = "scoped-scheduled" if gpu else "native-scoped-scheduled"
    lines.append(f"FORT_OFFLOAD entry=region mode={mode} gpu_units={gpu} cpu_units={cpu} unit_kind=regions")
    return manifest, lines


@pytest.mark.parametrize(("gpu", "cpu"), [(0, 0), (0, 3), (2, 0), (2, 3)])
@pytest.mark.parametrize("composed", [False, True])
def test_schedule_summary_matches_domain_decision_without_source_or_launch_credit(gpu, cpu, composed):
    manifest, lines = scheduled(gpu, cpu)
    if composed:
        manifest = {"component_manifests": [manifest], "runtime_provenance": manifest["runtime_provenance"]}
    before = deepcopy(manifest)
    activity = gpu_activity("\n".join(lines), "local", [], scope_manifest=manifest)
    assert not activity["verified"]
    assert activity["kernel_launches"] == 0
    decisions = activity["offload_decisions"]
    row, = decisions["matched_inline_schedule_summaries"]
    assert row["candidate_entries"] == [ENTRY]
    assert row["aggregate_only"]
    assert not row["source_attribution_available"]
    assert not row["gpu_execution_confirmed"]
    assert (row["context"], row["domain"], row["gpu_units"], row["cpu_units"]) == (2, 1, gpu, cpu)
    assert decisions["calls"] == 1
    assert "authenticated_inline_fallbacks" not in decisions
    assert manifest == before


def test_missing_hardware_or_costs_retains_native_schedule_without_available_estimate():
    manifest, lines = scheduled(0, 1)
    lines[2] = lines[2].replace("available=1", "available=0")
    activity = gpu_activity("\n".join(lines), "local", [], scope_manifest=manifest)
    row, = activity["offload_decisions"]["matched_inline_schedule_summaries"]
    assert not row["estimate_available"]
    assert not activity["verified"]
    assert row["gpu_units"] == 0
    assert row["mode"] == "native-scoped-scheduled"


@pytest.mark.parametrize("change", ["decision", "domain", "context", "retired", "counts", "unavailable",
                                  "interposed", "duplicate", "timing_domain", "timing_nonfinite",
                                  "source", "unknown_entry", "mode", "units"])
def test_schedule_summary_cannot_borrow_foreign_or_missing_evidence(change):
    manifest, lines = scheduled(2, 3)
    if change == "decision":
        del lines[2]
    elif change == "domain":
        del lines[:2]
    elif change == "context":
        lines[2] = lines[2].replace("context=2", "context=9")
    elif change == "retired":
        import json
        attach = json.loads(lines[1].removeprefix("FORT_SCOPED domain "))
        lines.insert(2, "FORT_SCOPED domain " + json.dumps({**attach, "event": "domain_detach", "last_context": 2}))
    elif change == "counts":
        lines[-1] = lines[-1].replace("gpu_units=2", "gpu_units=3")
    elif change == "unavailable":
        lines[2] = lines[2].replace("available=1", "available=0")
    elif change == "interposed":
        lines.insert(-1, "unrelated source work")
    elif change == "duplicate":
        lines.append(lines[-1])
    elif change == "timing_domain":
        lines[3] = lines[3].replace("domain=1", "domain=9")
    elif change == "timing_nonfinite":
        lines[3] = lines[3].replace("seconds=0.001", "seconds=1e999")
    elif change == "source":
        manifest["inline_numerical_regions"]["regions"][0]["source_identity"] = "f" * 64
    elif change == "unknown_entry":
        lines[-1] = lines[-1].replace("entry=region", "entry=unrelated")
    elif change == "mode":
        lines[-1] = lines[-1].replace("mode=scoped-scheduled", "mode=native-scoped-scheduled")
    elif change == "units":
        lines[-1] = lines[-1].replace("unit_kind=regions", "unit_kind=slabs")
    with pytest.raises(ValueError, match="(?i)inline|scoped|offload|residency|borrower"):
        offload_decisions("\n".join(lines), [], scope_manifest=manifest)
