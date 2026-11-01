"""Native orphan worksharing constraints survive diagnostic package transport."""
from copy import deepcopy
from hashlib import sha256

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.local_adapter import LocalAdapterError, compose_scope_packages
from test_checked_execution_plan import digest, seal_epoch_plan, with_epoch
from test_scope_composition import package, save, unchanged
from test_scope_composition import sources as _sources

sources = _sources


def stamp(text, line):
    return {"source_span": [line, line], "original_source": text,
            "syntax_sha256": sha256(text.encode()).hexdigest()}


def conditional(metadata, *, private=False):
    plan = with_epoch(metadata)
    source = next(iter(plan["sources"]))
    requirement = {
        "version": 1, "kind": "original_orphan_worksharing_storage_structure", "procedure": "filter::work",
        "source_inputs": deepcopy(plan["storage_epoch"]["source_inputs"]), "structured_identity": "a" * 64,
        "whole_original_procedure_node_id": "filter::work/body",
        "regions": [{"directive": stamp("!$omp do collapse(3)", 12),
                     "associated_loop": {"source_span": [13, 18], "syntax_sha256": "c" * 64},
                     "end_directive": stamp("!$omp end do nowait", 19), "collapse": 3, "retained_nowait": True}],
        "callee_source": {"path": source, "source_sha256": plan["sources"][source],
                          "source_span": [10, 20], "original_callee_sha256": "d" * 64},
        **{key: False for key in (
            "internal_cuts_authorized", "descriptor_preservation_authorized", "storage_lifetime_authority",
            "caller_participation_authority", "work_completion_authority", "execution_authorized",
            "payload_access_authorized", "entry_definition_authority", "alias_authority", "gpu_legality_established")},
        **{key: True for key in ("requires_complete_original_callee_and_body_rescan",
                               "original_native_source_unchanged", "requires_whole_original_native_callee")},
    }
    wrapper = {"descriptor_identity": "e" * 64, "descriptor_procedure": "stepper::step",
               "descriptor_structured_identity": "f" * 64,
               "descriptor_source_call_closure": {"filter::work": "a" * 64},
               "worksharing_requirements": [requirement],
               "native_source_mode": "unchanged original callee; no generated body or companion",
               **{key: False for key in ("internal_cuts_authorized", "execution_authorized",
                                        "payload_access_authorized", "entry_definition_authority")}}
    plan["storage_epoch"].update(version=3, original_worksharing_storage_obligations=[wrapper])
    if private:
        plan["storage_epoch"]["private_allocation_obligations"] = [{
            "descriptor_identity": "e" * 64, "descriptor_procedure": "stepper::step",
            "unregistered_private_allocations": [{"resource": "filter::work::scratch",
                                                 "declaring_procedure": "filter::work"}]}]
    reseal(plan)
    return plan, wrapper, requirement


def reseal(plan):
    for item in plan["storage_epoch"]["original_worksharing_storage_obligations"]:
        for record in item["worksharing_requirements"]:
            if isinstance(record, dict):
                record["proof_identity"] = digest({k: v for k, v in record.items() if k != "proof_identity"})
    seal_epoch_plan(plan)


@pytest.mark.parametrize("private", [False, True])
def test_original_worksharing_transports_without_execution_or_registration(tmp_path, sources, private):
    metadata = package(tmp_path, sources, "one")
    plan, _, _ = conditional(metadata, private=private)
    original = deepcopy(plan)
    validate_checked_execution_plans(metadata["scope_manifest"])
    assert plan == original
    assert len(plan["storage_epoch"]["resources"]) == 1


@pytest.mark.parametrize(("change", "reason"), [
    (lambda w, r: r.update(version=True), "version"),
    (lambda w, r: r.update(execution_authorized=True), "cannot grant"),
    (lambda w, r: r.update(work_completion_authority=True), "cannot grant"),
    (lambda w, r: r.update(original_native_source_unchanged=1), "cannot grant"),
    (lambda w, r: r.update(source_inputs="0" * 64), "input identity"),
    (lambda w, r: r.update(whole_original_procedure_node_id="filter::work/body/1"), "input identity"),
    (lambda w, r: r.update(procedure="foreign"), "input identity"),
    (lambda w, r: r.update(regions=[]), "nonempty original"),
    (lambda w, r: r["regions"][0].update(collapse=True), "collapse"),
    (lambda w, r: r["regions"][0].update(collapse=2), "collapse"),
    (lambda w, r: r["regions"][0].update(retained_nowait=False), "NOWAIT"),
    (lambda w, r: r["regions"][0]["associated_loop"].update(source_span=[13, True]), "span"),
    (lambda w, r: r["regions"][0]["associated_loop"].update(source_span=[12, 18]), "overlap"),
    (lambda w, r: r["regions"].append(deepcopy(r["regions"][0])), "overlap"),
    (lambda w, r: r["callee_source"].update(source_span=[14, 20]), "overlap"),
    (lambda w, r: r["callee_source"].update(source_sha256="0" * 64), "source input"),
    (lambda w, r: r["callee_source"].update(original_callee_sha256="bad"), "source input"),
    (lambda w, r: r["regions"][0]["directive"].update(original_source="!$omp do schedule(dynamic)"), "digest"),
    (lambda w, r: r["regions"][0].update(directive=stamp("!$omp do schedule(dynamic)", 12)), "collapse"),
    (lambda w, r: r["regions"][0].update(end_directive=stamp("!$omp barrier", 19)), "NOWAIT"),
    (lambda w, r: r.update(extra_authority=False), "exact schema"),
    (lambda w, r: w.update(internal_cuts_authorized=True), "preserve"),
    (lambda w, r: w.update(native_source_mode="generated companion"), "preserve"),
    (lambda w, r: w.update(descriptor_source_call_closure={}), "call closure"),
    (lambda w, r: w["descriptor_source_call_closure"].update({"filter::work": "b" * 64}), "input identity"),
    (lambda w, r: w["worksharing_requirements"].append(deepcopy(r)), "repeat"),
    (lambda w, r: w.update(worksharing_requirements=[]), "nonempty structure"),
])
def test_resealed_record_cannot_change_source_or_grant_authority(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    plan, wrapper, requirement = conditional(metadata)
    change(wrapper, requirement)
    reseal(plan)
    with pytest.raises(ValueError, match="original worksharing .*" + reason):
        validate_checked_execution_plans(metadata["scope_manifest"])


@pytest.mark.parametrize(("first", "last", "accepted"), [
    (1, 1, True), (21, 21, True), (21, 20, True), (9, 8, True),
    (10, 9, False), (20, 19, False), (9, 10, False), (20, 21, False), (12, 17, False),
])
def test_global_source_edits_preserve_whole_callee_including_insertion_endpoints(
        tmp_path, sources, first, last, accepted):
    metadata = package(tmp_path, sources, "one")
    _, _, requirement = conditional(metadata)
    source = requirement["callee_source"]
    metadata["scope_manifest"]["source_edits"] = [{"source": source["path"],
        "source_sha256": source["source_sha256"], "first_line": first, "last_line": last, "replacement": "! edit\n"}]
    if accepted:
        validate_checked_execution_plans(metadata["scope_manifest"])
    else:
        with pytest.raises(ValueError, match="source edit overlaps"):
            validate_checked_execution_plans(metadata["scope_manifest"])


def test_sibling_scope_cannot_manage_protected_callee(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    plan, _, _ = conditional(metadata)
    sibling = deepcopy(metadata["scope_manifest"]["scopes"][0])
    # An independently valid sibling plan selecting a statement inside the
    # protected native procedure must be rejected by package-wide checking.
    sibling_plan = sibling["checked_execution_plan"]
    sibling_plan["storage_epoch"].pop("original_worksharing_storage_obligations")
    sibling_plan["storage_epoch"]["version"] = 1
    unit = sibling_plan["procedures"][0]["units"][0]
    unit.update(first_line=12, last_line=12)
    seal_epoch_plan(sibling_plan)
    metadata["scope_manifest"]["scopes"].append(sibling)
    with pytest.raises(ValueError, match="managed unit overlaps"):
        validate_checked_execution_plans(metadata["scope_manifest"])
    assert plan["storage_epoch"]["version"] == 3


@pytest.mark.parametrize("mutation", ["missing", "empty", "old_version", "private_empty", "repeated"])
def test_conditional_epoch3_schema_is_closed(tmp_path, sources, mutation):
    metadata = package(tmp_path, sources, "one")
    plan, wrapper, _ = conditional(metadata)
    epoch = plan["storage_epoch"]
    if mutation == "missing":
        epoch.pop("original_worksharing_storage_obligations")
    elif mutation == "empty":
        epoch["original_worksharing_storage_obligations"] = []
    elif mutation == "old_version":
        epoch["version"] = 2
    elif mutation == "private_empty":
        epoch["private_allocation_obligations"] = []
    else:
        epoch["original_worksharing_storage_obligations"].append(deepcopy(wrapper))
    seal_epoch_plan(plan)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        validate_checked_execution_plans(metadata["scope_manifest"])


@pytest.mark.parametrize("keep_checked_plan", [False, True])
def test_duplicate_epoch_cannot_hide_obligations_by_downgrading_its_checked_plan(
        tmp_path, sources, keep_checked_plan):
    metadata = package(tmp_path, sources, "one")
    plan, _, _ = conditional(metadata)
    scope = metadata["scope_manifest"]["scopes"][0]
    scope["storage_epoch"] = deepcopy(plan["storage_epoch"])
    plan["storage_epoch"].pop("original_worksharing_storage_obligations")
    plan["storage_epoch"]["version"] = 1
    seal_epoch_plan(plan)
    if not keep_checked_plan:
        scope.pop("checked_execution_plan")
    with pytest.raises(ValueError, match="epoch .*checked plan"):
        validate_checked_execution_plans(metadata["scope_manifest"])


@pytest.mark.parametrize("conflict", [False, True])
def test_independent_packages_obey_each_others_native_callee_constraint(tmp_path, sources, conflict):
    first = package(tmp_path, sources, "one")
    plan, _, record = conditional(first)
    shared = next(path for path in sources[0] if path.endswith("shared.f90"))
    record["callee_source"].update(path=shared, source_sha256=sources[0][shared])
    reseal(plan)
    save(first)
    second = package(tmp_path, sources, "two", source="shared.f90")
    with_epoch(second)
    second["scope_manifest"]["source_edits"] = [{"source": shared, "source_sha256": sources[0][shared],
        "first_line": 10 if conflict else 1, "last_line": 9 if conflict else 1, "replacement": "! native edit\n"}]
    save(second)
    if conflict:
        with pytest.raises(LocalAdapterError, match="source edit overlaps"):
            compose_scope_packages([first, second], sources[1])
        unchanged(sources)
    else:
        result = compose_scope_packages([first, second], sources[1])
        assert len(result["patched_sources"]) == 2
