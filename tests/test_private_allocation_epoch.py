"""Transported lifetime obligations cannot hide a live canonical handle."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from test_checked_execution_plan import seal_epoch_plan, with_epoch
from test_scope_composition import package
from test_scope_composition import sources as _sources

sources = _sources
ROOT = "private_scratch::work::scratch"


def conditional(metadata, **options):
    plan = with_epoch(metadata, **options)
    plan["storage_epoch"].update(version=2, private_allocation_obligations=[{
        "descriptor_identity": "e" * 64, "descriptor_procedure": "private_scratch::step",
        "unregistered_private_allocations": [{"resource": ROOT, "declaring_procedure": "private_scratch::work"}]}])
    seal_epoch_plan(plan)
    return plan


@pytest.mark.parametrize("variant", ["managed", "opaque", "transfer"])
def test_original_lifetime_obligation_transports_without_adding_a_registration(tmp_path, sources, variant):
    metadata = package(tmp_path, sources, "one")
    plan = conditional(metadata, opaque=variant == "opaque", transfer=variant == "transfer")
    expected = deepcopy(plan)
    validate_checked_execution_plans(metadata["scope_manifest"])
    assert plan == expected
    assert ROOT not in {item["resource"] for item in plan["resources"]}
    assert plan["storage_epoch"]["definitions_authorized"] is False


def test_a_future_host_only_registration_also_conflicts_with_private_reallocation(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    plan = conditional(metadata)
    resource = {"resource": ROOT, "registration_identity": 2, "descriptor": {"rank": 1}}
    plan["resources"].append(resource)
    plan["storage_epoch"]["resources"].append({**deepcopy(resource),
        "storage": "opaque host reservation", "lifetime": "original checked native access",
        "device_replica_authorized": False, "initialization_authorized": False})
    seal_epoch_plan(plan)
    with pytest.raises(ValueError, match="private allocation overlaps a registration"):
        validate_checked_execution_plans(metadata["scope_manifest"])


@pytest.mark.parametrize(("change", "reason"), [
    (lambda e: e.update(version=1), "complete v1 schema"),
    (lambda e: e.pop("private_allocation_obligations"), "complete v2 schema"),
    (lambda e: e.update(private_allocation_obligations=[]), "nonempty private"),
    (lambda e: e["private_allocation_obligations"].append(deepcopy(e["private_allocation_obligations"][0])), "repeat or change order"),
    (lambda e: e["private_allocation_obligations"][0].update(descriptor_identity="not-a-digest"), "descriptor identity"),
    (lambda e: e["private_allocation_obligations"][0].update(descriptor_procedure="foreign"), "descriptor identity"),
    (lambda e: e["private_allocation_obligations"][0]["unregistered_private_allocations"][0].update(resource="argument::a"), "declaring identity"),
    (lambda e: e["private_allocation_obligations"][0]["unregistered_private_allocations"][0].update(resource="foreign::work::scratch"), "declaring identity"),
    (lambda e: e["private_allocation_obligations"][0]["unregistered_private_allocations"].append(deepcopy(e["private_allocation_obligations"][0]["unregistered_private_allocations"][0])), "declaring identity"),
    (lambda e: e["private_allocation_obligations"][0].update(unregistered_private_allocations=[]), "declaring identity"),
    (lambda e: e["private_allocation_obligations"][0].update(unregistered_private_allocations=[True]), "declaring identity"),
    (lambda e: e["private_allocation_obligations"][0].update(unregistered_private_allocations=ROOT), "declaring identity"),
    (lambda e: e["private_allocation_obligations"][0].update(execution_authorized=True), "descriptor identity"),
])
def test_resealing_diagnostic_hashes_cannot_remove_lifetime_obligations(tmp_path, sources, change, reason):
    metadata = package(tmp_path, sources, "one")
    plan = conditional(metadata)
    change(plan["storage_epoch"])
    seal_epoch_plan(plan)
    with pytest.raises(ValueError, match=reason):
        validate_checked_execution_plans(metadata["scope_manifest"])


def test_legacy_epoch_keeps_its_original_schema_and_contents(tmp_path, sources):
    metadata = package(tmp_path, sources, "one")
    plan = with_epoch(metadata)
    expected = deepcopy(plan)
    validate_checked_execution_plans(metadata["scope_manifest"])
    assert plan == expected
    assert plan["storage_epoch"]["version"] == 1
    assert "private_allocation_obligations" not in plan["storage_epoch"]
