"""Projected atomic v4 transport cannot replace checked invocation bindings."""

import json
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.source_task_record import validate_source_task

from compiler.tests.test_atomic_source_call_bindings import atomic_builder


@pytest.fixture(scope="module")
def authentic(tmp_path_factory):
    builder = atomic_builder(tmp_path_factory.mktemp("atomic-call-reader"))
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    return json.loads(outputs["scope-manifest.json"]), builder.facts


def values(manifest):
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    member = next(
        m for m in plan["storage_epoch"]["members"] if m.get("source_device_tasks")
    )
    (task,) = member["source_device_tasks"]
    procedure = next(
        p for p in plan["procedures"] if p["procedure"] == member["procedure"]
    )
    return (
        task,
        member,
        procedure,
        plan["storage_epoch"]["source_inputs"],
        {r["resource"]: r for r in plan["storage_epoch"]["resources"]},
        plan["entry"],
        manifest,
        scope["graph_execution"],
    )


def check(args):
    task, member, procedure, inputs, resources, owner, manifest, graph = args
    validate_source_task(
        task,
        member,
        procedure,
        inputs,
        resources,
        owner=owner,
        manifest=manifest,
        graph_execution=graph,
        task_ordinal=0,
    )


def seal(args):
    task = args[0]
    projection = task["execution_binding"].get("call_projection")
    if projection is not None:
        projection["identity"] = _digest(
            {k: v for k, v in projection.items() if k != "identity"}
        )
    task["identity"] = _digest({k: v for k, v in task.items() if k != "identity"})


def test_atomic_v4_saved_package_round_trips_without_rewriting_source_providers(
    authentic,
):
    manifest, facts = authentic
    before = deepcopy(manifest)
    validate_checked_execution_plans(manifest, capture_facts=facts)
    args = values(manifest)
    check(args)
    assert args[0]["version"] == 4 and args[0]["source_proof"]["version"] == 3
    assert all(
        v["provider"]["metadata"]["version"] == 3 for v in args[0]["alternatives"]
    )
    assert manifest == before


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "legacy_with_projection",
        "legacy_without_projection",
        "bool_version",
        "source_proof",
        "owner",
        "procedure",
        "empty_calls",
        "missing_call",
        "duplicate_call",
        "call_handle",
        "call_actual",
        "call_source",
        "predicate_actual",
        "extent_actual",
        "slot",
        "canonical",
        "field_roots",
        "effects_read",
        "effects_write",
        "effects_definition",
        "member_source",
        "member_namespace",
        "foreign_root",
        "source_kind",
        "provider_version",
        "parent_join",
        "unknown",
    ],
)
def test_resealed_atomic_projection_rejects_changed_semantics(authentic, fault):
    args = values(deepcopy(authentic[0]))
    check(args)
    task = args[0]
    binding = task["execution_binding"]
    projection = binding["call_projection"]
    edge = projection["calls"][0]
    if fault == "missing":
        binding.pop("call_projection")
    elif fault == "legacy_with_projection":
        task["version"] = 3
    elif fault == "legacy_without_projection":
        task["version"] = 3
        binding.pop("call_projection")
    elif fault == "bool_version":
        task["version"] = True
    elif fault == "source_proof":
        projection["source_proof_identity"] = "0" * 64
    elif fault == "owner":
        projection["owner"] = "foreign::owner"
    elif fault == "procedure":
        projection["procedure"] = "foreign::helper"
    elif fault == "empty_calls":
        projection["calls"] = []
    elif fault == "missing_call":
        projection["calls"].pop()
    elif fault == "duplicate_call":
        projection["calls"].append(deepcopy(edge))
    elif fault == "call_handle":
        edge["handle_bindings"][next(iter(edge["handle_bindings"]))] += 999
    elif fault == "call_actual":
        edge["resource_mappings"]["argument::a"] = "argument::p"
    elif fault == "call_source":
        edge["source_call"]["actual_arguments"][0] = "other"
    elif fault in {"predicate_actual", "extent_actual"}:
        edge["source_call"]["actual_arguments"][
            3 if fault == "predicate_actual" else 2
        ] = "foreign_actual"
    elif fault == "slot":
        binding["registration_slots"]["argument::a"] += 999
    elif fault == "canonical":
        binding["canonical_resources"]["argument::a"] = "argument::p"
    elif fault == "field_roots":
        projection["field_roots"]["argument::a"].pop()
    elif fault == "effects_read":
        projection["conservative_effects"]["reads"].pop()
    elif fault == "effects_write":
        projection["conservative_effects"]["may_writes"].append("argument::p")
    elif fault == "effects_definition":
        projection["conservative_effects"]["must_writes"] = ["argument::a"]
    elif fault == "member_source":
        projection["members"][0]["source_sha256"] = "0" * 64
    elif fault == "member_namespace":
        projection["members"][0]["parameterized"] = not projection["members"][0][
            "parameterized"
        ]
    elif fault == "foreign_root":
        projection["root_registrations"]["foreign::array"] = 999
    elif fault == "source_kind":
        task["source_proof"]["kind"] = "caller claimed team"
    elif fault == "provider_version":
        task["alternatives"][0]["provider"]["metadata"]["version"] = 4
    elif fault == "parent_join":
        task["source_proof"]["parent_completion"]["join"] = "caller claimed completion"
    elif fault == "unknown":
        binding["motion_authorized"] = True
    seal(args)
    with pytest.raises(ValueError):
        check(args)
