"""Versioned graph dispatch records crosslink issued source, provider and schedule."""
from copy import deepcopy
from hashlib import sha256

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.source_task_record import validate_source_task

from compiler.tests.test_source_native_pricing import live_emitted


@pytest.fixture(scope="module", params=[False, True], ids=["root", "nested"])
def issued(request, tmp_path_factory):
    # Real source issuance with synthetic cost observations: integrity only.
    directory = tmp_path_factory.mktemp("graph-placement-reader")
    _path, (_outputs, manifest) = live_emitted(directory, request.param)
    return manifest


def parts(manifest):
    scope = manifest["scopes"][0]
    task = scope["source_device_tasks"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    member = next(item for item in epoch["members"] if item["procedure"] == task["source_proof"]["procedure"])
    procedure = next(item for item in plan["procedures"] if item["procedure"] == member["procedure"])
    graph = next((item for item in scope.get("graph_execution", {}).get("procedures", [])
                  if item["procedure"] == member["procedure"]), None)
    return scope, task, plan, epoch, member, procedure, graph


def check(manifest):
    scope, task, plan, epoch, member, procedure, _graph = parts(manifest)
    validate_source_task(task, member, procedure, epoch["source_inputs"],
                         {item["resource"]: item for item in epoch["resources"]}, owner=plan["entry"],
                         manifest=manifest, graph_execution=scope.get("graph_execution"), task_ordinal=0)


def reseal(manifest, *, provider=True, graph=True):
    _scope, task, _plan, _epoch, _member, _procedure, linked = parts(manifest)
    placement = task["placement"]
    placement["identity"] = _digest({k: v for k, v in placement.items() if k != "identity"})
    if provider:
        record = task["execution_binding"]["provider"]
        metadata = record["metadata"]
        old_entry, old_module = record["entry"], record["module"]
        suffix = task["source_proof"]["team"]["identity"][:12]
        suffix += "_cta" if metadata["mapping"] == "cooperative" else "_thread"
        suffix += "_g_" + placement["identity"][:16]
        record["entry"] = metadata["entry"] = "fort_tile_task_" + suffix
        record["module"] = metadata["module"] = "fort_tile_module_" + suffix
        metadata["native_prepare_entry"] = "fort_tile_native_" + sha256(record["entry"].encode()).hexdigest()[:20]
        metadata["placement"] = deepcopy(placement)
        artifacts = manifest["artifacts_sha256"]
        artifacts[record["entry"] + ".cu"] = artifacts.pop(old_entry + ".cu")
        artifacts[record["module"] + ".f90"] = artifacts.pop(old_module + ".f90")
    if graph:
        deferred = {"version": 1, "policy": placement["policy"], "identity": placement["identity"]}
        linked["task_graph"]["tasks"][0]["deferred_placement"] = deepcopy(deferred)
        linked["schedule"]["placements"][0]["deferred_placement"] = deepcopy(deferred)
    task["identity"] = _digest({k: v for k, v in task.items() if k != "identity"})


def test_actual_issued_graph_has_both_implementations_and_no_invented_static_target(issued):
    validate_checked_execution_plans(issued)
    check(issued)
    _scope, task, plan, _epoch, _member, _procedure, graph = parts(issued)
    assert plan["schema_version"] == 2
    assert task["version"] == 4
    assert task["source_proof"]["version"] == 1
    assert task["execution_binding"]["provider"]["metadata"]["version"] == 6
    assert graph["task_graph"]["tasks"][0]["implementations"] == ["cpu", "gpu"]
    assert graph["schedule"]["placements"][0]["target"] is None
    assert task["placement"]["execution_authorized"] is False


@pytest.mark.parametrize("fault", ["version_bool", "kind", "policy", "owner", "procedure", "source", "pricing",
                                  "canonical", "slot_bool", "implementations", "unavailable", "native", "gpu",
                                  "execution", "extra", "ordinal"])
def test_resealed_placement_cannot_change_source_storage_endpoints_or_dispatch_policy(issued, fault):
    data = deepcopy(issued)
    placement = parts(data)[1]["placement"]
    if fault == "canonical":
        placement["canonical_resources"][next(iter(placement["canonical_resources"]))] = "forged::root"
    elif fault == "slot_bool":
        placement["registration_slots"][next(iter(placement["registration_slots"]))] = True
    else:
        key, value = {"version_bool": ("version", True), "kind": ("kind", "unchecked"),
            "policy": ("policy", "cpu_only"), "owner": ("owner", "foreign"), "procedure": ("procedure", "foreign"),
            "source": ("source_proof_identity", "0" * 64), "pricing": ("pricing_identity", "0" * 64),
            "implementations": ("implementations", ["cpu"]), "unavailable": ("unavailable_target", "gpu"),
            "native": ("native_endpoint", "close"), "gpu": ("gpu_endpoint", "close"),
            "execution": ("execution_authorized", True), "extra": ("extra", False),
            "ordinal": ("task_id", placement["procedure"] + ":source_device:1")}[fault]
        placement[key] = value
    reseal(data)
    with pytest.raises(ValueError, match="source placement"):
        check(data)


@pytest.mark.parametrize("fault", ["missing_graph", "missing_task", "duplicate_task", "missing_schedule", "duplicate_schedule",
                                  "task_binding", "schedule_binding", "only_gpu", "unordered", "opaque", "reads",
                                  "writes", "metadata_reads", "target", "cost", "transfers", "children"])
def test_graph_and_schedule_cannot_diverge_from_the_reached_source_dispatch(issued, fault):
    data = deepcopy(issued)
    scope, _task, _plan, _epoch, _member, _procedure, graph = parts(data)
    task, planned = graph["task_graph"]["tasks"][0], graph["schedule"]["placements"][0]
    if fault == "missing_graph":
        scope.pop("graph_execution")
    elif fault == "missing_task":
        graph["task_graph"]["tasks"].clear()
    elif fault == "duplicate_task":
        graph["task_graph"]["tasks"].append(deepcopy(task))
    elif fault == "missing_schedule":
        graph["schedule"]["placements"].clear()
    elif fault == "duplicate_schedule":
        graph["schedule"]["placements"].append(deepcopy(planned))
    elif fault in {"task_binding", "schedule_binding"}:
        (task if fault == "task_binding" else planned)["deferred_placement"]["identity"] = "0" * 64
    elif fault in {"only_gpu", "unordered", "opaque", "reads", "writes", "metadata_reads"}:
        key, value = {"only_gpu": ("implementations", ["gpu"]), "unordered": ("ordered", False),
                      "opaque": ("opaque", True), "reads": ("reads", []), "writes": ("may_writes", []),
                      "metadata_reads": ("metadata_reads", [])}[fault]
        task[key] = value
    else:
        planned[{"target": "target", "cost": "estimated_cost", "transfers": "transfers", "children": "children"}[fault]] = {
            "target": "gpu", "cost": 0.0, "transfers": [{"resource": "forged"}], "children": [{}]}[fault]
    with pytest.raises(ValueError, match="source placement"):
        check(data)


@pytest.mark.parametrize("fault", ["version", "selected_cpu_bool", "selected_gpu_bool", "selected_extra", "placement", "entry"])
def test_provider_dispatch_out_parameter_is_closed_and_crosslinked(issued, fault):
    data = deepcopy(issued)
    task = parts(data)[1]
    provider = task["execution_binding"]["provider"]
    metadata = provider["metadata"]
    if fault == "version":
        metadata["version"] = 4
    elif fault.startswith("selected"):
        metadata["selected_backend"][{"selected_cpu_bool": "cpu", "selected_gpu_bool": "gpu", "selected_extra": "extra"}[fault]] = True
    elif fault == "placement":
        metadata["placement"]["identity"] = "0" * 64
    else:
        provider["entry"] = metadata["entry"] = "unchecked_dispatch"
    task["identity"] = _digest({k: v for k, v in task.items() if k != "identity"})
    with pytest.raises(ValueError, match="source"):
        check(data)
