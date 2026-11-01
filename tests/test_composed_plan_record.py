"""Composed artifacts retain each compiler root's original proof identity."""
import copy
import hashlib
import json
from pathlib import Path

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.local_adapter import apply_scope_sources


@pytest.fixture(scope="module")
def composed(tmp_path_factory):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.compose import compose_graph_scopes
    from compiler.scopes.source import ScopeBuilder
    from compiler.tests import test_provider_only_lifetime as fixture

    directory = tmp_path_factory.mktemp("composed_provider")
    source = fixture.PARENT + """module other_work
contains
subroutine increment(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
do i=1,n
a(i)=a(i)+1.0_8
enddo
end subroutine
end module
"""
    paths, provider = fixture.fixture(directory, source=source)
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): fixture.digest(path) for path in paths},
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole",
                                           "escapes": False, "allocation_changes": False}}}
    ordinary = ScopeBuilder(paths, "other_work::increment", facts=facts, options=CompilerOptions(),
                            config=OffloadConfig(policy="sections", scope_execution="graph"))
    components = [provider.run(), ordinary.run()]
    for _, report in components:
        validate_checked_execution_plans(report)
    outputs, report = compose_graph_scopes(components)
    report = json.loads(outputs["scope-manifest.json"])
    root = directory / "artifacts"
    for name, content in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return paths, outputs, report, root, components


def test_real_composition_reads_plans_in_their_original_component(composed):
    _, _, report, _, _ = composed
    validate_checked_execution_plans(report)
    assert report["component_manifests"][0]["provider_only_lifetime"]["owner"] == "native_parent::owner"


def test_composed_provider_and_gpu_root_apply_one_checked_source_union(composed, tmp_path):
    paths, outputs, report, root, _ = composed
    destinations = {}
    for source in paths:
        destination = tmp_path / source.name
        destination.write_bytes(source.read_bytes())
        destinations[str(source)] = destination
    metadata = {"scope_manifest": report, "artifact_root": str(root)}
    changed = apply_scope_sources(metadata, destinations)
    assert set(changed) == {destinations[path] for path in report["sources"]}
    for source, descriptor in report["sources"].items():
        assert destinations[source].read_text() == outputs[descriptor["replacement"]]


def _reseal(report):
    for record, component in zip(report["composition"]["components"], report["component_manifests"], strict=True):
        record["manifest_sha256"] = hashlib.sha256(json.dumps(component, sort_keys=True).encode()).hexdigest()


@pytest.mark.parametrize("fault", ["component_hash", "component_entry", "plan", "scope_union", "edit_union",
    "sources", "build", "runtime", "helper", "provider", "missing_components", "missing_marker",
    "policy_bool", "count_bool", "non_graph", "nested", "measurement", "exact_build", "extra_artifact"])
def test_composed_record_rejects_changed_component_or_aggregate(composed, fault):
    _, _, original, _, _ = composed
    report = copy.deepcopy(original)
    component = report["component_manifests"][0]
    if fault == "component_hash":
        component["entry"] = "other"
    elif fault == "component_entry":
        report["entry"][0] = "other"
    elif fault == "plan":
        component["scopes"][0]["checked_execution_plan"]["identity"] = "0" * 64
        _reseal(report)
    elif fault == "scope_union":
        report["scopes"].pop()
    elif fault == "edit_union":
        report["source_edits"].pop()
    elif fault == "sources":
        report["sources"].pop(next(iter(report["sources"])))
    elif fault == "build":
        report["build_sources"].pop()
    elif fault == "runtime":
        component["runtime"]["runtime_id"] = "0" * 64
        _reseal(report)
    elif fault == "helper":
        helper = component["provider_only_lifetime"]["helper"]["path"]
        report["artifacts_sha256"][helper] = "0" * 64
    elif fault == "provider":
        component["provider_only_lifetime"]["field_retention_authorized"] = True
        _reseal(report)
    elif fault == "missing_components":
        del report["component_manifests"]
    elif fault == "missing_marker":
        del report["composition"]
    elif fault == "policy_bool":
        report["composition"]["schema_version"] = True
    elif fault == "count_bool":
        report["scope_count"] = True
    elif fault == "non_graph":
        component["scope_execution"] = "reached"
        _reseal(report)
    elif fault == "nested":
        component["component_manifests"] = []
        _reseal(report)
    elif fault == "measurement":
        component["scope_measurement"] = "resident"
        _reseal(report)
    elif fault == "exact_build":
        component["exact_window_build_contract"] = {"schema_version": 1}
        _reseal(report)
    else:
        report["artifacts_sha256"]["unbound.f90"] = "0" * 64
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report)


@pytest.fixture(scope="module")
def configured_composed(composed, tmp_path_factory):
    from compiler.driver.options import CompilerOptions
    from compiler.offload.config import OffloadConfig
    from compiler.scopes.compose import compose_graph_scopes
    from compiler.scopes.source import ScopeBuilder

    directory = tmp_path_factory.mktemp("configured_composition")
    source, prepared, include = [directory / name for name in ("configured.f90", "prepared.f90", "body.inc")]
    text = """module configured
contains
subroutine increment(a,n)
real(8),intent(inout)::a(:)
integer,intent(in)::n
integer::i
#include "body.inc"
do i=1,n
a(i)=a(i)+1.0_8
enddo
end subroutine
end module
"""
    source.write_text(text)
    include.write_text("! configured dependency\n")
    prepared.write_text(text.replace('#include "body.inc"', include.read_text().rstrip()))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    analysis = {"schema_version": 1, "preserves_source_order": True, "configuration": {"defines": []},
                "source_inputs": {str(source): digest(source)}, "dependencies": {str(include): digest(include)},
                "entries": [{"source": str(source), "path": str(prepared), "sha256": digest(prepared),
                             "line_map": [None if line.startswith("#include") else n
                                          for n, line in enumerate(text.splitlines(), 1)]}]}
    facts = {"schema_version": 1, "participation": "serial", "sources": analysis["source_inputs"],
             "captures": {"argument::a": {"storage": "stable", "initialized": "whole",
                                           "escapes": False, "allocation_changes": False}}}
    builder = ScopeBuilder([source], "configured::increment", facts=facts, options=CompilerOptions(),
                           config=OffloadConfig(policy="sections", scope_execution="graph"), analysis_sources=analysis)
    outputs, _report = compose_graph_scopes([*composed[4], builder.run()])
    root = directory / "artifacts"
    for name, content in outputs.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return outputs, json.loads(outputs["scope-manifest.json"]), root, include


def test_each_component_keeps_its_configured_input_identity(configured_composed, tmp_path):
    outputs, report, root, _ = configured_composed
    validate_checked_execution_plans(report)
    destinations = {}
    for index, source in enumerate(report["source_inputs"]):
        destination = tmp_path / f"source-{index}.f90"
        destination.write_bytes(Path(source).read_bytes())
        destinations[source] = destination
    apply_scope_sources({"scope_manifest": report, "artifact_root": str(root)}, destinations)
    for source, descriptor in report["sources"].items():
        assert destinations[source].read_text() == outputs[descriptor["replacement"]]


@pytest.mark.parametrize("mapped", [False, True])
def test_stale_component_dependency_rejects_before_any_source_write(configured_composed, tmp_path, mapped):
    _, report, root, include = configured_composed
    destinations, original = {}, {}
    for index, source in enumerate(report["source_inputs"]):
        destination = tmp_path / f"source-{index}.f90"
        original[destination] = Path(source).read_bytes()
        destination.write_bytes(original[destination])
        destinations[source] = destination
    saved = include.read_bytes()
    dependency = tmp_path / "mapped.inc" if mapped else include
    if mapped:
        destinations[str(include)] = dependency
    dependency.write_text("! changed configuration dependency\n")
    try:
        with pytest.raises(RuntimeError, match="Configured source dependency changed"):
            apply_scope_sources({"scope_manifest": report, "artifact_root": str(root)}, destinations)
        assert all(path.read_bytes() == content for path, content in original.items())
    finally:
        include.write_bytes(saved)


def test_composition_cannot_drop_component_build_requirement_before_application(composed, tmp_path):
    paths, _, original, _, _ = composed
    report = copy.deepcopy(original)
    report["component_manifests"][0]["exact_window_build_contract"] = {"schema_version": 1}
    _reseal(report)
    destinations = {}
    for path in paths:
        destination = tmp_path / path.name
        destination.write_bytes(path.read_bytes())
        destinations[str(path)] = destination
    with pytest.raises(RuntimeError, match="unsupported exact-window build contract"):
        apply_scope_sources({"scope_manifest": report, "artifact_root": str(tmp_path / "absent")}, destinations)
    assert all(destination.read_bytes() == Path(source).read_bytes() for source, destination in destinations.items())


def test_composition_does_not_accept_one_capture_fact_set_for_distinct_roots(composed):
    with pytest.raises(ValueError, match="each original root's capture identity"):
        validate_checked_execution_plans(composed[2], capture_facts={})
