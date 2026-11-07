"""Issued source work stays bound to whole-root effects and compiled ABI metadata."""
from copy import deepcopy
from hashlib import sha256

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.source_task_record import validate_source_task
from test_checked_execution_plan import seal_epoch_plan

from compiler.driver.options import CompilerOptions
from compiler.emission.cuda.tile_task_provider import emit_tile_task_provider
from compiler.frontend.source_device_tasks import source_device_task
from compiler.frontend.source_effects import SourceEffects, _children
from compiler.offload.config import OffloadConfig
from compiler.scopes.segments import directive
from compiler.scopes.source import ScopeBuilder

SOURCE = """module stencil
contains
subroutine advance(a,b,n,t)
integer,intent(in)::n,t
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
real(8)::tmp(-1:t+2)
integer::bi,i,li
!$omp parallel private(bi,i,li,tmp)
!$omp do schedule(runtime)
do bi=1,n,t
 do i=bi-1,min(bi+t-1,n)+2
  li=i-bi+1
  tmp(li)=3.0_8*a(i)
 enddo
 do i=bi,min(bi+t-1,n)
  li=i-bi+1
  b(i)=b(i)+(tmp(li+1)-tmp(li-1))
 enddo
enddo
!$omp end do
!$omp end parallel
end subroutine
end module
"""


def reseal(record):
    team = record["team"]
    team["identity"] = _digest({k: v for k, v in team.items() if k != "identity"})
    record["source_proof_identity"] = _digest({
        k: v for k, v in record.items() if k not in {"identity", "source_proof_identity", "execution_binding"}})
    record["identity"] = _digest({k: v for k, v in record.items() if k != "identity"})


def _issued(tmp_path, *, envelopes=False, source=SOURCE, facts=None, storage_transport=False):
    path = tmp_path / "source.f90"
    path.write_text(source)
    analysis = SourceEffects([path])
    routine = analysis.routines["stencil::advance"]
    nodes = tuple(_children(routine.execution))
    first = next(i for i, node in enumerate(nodes) if (directive(node) or "").startswith("parallel"))
    last = next(i for i, node in enumerate(nodes) if directive(node) == "end parallel")
    if facts is None:
        proof = source_device_task(analysis, routine, nodes[first:last + 1], storage_transport=storage_transport)
    else:
        builder = ScopeBuilder([path], routine.qualified, facts=facts, options=CompilerOptions(),
                               config=OffloadConfig(policy="sections", scope_execution="graph"))
        analysis, routine = builder.analysis, builder.entry
        nodes = tuple(_children(routine.execution))
        proof = source_device_task(analysis, routine, nodes[first:last + 1], builder=builder, storage_transport=storage_transport)
    provider = emit_tile_task_provider(proof.team, readable_scalars=proof, access_envelopes=envelopes)
    metadata = provider.public()
    record = proof.public()
    record["source_proof_identity"] = record.pop("identity")
    resources = {binding.root: {"resource": binding.root, "descriptor": binding.public(),
                 "device_replica_authorized": True, "registration_identity": index}
                 for index, (binding, _written) in enumerate(proof.fields, 1)}
    record["execution_binding"] = {"owner": routine.qualified,
        "canonical_resources": {root: root for root in resources},
        "registration_slots": {root: item["registration_identity"] for root, item in resources.items()},
        "provider": {"entry": provider.entry, "module": provider.module,
            "cuda_source_sha256": metadata["cuda_sha256"], "fortran_source_sha256": metadata["fortran_sha256"],
            "metadata": metadata}}
    reseal(record)
    member = {"procedure": routine.qualified, "source_graph": record["source_graph"]}
    procedure = {"procedure": routine.qualified, "calls": []}
    manifest = {"artifacts_sha256": {provider.entry + ".cu": metadata["cuda_sha256"],
                                    provider.module + ".f90": metadata["fortran_sha256"]}}
    if storage_transport:
        from compiler.emission.common.resources import read_scoped_runtime

        manifest["runtime"] = read_scoped_runtime()[1]
    return record, member, procedure, analysis.inputs.identity(), resources, routine.qualified, manifest


@pytest.fixture
def issued(tmp_path):
    return _issued(tmp_path)


def check(values):
    record, member, procedure, inputs, resources, owner, manifest = values
    validate_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest)


def test_real_original_team_proof_and_emitted_typed_provider_round_trip(issued):
    check(issued)
    record = issued[0]
    assert [item["access"] for item in record["resources"]] == ["read", "readwrite"]
    assert record["definitions_authorized"] is False
    assert record["placement_authorized"] is False
    assert record["motion_authorized"] is False


@pytest.fixture
def module_child(tmp_path):
    source = SOURCE.replace("module stencil\ncontains", "module stencil\nreal(8),allocatable::saved(:)\ncontains")
    source = source.replace("tmp(li)=3.0_8*a(i)", "tmp(li)=3.0_8*a(i)+saved(i)")
    source = source.replace("end module", """subroutine run(a,b,n,t)
integer,intent(in)::n,t
real(8),intent(in)::a(-3:)
real(8),intent(inout)::b(-3:)
call advance(a,b,n,t)
end subroutine
end module""")
    values = list(_issued(tmp_path, envelopes=True, source=source))
    record, _member, procedure, _inputs, _resources, _owner, _manifest = values
    # Actual source and provider establish this child/global inventory. This
    # fixture checks portable association links, not a complete owner plan.
    values[5] = record["execution_binding"]["owner"] = "stencil::run"
    procedure["calls"] = [{"resource_mappings": {"argument::a": "argument::a", "argument::b": "argument::b"}}]
    reseal(record)
    return values


def test_original_module_capture_keeps_identity_without_a_formal_call_mapping(module_child):
    check(module_child)
    record = module_child[0]
    assert record["execution_binding"]["canonical_resources"]["stencil::saved"] == "stencil::saved"


@pytest.mark.parametrize("fault", ["remapped", "descriptor", "intent", "pointer", "formal_missing",
                                  "formal_wrong", "global_in_formals", "no_calls", "procedure_local"])
def test_module_capture_cannot_bypass_formal_identity_or_physical_registration(module_child, fault):
    values = deepcopy(module_child)
    record, _member, procedure, _inputs, resources, _owner, _manifest = values
    field = next(item["binding"] for item in record["team"]["fields"] if item["binding"]["resource"] == "stencil::saved")
    if fault == "remapped":
        record["execution_binding"]["canonical_resources"]["stencil::saved"] = "argument::a"
    elif fault == "descriptor":
        resources["stencil::saved"]["descriptor"]["lower_bounds"] = ["17"]
    elif fault == "intent":
        field["intent"] = resources["stencil::saved"]["descriptor"]["intent"] = "in"
    elif fault == "pointer":
        field["attributes"].append("pointer")
        resources["stencil::saved"]["descriptor"]["attributes"].append("pointer")
    elif fault == "formal_missing":
        procedure["calls"][0]["resource_mappings"].pop("argument::a")
    elif fault == "formal_wrong":
        procedure["calls"][0]["resource_mappings"]["argument::a"] = "argument::b"
    elif fault == "global_in_formals":
        procedure["calls"][0]["resource_mappings"]["stencil::saved"] = "stencil::saved"
    elif fault == "no_calls":
        procedure["calls"] = []
    else:
        resources["stencil::advance::saved"] = resources.pop("stencil::saved")
        record["execution_binding"]["canonical_resources"]["stencil::saved"] = "stencil::advance::saved"
    if fault in {"intent", "pointer"}:
        next(item for item in record["resources"] if item["resource"] == "stencil::saved")["descriptor"] = deepcopy(field)
    reseal(record)
    reason = "owning storage projection" if fault == "remapped" else "child storage projection"
    with pytest.raises(ValueError, match=reason):
        check(values)


def provider_version_fixture(issued, version):
    """Exercise closed record schemas; this projection is not codegen evidence."""
    values = deepcopy(issued)
    record, _member, _procedure, _inputs, _resources, _owner, manifest = values
    provider = record["execution_binding"]["provider"]
    metadata = provider["metadata"]
    old_entry = provider["entry"]
    entry = old_entry.removesuffix("_batch") + ("_batch" if version == 2 else "")
    provider["entry"] = metadata["entry"] = entry
    manifest["artifacts_sha256"][entry + ".cu"] = manifest["artifacts_sha256"].pop(old_entry + ".cu")
    metadata["version"] = version
    if version == 1:
        metadata.pop("batching", None)
        metadata["scratch"] = "one bounded logical tile; same-stream source-order units"
    else:
        metadata["batching"] = {"version": 1, "target_bytes": 4194304, "minimum_tiles": 1,
                                "planned_launches": "sum ceil(unit_tiles/batch_tiles)"}
        metadata["scratch"] = "bounded original tile batches; same-stream source-order units"
    reseal(record)
    return values


@pytest.mark.parametrize("version", [1, 2])
def test_provider_versions_have_distinct_exact_scratch_and_batching_records(issued, version):
    values = provider_version_fixture(issued, version)
    check(values)
    metadata = values[0]["execution_binding"]["provider"]["metadata"]
    if version == 1:
        metadata["batching"] = {"version": 1}
    else:
        metadata.pop("batching")
    reseal(values[0])
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        check(values)


@pytest.mark.parametrize("mutation", ["target", "target_bool", "minimum", "minimum_bool", "version_bool",
                                    "planned", "extra_authority", "scratch", "entry", "definitions", "placement"])
def test_resealed_batching_protocol_cannot_change_budget_mapping_or_authority(issued, mutation):
    values = provider_version_fixture(issued, 2)
    record = values[0]
    provider = record["execution_binding"]["provider"]
    metadata = provider["metadata"]
    batching = metadata["batching"]
    if mutation == "target":
        batching["target_bytes"] += 1
    elif mutation == "target_bool":
        batching["target_bytes"] = True
    elif mutation == "minimum":
        batching["minimum_tiles"] = 0
    elif mutation == "minimum_bool":
        batching["minimum_tiles"] = True
    elif mutation == "version_bool":
        batching["version"] = True
    elif mutation == "planned":
        batching["planned_launches"] = "one arbitrary launch"
    elif mutation == "extra_authority":
        batching["definitions_authorized"] = True
    elif mutation == "scratch":
        metadata["scratch"] = "unbounded workspace"
    elif mutation == "entry":
        provider["entry"] = metadata["entry"] = provider["entry"].removesuffix("_batch")
    else:
        metadata[mutation + "_authorized"] = True
    reseal(record)
    with pytest.raises(ValueError, match="Compiler checked execution plan"):
        check(values)


def test_source_task_claim_cannot_exist_without_its_checked_plan(issued):
    with pytest.raises(ValueError, match="source device tasks require their checked plan"):
        validate_checked_execution_plans({"scopes": [{"source_device_tasks": [issued[0]]}]})


def test_actual_generated_source_task_round_trips_whole_checked_plan(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE)
    fact = {"storage": "stable", "initialized": "whole", "allocation_changes": False, "escapes": False}
    builder = ScopeBuilder([path], "stencil::advance", options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"),
        facts={"schema_version": 1, "participation": "serial",
               "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
               "captures": {"argument::a": dict(fact), "argument::b": dict(fact)}})
    _outputs, manifest = builder.run()
    assert manifest["scope_count"] == 1
    scope, = manifest["scopes"]
    assert len(scope["source_device_tasks"]) == 1
    validate_checked_execution_plans(manifest)
    # The explicit scope list cannot replace its sealed epoch authority.
    changed = deepcopy(manifest)
    changed["scopes"][0]["source_device_tasks"] = []
    with pytest.raises(ValueError, match="source device tasks differ"):
        validate_checked_execution_plans(changed)


def scalar_manifest(tmp_path):
    path = tmp_path / "source.f90"
    path.write_text(SOURCE.replace("advance(a,b,n,t)", "advance(a,b,n,t,alpha)").replace(
        "integer,intent(in)::n,t", "integer,intent(in)::n,t\nreal(8),intent(in)::alpha").replace(
        "3.0_8*a(i)", "alpha*a(i)"))
    fact = {"storage": "stable", "initialized": "whole", "allocation_changes": False, "escapes": False}
    facts = {"schema_version": 1, "participation": "serial",
             "sources": {str(path): sha256(path.read_bytes()).hexdigest()},
             "captures": {"argument::" + name: dict(fact) for name in ("a", "b", "alpha")}}
    builder = ScopeBuilder([path], "stencil::advance", options=CompilerOptions(),
        config=OffloadConfig(policy="sections", scope_execution="graph"), facts=facts)
    _outputs, manifest = builder.run()
    return manifest, facts


def bind_request_identity(manifest, facts):
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    plan["capture_facts_identity"] = _digest(facts)
    seal_epoch_plan(plan)
    applicability = scope["interval_applicability"]
    applicability["plan_identity"] = plan["identity"]
    applicability["execution_identity"] = plan["execution_identity"]
    applicability["capture_facts_identity"] = plan["capture_facts_identity"]
    applicability["identity"] = _digest({k: v for k, v in applicability.items() if k != "identity"})


def test_explicit_scalar_fact_matches_request_with_default_lifetime(tmp_path):
    manifest, facts = scalar_manifest(tmp_path)
    validate_checked_execution_plans(manifest, capture_facts=facts)
    facts["captures"]["argument::alpha"]["lifetime"] = "invocation"
    bind_request_identity(manifest, facts)
    validate_checked_execution_plans(manifest, capture_facts=facts)


@pytest.mark.parametrize("change", ["missing", "undefined", "escape", "allocation", "storage", "lifetime", "numeric_false"])
def test_resealed_scalar_entry_claim_cannot_override_supplied_request(tmp_path, change):
    manifest, facts = scalar_manifest(tmp_path)
    supplied = facts["captures"]["argument::alpha"]
    if change == "missing":
        del facts["captures"]["argument::alpha"]
    elif change == "undefined":
        supplied["initialized"] = "none"
    elif change == "escape":
        supplied["escapes"] = True
    elif change == "allocation":
        supplied["allocation_changes"] = True
    elif change == "storage":
        supplied["storage"] = "unknown"
    elif change == "numeric_false":
        supplied["escapes"] = 0
    else:
        supplied["lifetime"] = "selected_window"
    bind_request_identity(manifest, facts)
    with pytest.raises(ValueError, match="scalar entry fact differs from the supplied request"):
        validate_checked_execution_plans(manifest, capture_facts=facts)


@pytest.mark.parametrize("change", [
    "unknown", "version", "definitions", "motion", "pricing", "overwrite", "root_alias", "slot_bool",
    "graph", "selection", "empty_units", "team_guard", "provider_artifact", "provider_mapping", "parameter_order",
    "parameter_value", "parameter_bounds", "unknown_provider", "missing_provider", "kind_bool",
    "readability_identity", "scalar_finite", "scalar_undefined",
    "unhashable_type", "unhashable_lifetime",
])
def test_resealed_source_record_cannot_change_authority_mapping_or_abi(issued, change):
    values = deepcopy(issued)
    record, _member, _procedure, _inputs, _resources, _owner, _manifest = values
    binding = record["execution_binding"]
    provider = binding["provider"]
    metadata = provider["metadata"]
    if change == "unknown":
        record["unknown"] = False
    elif change == "version":
        record["version"] = True
    elif change == "definitions":
        record["definitions_authorized"] = True
    elif change == "motion":
        record["motion_authorized"] = True
    elif change == "pricing":
        record["pricing"] = "available"
    elif change == "overwrite":
        record["resources"][1]["access"] = "overwrite"
    elif change == "root_alias":
        binding["canonical_resources"]["argument::b"] = "argument::a"
    elif change == "slot_bool":
        binding["registration_slots"]["argument::a"] = True
    elif change == "graph":
        record["source_graph"] = record["team"]["source_graph"] = "f" * 64
    elif change == "selection":
        record["selected_node_ids"] = record["team"]["selected_node_ids"] = ["other::step/0"]
    elif change == "empty_units":
        record["team"]["unit_proof_identities"] = []
    elif change == "team_guard":
        record["team"]["requires_existing_capture_alias_bounds_definition_and_FP_guards"] = False
    elif change == "provider_artifact":
        provider["cuda_source_sha256"] = metadata["cuda_sha256"] = "f" * 64
    elif change == "provider_mapping":
        metadata["mapping"] = "unchecked"
    elif change == "parameter_order":
        metadata["parameters"].reverse()
    elif change == "parameter_value":
        metadata["parameters"][0]["role"] = "value"
    elif change == "parameter_bounds":
        metadata["parameters"][-1]["axis"] += 1
    elif change == "unknown_provider":
        provider["initialization_authorized"] = True
    elif change == "missing_provider":
        del binding["provider"]
    elif change == "readability_identity":
        metadata["scalar_readability_identity"] = "f" * 64
    elif change == "scalar_finite":
        record["scalar_captures"][0]["authority"]["finite_or_constant_authorized"] = True
    elif change == "scalar_undefined":
        record["scalar_captures"][0]["authority"] = {
            "kind": "explicit checked whole-defined scalar entry fact", "finite_or_constant_authorized": False,
            "fact": {"storage": "stable", "initialized": "none", "allocation_changes": False, "escapes": False}}
    elif change == "unhashable_type":
        record["scalars"][0]["type"] = record["team"]["scalars"][0]["type"] = []
    elif change == "unhashable_lifetime":
        record["scalar_captures"][0]["authority"] = {
            "kind": "explicit checked whole-defined scalar entry fact", "finite_or_constant_authorized": False,
            "fact": {"storage": "stable", "initialized": "whole", "allocation_changes": False,
                     "escapes": False, "lifetime": {}}}
    else:
        record["scalars"][0]["kind"] = record["team"]["scalars"][0]["kind"] = True
    reseal(record)
    with pytest.raises(ValueError, match="source device task"):
        check(values)
