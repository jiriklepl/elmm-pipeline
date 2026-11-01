"""Public preparation binds an explicit library premise, never infers one."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from elmm_pipeline import gpu_pressure_lazy
from elmm_pipeline.gpu_pressure_lazy import prepare_lazy_poisfft_contract

ROOT = Path(__file__).resolve().parents[2]
LIBRARY = """module PoisFFT
implicit none
integer,parameter::PoisFFT_FiniteDifference2=2,PoisFFT_FiniteDifference4=4
type PoisFFT_Solver3D_DP
integer::nxyz(3)
end type
interface PoisFFT_Solver3D_DP
module procedure make_solver
end interface
interface Execute
module procedure native_execute
end interface
interface Finalize
module procedure native_finalize
end interface
contains
function make_solver(nxyz,lengths,bcs,approximation) result(d)
integer,intent(in)::nxyz(3),bcs(6),approximation
real(8),intent(in)::lengths(3)
type(PoisFFT_Solver3D_DP)::d
d%nxyz=nxyz
end function
subroutine native_execute(d,phi,rhs)
type(PoisFFT_Solver3D_DP),intent(inout)::d
real(8),intent(out)::phi(:,:,:)
real(8),intent(in)::rhs(:,:,:)
phi=rhs
end subroutine
subroutine native_finalize(d)
type(PoisFFT_Solver3D_DP),intent(inout)::d
d%nxyz=0
end subroutine
end module
"""
WRAPPER = """module poissonsolvers
use elmm_gpu_pressure_lazy_poisfft, only: PoisFFT_Solver3D_DP,Execute
implicit none
contains
subroutine Poiss_PoisFFT(phi,rhs)
real(8),intent(inout)::phi(-1:,-1:,-1:)
real(8),intent(in)::rhs(0:,0:,0:)
type(PoisFFT_Solver3D_DP),save::solver
logical,save::called=.false.
if(.not.called)then
solver=PoisFFT_Solver3D_DP([8,8,8],[1.0_8,1.0_8,1.0_8],[0,0,0,0,0,0],2)
called=.true.
endif
call Execute(solver,phi,rhs)
end subroutine
end module
"""
CLAIM = "unconditional original entry reads before any same-root write under accepted query and layout predicates"
FLAGS = {"may_read_permissions_are_entry_requirements": False, "unreached_definition_authority": False,
         "halo_definition_authority": False, "alias_authority": False,
         "lifetime_authority": False, "execution_authority": False}
QUERY = {"module": "elmm_gpu_pressure_lazy_poisfft", "procedure": "elmm_gpu_pressure_configure_lazy_v1",
         "opaque_formals": ["d"], "configuration_argument": "config", "read_only": True,
         "configuration_components": [{"component": "solver%sizes", "kind": "int32", "count": 3},
             {"component": "solver%lengths", "kind": "real64", "count": 3},
             {"component": "solver%boundary_conditions", "kind": "int32", "count": 6},
             {"component": "solver%approximation", "kind": "int32", "count": 1}]}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ref(path):
    return {"path": str(path), "sha256": digest(path)}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")
    return ref(path)


def fixture(tmp_path, monkeypatch):
    from compiler.frontend.source_effects import SourceEffects

    library = tmp_path / "poisfft.f90"
    library.write_text(LIBRARY)
    wrapper = tmp_path / "poisson.f90"
    wrapper.write_text(WRAPPER)
    package = Path(gpu_pressure_lazy.__file__).parent
    implementation = package / "gpu_pressure_sources/gpu_pressure_lazy_poisfft.f90"
    paths = [library, wrapper, implementation, package / "gpu_pressure_sources/gpu_pressure.f90",
             ROOT / "elmm-pipeline/vendor/poisson-solver/src/device_binding.f90", ROOT / "compiler/runtime/scoped_memory.f90"]
    document = {"schema_version": 1, "source_inputs": {str(path): digest(path) for path in paths},
        "preserves_source_order": True, "configuration": {"defines": ["DPREC"]}, "dependencies": {},
        "entries": [{"source": str(path), "path": str(path), "sha256": digest(path),
                     "line_map": list(range(1, len(path.read_text().splitlines()) + 1))} for path in paths]}
    opaque = write(tmp_path / "opaque.json", {"kind": "explicit_lazy_cpu_poisfft_receiver_qualification",
        "implementation": ref(implementation), "source_contract": {
            "caller": {"procedure": "poissonsolvers::poiss_poisfft", "sha256": digest(wrapper)},
            "pinned_reference": {"sources": [ref(library)]}}})
    backend = {"manifest": str(tmp_path / "backend.json"), "sha256": ""}
    write(Path(backend["manifest"]), {"test_only_packaging": True})
    backend["sha256"] = digest(backend["manifest"])
    numerical = write(tmp_path / "numeric.json", {"source_and_receiver_observability_qualification": opaque,
                                                "prepared_backend": ref(backend["manifest"])})
    analysis = SourceEffects(paths, analysis_sources=document)
    callee = analysis.routines["elmm_gpu_pressure_lazy_poisfft::lazy_execute"]
    evidence = write(tmp_path / "review.json", {"test_only": True,
        "claim": "explicit fixture library read precondition, no execution qualification", "library": ref(library)})
    premise = {"version": 1, "kind": "reviewed_pinned_poisfft_entry_read_prerequisite", "claim": CLAIM,
        "authority": deepcopy(FLAGS), "pinned_sources": [ref(library)], "implementation_sha256": digest(implementation),
        "callee": {"procedure": callee.qualified,
            "body_sha256": hashlib.sha256(str(callee.scope.node).encode()).hexdigest()},
        "metadata_query": deepcopy(QUERY), "required_reads": [{"formal": "rhs", "lower": [1] * 3,
            "upper": [{"descriptor": "size", "formal": "rhs", "axis": axis, "offset": -1} for axis in range(1, 4)]}],
        "layout_equalities": [{"component": "solver%sizes", "index": axis, "descriptor": {
            "descriptor": "size", "formal": formal, "axis": axis, "offset": offset}}
            for formal, offset in (("rhs", -2), ("phi", -4)) for axis in range(1, 4)],
        "configuration": {"required_defines": ["DPREC"], "forbidden_defines": ["PAR", "MPI"]},
        "evidence": [evidence]}
    prerequisite = write(tmp_path / "prerequisite.json", premise)
    monkeypatch.setattr("elmm_pipeline.gpu_pressure_build.require_pressure_backend", lambda *args, **kwargs:
        {"requirements": {"cpu_fallback": {"implementation": "lazy_poisfft_v1", "source": ref(implementation)}},
         "extra_sources": [str(implementation)]})
    packages = []

    def package(request, destination, *, fortran, qualification, original_body_sha256, task_contract_sha256):
        assert request == backend
        assert fortran == "/selected/gfortran"
        assert qualification == numerical
        provider = {"version": 2, "kind": "external_device_provider", "fortran": {"module": "elmm_gpu_pressure",
            "config_type": "elmm_gpu_pressure_config_v1", "create": "elmm_gpu_pressure_create_v1",
            "execute": "elmm_gpu_pressure_solve_v1", "destroy": "elmm_gpu_pressure_destroy_v1"},
            "metadata_query": deepcopy(QUERY), "artifacts": {str(path): digest(path) for path in paths[2:]},
            "numerical_contract": {"identity": "a" * 64, "qualification": numerical,
                "original_bodies": [original_body_sha256], "task_contracts": [task_contract_sha256]},
            "execution": {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim",
                "fallback": "prework_only"}, "pressure_backend": dict(backend)}
        packages.append(provider)
        return write(destination, provider)

    monkeypatch.setattr("elmm_pipeline.gpu_pressure_contract.prepare_poisfft_provider_manifest", package)
    kwargs = {"backend": backend, "fortran": "/selected/gfortran", "numerical_qualification": numerical,
        "opaque_qualification": opaque, "workspace_cap_bytes": 123456,
        "required_entry_read_prerequisite": prerequisite}
    return paths, document, kwargs, premise, packages


def test_public_preparation_returns_final_provider_and_fresh_read_proof(tmp_path, monkeypatch):
    from elmm_pipeline.external_entry_read_record import validate_external_entry_reads

    from compiler.frontend.device_tasks import task_scope_identity

    paths, document, kwargs, premise, packages = fixture(tmp_path, monkeypatch)
    result = prepare_lazy_poisfft_contract(paths, document, tmp_path / "prepared", **kwargs)
    task, = json.loads(Path(result["effect_contracts"]["path"]).read_text())["procedures"][result["entry"]]["device_tasks"]
    proof = json.loads(Path(result["source_proof"]["path"]).read_text())
    reads = proof["source_effects"]["entry_read_requirements"]
    facts = json.loads(Path(result["capture_facts"]["path"]).read_text())
    assert {fact["initialized"] for fact in facts["captures"].values()} == {"none"}
    assert reads["canonical_resources"] == {"rhs": "argument::rhs"}
    assert reads["qualification"] == result["entry_read_requirements"] == task["entry_read_requirements"]
    assert reads["qualification_record"]["layout_equalities"] == premise["layout_equalities"]
    assert proof["source_effects"]["entry_definition_kills"] == ["phi"]
    assert all(reads[key] is False for key in FLAGS)
    assert result["provider_manifest"] == task["provider_manifest"]
    assert Path(result["provider_manifest"]["path"]).name == "provider-entry-reads.json"
    assert len(packages) == 2
    assert packages[-1]["numerical_contract"]["task_contracts"] == [task_scope_identity(task)]
    assert packages[0]["numerical_contract"]["task_contracts"] != packages[-1]["numerical_contract"]["task_contracts"]
    ranks = {item["formal"]: item["formal_descriptor"]["rank"] for item in proof["call"]["associations"]
             if item["formal_descriptor"]["rank"]}
    validate_external_entry_reads(reads, proof, ranks, proof["call"]["associations"])


def test_no_optional_premise_preserves_the_original_no_admission_contract(tmp_path, monkeypatch):
    paths, document, kwargs, _, packages = fixture(tmp_path, monkeypatch)
    kwargs.pop("required_entry_read_prerequisite")
    result = prepare_lazy_poisfft_contract(paths, document, tmp_path / "prepared", **kwargs)
    task, = json.loads(Path(result["effect_contracts"]["path"]).read_text())["procedures"][result["entry"]]["device_tasks"]
    proof = json.loads(Path(result["source_proof"]["path"]).read_text())
    assert "entry_read_requirements" not in task
    assert proof["source_effects"].get("entry_read_requirements") is None
    assert "required_entry_read_prerequisite" not in result
    assert "entry_read_requirements" not in result
    assert len(packages) == 1


@pytest.mark.parametrize("change", ["may_read", "conditional", "kind", "library", "implementation", "body", "query",
    "region", "phi", "missing_guard", "guard_offset", "authority_integer", "configuration", "evidence", "PAR", "MPI", "DPREC"])
def test_explicit_premise_must_match_the_full_pinned_read_precondition(tmp_path, monkeypatch, change):
    paths, document, kwargs, premise, _ = fixture(tmp_path, monkeypatch)
    if change == "may_read":
        premise["authority"]["may_read_permissions_are_entry_requirements"] = True
    elif change == "conditional":
        premise["claim"] = "read only under an additional source condition"
    elif change == "kind":
        premise["kind"] = "numerical_equivalence"
    elif change == "library":
        premise["pinned_sources"][0]["sha256"] = "0" * 64
    elif change == "implementation":
        premise["implementation_sha256"] = "0" * 64
    elif change == "body":
        premise["callee"]["body_sha256"] = "0" * 64
    elif change == "query":
        premise["metadata_query"]["procedure"] = "different_query"
    elif change == "region":
        premise["required_reads"][0]["lower"][0] = 0
    elif change == "phi":
        premise["required_reads"][0]["formal"] = "phi"
    elif change == "missing_guard":
        premise["layout_equalities"].pop()
    elif change == "guard_offset":
        premise["layout_equalities"][0]["descriptor"]["offset"] = -1
    elif change == "authority_integer":
        premise["authority"]["halo_definition_authority"] = 0
    elif change == "configuration":
        premise["configuration"]["forbidden_defines"] = ["PAR"]
    elif change == "evidence":
        premise["evidence"][0]["sha256"] = "0" * 64
    elif change == "DPREC":
        document["configuration"]["defines"] = []
    else:
        document["configuration"]["defines"].append(change)
    kwargs["required_entry_read_prerequisite"] = write(tmp_path / "changed-prerequisite.json", premise)
    with pytest.raises(ValueError, match="PoisFFT|Pressure"):
        prepare_lazy_poisfft_contract(paths, document, tmp_path / "rejected", **kwargs)
    assert not (tmp_path / "rejected/prepared-contract.json").exists()


@pytest.mark.parametrize("change", ["provider", "prerequisite", "evidence", "source", "required_record"])
def test_final_preparation_rejects_changes_during_provider_binding(tmp_path, monkeypatch, change):
    import elmm_pipeline.gpu_pressure_contract as contracts

    paths, document, kwargs, premise, _ = fixture(tmp_path, monkeypatch)
    package = contracts.prepare_poisfft_provider_manifest
    count = 0

    def changing_package(*args, **kw):
        nonlocal count
        count += 1
        result = package(*args, **kw)
        if count == 2:
            if change == "provider":
                record = json.loads(Path(result["path"]).read_text())
                record["execution"]["fallback"] = "different_implementation"
                result = write(Path(result["path"]), record)
            elif change == "prerequisite":
                Path(kwargs["required_entry_read_prerequisite"]["path"]).write_text("{}\n")
            elif change == "evidence":
                Path(premise["evidence"][0]["path"]).write_text("{}\n")
            elif change == "source":
                paths[1].write_text(paths[1].read_text() + "! changed after fresh source binding\n")
            else:
                (tmp_path / "prepared/required-entry-reads.json").write_text("{}\n")
        return result

    monkeypatch.setattr(contracts, "prepare_poisfft_provider_manifest", changing_package)
    from compiler.ir import CompilationError
    with pytest.raises((ValueError, CompilationError)):
        prepare_lazy_poisfft_contract(paths, document, tmp_path / "prepared", **kwargs)
    assert not (tmp_path / "prepared/prepared-contract.json").exists()


def test_entry_read_premise_cannot_qualify_a_different_original_call(tmp_path, monkeypatch):
    paths, document, kwargs, _, _ = fixture(tmp_path, monkeypatch)
    paths[1].write_text(WRAPPER.replace("call Execute(solver,phi,rhs)", "call Execute(solver,rhs,phi)"))
    # A refreshed configured export cannot silently refresh the explicitly
    # reviewed caller/source qualification for changed actual associations.
    document["source_inputs"][str(paths[1])] = digest(paths[1])
    document["entries"][1]["sha256"] = digest(paths[1])
    from compiler.ir import CompilationError
    with pytest.raises((ValueError, CompilationError)):
        prepare_lazy_poisfft_contract(paths, document, tmp_path / "rejected", **kwargs)
