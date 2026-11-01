"""Fresh source issuance stays separate from provider packaging and placement."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from elmm_pipeline.gpu_pressure_contract import (
    _qualified_sources,
    prepare_poisfft_contract,
)

SOURCE = """module fft_dp
implicit none
type poisfft_solver3d
real(8)::scratch
end type
end module
module poisfft_dp
use fft_dp
implicit none
interface Execute
module procedure poisfft_solver3d__execute
end interface
contains
function make_solver() result(value)
type(poisfft_solver3d)::value
value%scratch=1.0_8
end function
subroutine poisfft_solver3d__execute(d,phi,rhs)
type(poisfft_solver3d),intent(inout)::d
real(8),intent(out)::phi(-1:,-1:,-1:)
real(8),intent(in)::rhs(0:,0:,0:)
d%scratch=2.0_8
phi=0.0_8
end subroutine
end module
module poissonsolvers
use poisfft_dp
implicit none
contains
subroutine poiss_poisfft(phi,rhs)
real(8),intent(inout)::phi(-1:,-1:,-1:)
real(8),intent(in)::rhs(0:,0:,0:)
type(poisfft_solver3d),save::solver
logical,save::called=.false.
if(.not.called)then
solver=make_solver()
called=.true.
endif
call Execute(solver,phi,rhs)
end subroutine
end module
"""


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return {"path": str(path), "sha256": digest(path)}


def write_json(path, record):
    path.write_text(json.dumps(record))
    return reference(path)


def fixture(tmp_path, monkeypatch, *, source=SOURCE):
    path = tmp_path / "poisson.f90"
    path.write_text(source)
    document = {"schema_version": 1, "source_inputs": {str(path): digest(path)}, "preserves_source_order": True,
                "configuration": {"defines": ["DPREC"]}, "dependencies": {}, "entries": [
                    {"source": str(path), "path": str(path), "sha256": digest(path),
                     "line_map": list(range(1, len(source.splitlines()) + 1))}]}
    opaque = write_json(tmp_path / "opaque.json", {"source_contract": {
        "caller": {"procedure": "poissonsolvers::poiss_poisfft", "sha256": digest(path)},
        "pinned_reference": {"sources": [reference(path)]}}})
    backend_path = tmp_path / "backend.json"
    backend_path.write_text('{"packaging_only":true}')
    backend = {"manifest": str(backend_path), "sha256": digest(backend_path)}
    numerical = write_json(tmp_path / "numeric.json", {
        "source_and_receiver_observability_qualification": dict(opaque), "prepared_backend": reference(backend_path)})

    def package(request, destination, *, fortran, qualification, original_body_sha256, task_contract_sha256):
        assert request == backend
        assert fortran == "/selected/gfortran"
        assert qualification == numerical
        record = {"version": 2, "kind": "external_device_provider",
                  "fortran": {"module": "pressure_interface", "config_type": "config_type",
                              "create": "create", "execute": "solve", "destroy": "destroy"},
                  "metadata_query": {"module": "pressure_query", "procedure": "query",
                      "opaque_formals": ["d"], "configuration_argument": "config", "read_only": True,
                      "configuration_components": [{"component": "solver%sizes", "kind": "int32", "count": 3}]},
                  "artifacts": {str(path): digest(path)},
                  "numerical_contract": {"identity": "a" * 64, "qualification": numerical,
                      "original_bodies": [original_body_sha256], "task_contracts": [task_contract_sha256]},
                  "execution": {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim",
                                "fallback": "prework_only"}, "pressure_backend": dict(backend)}
        return write_json(destination, record)

    # The real compiler resolver/proof still runs. Only the separately tested
    # native-library packaging step is replaced, so these tests need no NVCC.
    monkeypatch.setattr("elmm_pipeline.gpu_pressure_contract.prepare_poisfft_provider_manifest", package)
    return path, document, {"backend": backend, "fortran": "/selected/gfortran",
        "numerical_qualification": numerical, "opaque_qualification": opaque, "workspace_cap_bytes": 123456}


def test_fresh_contract_binds_original_call_and_partial_out_without_guessing_operator(tmp_path, monkeypatch):
    path, document, kwargs = fixture(tmp_path, monkeypatch)
    before = path.read_bytes()
    result = prepare_poisfft_contract([path], document, tmp_path / "prepared", **kwargs)
    contract = json.loads(Path(result["effect_contracts"]["path"]).read_text())
    task, = contract["procedures"][result["entry"]]["device_tasks"]
    proof = json.loads(Path(result["source_proof"]["path"]).read_text())
    facts = json.loads(Path(result["capture_facts"]["path"]).read_text())
    assert path.read_bytes() == before
    assert task["source"]["sha256"] == digest(path)
    assert task["source"]["span"] == [SOURCE.splitlines().index("call Execute(solver,phi,rhs)") + 1] * 2
    assert task["resources"][0]["effects"][0]["lower"] == [2, 2, 2]
    assert task["resources"][0]["effects"][0]["upper"][0] == {
        "descriptor": "size", "formal": "phi", "axis": 1, "offset": -2}
    assert proof["source_effects"]["entry_definition_kills"] == ["phi"]
    assert facts["captures"]["argument::rhs"]["initialized"] == "none"
    assert facts["captures"]["argument::phi"]["initialized"] == "none"
    assert [item["component"] for item in task["configuration"]] == [
        "solver%device_budget_bytes", "phi_lower_bounds", "rhs_lower_bounds"]
    assert result["gpu_policy"] == "sections"
    assert result["scope_execution"] == "graph"
    assert result["performance_claim"] is False
    with pytest.raises(FileExistsError):
        prepare_poisfft_contract([path], document, tmp_path / "prepared", **kwargs)


def test_intent_in_does_not_supply_caller_payload_definitions(tmp_path, monkeypatch):
    path, document, kwargs = fixture(tmp_path, monkeypatch)
    assert "real(8),intent(in)::rhs(0:,0:,0:)" in path.read_text()
    result = prepare_poisfft_contract([path], document, tmp_path / "prepared", **kwargs)
    facts = json.loads(Path(result["capture_facts"]["path"]).read_text())
    rhs = facts["captures"]["argument::rhs"]
    assert rhs["initialized"] == "none"
    assert "sections" not in rhs
    assert "source_sections" not in rhs
    assert "requires_registered_handle" not in rhs
    # A qualified exact read box is an obligation, not evidence that the
    # original caller initialized those bytes or the surrounding ghost cells.
    contract = json.loads(Path(result["effect_contracts"]["path"]).read_text())
    task, = contract["procedures"][result["entry"]]["device_tasks"]
    assert task["resources"][1]["effects"][0]["kind"] == "read"
    assert task["resources"][1]["effects"][0]["lower"] == [1, 1, 1]


def test_identical_source_at_new_path_receives_new_original_bindings(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    path, document, kwargs = fixture(old, monkeypatch)
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    new_path = fresh / path.name
    shutil.copyfile(path, new_path)
    item = document["entries"][0]
    document["source_inputs"] = {str(new_path): digest(new_path)}
    item.update(source=str(new_path), path=str(new_path))
    result = prepare_poisfft_contract([new_path], document, fresh / "prepared", **kwargs)
    facts = json.loads(Path(result["capture_facts"]["path"]).read_text())
    assert result["source"] == str(new_path)
    assert facts["sources"] == {str(new_path): digest(new_path)}
    assert str(path) not in facts["sources"]


@pytest.mark.parametrize("change", ["wrapper", "library", "receiver_qualification", "backend", "cap", "raw_analysis"])
def test_stale_or_unqualified_sources_never_issue_a_contract(tmp_path, monkeypatch, change):
    path, document, kwargs = fixture(tmp_path, monkeypatch)
    if change == "wrapper":
        original = json.loads(Path(kwargs["opaque_qualification"]["path"]).read_text())
        original["source_contract"]["caller"]["sha256"] = "b" * 64
        kwargs["opaque_qualification"] = write_json(tmp_path / "changed-opaque.json", original)
    elif change == "library":
        original = json.loads(Path(kwargs["opaque_qualification"]["path"]).read_text())
        original["source_contract"]["pinned_reference"]["sources"][0]["sha256"] = "b" * 64
        kwargs["opaque_qualification"] = write_json(tmp_path / "changed-opaque.json", original)
        numeric = json.loads(Path(kwargs["numerical_qualification"]["path"]).read_text())
        numeric["source_and_receiver_observability_qualification"] = kwargs["opaque_qualification"]
        kwargs["numerical_qualification"] = write_json(tmp_path / "changed-numeric.json", numeric)
    elif change == "receiver_qualification":
        original = json.loads(Path(kwargs["numerical_qualification"]["path"]).read_text())
        original["source_and_receiver_observability_qualification"]["sha256"] = "b" * 64
        kwargs["numerical_qualification"] = write_json(tmp_path / "changed-numeric.json", original)
    elif change == "backend":
        kwargs["backend"] = {**kwargs["backend"], "sha256": "b" * 64}
    elif change == "cap":
        kwargs["workspace_cap_bytes"] = True
    else:
        document = None
    with pytest.raises((ValueError, TypeError)):
        prepare_poisfft_contract([path], document, tmp_path / "rejected", **kwargs)
    assert not (tmp_path / "rejected").exists()


def test_extra_opaque_receiver_observer_is_rejected_by_real_compiler_proof(tmp_path, monkeypatch):
    source = SOURCE.replace("call Execute(solver,phi,rhs)", "call Execute(solver,phi,rhs)\nphi(0,0,0)=solver%scratch")
    path, document, kwargs = fixture(tmp_path, monkeypatch, source=source)
    from compiler.ir import CompilationError
    with pytest.raises(CompilationError, match="unqualified observer"):
        prepare_poisfft_contract([path], document, tmp_path / "rejected", **kwargs)


def test_library_path_identity_distinguishes_elmm_and_poisfft_main_sources(tmp_path):
    caller = str(tmp_path / "elmm/src/poisson.f90")
    library = str(tmp_path / "PoisFFT/src/main.f90")
    unrelated = str(tmp_path / "elmm/src/main.f90")
    analysis = SimpleNamespace(sources={caller: "a" * 64, library: "b" * 64, unrelated: "c" * 64},
                               inputs=SimpleNamespace(dependencies={}))
    routine = SimpleNamespace(qualified="poissonsolvers::poiss_poisfft", scope=SimpleNamespace(path=caller))
    opaque = {"source_contract": {"caller": {"procedure": routine.qualified, "sha256": "a" * 64},
              "pinned_reference": {"sources": [{"path": "/old/PoisFFT/src/main.f90", "sha256": "b" * 64}]}}}
    qualification = {"path": "/qualification.json", "sha256": "d" * 64}
    backend = {"manifest": "/backend.json", "sha256": "e" * 64}
    numerical = {"source_and_receiver_observability_qualification": qualification,
                 "prepared_backend": {"path": backend["manifest"], "sha256": backend["sha256"]}}
    _qualified_sources(analysis, routine, opaque, numerical, backend, qualification)
    analysis.sources[library] = "c" * 64
    with pytest.raises(ValueError, match="qualified library"):
        _qualified_sources(analysis, routine, opaque, numerical, backend, qualification)
