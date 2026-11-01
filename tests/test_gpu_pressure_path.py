"""The explicit path overlay retains one original frame and every native call."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from elmm_pipeline.build_receipts import file_identity
from elmm_pipeline.gpu_pressure_path import (
    materialize_supported_pressure_path,
    prepare_supported_pressure_selection,
)


def solver(name):
    return (f"subroutine {name}(phi,rhs)\nreal(8),intent(inout)::phi(-1:,-1:,-1:)\n"
            "real(8),intent(in)::rhs(0:,0:,0:)\nphi(1,1,1)=rhs(1,1,1)+1.0_8\nend subroutine\n")


SOURCE = ("module poissonsolvers\nimplicit none\ncontains\n" +
          "".join(solver(name) for name in ("poisssor", "poiss_poisfft", "poiss_poisfft_variable_z")) +
          "end module\nmodule multigrid2d\nimplicit none\ncontains\n" + solver("poissmg2d") +
          "end module\nmodule multigrid\nimplicit none\ncontains\n" + solver("poissmg") + "end module\n" +
          """module pressure
use poissonsolvers
use multigrid2d
use multigrid
implicit none
integer::mode=2,grid=0
contains
subroutine PressureCorrection(u,n)
real(8),intent(inout)::u(:)
integer,intent(in)::n
real(8),save,allocatable::Phi(:,:,:),RHS(:,:,:)
integer,save::called=0
if(called==0)then
 allocate(Phi(-1:n+2,-1:n+2,-1:n+2))
 allocate(RHS(0:n+1,0:n+1,0:n+1))
 Phi=0
endif
called=called+1
call PrePoisson(u,RHS,n)
if(mode==1)then
 call PoissSOR(Phi,RHS)
else if(mode==2)then
 if(grid==1)then
  call Poiss_PoisFFT_variable_z(Phi,RHS)
 else
  call Poiss_PoisFFT(Phi,RHS)
 endif
else if(mode==3)then
 if(n==1)then
  call PoissMG2d(Phi,RHS)
 else
  call PoissMG(Phi,RHS)
 endif
endif
call PostPoisson(u,Phi)
end subroutine
subroutine PrePoisson(u,rhs,called)
real(8),intent(in)::u(:)
real(8),intent(out)::rhs(0:,0:,0:)
integer,intent(in)::called
rhs=real(called,8)+u(1)
end subroutine
subroutine PostPoisson(u,phi)
real(8),intent(inout)::u(:)
real(8),intent(in)::phi(-1:,-1:,-1:)
u=u+phi(1,1,1)
end subroutine
end module
""")


def fixture(tmp_path, source=SOURCE):
    path = tmp_path / "pressure.f90"
    path.write_text(source)
    document = {"schema_version": 1, "source_inputs": {str(path): file_identity(path)["sha256"]},
                "preserves_source_order": True, "configuration": {"defines": ["DPREC"]}, "dependencies": {},
                "entries": [{"source": str(path), **file_identity(path),
                             "line_map": list(range(1, len(source.splitlines())+1))}]}
    return path, document


def prepare(tmp_path, path, document):
    return materialize_supported_pressure_path([path], document, tmp_path / "selected",
                                              original_source=file_identity(path))


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_overlay_preserves_every_original_byte_and_keeps_one_save_frame(tmp_path, newline):
    source = SOURCE.replace("\n", newline)
    path, document = fixture(tmp_path, source)
    before = path.read_bytes()
    record = prepare(tmp_path, path, document)
    changed = Path(record["replacement"]["path"]).read_bytes().decode()
    restored = changed.splitlines(keepends=True)
    for edit in reversed(record["edits"]):
        lower, upper = edit["replacement_span"]
        assert "".join(restored[lower-1:upper]) == edit["text"]
        del restored[lower-1:upper]
    assert "".join(restored).encode() == before
    assert path.read_bytes() == before
    assert changed.count("subroutine PressureCorrection(") == 1
    assert changed.count("real(8),save,allocatable::Phi(:,:,:),RHS(:,:,:)") == 1
    assert changed.lower().count("call ") == SOURCE.lower().count("call ")
    assert record["requires_fresh_configured_source_export"] is True
    assert record["ready_for_execution"] is False
    assert record["eligibility"]["rearm_after_excluded_edge"] is False
    assert len(record["edits"]) == 6
    for field in ("source_authority", "lifetime_authority", "payload_definition_authority",
                  "execution_authority", "performance_claim"):
        assert record[field] is False
    assert json.loads((tmp_path / "selected/source-implementation.json").read_text()) == record


def test_fresh_replacement_issues_typed_sticky_gate_but_no_execution_authority(tmp_path):
    from fparser.two.utils import walk

    from compiler.frontend.allocation_epoch_eligibility import prove_allocation_gate
    from compiler.frontend.call_bindings import resolve_original_call
    from compiler.frontend.source_effects import SourceEffects

    path, document = fixture(tmp_path)
    record = prepare(tmp_path, path, document)
    # This fixture has no preprocessing. A new identity export is authentic
    # for these new bytes; production requires the real extractor anew.
    replacement = Path(record["replacement"]["path"])
    analysis = SourceEffects([replacement])
    routine = analysis.routines["pressure::pressurecorrection"]
    barred = set(record["eligibility"]["excluded_calls"])
    calls = [node for node in walk(routine.execution) if type(node).__name__ == "Call_Stmt"]
    barriers = [node for node in calls if resolve_original_call(
        analysis, analysis.source_scope_for(node, routine.scope), node).procedure in barred]
    proof = prove_allocation_gate(analysis, routine, routine.scope.bindings[record["eligibility"]["binding"]],
        [routine.scope.bindings[name] for name in ("phi", "rhs")], barriers)
    assert len(proof.public()["barriers"]) == 4
    assert proof.public()["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["reference", "analysis", "missing_edge", "existing_gate", "different_init", "target"])
def test_source_pin_or_frame_mismatch_rejects_before_publishing(tmp_path, mutation):
    source = SOURCE
    if mutation == "missing_edge":
        source = source.replace("call PoissSOR(Phi,RHS)", "call Poiss_PoisFFT(Phi,RHS)")
    elif mutation == "existing_gate":
        source = source.replace("integer,save::called=0", "integer,save::called=0\nlogical::elmm_pressure_epoch_eligible")
    elif mutation == "different_init":
        source = source.replace("Phi=0", "Phi=1")
    elif mutation == "target":
        source = source.replace("real(8),save,allocatable::Phi", "real(8),save,target,allocatable::Phi")
    path, document = fixture(tmp_path, source)
    reference = file_identity(path)
    if mutation == "reference":
        reference["sha256"] = "0"*64
    elif mutation == "analysis":
        document = None
    with pytest.raises((ValueError, TypeError)):
        materialize_supported_pressure_path([path], document, tmp_path / "rejected", original_source=reference)
    assert not (tmp_path / "rejected").exists()


def test_original_and_overlay_keep_native_state_and_arithmetic_bitwise(tmp_path):
    compiler = shutil.which("gfortran")
    if compiler is None:
        pytest.skip("gfortran unavailable")
    path, document = fixture(tmp_path)
    record = prepare(tmp_path, path, document)
    driver = tmp_path / "driver.f90"
    driver.write_text("""program driver
use iso_fortran_env,only:int64
use pressure
implicit none
real(8)::u(4)=[1.0_8,2.0_8,3.0_8,4.0_8]
integer::i
do i=1,6
mode=2
if(i==2)mode=1
if(i==4)mode=3
grid=0
if(i==5)grid=1
call PressureCorrection(u,2)
write(*,'(4z18.16)')transfer(u,[0_int64,0_int64,0_int64,0_int64])
enddo
end program
""")
    outputs = []
    for index, source in enumerate((path, Path(record["replacement"]["path"]))):
        build = tmp_path / ("build"+str(index))
        build.mkdir()
        executable = build / "fixture"
        subprocess.run([compiler, "-O0", "-fcheck=all", "-J", str(build), "-I", str(build),
                        str(source), str(driver), "-o", str(executable)], check=True, capture_output=True, text=True)
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert len(outputs[0].splitlines()) == 6


def fresh_export(path):
    source = path.read_text()
    return {"schema_version": 1, "source_inputs": {str(path): file_identity(path)["sha256"]},
        "preserves_source_order": True, "configuration": {"defines": ["DPREC"]}, "dependencies": {},
        "entries": [{"source": str(path), **file_identity(path),
                     "line_map": list(range(1, len(source.splitlines())+1))}]}


def test_fresh_export_prepares_explicit_selector_but_never_fabricates_capture_or_provider_facts(tmp_path):
    original, export = fixture(tmp_path)
    implementation = prepare(tmp_path, original, export)
    changed = Path(implementation["replacement"]["path"])
    prepared = prepare_supported_pressure_selection([changed], fresh_export(changed), tmp_path / "selector",
        implementation=file_identity(tmp_path / "selected/source-implementation.json"))
    assert prepared["ready_for_execution"] is False
    assert not (tmp_path / "selector/capture-facts.json").exists()
    contracts = json.loads(Path(prepared["effect_contracts"]["path"]).read_text())
    selector = contracts["procedures"]["pressure::pressurecorrection"]["native_ancestor"]
    assert selector["version"] == 4
    assert selector["lifetime"] == "supported_original_path"
    request = selector["path"]
    assert request["source"] == implementation["replacement"]
    assert request["eligibility"]["binding"] == "elmm_pressure_epoch_eligible"
    assert request["eligibility"]["resources"] == ["pressure::pressurecorrection::phi", "pressure::pressurecorrection::rhs"]
    assert [anchor["span"] for anchor in request["eligibility"]["barred_calls"]] == sorted(
        anchor["span"] for anchor in request["eligibility"]["barred_calls"])
    gate = json.loads(Path(prepared["source_gate"]["path"]).read_text())
    assert gate["execution_authorized"] is False
    assert "independent caller facts" in prepared["capture_facts"]
    assert "separately prepared" in prepared["provider"]


def test_original_first_call_clock_precedes_the_last_arming_and_is_qualified_as_metadata(tmp_path):
    source = SOURCE.replace("integer,save::called=0", "integer,save::called=0\ninteger,save::rate").replace(
        " Phi=0\nendif", " Phi=0\n if(mode>=1)call system_clock(count_rate=rate)\nendif")
    original, export = fixture(tmp_path, source)
    implementation = prepare(tmp_path, original, export)
    changed = Path(implementation["replacement"]["path"])
    text = changed.read_text()
    assert text.index("call system_clock(count_rate=rate)") < text.index("elmm_pressure_epoch_eligible = .true.")
    prepared = prepare_supported_pressure_selection([changed], fresh_export(changed), tmp_path / "selector",
        implementation=file_identity(tmp_path / "selected/source-implementation.json"))
    gate = json.loads(Path(prepared["source_gate"]["path"]).read_text())
    assert gate["version"] == 2
    metadata, = gate["prearm_metadata"]
    assert "SYSTEM_CLOCK" in metadata["original_node"]["source"].upper()
    assert gate["execution_authorized"] is False


@pytest.mark.parametrize("mutation", ["old_export", "record_authority", "recipe_reason", "recipe_text", "recipe_order"])
def test_selection_cannot_retag_old_export_or_insert_physics_through_a_changed_recipe(tmp_path, mutation):
    from compiler.ir import CompilationError

    original, export = fixture(tmp_path)
    implementation = prepare(tmp_path, original, export)
    changed = Path(implementation["replacement"]["path"])
    document = fresh_export(changed)
    manifest = tmp_path / "selected/source-implementation.json"
    if mutation == "old_export":
        document = export
    else:
        record = json.loads(manifest.read_text())
        if mutation == "record_authority":
            record["payload_definition_authority"] = True
        elif mutation == "recipe_reason":
            record["edits"][0]["reason"] = "ordinary arithmetic"
        elif mutation == "recipe_text":
            record["edits"][1]["text"] += "Phi=9\n"
        else:
            record["edits"] = list(reversed(record["edits"]))
        manifest.write_text(json.dumps(record)+"\n")
    with pytest.raises((ValueError, CompilationError)):
        prepare_supported_pressure_selection([changed], document, tmp_path / "rejected-selector",
                                            implementation=file_identity(manifest))
    assert not (tmp_path / "rejected-selector").exists()
