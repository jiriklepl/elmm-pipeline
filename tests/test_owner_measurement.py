"""Diagnostic call clocks never supply pricing or placement authority."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal(record):
    record["identity"] = digest({key: value for key, value in record.items() if key not in {"identity", "variant"}})


def fixture(tmp_path, variant="native"):
    source = tmp_path / "caller.f90"
    source.write_text("subroutine caller\ncall step()\nend subroutine\n")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    plan = {name: digest(name) for name in (
        "execution_identity", "source_inputs_identity", "configuration_identity", "capture_facts_identity")}
    call = {"caller": "caller", "source": str(source), "source_sha256": sha,
            "first_line": 2, "last_line": 2, "call_sha256": hashlib.sha256(b"call step()\n").hexdigest()}
    call["identity"] = digest(call)
    runtime = {"schema_version": 1, "runtime_id": "synthetic runtime"}
    record = {"schema_version": 1, "protocol": "complete-lexical-call-clock-v1", "variant": variant,
              **plan, "runtime_identity": digest(runtime), "compiler_identity": digest("compiler"),
              "call_sites": [call], "placement_authority": False, "automatic_estimate_available": False,
              "calibration_evidence": False,
              "interval": "before original call argument evaluation through return and finalization",
              "clock": "system_clock:int64", "observation_prefix": "FORT_SCOPE_MEASUREMENT ",
              "limitations": ["No runtime workload key; observations are not automatic placement evidence.",
                  "Resident variant may execute guarded native fallbacks; verify runtime traces separately.",
                  "Only listed direct call sites in registered subroutines of the authenticated supplied source graph are observed."]}
    seal(record)
    return {"scope_measurement": variant, "scope_execution": "reached", "scope_count": 1,
            "automatic_estimate_available": False, "scopes": [{"checked_execution_plan": plan,
                "owner_measurement": record}], "source_inputs": {str(source): sha},
            "sources": {str(source): {"sha256": sha, "replacement": "sources/caller.f90"}},
            "build_sources": [{"path": "sources/caller.f90", "language": "fortran", "role": "original_source"}],
            "runtime": runtime if variant == "resident" else None}


def test_public_compiler_rejects_removed_measurement_options_before_source_edits(tmp_path):
    from elmm_pipeline.adaptation import adapt_project
    from elmm_pipeline.local_adapter import LocalAdapterError, transform_scopes

    application = tmp_path / "application"
    application.mkdir()
    source = application / "original.f90"
    source.write_text("module measurement_application\ncontains\nsubroutine step(a,b,n)\n"
                      "integer,intent(in)::n\nreal(8),intent(in)::a(n)\nreal(8),intent(out)::b(n)\ninteger::i\n"
                      "!$omp parallel do\ndo i=1,n\nb(i)=2*a(i)\nenddo\n!$omp end parallel do\n"
                      "end subroutine\nend module\n")
    original = source.read_bytes()
    stable = {"storage": "stable", "initialized": "whole", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial", "captures": {
        "argument::a": stable, "argument::b": {**stable, "initialized": "none"}}}
    for variant in ("native", "resident"):
        options = {"gpu_policy": "sections", "scope_execution": "reached", "scope_measurement": variant}
        output = tmp_path / variant
        with patch("elmm_pipeline.local_adapter._invoke", side_effect=AssertionError("removed producer cannot run")), \
                pytest.raises(LocalAdapterError, match="Retired public source options: scope_measurement"):
            transform_scopes(source, tmp_path, Path(sys.executable), output,
                entry="measurement_application::step", capture_facts=facts, gpu_options=options)
        assert source.read_bytes() == original
        assert not (output / "scope-transformation.json").exists()
        with patch("elmm_pipeline.gpu_toolchain.compile_gpu_sources", side_effect=AssertionError("removed producer cannot build")), \
                pytest.raises(LocalAdapterError, match="Retired public source options: scope_measurement"):
            adapt_project("local", application, Path(sys.executable), tmp_path / (variant + "-adaptation"),
                compiler_root=Path(__file__).resolve().parents[2], target="gpu", memory_model="scoped",
                gpu_options=options, scope_options={"entry": "measurement_application::step",
                                                  "source": source.name, "capture_facts": facts})
        assert source.read_bytes() == original
