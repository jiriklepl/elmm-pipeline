"""Diagnostic call clocks never supply pricing or placement authority."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


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
                compiler_root=Path(__file__).resolve().parents[2], target="gpu", gpu_options=options, scope_options={"entry": "measurement_application::step",
                                                  "source": source.name, "capture_facts": facts})
        assert source.read_bytes() == original
