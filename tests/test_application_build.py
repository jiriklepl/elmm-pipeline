"""Transport build authority through the explicit compiler snapshot only."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from elmm_pipeline.application_build import selection_builds, verify
from elmm_pipeline.build import build
from elmm_pipeline.local_adapter import LocalAdapterError, transform_scopes


def test_public_verifier_transport_preserves_paths_and_blocks_failure(tmp_path):
    compiler = tmp_path / "compiler with spaces"
    (compiler / "compiler/offload").mkdir(parents=True)
    (compiler / "compiler/offload/application_build.py").touch()
    transport = {"compiler_root": str(compiler), "python": sys.executable}
    calls = []

    def invoke(argv, **kwargs):
        calls.append((argv, kwargs))
        assert kwargs["cwd"] == compiler and kwargs["env"]["PYTHONPATH"] == str(compiler)
        request = json.loads(Path(argv[argv.index("--request") + 1]).read_text())
        assert request == {"manifest": "literal $(unchanged) path"}
        Path(argv[argv.index("--output") + 1]).write_text(json.dumps({"identity": "verified"}))
        return subprocess.CompletedProcess(argv, 0, json.dumps({"status": "verified", "identity": "verified"}), "")

    with patch("elmm_pipeline.application_build.subprocess.run", side_effect=invoke):
        assert verify(transport, "preflight", {"manifest": "literal $(unchanged) path"}, tmp_path / "evidence")["identity"] == "verified"
    assert calls[0][0][:4] == [sys.executable, "-m", "compiler.offload.application_build", "preflight"]
    with patch("elmm_pipeline.application_build.subprocess.run", return_value=subprocess.CompletedProcess([], 1, "", "actual argv differs")), \
            pytest.raises(ValueError, match="binary delivery blocked"):
        verify(transport, "attest", {}, tmp_path / "evidence")


def test_old_exact_fixture_protocol_remains_rejected():
    with pytest.raises(ValueError, match="supported executed application"):
        selection_builds({"scopes": [{"exact_window": {"contract": {"protocol": "complete-original-window-cold-v4"}}}]})


def test_static_automatic_native_diagnostic_requires_no_selector_build():
    record = {"selection": "native", "capture": False, "reason": "unavailable calibration",
              "contract": {"protocol": "checked-selection-process-v1",
                           "build_contract": {"protocol": "executed-application-build-v1"}}}
    assert selection_builds({"scopes": [{"exact_window": record}]}) == []
    record["capture"] = True
    record["capture_variant"] = "native"
    assert selection_builds({"scopes": [{"exact_window": record}]}) == [record]


def test_scope_cost_transport_is_hashed_and_not_reinterpreted(tmp_path):
    source = tmp_path / "source.f90"
    source.write_text("subroutine step\nend\n")
    cost = tmp_path / "cost.json"
    cost.write_text('{"coverage":"original_selection"}')
    with patch("elmm_pipeline.local_adapter._invoke", side_effect=LocalAdapterError("transport sentinel")) as invoke, \
            pytest.raises(LocalAdapterError, match="transport sentinel"):
        transform_scopes(source, tmp_path, Path(sys.executable), tmp_path / "artifacts", entry="step", capture_facts={},
                         gpu_options={"gpu_policy": "auto", "scope_cost": str(cost)})
    arguments = invoke.call_args.args[-2]
    assert arguments[arguments.index("--scope-cost") + 1] == str(cost)

    def change_request(*_args):
        cost.write_text("{}")
        return {}, []

    with patch("elmm_pipeline.local_adapter._invoke", side_effect=change_request), pytest.raises(LocalAdapterError, match="cost request changed"):
        transform_scopes(source, tmp_path, Path(sys.executable), tmp_path / "other", entry="step", capture_facts={},
                         gpu_options={"gpu_policy": "auto", "scope_cost": str(cost)})


def test_post_build_failure_prevents_returning_binary(tmp_path):
    application, fft, directory = tmp_path / "ELMM", tmp_path / "PoisFFT", tmp_path / "build"
    (application / "src").mkdir(parents=True)
    (fft / "src").mkdir(parents=True)
    (application / "src/SConstruct").write_text("t = env.Program('program', [])\n")
    (fft / "src/poisfft.f90").touch()

    def execute(argv, **_kwargs):
        (directory / "ELMM").write_text("compiled binary exists")
        return subprocess.CompletedProcess(argv, 0)

    with patch("elmm_pipeline.application_build.preflight", return_value={"macros": {}}) as before, \
            patch("elmm_pipeline.build.subprocess.run", side_effect=execute), \
            patch("elmm_pipeline.build.summarize_build", return_value={"status": "complete", "identity": "receipt"}), \
            patch("elmm_pipeline.application_build.attest", side_effect=ValueError("actual link changed")) as after, \
            pytest.raises(ValueError, match="actual link changed"):
        build(application, fft, directory, Path(sys.executable), tmp_path, 1, 4, [], application_build={})
    assert before.call_count == after.call_count == 1
    assert (directory / "ELMM").is_file()
    assert "application_build" not in json.loads((directory / "build-command.json").read_text())
