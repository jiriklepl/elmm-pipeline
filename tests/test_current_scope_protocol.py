"""Current graph transport and explicit frozen-control compatibility."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from elmm_pipeline import cli, local_adapter
from elmm_pipeline.common import COMPILER_PHASE_TIMEOUT_SECONDS
from test_local_scopes import FACTS


def test_compiler_phase_cap_is_240_seconds():
    assert COMPILER_PHASE_TIMEOUT_SECONDS == 240


@pytest.mark.parametrize("options", [
    {"scope_execution": "bounded"}, {"scope_execution": "reached"},
    {"scope_cost": None}, {"scope_measurement": None},
    {"scope_cost": "missing.json"}, {"scope_measurement": "native"},
])
def test_public_adapter_rejects_retired_options_before_creating_artifacts(tmp_path, options):
    output = tmp_path / "output"
    with (patch.object(local_adapter, "_invoke", side_effect=AssertionError("no compiler invocation")),
          pytest.raises(local_adapter.LocalAdapterError, match="[Rr]etired|graph execution")):
        local_adapter.transform_scopes(tmp_path / "absent.f90", tmp_path, Path(sys.executable),
                                      output, entry="step", capture_facts=FACTS, gpu_options=options)
    assert not output.exists()


@pytest.mark.parametrize("extra", [["--scope-execution", "bounded"], ["--scope-execution", "reached"],
                                   ["--scope-cost", "absent.json"], ["--scope-measurement", "native"]])
def test_public_cli_rejects_retired_arguments_before_run(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["pipeline", "run", *extra])
    with (patch.object(cli, "run", side_effect=AssertionError("no application build")),
          pytest.raises(SystemExit) as error):
        cli.main()
    assert error.value.code == 2


def test_public_cli_accepts_graph_and_defaults_to_shared_ownership(monkeypatch):
    for extra, expected in [([], None), (["--scope-execution", "graph"], "graph")]:
        monkeypatch.setattr(sys, "argv", ["pipeline", "run", *extra])
        with patch.object(cli, "run") as run:
            cli.main()
        assert run.call_args.args[0].scope_execution == expected
        assert run.call_args.args[0].memory_model == "scoped"
        assert run.call_args.args[0].gpu_policy == "sections"
        assert run.call_args.args[0].tools == ["baseline"]


def test_scoped_configuration_derives_graph_without_retired_fields(tmp_path):
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(FACTS))
    args = SimpleNamespace(memory_model="scoped", target="gpu", tools=["baseline", "local"],
                           gpu_policy="auto", scope_entry="step", scope_source="application.f90",
                           scope_facts=facts)
    assert cli.scoped_configuration(args)["execution"] == "graph"
    assert not {"measurement", "cost"} & cli.scoped_configuration(args).keys()
    with pytest.raises(ValueError, match="removed"):
        cli.scoped_configuration(SimpleNamespace(memory_model="call"))
