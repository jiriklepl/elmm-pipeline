"""Fresh comparison processes share the frozen wait/spin launch settings."""
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from test_campaign_cpu_placement import CPUS
from test_campaign_cpu_placement import inputs as campaign_inputs
from test_campaign_profile import fixture as profiler_fixture

from elmm_pipeline.campaigns import __main__ as commands
from elmm_pipeline.campaigns import halos
from elmm_pipeline.campaigns import profile as profiling
from elmm_pipeline.campaigns.identity import read, write
from elmm_pipeline.campaigns.launch_environment import (
    apply_launch_environment,
    freeze_launch_environment,
)
from elmm_pipeline.campaigns.prepare import check, prepare
from elmm_pipeline.campaigns.run import environment, run_one

VALUES = {"OMP_WAIT_POLICY": "PASSIVE", "GOMP_SPINCOUNT": None}


inputs = campaign_inputs


@pytest.fixture(autouse=True)
def unset_inherited_thread_limit(monkeypatch):
    monkeypatch.delenv("OMP_THREAD_LIMIT", raising=False)


def section(profile, values=None):
    values = VALUES if values is None else values
    return {"schema_version": 1, "status": "rejected", "execution_identity": {
        "cpu_threads": profile["cpu_threads"], "precision_bits": profile["precision_bits"],
        "cpu_affinity": CPUS, "omp_wait_policy": values["OMP_WAIT_POLICY"],
        "gomp_spincount": values["GOMP_SPINCOUNT"]}}


@pytest.mark.parametrize("sections", [["cpu_execution_protocol"], ["cpu_dependency"],
                                     ["cpu_dependency", "cpu_execution_protocol"]])
def test_prepare_transports_profile_launch_identity_even_when_costs_reject(inputs, sections, monkeypatch):
    source, output, path = inputs
    profile = read(path)
    for name in sections:
        profile[name] = section(profile)
    write(path, profile)
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    monkeypatch.setenv("GOMP_SPINCOUNT", "900")
    prepare(source, output)
    state = check(output)
    contract = state["openmp_launch_environment"]
    assert contract["values"] == VALUES
    assert set(contract["profile_sections"]) == set(sections)
    assert set(contract["provenance"].values()) == {"profile"}
    for traced in (False, True):
        observed = environment(4, traced=traced, cpu_placement=state["cpu_placement"],
                               openmp_launch_environment=contract)
        assert observed["OMP_WAIT_POLICY"] == "PASSIVE" and "GOMP_SPINCOUNT" not in observed
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE" and os.environ["GOMP_SPINCOUNT"] == "900"


def test_preparation_freezes_current_values_and_explicit_unset(inputs, monkeypatch):
    source, output, _ = inputs
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    monkeypatch.setenv("GOMP_SPINCOUNT", "500")
    config = read(source)
    config["openmp_launch_environment"] = {"GOMP_SPINCOUNT": None}
    write(source, config)
    prepare(source, output)
    contract = check(output)["openmp_launch_environment"]
    assert contract["values"] == {"OMP_WAIT_POLICY": "ACTIVE", "GOMP_SPINCOUNT": None}
    assert contract["provenance"] == {"OMP_WAIT_POLICY": "preparation_environment",
                                      "GOMP_SPINCOUNT": "configuration"}
    monkeypatch.setenv("OMP_WAIT_POLICY", "PASSIVE")
    assert environment(4, traced=False, openmp_launch_environment=contract)["OMP_WAIT_POLICY"] == "ACTIVE"
    state = read(output / "campaign.json")
    state["openmp_launch_environment"]["values"]["GOMP_SPINCOUNT"] = "1000"
    write(output / "campaign.json", state)
    with pytest.raises(ValueError, match="configuration changed"):
        check(output)


@pytest.mark.parametrize("mutation", ["profile_conflict", "configured_conflict", "precision", "threads",
                                     "affinity", "boolean_cpu", "schema", "missing_value", "malformed"])
def test_preparation_rejects_incompatible_or_conflicting_profile_identity(inputs, mutation):
    source, output, path = inputs
    profile = read(path)
    profile["cpu_execution_protocol"] = section(profile)
    identity = profile["cpu_execution_protocol"]["execution_identity"]
    if mutation == "profile_conflict":
        profile["cpu_dependency"] = section(profile, {**VALUES, "GOMP_SPINCOUNT": "1"})
    elif mutation == "configured_conflict":
        config = read(source)
        config["openmp_launch_environment"] = {"OMP_WAIT_POLICY": "ACTIVE"}
        write(source, config)
    elif mutation == "precision":
        identity["precision_bits"] = 32
    elif mutation == "threads":
        identity["cpu_threads"] = 2
    elif mutation == "affinity":
        identity["cpu_affinity"] = [6, 7, 8, 9]
    elif mutation == "boolean_cpu":
        profile["numerical"]["identity"]["cpu_affinity"] = [0, 1, 2, 3]
        profile["numerical"]["protocol"]["cpu_affinity"] = [0, 1, 2, 3]
        identity["cpu_affinity"] = [False, 1, 2, 3]
    elif mutation == "schema":
        profile["cpu_execution_protocol"]["schema_version"] = True
    elif mutation == "missing_value":
        identity.pop("omp_wait_policy")
    else:
        profile["cpu_execution_protocol"] = None
    write(path, profile)
    if mutation == "boolean_cpu":
        # Direct transport test avoids the unrelated actual affinity guard.
        with pytest.raises(ValueError, match="OpenMP launch"):
            freeze_launch_environment(profile)
    else:
        with pytest.raises(ValueError, match="OpenMP launch"):
            prepare(source, output)
        assert not (output / "identity.json").exists()


@pytest.mark.parametrize("values", [{"OTHER": "1"}, {"OMP_WAIT_POLICY": True},
                                    {"GOMP_SPINCOUNT": 1}, {"GOMP_SPINCOUNT": "x" * 129},
                                    {"OMP_WAIT_POLICY": "PASSIVE\n"}, {"OMP_WAIT_POLICY": "é"}])
def test_unbounded_or_unrelated_configured_launch_values_reject(values):
    with pytest.raises(ValueError, match="OpenMP launch"):
        freeze_launch_environment({}, values)


def test_fresh_subprocess_receives_exact_values_without_mutating_parent(monkeypatch):
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    monkeypatch.setenv("GOMP_SPINCOUNT", "900")
    contract = freeze_launch_environment({}, VALUES)
    observed = subprocess.check_output([sys.executable, "-c", ("import json,os;print(json.dumps("
        "{key:os.getenv(key) for key in ('OMP_WAIT_POLICY','GOMP_SPINCOUNT')}))")],
        env=environment(4, traced=False, openmp_launch_environment=contract), text=True)
    assert json.loads(observed) == VALUES
    assert os.environ["OMP_WAIT_POLICY"] == "ACTIVE" and os.environ["GOMP_SPINCOUNT"] == "900"
    legacy = environment(4, traced=False)
    assert legacy["OMP_WAIT_POLICY"] == "ACTIVE" and legacy["GOMP_SPINCOUNT"] == "900"


@pytest.mark.parametrize("mode", ["native", "frozen", "candidate"])
def test_application_receipt_records_launch_values_for_every_mode(tmp_path, monkeypatch, mode):
    (tmp_path / "config").mkdir()
    monkeypatch.setenv("GOMP_SPINCOUNT", "900")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "+0008")
    monkeypatch.setattr("elmm_pipeline.campaigns.run.available", lambda: 4 * 2**30)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.read_vtk", lambda path: None)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.compare_vtk", lambda *args: {"passed": True})
    monkeypatch.setattr("elmm_pipeline.campaigns.run.validate_log", lambda *args, **kwargs: {})

    def execute(command, **kwargs):
        assert kwargs["env"]["OMP_WAIT_POLICY"] == "PASSIVE" and "GOMP_SPINCOUNT" not in kwargs["env"]
        rss = next(argument.split("=", 1)[1] for argument in command if argument.startswith("--output="))
        Path(rss).write_text("1024\n")
        return subprocess.CompletedProcess(command, 0, stdout="complete\n", stderr="")

    monkeypatch.setattr("elmm_pipeline.campaigns.run.subprocess.run", execute)
    state = {"threads": 4, "selected_gpu_uuid": "GPU-fixture", "tools": {"time": {"path": "/fixture/time"}},
             "openmp_launch_environment": freeze_launch_environment({}, VALUES)}
    case = {"name": "case", "config": "config", "reference": "reference.vtk", "completed_steps": 1,
            "configuration": {"runtime_arguments": [], "expected_max_steps": 1}}
    record = run_one(tmp_path, state, case, {"name": mode, "binary": mode}, tmp_path / "run", traced=False)
    expected = {**VALUES, "OMP_THREAD_LIMIT": "+0008"}
    assert record["openmp_launch_environment"] == expected
    assert read(tmp_path / "run/run.json")["openmp_launch_environment"] == expected


@pytest.mark.parametrize("override", [None, "matching", "conflicting", "unset_conflict"])
def test_external_phase_transports_frozen_values_or_rejects_conflict(tmp_path, monkeypatch, override):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "8")
    explicit = {} if override is None else {"OMP_WAIT_POLICY": "PASSIVE" if override == "matching" else "ACTIVE"}
    if override == "unset_conflict":
        explicit = {"GOMP_SPINCOUNT": "0"}
    state = {"threads": 4, "selected_gpu_uuid": "GPU-fixture", "commands": {
        "fixture": {"argv": ["fixture"], "environment": explicit}},
        "openmp_launch_environment": freeze_launch_environment({}, VALUES)}
    monkeypatch.setattr(commands, "check", lambda root: state)
    invoked = []
    monkeypatch.setattr(commands.subprocess, "run", lambda *args, **kwargs: invoked.append(kwargs["env"]))
    if override in {"conflicting", "unset_conflict"}:
        with pytest.raises(ValueError, match="conflicts with frozen OpenMP launch"):
            commands.external(tmp_path, "fixture")
        assert not invoked and not (tmp_path / "fixture-command.json").exists()
    else:
        commands.external(tmp_path, "fixture")
        assert invoked[0]["OMP_WAIT_POLICY"] == "PASSIVE" and "GOMP_SPINCOUNT" not in invoked[0]
        assert read(tmp_path / "fixture-command.json")["openmp_launch_environment"] == {
            **VALUES, "OMP_THREAD_LIMIT": "8"}


def test_profiler_target_inherits_frozen_settings_and_records_them(tmp_path, monkeypatch):
    root, output, profiler, _ = profiler_fixture(tmp_path, monkeypatch)
    state = profiling.check(root)
    state["openmp_launch_environment"] = freeze_launch_environment({}, VALUES)
    application = root / "application"
    application.write_text(application.read_text().replace("import sys\n", "import sys\nimport json,os\n"
        "Path('launch-settings.json').write_text(json.dumps({key:os.getenv(key) for key in "
        "('OMP_WAIT_POLICY','GOMP_SPINCOUNT')}))\n"))
    monkeypatch.setenv("OMP_WAIT_POLICY", "ACTIVE")
    monkeypatch.setenv("GOMP_SPINCOUNT", "900")
    monkeypatch.setenv("OMP_THREAD_LIMIT", "4")
    record = profiling.profile(root, output, case="case", mode="candidate", profiler=profiler, seconds=1)
    assert read(output / "launch-settings.json") == VALUES
    assert record["openmp_launch_environment"] == {**VALUES, "OMP_THREAD_LIMIT": "4"}


def test_generic_halo_driver_receives_frozen_launch_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "16")
    contract = freeze_launch_environment({}, VALUES)
    driver = {"name": "fixture", "coverage": "complete", "sources": {"native": "driver.f90"},
              "link": {"native": {"include_dirs": [], "objects": [], "flags": []}},
              "flags": [], "success_marker": "COMPLETE", "dtype": "float64", "field_count": 1,
              "rtol": 0, "atol": 0}
    state = {"threads": 4, "selected_gpu_uuid": "GPU-fixture", "halo_drivers": [driver],
             "tools": {"fortran": {"path": "fixture-compiler"}}, "openmp_launch_environment": contract}
    monkeypatch.setattr(halos, "check", lambda root: state)
    monkeypatch.setattr(halos, "fields", lambda *args: {})
    monkeypatch.setattr(halos, "compare_fields", lambda *args: {"field": {"passed": True}})
    invocations = []

    def execute(command, **kwargs):
        invocations.append(kwargs)
        if len(invocations) == 2:
            assert kwargs["env"]["OMP_WAIT_POLICY"] == "PASSIVE" and "GOMP_SPINCOUNT" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, stdout="COMPLETE", stderr="")

    monkeypatch.setattr(halos.subprocess, "run", execute)
    halos.run(tmp_path)
    assert len(invocations) == 2
    assert read(tmp_path / "halos/report.json")["drivers"][0]["runs"][0]["openmp_launch_environment"] == {
        **VALUES, "OMP_THREAD_LIMIT": "16"}


@pytest.mark.parametrize("bad", [{"schema_version": True, "values": VALUES},
                                {"schema_version": 1, "values": {"OMP_WAIT_POLICY": "PASSIVE"}}])
def test_malformed_frozen_contract_rejects_before_changing_environment(bad):
    observed = {"OMP_WAIT_POLICY": "ACTIVE", "CUSTOM": "preserved"}
    before = deepcopy(observed)
    with pytest.raises(ValueError, match="OpenMP launch"):
        apply_launch_environment(observed, bad)
    assert observed == before


@pytest.mark.parametrize("raw", ["0", "-1", "", "invalid", "4.0", "1,4", "４", "4\n", "9" * 129,
                                str(1 << 31), None, 4, True])
def test_new_campaign_rejects_malformed_thread_limit_without_changing_environment(raw):
    observed = {"OMP_THREAD_LIMIT": raw, "OMP_WAIT_POLICY": "ACTIVE"}
    before = deepcopy(observed)
    with pytest.raises(ValueError, match="OMP_THREAD_LIMIT.*positive decimal"):
        apply_launch_environment(observed, freeze_launch_environment({}, VALUES))
    assert observed == before


@pytest.mark.parametrize("raw", ["1", "2", "3"])
def test_new_campaign_rejects_insufficient_thread_limit_but_legacy_is_unchanged(raw):
    observed = {"OMP_THREAD_LIMIT": raw, "OMP_WAIT_POLICY": "ACTIVE"}
    with pytest.raises(ValueError, match="four-thread campaign budget"):
        apply_launch_environment(observed, freeze_launch_environment({}, VALUES))
    apply_launch_environment(observed, None)
    assert observed == {"OMP_THREAD_LIMIT": raw, "OMP_WAIT_POLICY": "ACTIVE"}


@pytest.mark.parametrize("raw", ["4", "8", "  +0004\t", str((1 << 31) - 1)])
def test_new_campaign_retains_valid_raw_thread_limit(raw):
    observed = {"OMP_THREAD_LIMIT": raw}
    apply_launch_environment(observed, freeze_launch_environment({}, VALUES))
    assert observed == {"OMP_THREAD_LIMIT": raw, "OMP_WAIT_POLICY": "PASSIVE"}


def test_application_rejects_insufficient_thread_limit_before_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    monkeypatch.setattr("elmm_pipeline.campaigns.run.available", lambda: 4 * 2**30)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.subprocess.run", lambda *args, **kwargs: pytest.fail("launched"))
    state = {"threads": 4, "openmp_launch_environment": freeze_launch_environment({}, VALUES)}
    with pytest.raises(ValueError, match="four-thread campaign budget"):
        run_one(tmp_path, state, {}, {"name": "native"}, tmp_path / "run", traced=False)
    assert not (tmp_path / "run").exists()


def test_halo_rejects_insufficient_thread_limit_before_build_or_launch(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    state = {"threads": 4, "halo_drivers": [{"name": "fixture", "coverage": "complete",
             "sources": {"native": "driver.f90"}}],
             "openmp_launch_environment": freeze_launch_environment({}, VALUES)}
    monkeypatch.setattr(halos, "check", lambda root: state)
    monkeypatch.setattr(halos.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched"))
    with pytest.raises(ValueError, match="four-thread campaign budget"):
        halos.run(tmp_path)
    assert not (tmp_path / "halos/fixture").exists()


def test_profiler_rejects_insufficient_thread_limit_before_launch(tmp_path, monkeypatch):
    root, output, profiler, _ = profiler_fixture(tmp_path, monkeypatch)
    profiling.check(root)["openmp_launch_environment"] = freeze_launch_environment({}, VALUES)
    monkeypatch.setenv("OMP_THREAD_LIMIT", "1")
    monkeypatch.setattr(profiling.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched"))
    with pytest.raises(ValueError, match="four-thread campaign budget"):
        profiling.profile(root, output, case="case", mode="candidate", profiler=profiler, seconds=1)


@pytest.mark.parametrize("raw", ["1", "invalid"])
def test_external_phase_rejects_insufficient_or_malformed_thread_override(tmp_path, monkeypatch, raw):
    state = {"threads": 4, "selected_gpu_uuid": "GPU-fixture", "commands": {
        "fixture": {"argv": ["fixture"], "environment": {"OMP_THREAD_LIMIT": raw}}},
        "openmp_launch_environment": freeze_launch_environment({}, VALUES)}
    monkeypatch.setattr(commands, "check", lambda root: state)
    monkeypatch.setattr(commands.subprocess, "run", lambda *args, **kwargs: pytest.fail("launched"))
    with pytest.raises(ValueError, match="OMP_THREAD_LIMIT"):
        commands.external(tmp_path, "fixture")
    assert not (tmp_path / "fixture-command.json").exists()
