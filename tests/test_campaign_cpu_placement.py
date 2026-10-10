"""Optional numerical-v2 placement is frozen and enforced without repinning."""

from pathlib import Path
import subprocess
import sys

import pytest

from elmm_pipeline.campaigns.identity import digest, read, write
from elmm_pipeline.campaigns.prepare import (
    calibrated_cpu_placement, check, prepare, validate_cpu_placement,
)
from elmm_pipeline.campaigns.run import environment, run_one


CPUS = [2, 3, 4, 5]


def numerical(cpus=None):
    cpus = CPUS if cpus is None else cpus
    return {"schema_version": 2, "identity": {"cpu_affinity": cpus},
            "protocol": {"cpu_affinity": cpus, "omp_proc_bind": "false"}}


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(CPUS))
    repo = tmp_path / "checkout"
    repo.mkdir()
    (repo / "code.py").write_text("fixture\n")
    monkeypatch.setattr("elmm_pipeline.campaigns.prepare.repository", lambda root, paths: {
        "root": str(root), "commit": "a" * 40, "paths": paths,
        "sha256": {"code.py": digest(root / "code.py")}})
    config_dir = tmp_path / "configuration"
    config_dir.mkdir()
    (config_dir / "options.nml").write_text("&inputs /\n")
    reference, binary, profile = (tmp_path / name for name in ("reference.vtk", "program", "profile.json"))
    reference.write_text("fields\n")
    binary.write_text("binary\n")
    write(profile, {"cpu_threads": 4, "precision_bits": 64, "scoped": {"runtime_id": "runtime"},
                    "hardware": {"gpu_uuid": "GPU-fixture"}, "numerical": numerical()})
    config = {"schema_version": 1, "threads": 4, "runtime_id": "runtime", "profile": str(profile),
              "cases": [{"name": "case", "config": str(config_dir), "reference": str(reference)}],
              "modes": [{"name": "native", "binary": str(binary), "sha256": digest(binary)}],
              "baseline_modes": ["native"], "repositories": {
                  name: {"path": str(repo), "paths": ["code.py"]} for name in ("compiler", "pipeline")},
              "tools": {"time": [sys.executable, "--version"]}}
    source, output = tmp_path / "config.json", tmp_path / "campaign"
    write(source, config)
    return source, output, profile


def test_prepare_derives_and_freezes_calibration_placement(inputs):
    source, output, _ = inputs
    prepare(source, output)
    state = check(output)
    assert state["cpu_placement"] == {"schema_version": 1, "cpu_affinity": CPUS,
                                     "environment": {"OMP_PROC_BIND": "false"}}
    assert state["host"]["affinity"] == CPUS
    state["cpu_placement"]["cpu_affinity"] = [6, 7, 8, 9]
    write(output / "campaign.json", state)
    with pytest.raises(ValueError, match="configuration changed"):
        check(output)


@pytest.mark.parametrize("cpus", [[2, 3, 4], [2, 3, 4, 4], [3, 2, 4, 5],
                                   [2, 3, 4, True], [2, 3, 4, -1], [2, 3, 4, 5.0]])
def test_v2_profile_requires_four_sorted_unique_integer_cpus(cpus, monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(CPUS))
    with pytest.raises(ValueError, match="four sorted unique"):
        calibrated_cpu_placement({"numerical": numerical(cpus)})


def test_preparation_rejects_different_process_mask_without_repinning(inputs, monkeypatch):
    source, output, _ = inputs
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {2, 3, 4, 6})
    monkeypatch.setattr("os.sched_setaffinity", lambda *args: pytest.fail("must not repin"))
    with pytest.raises(ValueError, match="exactly match numerical calibration"):
        prepare(source, output)
    assert read(output / "campaign.json")["status"] == "preparing"
    assert not (output / "identity.json").exists()


@pytest.mark.parametrize("field,bad", [("omp_proc_bind", "true"), ("cpu_affinity", [6, 7, 8, 9])])
def test_profile_protocol_must_match_calibrated_placement(field, bad, monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(CPUS))
    section = numerical()
    section["protocol"][field] = bad
    with pytest.raises(ValueError, match="protocol mismatch"):
        calibrated_cpu_placement({"numerical": section})


@pytest.mark.parametrize("section", [None, {"schema_version": 1, "identity": {"cpu_affinity": [1]}}])
def test_legacy_profiles_have_no_new_placement_contract(inputs, section):
    source, output, path = inputs
    profile = read(path)
    if section is None:
        profile.pop("numerical")
    else:
        profile["numerical"] = section
    write(path, profile)
    prepare(source, output)
    assert "cpu_placement" not in check(output)


def test_calibrated_environment_removes_overrides_but_legacy_is_unchanged(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(CPUS))
    for key, value in {"OMP_PROC_BIND": "spread", "OMP_PLACES": "cores", "GOMP_CPU_AFFINITY": "9-12"}.items():
        monkeypatch.setenv(key, value)
    placement = calibrated_cpu_placement({"numerical": numerical()})
    for traced in (False, True):
        observed = environment(4, traced=traced, cpu_placement=placement)
        assert observed["OMP_PROC_BIND"] == "false"
        assert "OMP_PLACES" not in observed and "GOMP_CPU_AFFINITY" not in observed
    legacy = environment(4, traced=False)
    assert legacy["OMP_PROC_BIND"] == "spread" and legacy["OMP_PLACES"] == "cores"
    assert legacy["GOMP_CPU_AFFINITY"] == "9-12"


def test_run_one_enforces_and_records_environment_before_any_application_work(tmp_path, monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(CPUS))
    placement = calibrated_cpu_placement({"numerical": numerical()})
    (tmp_path / "config").mkdir()
    monkeypatch.setattr("elmm_pipeline.campaigns.run.available", lambda: 4 * 2**30)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.read_vtk", lambda path: None)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.compare_vtk", lambda *args: {"passed": True})
    monkeypatch.setattr("elmm_pipeline.campaigns.run.validate_log", lambda *args, **kwargs: {})
    invoked = []

    def execute(command, **kwargs):
        invoked.append(command)
        assert kwargs["env"]["OMP_PROC_BIND"] == "false"
        assert "GOMP_CPU_AFFINITY" not in kwargs["env"] and "OMP_PLACES" not in kwargs["env"]
        rss = next(argument.split("=", 1)[1] for argument in command if argument.startswith("--output="))
        Path(rss).write_text("1024\n")
        return subprocess.CompletedProcess(command, 0, stdout="complete\n", stderr="")

    monkeypatch.setattr("elmm_pipeline.campaigns.run.subprocess.run", execute)
    state = {"threads": 4, "selected_gpu_uuid": "GPU-fixture", "cpu_placement": placement,
             "tools": {"time": {"path": "/fixture/time"}}}
    case = {"name": "case", "config": "config", "reference": "reference.vtk", "completed_steps": 1,
            "configuration": {"runtime_arguments": [], "expected_max_steps": 1}}
    mode = {"name": "native", "binary": "native"}
    record = run_one(tmp_path, state, case, mode, tmp_path / "run", traced=False)
    assert record["environment"]["OMP_PROC_BIND"] == "false"
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: {2, 3, 4, 6})
    with pytest.raises(ValueError, match="exactly match numerical calibration"):
        run_one(tmp_path, state, case, mode, tmp_path / "invalid-run", traced=False)
    assert len(invoked) == 1 and not (tmp_path / "invalid-run").exists()


def test_tampered_contract_does_not_change_openmp_binding(monkeypatch):
    monkeypatch.setattr("os.sched_getaffinity", lambda pid: set(CPUS))
    with pytest.raises(ValueError, match="invalid calibrated CPU placement"):
        validate_cpu_placement({"schema_version": 1, "cpu_affinity": CPUS,
                                "environment": {"OMP_PROC_BIND": "true"}})
