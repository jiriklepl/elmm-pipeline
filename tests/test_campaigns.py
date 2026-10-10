"""Campaign identity, clean timing and paired baseline regression contracts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from elmm_pipeline.campaigns.identity import digest, guard_passed, read, repository, write
from elmm_pipeline.campaigns.prepare import check, prepare, validate
from elmm_pipeline.campaigns.run import borderline, environment, gates, planning_diagnostics, run_one, summaries
from elmm_pipeline.campaigns.profile import command as profile_command


def test_repository_identity_rejects_changed_committed_input(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "code.py"
    source.write_text("original\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "code.py"], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "fixture"], check=True)
    before = repository(tmp_path, ["code.py"])
    assert before["sha256"] == {"code.py": digest(source)}
    source.write_text("changed\n")
    with pytest.raises(ValueError, match="differs from its commit"):
        repository(tmp_path, ["code.py"])


def test_prepare_freezes_explicit_inputs_and_rejects_later_mutation(tmp_path, monkeypatch):
    repo = tmp_path / "independent-checkout"
    repo.mkdir()
    (repo / "code.py").write_text("original\n")
    monkeypatch.setattr("elmm_pipeline.campaigns.prepare.repository", lambda root, paths: {
        "root": str(root), "commit": "a" * 40, "paths": paths, "sha256": {"code.py": digest(root / "code.py")}})
    config_dir = tmp_path / "configuration with spaces"
    config_dir.mkdir()
    (config_dir / "options.nml").write_text("&inputs /\n")
    reference, binary, profile = (tmp_path / name for name in ("reference.vtk", "ELMM", "profile.json"))
    reference.write_text("complete fields\n")
    binary.write_text("binary fixture\n")
    write(profile, {"cpu_threads": 4, "precision_bits": 64, "scoped": {"runtime_id": "test-runtime"},
                    "hardware": {"gpu_uuid": "GPU-test"}})
    config = {"schema_version": 1, "threads": 4, "runtime_id": "test-runtime", "profile": str(profile),
              "cases": [{"name": "case", "config": str(config_dir), "reference": str(reference)}],
              "modes": [{"name": "native", "binary": str(binary), "sha256": digest(binary)}],
              "baseline_modes": ["native"], "repositories": {
                  name: {"path": str(repo), "paths": ["code.py"]} for name in ("compiler", "pipeline")},
              "tools": {"time": [sys.executable, "--version"]}}
    source, output = tmp_path / "config.json", tmp_path / "campaign"
    write(source, config)
    prepare(source, output)
    assert check(output)["repositories"]["pipeline"]["commit"] == "a" * 40
    # Changing the original cannot change the selected independent snapshot.
    (config_dir / "options.nml").write_text("changed\n")
    check(output)
    (output / "configs/case/unexpected.nml").write_text("extra\n")
    with pytest.raises(ValueError, match="gained or lost"):
        check(output)
    (output / "configs/case/unexpected.nml").unlink()
    (output / "binaries/native").write_text("different binary\n")
    with pytest.raises(ValueError, match="frozen artifact changed"):
        check(output)
    with pytest.raises(FileExistsError):
        prepare(source, output)


def field_campaign():
    modes = ['native', 'frozen', 'candidate']
    return {'schema_version': 1, 'threads': 4, 'repositories': {'compiler': {}, 'pipeline': {}},
            'baseline_modes': ['native', 'frozen'], 'candidate_mode': 'candidate',
            'modes': [{'name': mode} for mode in modes],
            'cases': [{'name': case} for case in ('first', 'second')],
            'halo_drivers': [
                {'name': 'application-' + case, 'application_case': case,
                 'sources': {mode: mode + '.f90' for mode in modes},
                 'link': {mode: {} for mode in modes}} for case in ('first', 'second')]}


@pytest.mark.parametrize('missing_mode', ['candidate', 'frozen'])
def test_application_fields_require_candidate_and_other_configured_modes(missing_mode):
    config = field_campaign()
    config['halo_drivers'][0]['sources'].pop(missing_mode)
    config['halo_drivers'][0]['link'].pop(missing_mode)
    with pytest.raises(ValueError, match='every configured case and mode'):
        validate(config)


def test_application_fields_require_every_configured_case():
    config = field_campaign()
    config['halo_drivers'].pop()
    with pytest.raises(ValueError, match='every configured case and mode'):
        validate(config)


def test_application_fields_require_native_reference_and_matching_links():
    config = field_campaign()
    config['halo_drivers'][0]['sources'].pop('native')
    with pytest.raises(ValueError, match='native reference'):
        validate(config)
    config = field_campaign()
    config['halo_drivers'][0]['link'].pop('candidate')
    with pytest.raises(ValueError, match='sources and link modes'):
        validate(config)


@pytest.mark.parametrize('application', [False, True])
def test_duplicate_halo_driver_names_reject_before_freezing(tmp_path, application):
    config = field_campaign()
    config['halo_drivers'][1]['name'] = config['halo_drivers'][0]['name']
    if not application:
        for driver in config['halo_drivers']:
            driver.pop('application_case')
    path, output = tmp_path / 'config.json', tmp_path / 'campaign'
    write(path, config)
    with pytest.raises(ValueError, match='unique halo driver names'):
        prepare(path, output)
    assert not output.exists()


def test_complete_application_fields_and_joint_coverage_are_valid():
    config = field_campaign()
    validate(config)
    first = config['halo_drivers'][0]
    first['sources'].pop('candidate')
    first['link'].pop('candidate')
    config['halo_drivers'].append({'name': 'additional-first', 'application_case': 'first',
                                  'sources': {'native': 'native.f90', 'candidate': 'candidate.f90'},
                                  'link': {'native': {}, 'candidate': {}}})
    validate(config)


def test_generic_fixture_campaign_does_not_require_application_case_coverage():
    config = field_campaign()
    config['halo_drivers'] = [{'name': 'generic-fixture', 'sources': {'native': 'native.f90'}}]
    validate(config)
    config['halo_drivers'] = []
    validate(config)


def test_untraced_environment_clears_diagnostics_and_failure_injection(monkeypatch):
    for name in ("FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE", "FORT_PHASE_TIMING", "FORT_PROFILE",
                 "CUDA_LAUNCH_BLOCKING", "FORT_SCOPE_TEST_FAIL_ALLOC", "ELMM_SOMETHING"):
        monkeypatch.setenv(name, "1")
    env = environment(4, traced=False)
    assert env["OMP_NUM_THREADS"] == "4"
    assert not any(name.startswith(("FORT_", "ELMM_")) for name in env)
    assert "CUDA_LAUNCH_BLOCKING" not in env
    assert environment(4, traced=True)["FORT_RUNTIME_TRACE"] == "1"


def test_acceptance_uses_better_frozen_baseline_and_one_candidate():
    state = {"baseline_modes": ["native", "frozen"], "candidate_mode": "auto",
             "modes": [{"name": name} for name in ("native", "frozen", "auto")]}
    summary = {"native": {"median": 10}, "frozen": {"median": 8}, "auto": {"median": 6}}
    value = gates(state, summary)
    assert value["baseline_mode"] == "frozen"
    assert value["target_seconds"] == pytest.approx(5.6)
    assert not value["thirty_percent_lower"]  # 40% vs native alone is insufficient.
    assert value["within_five_percent"]
    summary["auto"]["median"] = 5.6
    assert gates(state, summary)["thirty_percent_lower"]
    assert borderline(state, summary)


def test_traced_samples_cannot_enter_summary():
    with pytest.raises(ValueError, match="diagnostics"):
        summaries([{"mode": "native", "warmup": False, "traced": True}], [{"name": "native"}])


def test_planning_phases_remain_separate_and_reject_nan():
    log = "FORT_SCOPED planning_timing context=1 phase=selection seconds=.001 cache_hit=1 query_generation=2 state_generation=3\n"
    result = planning_diagnostics(log)
    assert result["phases"]["selection"]["cache_hits"] == 1
    assert result["phases"]["query_construction"]["seconds"] == 0
    assert not result["timing_sample"]
    with pytest.raises(ValueError, match="finite"):
        planning_diagnostics(log.replace(".001", "nan"))


def test_campaign_diagnostics_transport_separate_scope_packages(tmp_path, monkeypatch):
    (tmp_path / "config").mkdir()
    packages = [{"scope_manifest": {
        "schema_version": 1, "scope_count": 1, "automatic_estimate_available": False,
        "scopes": [{"mode": "auto", "estimate_available": False, "gpu_leaves": ["renamed::leaf"]}]}},
        {"scope_manifest": {"schema_version": 1, "scope_count": 0, "scopes": []}}]
    write(tmp_path / "adapter.json", {"scope_packages": packages, "regions": []})
    monkeypatch.setattr("elmm_pipeline.campaigns.run.available", lambda: 4 * 2**30)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.read_vtk", lambda path: None)
    monkeypatch.setattr("elmm_pipeline.campaigns.run.compare_vtk", lambda *args: {"passed": True})
    monkeypatch.setattr("elmm_pipeline.campaigns.run.validate_log", lambda *args, **kwargs: {})

    def execute(command, **kwargs):
        rss = next(argument.split("=", 1)[1] for argument in command if argument.startswith("--output="))
        Path(rss).write_text("1024\n")
        log = "FORT_OFFLOAD entry=leaf mode=native-scoped-unestimated gpu_units=0 cpu_units=1\n"
        return subprocess.CompletedProcess(command, 0, stdout=log, stderr="")

    monkeypatch.setattr("elmm_pipeline.campaigns.run.subprocess.run", execute)
    state = {"threads": 4, "selected_gpu_uuid": "GPU-fixture", "tools": {"time": {"path": "/fixture/time"}}}
    case = {"name": "fixture", "config": "config", "reference": "reference.vtk", "completed_steps": 1,
            "configuration": {"runtime_arguments": [], "expected_max_steps": 1}}
    mode = {"name": "candidate", "binary": "candidate", "adapter": "adapter.json"}
    record = run_one(tmp_path, state, case, mode, tmp_path / "run", traced=True)
    assert record["activity"]["scope_manifest_count"] == 2
    assert record["activity"]["compiler_native_selection"]
    assert record["activity"]["offload_decisions"]["calls"] == 1
    assert "wall_seconds" not in record


def test_bounded_profile_continues_application_and_disables_sampling():
    argv = profile_command("/explicit/nsys", "/new/capture", "/frozen/program", ["x=1"], 2)
    assert "--duration=2" in argv
    assert "--kill=none" in argv
    assert "--trace=cuda,nvtx" in argv
    assert "--sample=none" in argv and "--cpuctxsw=none" in argv
    assert argv[-2:] == ["/frozen/program", "x=1"]
    for seconds in (0, 31, -1, 1.5, True):
        with pytest.raises(ValueError, match="duration"):
            profile_command("nsys", "capture", "program", [], seconds)


def test_shared_plot_scale_includes_later_case_outliers(tmp_path, monkeypatch):
    from elmm_pipeline.campaigns import report
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib.pyplot as plt
    modes = [{"name": "native", "label": "Native"}, {"name": "auto", "label": "Automatic"}]
    monkeypatch.setattr(report, "check", lambda root: {"modes": modes, "title": "Fixture"})
    cases = [{"case": name, "gates": {"baseline_seconds": 1}, "summary": {
        mode["name"]: {"median": 1, "min": .9, "max": maximum} for mode in modes}}
        for name, maximum in (("first", 1.1), ("later", 3))]
    write(tmp_path / "timings/report.json", {"status": "passed", "cases": cases})
    write(tmp_path / "verification/report.json", {"status": "passed", "cases": [
        {"case": row["case"], "runs": [{"mode": mode["name"], "activity": {"kernel_launches": 0}}
                                       for mode in modes]} for row in cases]})
    limits = []
    original_close = plt.close
    def close(figure=None):
        if hasattr(figure, "axes"):
            limits.extend(axis.get_ylim() for axis in figure.axes)
        original_close(figure)
    monkeypatch.setattr(plt, "close", close)
    report.plot(tmp_path)
    assert len(limits) == 2 and all(low == 0 and high >= 3 for low, high in limits)


@pytest.mark.parametrize("update", [
    {"cleanup_complete": False}, {"remaining_descendants": [{"pid": 123}]},
    {"minimum_available_bytes": 2 * 2**30}, {"stopped_for_memory": True}, {"exit_code": 1}])
def test_sealing_rejects_incomplete_guard_receipts(tmp_path, update):
    path = tmp_path / "guard.json"
    receipt = {"status": "finished", "exit_code": 0, "headroom_bytes": 3 * 2**30,
               "minimum_available_bytes": 4 * 2**30, "cleanup_complete": True, "remaining_descendants": []}
    write(path, receipt)
    guard_passed(path)
    write(path, {**receipt, **update})
    with pytest.raises(ValueError, match="cleanup required"):
        guard_passed(path)
