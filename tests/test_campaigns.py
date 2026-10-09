"""Campaign identity, clean timing and paired baseline regression contracts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

from elmm_pipeline.campaigns.identity import digest, guard_passed, read, repository, write
from elmm_pipeline.campaigns.prepare import check, prepare, validate
from elmm_pipeline.campaigns.run import borderline, environment, gates, planning_diagnostics, summaries
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
