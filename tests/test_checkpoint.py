"""Component snapshots retain evidence identities and historical claim boundaries."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from elmm_pipeline.campaigns import checkpoint


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def component(tmp_path):
    compiler = tmp_path / "independent compiler"
    compiler.mkdir()
    subprocess.run(["git", "init", "-q", str(compiler)], check=True)
    source = compiler / "source.py"
    source.write_text("previous source\n")
    git(compiler, "add", "source.py")
    git(compiler, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "previous")
    previous_commit = git(compiler, "rev-parse", "HEAD")
    source.write_text("new component source\n")
    git(compiler, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qam", "current")
    pipeline = Path(checkpoint.__file__).resolve().parents[2]
    pair = {"compiler": git(compiler, "rev-parse", "HEAD"), "pipeline": git(pipeline, "rev-parse", "HEAD")}
    previous_pair = {**pair, "compiler": previous_commit}
    application_pair = {"compiler": "a" * 40, "pipeline": "b" * 40}
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    receipt, log, guard = (evidence / name for name in ("accepted.json", "host.log", "guard.json"))
    write(receipt, {"status": "passed", "tests": 12, "performance_samples": 0})
    log.write_text("12 passed\n")
    write(guard, {"status": "finished", "exit_code": 0, "stopped_for_memory": False,
                  "interrupted_signal": 0, "headroom_bytes": 3 * 2**30,
                  "minimum_available_bytes": 4 * 2**30, "cleanup_complete": True, "remaining_descendants": []})
    application = tmp_path / "old application" / "snapshot.json"
    write(application, {"compatible_commits": application_pair, "full_application_gate_passed": True})
    previous = tmp_path / "previous component" / "snapshot.json"
    write(previous, {"compatible_commits": previous_pair,
                     "previous_application_snapshot_sha256": digest(application)})
    historical = tmp_path / "historical campaign"
    for name in ("campaign.json", "timings/report.json", "verification/report.json", "ordinary/report.json"):
        write(historical / name, {"status": "passed", "label": "unchanged original label", "seconds": 1.25})
    renderer = tmp_path / "ordinary renderer.py"
    renderer.write_text("raise AssertionError('tests must mock plotting')\n")
    helper = tmp_path / "obsolete helper.py"
    helper.write_text("raise AssertionError('historical helper must never be imported')\n")
    inputs = [receipt, log, guard, previous, helper]
    history = [renderer, *(historical / name for name in (
        "campaign.json", "timings/report.json", "verification/report.json", "ordinary/report.json"))]
    config = {"schema_version": 1, "component_only": True, "historical_plots_only": True,
              "draft": False, "new_application_observations": 0, "new_performance_samples": 0,
              "repositories": {
                  "compiler": {"path": str(compiler), "commit": pair["compiler"],
                               "verified_source_sha256": {"source.py": digest(source)}},
                  "pipeline": {"path": str(pipeline), "commit": pair["pipeline"],
                               "verified_source_sha256": {"elmm_pipeline/campaigns/report.py":
                                   digest(pipeline / "elmm_pipeline/campaigns/report.py")}}},
              "input_sha256": {str(path): digest(path) for path in inputs},
              "historical_input_sha256": {str(path): digest(path) for path in history},
              "receipt_helpers": str(helper), "successful_guards": [str(guard)],
              "json_checks": [{"path": str(receipt), "keys": ["status"], "equals": "passed"}],
              "text_checks": [{"path": str(log), "contains": "12 passed"}],
              "previous_snapshot": str(previous), "previous_checkpoint_commits": previous_pair,
              "previous_application_snapshot": str(application), "previous_application_commits": application_pair,
              "output": str(tmp_path / "new component"), "evidence_root": str(evidence),
              "copy_evidence": [str(receipt), str(log)], "historical_campaign": str(historical),
              "ordinary_renderer": str(renderer), "ordinary_label": "Original policies (historical timings)",
              "workspace": str(tmp_path), "component_report": "# Component correctness\n\nTwelve host tests passed.",
              "component_claims": {"host_tests_passed": 12, "automatic_placement_verified": False}}
    path = tmp_path / "component.json"
    write(path, config)
    return path, config


@pytest.fixture
def renderers(monkeypatch):
    calls = []
    original_run = subprocess.run

    def plot(root, output):
        calls.append(("scoped", str(root), str(output)))
        output.mkdir()
        (output / "complete-wall-relative.png").write_bytes(b"mock scoped historical chart")

    def ordinary(argv, **kwargs):
        if len(argv) > 1 and argv[1].endswith("ordinary renderer.py"):
            calls.append(("ordinary", argv, kwargs))
            output = Path(argv[argv.index("--output") + 1])
            output.mkdir()
            (output / "complete-wall-relative.png").write_bytes(b"mock ordinary historical chart")
            return subprocess.CompletedProcess(argv, 0, "rendered unchanged observations", "")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(checkpoint, "plot", plot)
    monkeypatch.setattr(checkpoint.subprocess, "run", ordinary)
    return calls


def test_verify_only_checks_commits_and_inputs_without_rendering_or_writes(component, monkeypatch, capsys):
    path, config = component
    before = {name: digest(name) for name in [str(path), *config["input_sha256"], *config["historical_input_sha256"]]}
    monkeypatch.setattr(checkpoint, "plot", lambda *_: pytest.fail("verify-only attempted to render"))
    result = checkpoint.seal(path, verify_only=True)
    assert result["status"] == "component_checkpoint_inputs_verified"
    assert result["compatible_commits"] == {k: v["commit"] for k, v in config["repositories"].items()}
    assert result["snapshot_written"] is False
    assert not Path(config["output"]).exists()
    assert {name: digest(name) for name in before} == before
    monkeypatch.setattr(sys, "argv", ["checkpoint", "--config", str(path), "--historical-plots-only", "--verify-only"])
    checkpoint.main()
    assert "NO_PLOTS_OR_SNAPSHOT_WRITTEN" in capsys.readouterr().out


def test_seal_keeps_separate_commit_pairs_history_and_zero_fresh_observations(component, renderers):
    path, config = component
    output = checkpoint.seal(path)
    receipt = json.loads((output / "snapshot.json").read_text())
    assert receipt["status"] == "sealed_component_correctness_only"
    assert receipt["compatible_commits"]["compiler"] == config["repositories"]["compiler"]["commit"]
    assert receipt["previous_full_application_commits"] == config["previous_application_commits"]
    assert receipt["full_application_gate_for_current_compiler_verified"] is False
    assert receipt["new_application_observations"] == receipt["new_performance_samples"] == 0
    assert receipt["historical_figures_only"] is True
    assert receipt["policy_promoted"] is False
    for name, expected in receipt["files_sha256"].items():
        assert digest(output / name) == expected
    assert digest(output / receipt["previous_snapshot"]) == digest(config["previous_snapshot"])
    assert digest(output / receipt["previous_application_snapshot"]) == digest(config["previous_application_snapshot"])
    inventory = json.loads((output / "evidence-inventory.json").read_text())
    assert inventory["retained_external_authorities_sha256"] == config["input_sha256"]
    assert inventory["large_arrays_objects_binaries_copied"] is False
    assert len(renderers) == 2
    assert renderers[1][1] == [sys.executable, config["ordinary_renderer"],
        str(Path(config["historical_campaign"]) / "ordinary/report.json"),
        "--output", str(output / "ordinary-figures"), "--label", config["ordinary_label"]]
    report = (output / "report.md").read_text()
    assert "Zero new performance samples" in report
    assert config["previous_application_commits"]["compiler"] in report
    before = digest(output / "snapshot.json")
    with pytest.raises(ValueError, match="Immutable component snapshot already exists"):
        checkpoint.seal(path)
    assert digest(output / "snapshot.json") == before
    assert len(renderers) == 2
    assert checkpoint.seal(path, verify_only=True)["snapshot_written"] is False


@pytest.mark.parametrize("change", [{"draft": True}, {"new_performance_samples": 1},
    {"new_application_observations": 1}, {"historical_plots_only": False}])
def test_draft_or_fresh_claims_reject_before_output(component, change):
    path, config = component
    config.update(change)
    write(path, config)
    with pytest.raises(ValueError, match="component-only, historical-plots-only"):
        checkpoint.seal(path, verify_only=True)
    assert not Path(config["output"]).exists()


def test_pipeline_only_milestone_retains_unchanged_compiler_identity(component, renderers):
    path, config = component
    current = {name: row["commit"] for name, row in config["repositories"].items()}
    previous_pair = {**current, "pipeline": "c" * 40}
    previous = Path(config["previous_snapshot"])
    previous_receipt = json.loads(previous.read_text())
    previous_receipt["compatible_commits"] = previous_pair
    write(previous, previous_receipt)
    config["previous_checkpoint_commits"] = previous_pair
    config["input_sha256"][str(previous)] = digest(previous)
    write(path, config)

    output = checkpoint.seal(path)
    receipt = json.loads((output / "snapshot.json").read_text())
    assert receipt["compatible_commits"] == current
    assert receipt["previous_full_application_commits"] == config["previous_application_commits"]
    assert receipt["new_application_observations"] == receipt["new_performance_samples"] == 0
    assert receipt["policy_promoted"] is False
    assert json.loads(previous.read_text())["compatible_commits"] == previous_pair
    assert len(renderers) == 2


@pytest.mark.parametrize("role", ["input", "history", "committed_source"])
def test_changed_evidence_history_or_committed_source_reject(component, role):
    path, config = component
    target = {"input": config["copy_evidence"][0],
              "history": str(Path(config["historical_campaign"]) / "timings/report.json"),
              "committed_source": str(Path(config["repositories"]["compiler"]["path"]) / "source.py")}[role]
    Path(target).write_text("changed after verification\n")
    with pytest.raises(ValueError, match="input changed|differs from committed bytes"):
        checkpoint.seal(path, verify_only=True)
    assert not Path(config["output"]).exists()


def test_foreign_pipeline_origin_rejects_without_importing_configured_code(component):
    path, config = component
    config["repositories"]["pipeline"] = config["repositories"]["compiler"]
    write(path, config)
    with pytest.raises(ValueError, match="Foreign maintained pipeline renderer"):
        checkpoint.seal(path, verify_only=True)


@pytest.mark.parametrize("change", [{"exit_code": 1}, {"cleanup_complete": False},
    {"remaining_descendants": [42]}, {"minimum_available_bytes": 2 * 2**30}])
def test_failed_guard_rejects_even_when_its_current_hash_is_supplied(component, change):
    path, config = component
    guard = Path(config["successful_guards"][0])
    value = json.loads(guard.read_text())
    value.update(change)
    write(guard, value)
    config["input_sha256"][str(guard)] = digest(guard)
    write(path, config)
    with pytest.raises(ValueError, match="successful memory guard"):
        checkpoint.seal(path, verify_only=True)


@pytest.mark.parametrize("role", ["previous_pair", "application_pair", "application_gate", "relabel"])
def test_previous_component_and_application_authorities_cannot_be_relabelled(component, role):
    path, config = component
    if role == "previous_pair":
        config["previous_checkpoint_commits"]["pipeline"] = "c" * 40
    elif role == "application_pair":
        config["previous_application_commits"]["compiler"] = "c" * 40
    elif role == "application_gate":
        application = Path(config["previous_application_snapshot"])
        value = json.loads(application.read_text())
        value["full_application_gate_passed"] = False
        write(application, value)
        config["input_sha256"][str(application)] = digest(application)
    else:
        config["previous_checkpoint_commits"] = {name: row["commit"] for name, row in config["repositories"].items()}
        previous = Path(config["previous_snapshot"])
        value = json.loads(previous.read_text())
        value["compatible_commits"] = config["previous_checkpoint_commits"]
        write(previous, value)
        config["input_sha256"][str(previous)] = digest(previous)
    write(path, config)
    with pytest.raises(ValueError, match="retain its original commit pair|not relabel"):
        checkpoint.seal(path, verify_only=True)


def test_previous_application_hash_chain_is_checked(component):
    path, config = component
    application = Path(config["previous_application_snapshot"])
    application.write_text(application.read_text() + " ")
    with pytest.raises(ValueError, match="Previous application snapshot changed"):
        checkpoint.seal(path, verify_only=True)


def test_copied_evidence_cannot_escape_declared_root(component):
    path, config = component
    config["copy_evidence"].append(config["previous_snapshot"])
    write(path, config)
    with pytest.raises(ValueError, match="Only declared small component evidence"):
        checkpoint.seal(path, verify_only=True)


def test_mid_render_evidence_change_leaves_failed_output_without_snapshot(component, renderers, monkeypatch):
    path, config = component
    ordinary_plot = checkpoint.plot

    def mutate(root, output):
        ordinary_plot(root, output)
        Path(config["copy_evidence"][0]).write_text("mid-render mutation\n")

    monkeypatch.setattr(checkpoint, "plot", mutate)
    with pytest.raises(ValueError, match="Component input changed"):
        checkpoint.seal(path)
    output = Path(config["output"])
    assert output.exists()
    assert (output / "historical-render-command.json").exists()
    assert not (output / "snapshot.json").exists()


def test_failed_renderer_does_not_publish_snapshot(component, renderers, monkeypatch):
    path, config = component
    original_run = checkpoint.subprocess.run

    def fail(argv, **kwargs):
        if len(argv) > 1 and argv[1] == config["ordinary_renderer"]:
            return subprocess.CompletedProcess(argv, 1, "", "renderer failure")
        return original_run(argv, **kwargs)

    monkeypatch.setattr(checkpoint.subprocess, "run", fail)
    with pytest.raises(ValueError, match="six-policy renderer failed"):
        checkpoint.seal(path)
    output = Path(config["output"])
    assert not (output / "snapshot.json").exists()
    assert json.loads((output / "historical-render-command.json").read_text())["exit_code"] == 1
