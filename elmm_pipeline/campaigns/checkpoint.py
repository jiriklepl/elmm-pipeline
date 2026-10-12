"""Seal component evidence while retaining separate historical application claims."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from .identity import guard_passed
from .report import plot


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    path = Path(path)
    _require(path.stat().st_size <= 32 * 2**20, "Component JSON exceeds 32 MiB: " + str(path))
    return json.loads(path.read_text())


def _write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _hashes(hashes):
    for path, expected in hashes.items():
        _require(_digest(path) == expected, "Component input changed: " + path)


def _validate(configuration):
    config = _read(configuration)
    _require(type(config.get("schema_version")) is int and config["schema_version"] == 1
             and config.get("component_only") is True and config.get("historical_plots_only") is True
             and config.get("draft", False) is False
             and all(type(config.get(key)) is int and config[key] == 0
                     for key in ("new_application_observations", "new_performance_samples")),
             "Bounded component-only, historical-plots-only configuration required")
    # Legacy receipt_helpers remains hashed provenance when declared; no code
    # from that path, or from any previous results directory, is imported.
    _hashes(config["input_sha256"])
    _hashes(config["historical_input_sha256"])
    _require(Path(config["repositories"]["pipeline"]["path"]).resolve(strict=True)
             == Path(__file__).resolve().parents[2], "Foreign maintained pipeline renderer")
    commits = {}
    for name, repository in config["repositories"].items():
        root = Path(repository["path"]).resolve(strict=True)
        actual = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        _require(actual == repository["commit"], "Committed component checkpoint required: " + name)
        commits[name] = actual
        for path, expected in repository.get("verified_source_sha256", {}).items():
            relative = Path(path)
            _require(not relative.is_absolute() and ".." not in relative.parts,
                     "Component source must be relative to its repository")
            contents = subprocess.check_output(["git", "-C", str(root), "show", actual + ":" + path])
            _require(hashlib.sha256(contents).hexdigest() == expected and _digest(root / path) == expected,
                     "Verified component source differs from committed bytes: " + path)
    for path in config["successful_guards"]:
        _require(path in config["input_sha256"], "Component guard requires an input hash: " + path)
        guard_passed(Path(path))
    for check in config["json_checks"]:
        _require(check["path"] in config["input_sha256"], "Component check requires an input hash")
        value = _read(check["path"])
        for key in check["keys"]:
            value = value[key]
        _require(value == check["equals"], "Component evidence check failed: " + check["path"])
    for check in config["text_checks"]:
        _require(check["path"] in config["input_sha256"], "Component log requires an input hash")
        _require(check["contains"] in Path(check["path"]).read_text(),
                 "Component log marker missing: " + check["path"])
    previous = Path(config["previous_snapshot"])
    _require(str(previous) in config["input_sha256"], "Previous component snapshot requires an input hash")
    previous_receipt = _read(previous)
    _require(previous_receipt["compatible_commits"] == config["previous_checkpoint_commits"],
             "Previous component checkpoint must retain its original commit pair")
    application = Path(config["previous_application_snapshot"])
    application_receipt = _read(application)
    expected = config["input_sha256"].get(str(application), previous_receipt.get("previous_application_snapshot_sha256"))
    _require(expected == _digest(application), "Previous application snapshot changed or lacks a hash")
    _require(application_receipt["compatible_commits"] == config["previous_application_commits"]
             and application_receipt["full_application_gate_passed"] is True,
             "Last full-application proof must retain its original commit pair")
    _require(commits["compiler"] != config["previous_checkpoint_commits"]["compiler"],
             "A new component checkpoint must not relabel the previous component proof")
    historical = Path(config["historical_campaign"])
    for path in (Path(config["ordinary_renderer"]), *(historical / name for name in (
            "campaign.json", "timings/report.json", "verification/report.json", "ordinary/report.json"))):
        _require(str(path) in config["historical_input_sha256"], "Historical input requires a hash: " + str(path))
    evidence_root = Path(config["evidence_root"]).resolve(strict=True)
    _require(len(set(config["copy_evidence"])) == len(config["copy_evidence"]), "Duplicate component evidence")
    for value in config["copy_evidence"]:
        path = Path(value).resolve(strict=True)
        _require(path.is_relative_to(evidence_root) and path.stat().st_size <= 2 * 2**20
                 and value in config["input_sha256"], "Only declared small component evidence may be copied")
    return config, commits


def seal(configuration, *, verify_only=False):
    """Verify explicit schema-1 inputs, or create one new immutable component snapshot.

    Verification does not render, execute the configured renderer, or write files.
    Sealing redraws historical reports only; it performs no application runs.
    """
    configuration = Path(configuration).resolve(strict=True)
    configuration_sha = _digest(configuration)
    config, commits = _validate(configuration)
    if verify_only:
        return {"status": "component_checkpoint_inputs_verified", "compatible_commits": commits,
                "configuration_sha256": configuration_sha, "snapshot_written": False,
                "new_application_observations": 0, "new_performance_samples": 0}
    output = Path(config["output"])
    _require(not output.exists() and not output.is_symlink(), "Immutable component snapshot already exists")
    output.mkdir()
    evidence_root = Path(config["evidence_root"]).resolve(strict=True)
    copied = {}
    for value in config["copy_evidence"]:
        path = Path(value).resolve(strict=True)
        target = output / "evidence" / path.relative_to(evidence_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        _require(_digest(target) == config["input_sha256"][value], "Component evidence copy changed")
        copied[value] = {"sha256": _digest(target), "snapshot_copy": str(target.relative_to(output))}
    historical = Path(config["historical_campaign"])
    plot(historical, output / "figures")
    command = [sys.executable, config["ordinary_renderer"], str(historical / "ordinary/report.json"),
               "--output", str(output / "ordinary-figures"), "--label", config["ordinary_label"]]
    result = subprocess.run(command, cwd=config["workspace"],
        env={**os.environ, "MPLCONFIGDIR": "/tmp/fort-component-historical-plots"},
        capture_output=True, text=True, check=False)
    _write(output / "historical-render-command.json", {"argv": command, "exit_code": result.returncode,
        "stdout": result.stdout, "stderr": result.stderr, "new_performance_samples": 0})
    _require(result.returncode == 0, "Maintained historical six-policy renderer failed")
    # Renderers are external to evidence validation. Recheck before publishing
    # a successful snapshot; failures retain the partial output for inspection.
    _require(_digest(configuration) == configuration_sha, "Component configuration changed")
    _validate(configuration)
    previous, application = Path(config["previous_snapshot"]), Path(config["previous_application_snapshot"])
    report = config["component_report"].rstrip() + f"""

[Historical scoped comparison](figures/complete-wall-relative.png) · [Historical six-policy comparison](ordinary-figures/complete-wall-relative.png) · [Previous component checkpoint]({os.path.relpath(previous.parent / 'report.md', output)}) · [Last full-application checkpoint]({os.path.relpath(application.parent / 'report.md', output)}).

Both charts redraw unchanged historical results with their original readable labels. **Zero new performance samples** were collected. This component checkpoint adds no whole-application execution or performance proof for compiler `{commits['compiler']}`. The previous three-case proof belongs to compiler `{config['previous_application_commits']['compiler']}` and pipeline `{config['previous_application_commits']['pipeline']}`; it is not attributed to this newer compiler. No automatic-placement, speedup or whole-stage residency promotion is claimed.
"""
    with (output / "report.md").open("x") as stream:
        stream.write(report)
    _write(output / "historical-plot-status.json", {"historical_only": True, "new_performance_samples": 0,
        "original_labels_preserved": True, "current_build_timing_reuse": False,
        "historical_input_sha256": config["historical_input_sha256"],
        "previous_snapshot": str(previous), "previous_snapshot_sha256": _digest(previous),
        "previous_application_snapshot": str(application), "previous_application_snapshot_sha256": _digest(application)})
    _write(output / "evidence-inventory.json", {"small_evidence": copied,
        "retained_external_authorities_sha256": config["input_sha256"], "large_arrays_objects_binaries_copied": False,
        "sealing_configuration": {"path": str(configuration), "sha256": configuration_sha},
        "sealing_script_sha256": _digest(__file__)})
    _write(output / "snapshot.json", {"schema_version": 1, "status": "sealed_component_correctness_only",
        "created_utc": datetime.now(timezone.utc).isoformat(), "compatible_commits": commits,
        "component_claims": config["component_claims"], "new_application_observations": 0,
        "new_performance_samples": 0, "historical_figures_only": True,
        "full_application_gate_for_current_compiler_verified": False,
        "previous_full_application_commits": config["previous_application_commits"],
        "previous_application_snapshot": os.path.relpath(application, output),
        "previous_application_snapshot_sha256": _digest(application),
        "previous_snapshot": os.path.relpath(previous, output), "previous_snapshot_sha256": _digest(previous),
        "policy_promoted": False,
        "files_sha256": {str(path.relative_to(output)): _digest(path) for path in sorted(output.rglob("*")) if path.is_file()}})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--historical-plots-only", action="store_true", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    result = seal(args.config, verify_only=args.verify_only)
    print("COMPONENT_CHECKPOINT_INPUTS_VERIFIED_NO_PLOTS_OR_SNAPSHOT_WRITTEN" if args.verify_only else result)


if __name__ == "__main__":
    main()
