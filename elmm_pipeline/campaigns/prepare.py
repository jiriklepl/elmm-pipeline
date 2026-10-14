"""Freeze explicitly selected repositories, configurations and approved binaries."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .identity import check_files, digest, read, repository, tree_hashes, write
from .launch_environment import freeze_launch_environment
from .targets import validate_target


def slug(value):
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]*", value):
        raise ValueError("names must be lowercase filename slugs")
    return value


def validate_cpu_placement(placement):
    """Check the optional calibrated process mask without changing affinity."""
    if placement is None:
        return
    if (not isinstance(placement, dict) or type(placement.get("schema_version")) is not int
            or placement["schema_version"] != 1
            or placement.get("environment") != {"OMP_PROC_BIND": "false"}):
        raise ValueError("invalid calibrated CPU placement contract")
    cpus = placement.get("cpu_affinity")
    if (not isinstance(cpus, list) or len(cpus) != 4
            or any(type(cpu) is not int or cpu < 0 for cpu in cpus)
            or cpus != sorted(set(cpus))):
        raise ValueError("calibrated CPU placement requires four sorted unique CPU numbers")
    if sorted(os.sched_getaffinity(0)) != cpus:
        raise ValueError("process CPU affinity must exactly match numerical calibration; launch the campaign with taskset")


def calibrated_cpu_placement(profile):
    numerical = profile.get("numerical")
    if not isinstance(numerical, dict) or numerical.get("schema_version") != 2:
        return None
    if type(numerical["schema_version"]) is not int:
        raise ValueError("invalid numerical calibration schema")
    identity, protocol = numerical.get("identity"), numerical.get("protocol")
    if not isinstance(identity, dict) or not isinstance(protocol, dict):
        raise ValueError("numerical calibration v2 requires CPU placement identity and protocol")
    placement = {"schema_version": 1, "cpu_affinity": identity.get("cpu_affinity"),
                 "environment": {"OMP_PROC_BIND": "false"}}
    validate_cpu_placement(placement)
    if protocol.get("cpu_affinity") != placement["cpu_affinity"] or protocol.get("omp_proc_bind") != "false":
        raise ValueError("numerical calibration CPU placement protocol mismatch")
    return placement


def validate(config):
    if config.get("schema_version") != 1 or config.get("threads") != 4:
        raise ValueError("campaign schema 1 and the four-thread comparison budget required")
    for kind in ("cases", "modes"):
        names = [slug(row["name"]) for row in config[kind]]
        if not names or len(names) != len(set(names)):
            raise ValueError("nonempty unique " + kind + " required")
    modes = {row["name"] for row in config["modes"]}
    if "native" not in modes or not config["baseline_modes"] or not set(config["baseline_modes"]) <= modes:
        raise ValueError("native and explicit frozen baseline modes required")
    if config.get("candidate_mode") and config["candidate_mode"] not in modes:
        raise ValueError("candidate must be one fixed mode across cases")
    if config.get("repetitions", 3) not in (3, 7):
        raise ValueError("three or seven measured repetitions required")
    if config.get("warmups", 1) != 1:
        raise ValueError("one matched warmup required")
    for case in config["cases"]:
        if "performance_target" in case:
            validate_target(case["performance_target"], modes, config.get("candidate_mode"))
            if config.get("repetitions", 3) != 7:
                raise ValueError("per-case performance targets require seven measured repetitions")
    if not {"compiler", "pipeline"} <= config["repositories"].keys():
        raise ValueError("explicit independent compiler and pipeline repositories required")
    for name in [*config["repositories"], *config.get("commands", {}), *config.get("input_trees", {})]:
        slug(name)
    if set(config.get("commands", {})) & {"verification", "halos", "timings"}:
        raise ValueError("additional commands cannot replace required correctness/timing phases")
    drivers = config.get('halo_drivers', [])
    names = [slug(driver['name']) for driver in drivers]
    if len(names) != len(set(names)):
        raise ValueError('unique halo driver names required')
    application_drivers = [driver for driver in drivers if 'application_case' in driver]
    if application_drivers:
        coverage = {row['name']: set() for row in config['cases']}
        for driver in application_drivers:
            case, selected = driver['application_case'], set(driver['sources'])
            if case not in coverage or not selected <= modes:
                raise ValueError('application field driver requires configured case and modes')
            if 'native' not in selected:
                raise ValueError('application field driver requires an original native reference')
            if set(driver.get('link', {})) != selected:
                raise ValueError('application field driver sources and link modes must match')
            coverage[case].update(selected)
        missing = [(case, sorted(modes - selected)) for case, selected in coverage.items() if selected != modes]
        if missing:
            raise ValueError('application field drivers must cover every configured case and mode: ' + str(missing))


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(strict=True), Path(output).resolve()
    config = read(config_path)
    validate(config)
    # Absolute paths are deliberate: a campaign never guesses a sibling checkout.
    def source(value):
        path = Path(value)
        if not path.is_absolute():
            path = config_path.parent / path
        return path.resolve(strict=True)

    repos = {name: repository(source(spec["path"]), spec["paths"])
             for name, spec in config["repositories"].items()}
    output.mkdir(parents=True, exist_ok=False)
    state = {**config, "status": "preparing", "created_utc": datetime.now(timezone.utc).isoformat(),
             "repositories": repos, "source_config": str(config_path), "source_config_sha256": digest(config_path),
             "files": {}, "external_files": {}, "timings_reused": False, "policy_promoted": False}
    write(output / "campaign.json", state)

    def copy(path, target):
        path, target = source(path), output / target
        target.parent.mkdir(parents=True, exist_ok=True)
        before = digest(path)
        shutil.copy2(path, target)
        if digest(target) != before or digest(path) != before:
            raise ValueError("input changed during snapshot: " + str(path))
        state["files"][str(target.relative_to(output))] = before
        return str(target.relative_to(output))

    for name, repo in repos.items():
        for path in repo["sha256"]:
            copy(Path(repo["root"]) / path, Path("sources") / name / path)
        check_files(Path(repo["root"]), repo["sha256"])
    for name, path in config.get("input_trees", {}).items():
        path = source(path)
        before = tree_hashes(path)
        for filename in before:
            copy(path / filename, Path("inputs") / name / filename)
        if tree_hashes(path) != before:
            raise ValueError("input tree changed during snapshot: " + str(path))
    state["profile"] = copy(config["profile"], "inputs/profile.json")
    profile = read(output / state["profile"])
    if profile.get("cpu_threads") != 4 or profile.get("precision_bits") != 64:
        raise ValueError("matching four-thread double-precision profile required")
    if profile.get("scoped", {}).get("runtime_id") != config["runtime_id"]:
        raise ValueError("profile/runtime identity mismatch")
    placement = calibrated_cpu_placement(profile)
    state.pop("cpu_placement", None)  # Derived from frozen evidence, not user overrides.
    if placement is not None:
        state["cpu_placement"] = placement
    state["openmp_launch_environment"] = freeze_launch_environment(
        profile, config.get("openmp_launch_environment"))
    state["profile_sha256"] = digest(output / state["profile"])
    state["selected_gpu_uuid"] = profile.get("hardware", {}).get("gpu_uuid")
    if not isinstance(state["selected_gpu_uuid"], str) or not state["selected_gpu_uuid"].startswith("GPU-"):
        raise ValueError("calibration must identify the selected GPU UUID")
    state["tools"] = {}
    for name, command in config["tools"].items():
        executable = shutil.which(command[0])
        if not executable:
            raise ValueError("missing configured tool: " + name)
        path = str(Path(executable).resolve())
        result = subprocess.run([path, *command[1:]], text=True, capture_output=True, check=True)
        state["tools"][name] = {"path": path, "sha256": digest(path), "argv": command,
                                "version": result.stdout + result.stderr}
        state["external_files"][path] = digest(path)
    if "time" not in state["tools"]:
        raise ValueError("explicit GNU time measurement tool required")
    # Runtime libraries and original build artifacts are recorded separately
    # from copied inputs. Mutation invalidates the receipt before the next job.
    for path, expected in config.get("external_files", {}).items():
        path = source(path)
        if digest(path) != expected:
            raise ValueError("external build artifact changed: " + str(path))
        state["external_files"][str(path)] = expected
    state["modes"] = []
    for mode in config["modes"]:
        binary = source(mode["binary"])
        if digest(binary) != mode["sha256"]:
            raise ValueError("approved binary changed: " + mode["name"])
        entry = {**mode, "original_binary": str(binary),
                 "binary": copy(binary, "binaries/" + mode["name"])}
        if "build_receipt" in mode:
            entry["build_receipt"] = copy(mode["build_receipt"], "inputs/" + mode["name"] + "-build.json")
        if "adapter" in mode:
            entry["adapter"] = copy(mode["adapter"], "inputs/" + mode["name"] + "-adapter.json")
        state["modes"].append(entry)
    state["cases"] = []
    for case in config["cases"]:
        path = source(case["config"])
        before = tree_hashes(path)
        target = output / "configs" / case["name"]
        shutil.copytree(path, target)
        if tree_hashes(path) != before or tree_hashes(target) != before:
            raise ValueError("configuration changed during snapshot")
        for name, expected in before.items():
            state["files"][str((target / name).relative_to(output))] = expected
        state["cases"].append({**case, "config": str(target.relative_to(output)),
                              "config_sha256": before,
                              "reference": copy(case["reference"], "references/" + case["name"] + ".vtk")})
    state["evidence"] = [copy(path, "evidence/" + str(index) + "-" + source(path).name)
                         for index, path in enumerate(config.get("evidence", []))]
    state["halo_drivers"] = []
    for driver in config.get("halo_drivers", []):
        name = slug(driver["name"])
        state["halo_drivers"].append({**driver, "sources": {
            mode: copy(path, "drivers/" + name + "/" + slug(mode) + ".f90")
            for mode, path in driver["sources"].items()}})
    state["host"] = {"uname": list(os.uname()), "affinity": sorted(os.sched_getaffinity(0))}
    state["status"] = "prepared"
    write(output / "campaign.json", state)
    write(output / "identity.json", {"campaign_sha256": digest(output / "campaign.json")})
    return output


def check(output):
    output = Path(output).resolve(strict=True)
    if digest(output / "campaign.json") != read(output / "identity.json")["campaign_sha256"]:
        raise ValueError("campaign configuration changed")
    state = read(output / "campaign.json")
    if state["status"] != "prepared":
        raise ValueError("campaign preparation did not finish")
    check_files(output, state["files"])
    check_files(Path("/"), state["external_files"])
    for case in state["cases"]:
        if tree_hashes(output / case["config"]) != case["config_sha256"]:
            raise ValueError("frozen configuration gained or lost files")
    return state
