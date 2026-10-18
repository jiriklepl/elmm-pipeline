"""Measure compiler-issued application alternatives and transport their evidence.

This collector owns process execution and immutable observations, not numerical
validation or placement. The explicitly selected compiler verifies build inputs
before execution and compares real output payloads before evidence is delivered.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path

from .common import (
    COMPILER_PHASE_TIMEOUT_SECONDS,
    clean_environment,
    python_environment,
)
from .validation import validate_log

PROTOCOL = "checked-interval-process-v4"
OBSERVATION = "FORT_SELECTION_OBSERVATION "
COMPLETION = "FORT_APPLICATION_COMPLETION "
MAX_LOG_BYTES = 64 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _reference(path):
    return {"path": str(Path(path).resolve(strict=True)), "sha256": _hash(path)}


def _read(reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("an explicit hashed JSON reference is required")
    path = Path(reference["path"])
    if not path.is_absolute() or path.stat().st_size > MAX_LOG_BYTES:
        raise ValueError("JSON reference must be absolute and bounded")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("referenced JSON changed: " + str(path))
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("referenced JSON must contain an object")
    return value


def _relative(name):
    if (type(name) is not str or not name or "\0" in name or "\\" in name
            or name.startswith("/") or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("workload paths must be contained relative files")
    return name


def _input(directory, name, expected):
    path = directory / _relative(name)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(directory.resolve(strict=True)):
        raise ValueError("workload input escapes its configuration directory")
    if _hash(path) != expected:
        raise ValueError("workload input changed: " + str(path))
    return path


def _verifier(transport, phase, request, directory):
    """Call a public compiler command; its timeout excludes application runs."""
    root = Path(transport["compiler_root"]).resolve(strict=True)
    # A virtualenv launcher is commonly a symlink to the system interpreter.
    python = Path(transport["python"]).absolute()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError("selected compiler Python launcher is unavailable")
    if not (root / "compiler/offload/selection_cost.py").is_file():
        raise ValueError("selected compiler has no interval evidence verifier")
    directory.mkdir(parents=True, exist_ok=False)
    request_path, output = directory / "request.json", directory / "response.json"
    _write(request_path, request)
    environment = python_environment()
    environment["PYTHONPATH"] = str(root)
    argv = [str(python), "-m", "compiler.offload.selection_cost", phase,
            "--request", str(request_path), "--output", str(output)]
    try:
        result = subprocess.run(argv, cwd=root, env=environment, text=True, capture_output=True,
                                check=False, timeout=COMPILER_PHASE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = (value.decode(errors="replace") if isinstance(value, bytes) else value or ""
                          for value in (error.stdout, error.stderr))
        (directory / "stdout.log").write_text(stdout)
        (directory / "stderr.log").write_text(stderr)
        _write(directory / "command.json", {"argv": argv, "cwd": str(root), "returncode": None,
                                            "timeout_seconds": COMPILER_PHASE_TIMEOUT_SECONDS, "timed_out": True})
        raise ValueError("compiler interval evidence verification exceeded the "
                         f"{COMPILER_PHASE_TIMEOUT_SECONDS}-second phase limit") from error
    (directory / "stdout.log").write_text(result.stdout)
    (directory / "stderr.log").write_text(result.stderr)
    _write(directory / "command.json", {"argv": argv, "cwd": str(root), "returncode": result.returncode,
                                        "timeout_seconds": COMPILER_PHASE_TIMEOUT_SECONDS})
    if result.returncode:
        raise ValueError("compiler interval evidence verification rejected the campaign: " + result.stderr[-2000:])
    response = _read(_reference(output))
    if (response.get("status") != "verified" or response.get("identity") !=
            _digest({key: value for key, value in response.items() if key != "identity"})):
        raise ValueError("compiler interval verifier returned an inconsistent response")
    return response


def _text(path):
    if path.stat().st_size > MAX_LOG_BYTES:
        raise ValueError("application log exceeds the supported evidence bound")
    return path.read_text(errors="strict")


def _completion(stdout, method):
    if method == {"kind": "elmm-log"}:
        observed = validate_log(stdout)
        return {key: observed[key] for key in ("completed_steps", "natural_end")}
    if method == {"kind": "json-stdout"}:
        values = [json.loads(line[len(COMPLETION):]) for line in stdout.splitlines() if line.startswith(COMPLETION)]
        if len(values) != 1 or type(values[0]) is not dict or not values[0]:
            raise ValueError("one observed application completion object is required")
        return values[0]
    raise ValueError("unsupported application completion observation")


def _outputs(directory, policy):
    result, hashes = {}, {}
    raw = policy["outputs"]["format"] == "raw-float64-storage-v1"
    for name, specification in policy["outputs"]["files"].items():
        paths = specification if raw else {"file": name}
        refs = {}
        for kind, relative in paths.items():
            path = directory / _relative(relative)
            if not path.resolve(strict=True).is_relative_to(directory):
                raise ValueError("output escapes the measured run directory")
            refs[kind] = _reference(path)
            hashes[relative] = refs[kind]["sha256"]
        result[name] = refs if raw else refs["file"]
    return result, hashes


def _run_process(directory, *, executable, workload, configuration, environment, threads,
                 timeout, completion, policy):
    """Time one actual process, retaining failure receipts before raising."""
    directory.mkdir(parents=True, exist_ok=False)
    for name, expected in workload["files"].items():
        if name in {"stdout.log", "stderr.log", "run.json"}:
            raise ValueError("workload input conflicts with collector evidence files")
        source = _input(configuration, name, expected)
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        _input(directory, name, expected)
    binary = Path(executable["binary"]).resolve(strict=True)
    if _hash(binary) != executable["binary_sha256"]:
        raise ValueError("verified executable changed before execution")
    argv = [str(binary), *workload["argv"][1:]]
    stdout, stderr = directory / "stdout.log", directory / "stderr.log"
    record = {"schema_version": 1, "kind": "measured-application-process", "argv": argv,
              "binary_sha256": executable["binary_sha256"], "cwd": str(directory),
              "workload_identity": workload["identity"], "configuration_sha256": workload["files"],
              "environment": {name: environment.get(name) for name in workload["environment"]},
              "environment_sha256": _digest(environment), "affinity": sorted(os.sched_getaffinity(0)),
              "threads": threads, "traced": False, "timed_out": False, "exit_code": None,
              "application_timeout_seconds": timeout, "completion": {}, "output_hashes": {}}
    started = time.perf_counter()
    try:
        with stdout.open("w") as out, stderr.open("w") as err:
            process = subprocess.run(argv, cwd=directory, env=environment, stdout=out, stderr=err,
                                     timeout=timeout, check=False)
        record["exit_code"] = process.returncode
    except subprocess.TimeoutExpired:
        record["timed_out"] = True
    finally:
        record["diagnostic_complete_process_seconds"] = time.perf_counter() - started
        _write(directory / "run.json", record)
    if record["timed_out"] or record["exit_code"] != 0:
        raise ValueError("application did not complete successfully: " + str(directory))
    try:
        if _hash(binary) != executable["binary_sha256"]:
            raise ValueError("verified executable changed during execution")
        for name, expected in workload["files"].items():
            _input(directory, name, expected)
            _input(configuration, name, expected)
        observed = _completion(_text(stdout), completion)
        record["completion"] = observed
        if any(type(observed.get(name)) is not type(expected) or observed.get(name) != expected
               for name, expected in workload["completion"].items()):
            raise ValueError("observed application completion differs from the compiler workload")
        outputs, record["output_hashes"] = _outputs(directory, policy)
        _write(directory / "run.json", record)
        return {"run": _reference(directory / "run.json"), "observations": _reference(stderr), "outputs": outputs}
    finally:
        _write(directory / "run.json", record)


def _invocations(reference, *, selection, mode):
    path = Path(reference["path"])
    if _hash(path) != reference["sha256"]:
        raise ValueError("application observation log changed")
    rows, identities = [], []
    for line in _text(path).splitlines():
        if not line.startswith(OBSERVATION):
            continue
        row = json.loads(line[len(OBSERVATION):])
        if (type(row) is not dict or row.get("selection") != selection or row.get("outcome") != mode
                or row.get("complete") is not True or type(row.get("fallbacks")) is not int
                or row["fallbacks"] != 0 or type(row.get("launches")) is not int or row["launches"] < 0
                or mode == "native" and row["launches"] != 0 or type(row.get("key")) is not list
                or any(type(value) is not int for value in row["key"])
                or type(row.get("seconds")) not in {int, float} or not math.isfinite(row["seconds"])
                or row["seconds"] <= 0):
            raise ValueError("application did not complete the selected alternative")
        identities.append(row["cpu_identity"] if mode == "native" else row["identity"])
        rows.append({"ordinal": len(rows), **{key: row[key] for key in
                    ("selection", "key", "seconds", "outcome", "launches", "fallbacks", "complete")}})
    if not rows or any(value != identities[0] for value in identities) or type(identities[0]) is not dict:
        raise ValueError("application selection observations or stable execution identity are missing")
    if mode == "resident" and not any(row["launches"] for row in rows):
        raise ValueError("resident process reported no completed GPU work")
    return rows, identities[0]


def _deliver(output, contract, result):
    _write(output / "evidence.json", result["evidence"])
    _write(output / "scope-cost.json", {"coverage": "checked_interval", "acceptance": contract["acceptance"],
           **{key: contract[key] for key in ("workloads", "build_contract", "validation_policy")},
           "evidence": result["evidence"]})
    return {"decision": result["decision"], "evidence": _reference(output / "evidence.json"),
            "scope_cost": _reference(output / "scope-cost.json")}


def run(request, output):
    """Collect one unchanged workload; the compiler makes the final decision."""
    request = deepcopy(request)
    if type(request.get("schema_version")) is not int or request["schema_version"] != 1:
        raise ValueError("selection campaign request schema 1 is required")
    contract = _read(request["contract"])
    if (contract.get("protocol") != PROTOCOL or contract.get("schema_version") != 2
            or contract.get("acceptance") != {"criterion": "paired_non_regression"}):
        raise ValueError("a compiler-issued v4 paired_non_regression contract is required")
    candidates = [item for item in contract["workloads"] if item["identity"] == request["workload_identity"]]
    if len(candidates) != 1:
        raise ValueError("one exact compiler workload is required")
    workload = candidates[0]
    timeout = request["application_timeout_seconds"]
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("the existing application runtime limit must be specified explicitly")
    configuration = Path(request["configuration_directory"]).resolve(strict=True)
    for name, expected in workload["files"].items():
        _input(configuration, name, expected)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "request.json", request)
    report = {"schema_version": 1, "status": "running", "placement_authority": False,
              "qualification": False, "warmups": [], "application_timeout_seconds": timeout,
              "compiler_verifier_timeout_seconds": COMPILER_PHASE_TIMEOUT_SECONDS}
    try:
        verified = _verifier(request["compiler"], "builds", {key: request[key] for key in
                             ("contract", "builds", "diagnostics")}, output / "build-verification")
        if verified["contract_sha256"] != _digest(contract):
            raise ValueError("compiler verified a different selection contract")
        environment = clean_environment(contract["threads"])
        for name in list(environment):
            if name.startswith("FORT_"):
                environment.pop(name)
        for name, value in workload["environment"].items():
            if name in {"FORT_RUNTIME_TRACE", "FORT_OFFLOAD_TRACE", "FORT_EXACT_CALIBRATION_MODE"} and value is not None:
                raise ValueError("tracing and runtime forcing cannot enter calibration timings")
            if value is None:
                environment.pop(name, None)
            else:
                environment[name] = value
        arguments = {"workload": workload, "configuration": configuration, "environment": environment,
                     "threads": contract["threads"], "timeout": timeout, "completion": request["completion"],
                     "policy": contract["validation_policy"]}

        def execute(path, mode, kind):
            result = _run_process(output / path, executable=verified["variants"][mode][kind], **arguments)
            _invocations(result["observations"], selection=contract["selection_identity"], mode=mode)
            return result

        for mode in ("native", "resident"):
            report["warmups"].append({"mode": mode, **execute("warmup/" + mode, mode, "application")})
        policy = contract["validation_policy"]
        numerical = {"schema_version": policy["schema_version"],
                     "protocol": "checked-selection-numerics-v" + str(policy["schema_version"]),
                     "contract_sha256": _digest(contract), "policy_identity": policy["identity"],
                     "workload_identity": workload["identity"],
                     "selection_bundle_identity": contract["selection_bundle"]["identity"],
                     "variants": {}, "pairs": []}
        for mode in ("native", "resident"):
            observed = execute("diagnostic/" + mode, mode, "diagnostic")
            directory = Path(observed["run"]["path"]).parent
            fields = {name: {kind: _reference(directory / _relative(path)) for kind, path in paths.items()}
                      for name, paths in policy["fields"].items()}
            numerical["variants"][mode] = {**observed, "fields": fields,
                                           "diagnostic_build": request["diagnostics"][mode]}
        observations, identities = [], []
        for round_index in range(9):
            pair = {"round": round_index}
            for turn in range(2):
                mode = ("native", "resident")[(round_index + turn) % 2]
                measured = execute(f"round-{round_index}/{mode}", mode, "application")
                pair[mode] = measured
                process = _read(measured["run"])
                invocations, identity = _invocations(measured["observations"], selection=contract["selection_identity"], mode=mode)
                if mode == "resident":
                    identities.append(identity)
                observations.append({"round": round_index, "mode": mode, "role": "fit" if round_index < 7 else "holdout",
                    "seconds": process["diagnostic_complete_process_seconds"], "build": request["builds"][mode],
                    "invocations": invocations, "observed_cpu_identity": identity if mode == "native" else None,
                    "process": {"returncode": process["exit_code"], "workload_identity": workload["identity"],
                                **{key: workload[key] for key in ("argv", "environment", "files")},
                                "completion": {key: process["completion"][key] for key in workload["completion"]},
                                "selection_bundle_identity": contract["selection_bundle"]["identity"],
                                "wall_time_includes": ["startup", "first_invocation", "all_invocations", "output", "teardown"],
                                "seconds": process["diagnostic_complete_process_seconds"]}})
            numerical["pairs"].append(pair)
        if not identities or any(value != identities[0] for value in identities):
            raise ValueError("resident execution identity changed across the campaign")
        identity = identities[0]
        for row in observations:
            # This is the shared campaign environment observed by the resident
            # processes; each native process retains its independent CPU event.
            row["identity"] = identity
            row["key"] = row["observed_key"] = row["invocations"][0]["key"]
            if row["mode"] == "resident":
                row["observed_cpu_identity"] = {key: identity[key] for key in
                    ("cpu_name", "fortran_version", "fortran_options", "affinity", "effective_schedule", "environment")}
                row["observed_cpu_identity"]["team"] = [identity[key] for key in
                    ("threads", "dynamic", "proc_bind", "thread_limit", "max_active_levels")]
        _write(output / "numerical.json", numerical)
        campaign = {"protocol": PROTOCOL, "contract": contract, "workload": workload, "identity": identity,
                    "selection_bundle": contract["selection_bundle"], "observations": observations,
                    "numerical_validation": _reference(output / "numerical.json")}
        _write(output / "raw-evidence.json", {"protocol": PROTOCOL, "campaigns": [campaign]})
        result = _verifier(request["compiler"], "evaluate", {"contract": request["contract"],
                           "evidence": _reference(output / "raw-evidence.json")}, output / "evaluation")
        report.update(status="complete", **_deliver(output, contract, result), measured_pairs=9)
        return report
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        _write(output / "report.json", report)


def combine(request, output):
    """Reverify complete workload coverage under one identical compiler contract."""
    if (type(request.get("schema_version")) is not int or request["schema_version"] != 1
            or type(request.get("scope_costs")) is not list or not 1 <= len(request["scope_costs"]) <= 32):
        raise ValueError("combination requires schema 1 and explicit scope-cost references")
    costs = [_read(reference) for reference in request["scope_costs"]]
    campaigns, contract, seen = [], None, set()
    for cost in costs:
        evidence = cost.get("evidence")
        if (cost.get("coverage") != "checked_interval"
                or cost.get("acceptance") != {"criterion": "paired_non_regression"}
                or type(evidence) is not dict or evidence.get("protocol") != PROTOCOL
                or type(evidence.get("campaigns")) is not list or not evidence["campaigns"]):
            raise ValueError("combination requires canonical checked-interval campaigns")
        for campaign in evidence["campaigns"]:
            if contract is None:
                contract = campaign["contract"]
            if campaign["contract"] != contract:
                raise ValueError("different compiler contracts require separate calibration packages")
            workload = campaign["workload"]["identity"]
            if workload in seen:
                raise ValueError("combination repeats a workload campaign")
            seen.add(workload)
            campaigns.append(campaign)
    if seen != {workload["identity"] for workload in contract["workloads"]}:
        raise ValueError("combination must cover every workload in the compiler contract")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "request.json", request)
    report = {"schema_version": 1, "status": "running", "placement_authority": False, "qualification": False}
    try:
        _write(output / "contract.json", contract)
        _write(output / "raw-evidence.json", {"protocol": PROTOCOL, "campaigns": campaigns})
        result = _verifier(request["compiler"], "evaluate", {"contract": _reference(output / "contract.json"),
                           "evidence": _reference(output / "raw-evidence.json")}, output / "evaluation")
        for reference in request["scope_costs"]:
            _read(reference)
        report.update(status="complete", measured_workloads=len(seen), **_deliver(output, contract, result))
        return report
    except BaseException as error:
        report.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        raise
    finally:
        _write(output / "report.json", report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--combine", action="store_true", help="Combine complete workload coverage without new executions")
    args = parser.parse_args(argv)
    try:
        result = (combine if args.combine else run)(_read(_reference(args.request)), args.output)
        print(json.dumps({"status": result["status"], "scope_cost": result["scope_cost"]}))
        return 0
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
