"""Assemble checked-interval capture requests without issuing compiler proofs.

One recorded analysis source set is reused by capture and production. Existing
executed origins supply build facts; the selected compiler owns their acceptance,
numerical eligibility, source edits and placement decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

from .build_receipts import file_identity, require_build_receipts

PROTOCOL = "checked-interval-request-session-v1"
MAX_BYTES = 64 * 1024 * 1024


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _read(reference):
    if type(reference) is not dict or set(reference) != {"path", "sha256"}:
        raise ValueError("an explicit hashed file reference is required")
    path = Path(reference["path"])
    if not path.is_absolute() or not path.is_file() or path.stat().st_size > MAX_BYTES:
        raise ValueError("a bounded absolute JSON file is required")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("referenced request file changed: " + str(path))
    value = json.loads(raw)
    if type(value) is not dict:
        raise ValueError("referenced JSON must be an object")
    return value


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return file_identity(path)


def _absolute(value):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("explicit absolute source/build paths are required")
    return str(path.resolve(strict=True))


def _remember(inventory, path, expected=None):
    identity = file_identity(_absolute(path))
    name, recorded = identity["path"], identity["sha256"]
    if expected is not None and expected != recorded or name in inventory and inventory[name] != recorded:
        raise ValueError("source/configuration input changed: " + name)
    inventory[name] = recorded
    return identity


def _json_input(inventory, reference):
    _remember(inventory, reference["path"], reference["sha256"])
    return _read(reference)


def _origin(inventory, requested, mode):
    receipt_ref = requested["receipt"]
    _json_input(inventory, receipt_ref)
    receipt = require_build_receipts(receipt_ref["path"])
    for path, recorded in receipt["files_sha256"].items():
        _remember(inventory, path, recorded)
    mapping = requested["source_mapping"]
    if type(mapping) is not dict or not mapping:
        raise ValueError("explicit original-source mappings are required")
    mapped = {_absolute(source): _absolute(target) for source, target in mapping.items()}
    if len(mapped) != len(mapping) or len(set(mapped.values())) != len(mapped):
        raise ValueError("ambiguous original-source mapping")
    if not set(mapped) <= set(receipt["source_inputs"]):
        raise ValueError("mapped source is absent from the executed origin inventory")
    for source, target in mapped.items():
        _remember(inventory, target, receipt["source_inputs"][source] if mode == "native" else None)
    build_root = _absolute(requested["build_root"])
    if Path(receipt["binary"]).parent != Path(build_root):
        raise ValueError("origin build root differs from its recorded binary")
    result = {"receipt": _absolute(receipt_ref["path"]), "receipt_sha256": receipt_ref["sha256"],
              "build_root": build_root, "source_mapping": mapped}
    if mode == "resident":
        manifest = requested["manifest"]
        _json_input(inventory, manifest)
        result.update(manifest=_absolute(manifest["path"]), manifest_sha256=manifest["sha256"])
    return result


def _workload(inventory, requested):
    directory = Path(_absolute(requested["configuration_directory"]))
    names = requested["files"]
    if type(names) is not list or not 1 <= len(names) <= 64 or len(set(names)) != len(names):
        raise ValueError("explicit distinct workload input files are required")
    files = {}
    for name in names:
        if (type(name) is not str or not name or "\\" in name or "\0" in name or name.startswith("/")
                or any(part in {"", ".", ".."} for part in name.split("/"))):
            raise ValueError("workload paths must be contained relative files")
        path = directory / name
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(directory) or path.stat().st_size > MAX_BYTES:
            raise ValueError("workload input must be a bounded contained file")
        files[name] = _remember(inventory, str(path))["sha256"]
    argv, environment, completion = (requested[key] for key in ("argv", "environment", "completion"))
    if (type(argv) is not list or not argv or any(type(arg) is not str or "\0" in arg for arg in argv)
            or type(environment) is not dict or type(completion) is not dict or not completion
            or any(type(name) is not str or not name or "\0" in name or "=" in name
                   or value is not None and (type(value) is not str or "\0" in value)
                   for name, value in environment.items())):
        raise ValueError("explicit workload argv, environment and completion are required")
    value = {"argv": argv, "environment": environment, "files": files, "completion": completion}
    return {"identity": _digest(value), **value}, str(directory)


def assemble(request, output):
    """Write native/resident requests; their contents grant no placement authority."""
    requested = deepcopy(request)
    if type(requested.get("schema_version")) is not int or requested["schema_version"] != 1:
        raise ValueError("selection request schema 1 is required")
    compiler = requested["compiler"]
    compiler_root = Path(_absolute(compiler["compiler_root"]))
    python = Path(compiler["python"])
    if (not (compiler_root / "compiler/__main__.py").is_file() or not python.is_absolute()
            or not python.is_file() or not os.access(python, os.X_OK)):
        raise ValueError("an explicit compiler checkout and executable Python launcher are required")
    inventory, analysis = {}, requested["analysis"]
    references = [analysis["source"], *analysis.get("source_files", [])]
    sources = {_remember(inventory, item["path"], item["sha256"])["path"]: item["sha256"] for item in references}
    if len(sources) != len(references):
        raise ValueError("analysis sources must be distinct exact paths")
    facts = _json_input(inventory, analysis["capture_facts"])
    if "sources" in facts and facts["sources"] != sources:
        raise ValueError("capture facts belong to another analysis source set")
    for key in ("analysis_sources", "numerical_sources", "effect_contracts"):
        if key not in analysis:
            continue
        package = _json_input(inventory, analysis[key])
        if key == "analysis_sources":
            if package.get("source_inputs") != sources:
                raise ValueError("configured analysis belongs to another source set")
            for path, recorded in package.get("dependencies", {}).items():
                _remember(inventory, path, recorded)
        if key in {"analysis_sources", "numerical_sources"}:
            for item in package.get("entries", []):
                _remember(inventory, item["path"], item["sha256"])
    origins = {mode: _origin(inventory, requested["origins"][mode], mode) for mode in ("native", "resident")}
    if not set(sources) <= set(origins["native"]["source_mapping"].values()):
        raise ValueError("analysis sources must use the immutable original paths mapped by the native origin")
    workloads, transport = [], []
    for specification in requested["workloads"]:
        workload, directory = _workload(inventory, specification)
        workloads.append(workload)
        transport.append({"workload_identity": workload["identity"], "configuration_directory": directory})
    if not 1 <= len(workloads) <= 32 or len({item["identity"] for item in workloads}) != len(workloads):
        raise ValueError("one to thirty-two distinct workloads are required")
    policy = _json_input(inventory, requested["validation_policy"])
    driver = policy["driver"]
    _remember(inventory, driver["original_source"], driver["original_sha256"])
    _remember(inventory, requested["diagnostic_source"]["path"], requested["diagnostic_source"]["sha256"])
    if requested["diagnostic_source"]["sha256"] != driver["replacement_sha256"]:
        raise ValueError("diagnostic source differs from the explicitly supplied numerical policy")
    options = requested.get("gpu_options", {})
    if (type(options) is not dict or set(options) - {"host_threads", "scope_transfers", "calibration_profile"}
            or type(options.get("host_threads", 4)) is not int or options.get("host_threads", 4) < 1):
        raise ValueError("only shared host-thread, transfer and calibration-profile options are supported")
    if options.get("calibration_profile"):
        _remember(inventory, options["calibration_profile"])
    base = {"coverage": "checked_interval", "acceptance": requested["acceptance"], "workloads": workloads,
            "build_contract": {"schema_version": 1, "protocol": "executed-application-build-v1", "origins": origins},
            "validation_policy": policy}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    costs = {mode: _write(output / ("capture-" + mode + ".json"), {**base, "calibration": mode})
             for mode in ("native", "resident")}
    for path, recorded in inventory.items():
        _remember({}, path, recorded)
    session = {"schema_version": 1, "protocol": PROTOCOL, "placement_authority": False,
               "request": requested, "source_inputs": sources, "files_sha256": inventory,
               "scope_costs": costs, "workload_transport": transport, "base": base}
    session["identity"] = _digest(session)
    return {"session": _write(output / "session.json", session), "scope_costs": costs,
            "workloads": transport, "placement_authority": False}


def generation_inputs(session_reference, mode, *, production_scope_cost=None):
    """Return existing transform_scopes arguments on the same original paths."""
    session = _read(session_reference)
    if (session.get("protocol") != PROTOCOL or session.get("identity") !=
            _digest({key: value for key, value in session.items() if key != "identity"})):
        raise ValueError("immutable selection request session differs")
    for path, recorded in session["files_sha256"].items():
        _remember({}, path, recorded)
    base = session["base"]
    if mode in {"native", "resident"}:
        if production_scope_cost is not None:
            raise ValueError("capture requests cannot consume production evidence")
        cost = session["scope_costs"][mode]
        if _read(cost) != {**base, "calibration": mode}:
            raise ValueError("capture request differs from its shared source session")
    elif mode == "production":
        cost = production_scope_cost
        value = _read(cost)
        if (value.get("coverage") != "checked_interval" or "calibration" in value
                or any(value.get(key) != base[key] for key in ("acceptance", "workloads"))
                or value.get("build_contract", {}).get("origins") != base["build_contract"]["origins"]
                or {key: item for key, item in value.get("validation_policy", {}).items() if key != "identity"} !=
                   {key: item for key, item in base["validation_policy"].items() if key != "identity"}
                or value.get("evidence", {}).get("protocol") != "checked-interval-process-v4"):
            raise ValueError("production evidence belongs to another capture request")
        campaigns = value["evidence"].get("campaigns", [])
        if not campaigns or any(row.get("contract", {}).get("sources") != session["source_inputs"] for row in campaigns):
            raise ValueError("production evidence belongs to another immutable analysis source set")
    else:
        raise ValueError("generation mode must be native, resident or production")
    requested = session["request"]
    analysis = requested["analysis"]
    kwargs = {"source_path": _absolute(analysis["source"]["path"]), "entry": analysis["entry"],
              "source_files": [_absolute(item["path"]) for item in analysis.get("source_files", [])],
              "capture_facts": _read(analysis["capture_facts"]),
              "compiler_root": _absolute(requested["compiler"]["compiler_root"]),
              # Resolving a virtualenv launcher symlink loses its environment.
              "python": str(Path(requested["compiler"]["python"]).absolute()),
              "gpu_options": {**requested.get("gpu_options", {}), "gpu_policy": "auto", "scope_execution": "reached",
                              "scope_cost": _absolute(cost["path"])}}
    for key in ("analysis_sources", "numerical_sources", "effect_contracts"):
        if key in analysis:
            kwargs[key] = _absolute(analysis[key]["path"])
    return kwargs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = assemble(_read(file_identity(args.request)), args.output)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
