"""Execute one source-selected diagnostic relink from structured build receipts.

This transports recorded build recipes; the selected compiler's verifier remains
responsible for authenticating the application contract and diagnostic policy.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ..build_receipts import (
    ENVIRONMENT,
    CommandRecorder,
    file_identity,
    read_receipts,
    require_build_receipts,
)

COMPILER_TIMEOUT_SECONDS = 180
FORTRAN_SUFFIXES = {".f", ".for", ".ftn", ".f90", ".f95", ".f03", ".f08"}


def _file(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("expected a regular build input: " + str(path))
    return path


def _read(path, expected):
    path = _file(path)
    if file_identity(path)["sha256"] != expected:
        raise ValueError("application build observation changed: " + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    return file_identity(path)


def _new_output(value):
    path = Path(value).absolute()
    if os.path.lexists(path):
        raise ValueError("diagnostic output already exists: " + str(path))
    path = path.resolve()
    if not path.parent.is_dir():
        raise ValueError("diagnostic output parent does not exist: " + str(path.parent))
    return path


def _operand(argv, path, cwd):
    indices = [index for index, arg in enumerate(argv[1:], 1)
               if not arg.startswith("-") and str((cwd / arg).resolve()) == path]
    if len(indices) != 1:
        raise ValueError("diagnostic replacement requires one exact original operand: " + path)
    return indices[0]


def _command(command, *, output, compile, environment):
    argv = command.get("argv")
    if (command.get("kind") != "executed-build-command" or command.get("schema_version") != 1
            or command.get("status") != "completed" or command.get("returncode") != 0
            or command.get("shell") is not False or not isinstance(argv, list) or not argv
            or any(not isinstance(arg, str) or "\0" in arg or arg.startswith("@") for arg in argv)
            or argv.count("-c") != int(compile) or argv.count("-o") != 1
            or argv.index("-o") + 1 == len(argv)):
        raise ValueError("successful direct executed compile/link command required")
    cwd = Path(command["cwd"])
    if not cwd.is_absolute() or not cwd.is_dir() or (cwd / argv[argv.index("-o") + 1]).resolve() != Path(output):
        raise ValueError("recorded command directory or output differs")
    if command.get("environment") != {name: environment.get(name) for name in ENVIRONMENT}:
        raise ValueError("diagnostic environment differs from the recorded build environment")
    if command.get("compiler", {}).get("version_returncode") != 0:
        raise ValueError("successful compiler identity required")
    for item in [command["compiler"], *command["inputs"], *command["outputs"]]:
        if file_identity(item["path"])["sha256"] != item["sha256"]:
            raise ValueError("recorded command input/output changed: " + item["path"])
    if {item["path"] for item in command["outputs"]} != {output}:
        raise ValueError("one recorded command output required")
    return cwd


def build_driver_link(application_build, *, original_source, source, object_path, output,
                      directory, environment=None, module_inputs=()):
    """Build one diagnostic and return the verifier's plan/build/two-command refs.

    Select the unchanged original driver explicitly, never by an object name.
    Supply its reviewed replacement and any current Fortran module dependencies.
    The supplied (or current) environment must match both recorded commands.
    Output paths and the evidence directory must be new. Failed builds retain
    their plan, logs and command receipts and never return a usable observation.
    """
    original_source, source = _file(original_source), _file(source)
    if original_source.suffix.lower() not in FORTRAN_SUFFIXES or source.suffix.lower() not in FORTRAN_SUFFIXES:
        raise ValueError("diagnostic driver must replace an explicitly selected Fortran source")
    object_path, output = _new_output(object_path), _new_output(output)
    if len({source, object_path, output}) != 3:
        raise ValueError("diagnostic source, object and executable must be distinct")
    receipt_path = _file(application_build["receipt"])
    _read(receipt_path, application_build["receipt_sha256"])
    receipt = require_build_receipts(receipt_path)
    attestation = _read(application_build["attestation"], application_build["attestation_sha256"])
    if (attestation["receipt"] != str(receipt_path)
            or attestation["receipt_sha256"] != application_build["receipt_sha256"]
            or attestation["receipt_identity"] != receipt["identity"]
            or attestation["binary"] != receipt["binary"]
            or any(attestation[name] != application_build[name]
                   for name in ("binary_sha256", "contract_identity", "variant"))):
        raise ValueError("application build attestation differs from the executed receipt")
    destination = attestation["preflight"]["request"]["destinations"].get(str(original_source))
    units = [unit for unit in receipt["units"] if unit["source"] == destination]
    original_hash = file_identity(original_source)["sha256"]
    if len(units) != 1 or receipt["source_inputs"].get(destination) != original_hash:
        raise ValueError("diagnostic driver must select one unchanged original compilation unit")
    unit = units[0]
    before_compile, before_link = unit["execution"], receipt["link"]
    environment = dict(os.environ if environment is None else environment)
    cwd = _command(before_compile, output=unit["object"], compile=True, environment=environment)
    if _command(before_link, output=receipt["binary"], compile=False, environment=environment) != cwd:
        raise ValueError("diagnostic compile and link require the same recorded working directory")
    compile_argv, link_argv = list(before_compile["argv"]), list(before_link["argv"])
    compile_argv[_operand(compile_argv, destination, cwd)] = str(source)
    compile_argv[compile_argv.index("-o") + 1] = str(object_path)
    link_argv[_operand(link_argv, unit["object"], cwd)] = str(object_path)
    link_argv[link_argv.index("-o") + 1] = str(output)
    modules = {}
    for path in module_inputs:
        item = file_identity(_file(path))
        if Path(item["path"]).suffix.lower() not in {".mod", ".smod"}:
            raise ValueError("diagnostic module inputs must be .mod or .smod files")
        modules[item["path"]] = item["sha256"]
    if any(str(path) in receipt["files_sha256"] for path in (source, object_path, output)):
        raise ValueError("diagnostic artifacts must not overwrite application inputs or outputs")
    identities = {**receipt["files_sha256"], **modules}
    for path in (receipt_path, _file(application_build["attestation"]), original_source, source):
        item = file_identity(path)
        identities[item["path"]] = item["sha256"]
    plan = {"schema_version": 1, "placement_authority": False,
            "application_build_identity": receipt["identity"], "application_build_receipt": str(receipt_path),
            "application_binary": receipt["binary"], "diagnostic_binary": str(output),
            "original_source": str(original_source), "original_object": unit["object"],
            "module_inputs": modules, "files_sha256": identities,
            "compiler_timeout_seconds": COMPILER_TIMEOUT_SECONDS,
            "relink": {"cwd": str(cwd), "compile_argv": compile_argv, "link_argv": link_argv}}
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    plan_ref = _write(directory / "plan.json", plan)
    recorder = CommandRecorder(directory / "commands")
    build = {"schema_version": 1, "status": "failed", "plan_sha256": plan_ref["sha256"],
             "binary": str(output), "compiler_timeout_seconds": COMPILER_TIMEOUT_SECONDS}

    def recheck():
        for path, expected in identities.items():
            if file_identity(path)["sha256"] != expected:
                raise ValueError("diagnostic build input changed: " + path)

    try:
        for phase, argv, previous, replaced in (("compile", compile_argv, before_compile, destination),
                                               ("link", link_argv, before_link, unit["object"])):
            recheck()
            inputs = {item["path"] for item in previous["inputs"] if item["path"] != replaced} | set(modules)
            with (directory / (phase + ".log")).open("wb") as log:
                result = recorder.run(argv, cwd=cwd, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                      additional_inputs=inputs, timeout=COMPILER_TIMEOUT_SECONDS)
            if result.returncode:
                raise ValueError("diagnostic " + phase + " failed; see " + str(directory / (phase + ".log")))
        recheck()
        rows = read_receipts(directory / "commands")
        if len(rows) != 2:
            raise ValueError("exactly two successful diagnostic commands required")
        for row, previous, argv in zip(rows, (before_compile, before_link), (compile_argv, link_argv), strict=True):
            current = row["execution"]
            if current["argv"] != argv or any(current[key] != previous[key] for key in ("cwd", "compiler", "environment")):
                raise ValueError("executed diagnostic command differs from its recorded recipe")
        build.update(status="passed", binary_sha256=file_identity(output)["sha256"])
    except BaseException as error:
        build["error"] = type(error).__name__ + ": " + str(error)
        _write(directory / "build.json", build)
        raise
    build_ref = _write(directory / "build.json", build)
    observation = {"plan": plan_ref, "build": build_ref, "commands": [row["receipt"] for row in rows]}
    _write(directory / "observation.json", observation)
    return observation
