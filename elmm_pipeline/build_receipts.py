"""Record actual compiler argv and file identities without invoking a shell.

This module is also copied into isolated SCons build directories. It has no
package dependencies and does not infer numerical legality or placement.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

ENVIRONMENT = (
    "PATH", "CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LIBRARY_PATH",
    "LD_LIBRARY_PATH", "GCC_EXEC_PREFIX", "COMPILER_PATH", "SOURCE_DATE_EPOCH",
    "LANG", "LC_ALL", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS",
    "FC", "CC", "CXX", "FFLAGS", "FCFLAGS", "F90FLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS",
)
SOURCE_SUFFIXES = {".f", ".for", ".ftn", ".f90", ".f95", ".f03", ".f08", ".c", ".cc", ".cpp", ".cxx", ".cu"}
INPUT_SUFFIXES = SOURCE_SUFFIXES | {".h", ".hh", ".hpp", ".cuh", ".inc"}


def file_identity(path):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "sha256": digest}


def _unchanged(item):
    try:
        return file_identity(item["path"])["sha256"] == item["sha256"]
    except OSError:
        # A failed compiler can remove a stale module file. Preserve its actual
        # exit status while making the changed-input receipt unusable.
        return False


def read_receipts(directory):
    """Return successful, still-current execution records and receipt hashes."""
    records = []
    paths = sorted(Path(directory).glob("*.json"))
    if not paths or list(Path(directory).glob("*.tmp")):
        raise ValueError("Complete executed build receipts are unavailable")
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or record.get("schema_version") != 1
                or record.get("kind") != "executed-build-command" or record.get("shell") is not False
                or record.get("status") != "completed" or record.get("returncode") != 0
                or not isinstance(record.get("argv"), list) or not record["argv"]
                or any(not isinstance(arg, str) for arg in record["argv"])
                or not isinstance(record.get("inputs"), list) or not isinstance(record.get("outputs"), list)
                or not record["outputs"] or not isinstance(record.get("compiler"), dict)
                or record["compiler"].get("version_returncode") != 0):
            raise ValueError("Build receipt does not describe a successful compiler command: " + str(path))
        for item in [record["compiler"], *record["inputs"], *record["outputs"]]:
            if (not isinstance(item, dict) or not isinstance(item.get("path"), str)
                    or not isinstance(item.get("sha256"), str)):
                raise ValueError("Build receipt requires explicit file identities: " + str(path))  # noqa: TRY004 - invalid JSON evidence
            try:
                current = file_identity(item["path"])
            except OSError as error:
                raise ValueError("Build receipt file is unavailable: " + item["path"]) from error
            if current["sha256"] != item["sha256"]:
                raise ValueError("Build receipt file changed: " + item["path"])
        records.append({"receipt": file_identity(path), "execution": record})
    return sorted(records, key=lambda item: item["execution"]["started_ns"])


def snapshot_inputs(roots, *, exclude=()):
    """Bind supplied source/header trees, conservatively including unused files."""
    excluded = [Path(path).resolve() for path in exclude]
    paths = set()
    for root in sorted({Path(path).resolve(strict=True) for path in roots}):
        for path in root.rglob("*"):
            if (not any(path.is_relative_to(item) for item in excluded) and path.is_file()
                    and (path.suffix.lower() in INPUT_SUFFIXES or path.name == "SConstruct")):
                paths.add(path.resolve())
                if len(paths) > 20000:
                    raise ValueError("Build source/header inventory exceeds 20000 files")
    return {str(path): file_identity(path)["sha256"] for path in sorted(paths)}


def summarize_build(binary, directories, inputs):
    """Require actual successful execution evidence for every linked object."""
    binary = Path(binary).resolve(strict=True)
    rows = [row for directory in directories for row in read_receipts(directory)]
    commands, identities = {}, dict(inputs)
    for path, digest in inputs.items():
        if file_identity(path)["sha256"] != digest:
            raise ValueError("Build source/header changed: " + path)
    for row in rows:
        record = row["execution"]
        identities[row["receipt"]["path"]] = row["receipt"]["sha256"]
        for item in [record["compiler"], *record["inputs"], *record["outputs"]]:
            identities[item["path"]] = item["sha256"]
        for output in record["outputs"]:
            if output["path"] in commands:
                raise ValueError("Build has ambiguous executed commands for " + output["path"])
            commands[output["path"]] = record
    link = commands.get(str(binary))
    if link is None or "-c" in link["argv"]:
        raise ValueError("Current build receipts lack executable link command")
    objects = [item["path"] for item in link["inputs"] if Path(item["path"]).suffix == ".o"]
    if not objects:
        raise ValueError("Build receipts require explicitly linked objects")
    units = []
    for path in objects:
        record = commands.get(path)
        if record is None or record["argv"].count("-c") != 1:
            raise ValueError("Current build receipts lack linked object command: " + path)
        source = [item for item in record["inputs"] if Path(item["path"]).suffix.lower() in SOURCE_SUFFIXES]
        if len(source) != 1 or inputs.get(source[0]["path"]) != source[0]["sha256"]:
            raise ValueError("Compiled source lacks pre-build identity: " + path)
        units.append({"object": path, "source": source[0]["path"], "execution": record})
    value = {"schema_version": 1, "kind": "application-build-receipts", "status": "complete",
             "placement_authority": False, "binary": str(binary), "units": units, "link": link,
             "source_inputs": inputs, "files_sha256": identities,
             "limitations": ["No placement authority", "Implicit system headers and libraries are not inventoried"]}
    value["identity"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value


def require_build_receipts(path):
    """Revalidate the immutable build receipt inventory before consuming it."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(value, dict) or value.get("schema_version") != 1
            or value.get("kind") != "application-build-receipts" or value.get("status") != "complete"
            or value.get("placement_authority") is not False or not value.get("files_sha256")):
        raise ValueError("Complete application build receipts are unavailable")
    payload = {key: item for key, item in value.items() if key != "identity"}
    if value.get("identity") != hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest():
        raise ValueError("Application build receipt identity differs")
    for name, digest in value["files_sha256"].items():
        if file_identity(name)["sha256"] != digest:
            raise ValueError("Application build receipt file changed: " + name)
    return value


class CommandRecorder:
    """One JSON receipt per executed command; parallel writers never share files."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._compilers = {}
        self._lock = threading.Lock()

    def _compiler(self, argv, environment, directory):
        executable = (str(directory / argv[0]) if os.path.dirname(argv[0])
                      else shutil.which(argv[0], path=environment.get("PATH", os.defpath)))
        if executable is None:
            raise FileNotFoundError("Build executable is unavailable: " + argv[0])
        identity = file_identity(executable)
        key = identity["path"], identity["sha256"]
        with self._lock:
            if key not in self._compilers:
                result = subprocess.run([identity["path"], "--version"], env=environment,
                                        capture_output=True, text=True, check=False, timeout=15)
                identity["version"] = (result.stdout + result.stderr).strip()
                identity["version_returncode"] = result.returncode
                self._compilers[key] = identity
            return dict(self._compilers[key])

    def run(self, argv, *, cwd=None, env=None, stdout=None, stderr=None, additional_inputs=(), timeout=None):
        """Execute precisely the recorded argv and retain failures as evidence."""
        argv = [str(argument) for argument in argv]
        if not argv or any("\0" in argument for argument in argv):
            raise ValueError("Build commands require nonempty argv without NUL bytes")
        directory = Path(cwd or os.getcwd()).resolve()
        environment = dict(os.environ if env is None else env)
        try:
            compiler, identity_error = self._compiler(argv, environment, directory), None
        except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
            # Evidence failure must not change the actual compiler invocation.
            compiler, identity_error = None, type(error).__name__ + ": " + str(error)
        record = {"schema_version": 1, "kind": "executed-build-command", "status": "running",
                  "argv": argv, "cwd": str(directory), "shell": False,
                  "environment": {name: environment.get(name) for name in ENVIRONMENT},
                  # Bind all environment entries without copying unrelated
                  # credentials into a public build receipt.
                  "environment_sha256": hashlib.sha256(json.dumps(environment, sort_keys=True,
                                                                   separators=(",", ":")).encode()).hexdigest(),
                  "compiler": compiler, "inputs": [], "outputs": [],
                  "started_ns": time.time_ns()}
        if timeout is not None:
            record.update(timeout_seconds=timeout, timed_out=False)
        if identity_error is not None:
            record["identity_error"] = identity_error
        output = None
        if argv.count("-o") == 1 and argv.index("-o") + 1 < len(argv):
            output = (directory / argv[argv.index("-o") + 1]).resolve()
        paths = {(directory / path).resolve(strict=True) for path in additional_inputs}
        for argument in argv[1:]:
            if argument.startswith("-"):
                continue
            path = (directory / argument).resolve()
            if path != output and path.is_file():
                paths.add(path)
        record["inputs"] = [file_identity(path) for path in sorted(paths)]
        receipt = self.directory / (uuid.uuid4().hex + ".json")

        def save():
            temporary = receipt.with_suffix(".tmp")
            temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
            temporary.replace(receipt)

        save()
        try:
            result = subprocess.run(argv, cwd=directory, env=environment, stdout=stdout,
                                    stderr=stderr, check=False, timeout=timeout)
            record["returncode"] = result.returncode
            record["status"] = "completed"
            if output is not None and output.is_file():
                record["outputs"] = [file_identity(output)]
            if ((compiler is not None and not _unchanged(compiler))
                    or any(not _unchanged(item) for item in record["inputs"])):
                record["status"] = "inputs_changed"
            return result
        except BaseException as error:
            record["status"] = "failed"
            record["error"] = type(error).__name__ + ": " + str(error)
            if isinstance(error, subprocess.TimeoutExpired):
                record["timed_out"] = True
            raise
        finally:
            record["finished_ns"] = time.time_ns()
            save()

    def scons_spawn(self, shell, escape, command, arguments, environment):
        """Decode SCons' already shell-quoted tokens, then execute without a shell."""
        del shell, escape, command
        argv = shlex.split(" ".join(arguments), posix=True)
        if any(token in {";", "|", "||", "&", "&&", ">", ">>", "<", "<<"} for token in argv):
            raise ValueError("Recorded application builds require direct compiler commands")
        return self.run(argv, env=environment).returncode
