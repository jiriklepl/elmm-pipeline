"""Derive a diagnostic-main relink from an application's own build receipts."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
from pathlib import Path


def _file(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("expected a regular build input: " + str(path))
    return path


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _resolve(value, cwd):
    path = Path(value)
    return (path if path.is_absolute() else cwd / path).resolve()


def _compiler(value, cwd):
    if os.path.dirname(str(value)):
        path = _resolve(value, cwd)
    else:
        found = shutil.which(str(value))
        if found is None:
            raise ValueError("Fortran compiler is unavailable: " + str(value))
        path = Path(found).resolve()
    return _file(path)


def _new_output(value):
    path = Path(value).absolute()
    if os.path.lexists(path):
        raise ValueError("diagnostic output already exists: " + str(path))
    path = path.resolve()
    if not path.parent.is_dir():
        raise ValueError("diagnostic output parent does not exist: " + str(path.parent))
    return path


def derive_driver_link(binary, *, fortran, source, object_path, output):
    """Return compile/link argv and provenance without running a compiler.

    Commands run in the recorded build cwd. Only the original main source,
    application main object and final output are replaced; every other argument
    retains its exact position and spelling. The caller owns execution and must
    keep the recorded inputs unchanged between this plan and execution.
    """
    binary, source = _file(binary), _file(source)
    if source.suffix.lower() not in {".f", ".for", ".ftn", ".f90", ".f95", ".f03", ".f08"}:
        raise ValueError("diagnostic main must be a Fortran source")
    object_path, output = _new_output(object_path), _new_output(output)
    if object_path == output:
        raise ValueError("diagnostic object and executable must be distinct")
    original_object = _file(binary.parent / "main.o")
    receipt, log = binary.parent / "build-command.json", binary.parent / "build.log"
    if receipt.stat().st_size > 2**20 or log.stat().st_size > 64 * 2**20:
        raise ValueError("application build receipts exceed the bounded interface")
    raw_receipt, raw_log = receipt.read_bytes(), log.read_bytes()
    if len(raw_receipt) > 2**20 or len(raw_log) > 64 * 2**20:
        raise ValueError("application build receipts exceed the bounded interface")
    metadata = json.loads(raw_receipt)
    if (not isinstance(metadata, dict) or not isinstance(metadata.get("cwd"), str)
            or not Path(metadata["cwd"]).is_absolute()
            or not isinstance(metadata.get("argv"), list) or not metadata["argv"]
            or any(not isinstance(arg, str) or not arg for arg in metadata["argv"])):
        raise ValueError("application build receipt requires explicit cwd and argv")
    cwd = Path(metadata["cwd"]).resolve(strict=True)
    if not cwd.is_dir():
        raise ValueError("application build cwd is not a directory")
    compiler = _compiler(fortran, Path.cwd())
    selected = {"compile": {}, "link": {}}
    for line in raw_log.decode("utf-8").splitlines():
        if "-o" not in line:
            continue
        try:
            lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|<>")
            lexer.whitespace_split = True
            lexer.commenters = ""
            argv = list(lexer)
        except ValueError as error:
            raise ValueError("malformed command in application build log") from error
        if "-o" not in argv:
            continue
        if argv.count("-o") != 1 or argv.index("-o") + 1 >= len(argv):
            raise ValueError("malformed output option in application build log")
        target = _resolve(argv[argv.index("-o") + 1], cwd)
        if target not in {original_object, binary}:
            continue
        if (not argv or any(set(arg) <= set(";&|<>") or arg.startswith("@") or "\0" in arg
                            for arg in argv)):
            raise ValueError("application main commands must be direct argv, without shell or response files")
        if _compiler(argv[0], cwd) != compiler:
            raise ValueError("application main compiler differs from the supplied Fortran compiler")
        role = "compile" if target == original_object else "link"
        if argv.count("-c") != (1 if role == "compile" else 0):
            raise ValueError("application main compile/link mode is malformed")
        selected[role][tuple(argv)] = argv
    if any(len(commands) != 1 for commands in selected.values()):
        raise ValueError("unique application main compile and executable link commands required")
    compile_argv, link_argv = (next(iter(selected[role].values())) for role in ("compile", "link"))
    compile_output = compile_argv.index("-o") + 1
    sources = [(index, _resolve(arg, cwd)) for index, arg in enumerate(compile_argv[1:], 1)
               if index != compile_output and not arg.startswith("-")
               and Path(arg).suffix.lower() in {".f", ".for", ".ftn", ".f90", ".f95", ".f03", ".f08"}]
    if len(sources) != 1:
        raise ValueError("unique original Fortran main source required")
    source_index, original_source = sources[0]
    original_source = _file(original_source)
    link_output = link_argv.index("-o") + 1
    objects = [index for index, arg in enumerate(link_argv[1:], 1)
               if index != link_output and not arg.startswith("-") and _resolve(arg, cwd) == original_object]
    if len(objects) != 1:
        raise ValueError("exactly one application main object required in the link command")
    compiled, linked = list(compile_argv), list(link_argv)
    compiled[source_index], compiled[compile_output] = str(source), str(object_path)
    linked[objects[0]], linked[link_output] = str(object_path), str(output)
    return {"schema_version": 1, "cwd": str(cwd), "compile_argv": compiled, "link_argv": linked,
            "provenance": {"original_compile_argv": compile_argv, "original_link_argv": link_argv,
                           "original_main_source": str(original_source), "original_main_object": str(original_object),
                           "original_executable": str(binary), "fortran_compiler": str(compiler),
                           "sha256": {str(path): _digest(path) for path in
                                      (binary, original_object, original_source, source, compiler)},
                           "build_receipts_sha256": {
                               str(receipt): hashlib.sha256(raw_receipt).hexdigest(),
                               str(log): hashlib.sha256(raw_log).hexdigest()}}}
