"""Ask the compiler whether one outlined entry is eligible, through its command line."""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

COMPILER_PHASE_TIMEOUT_SECONDS = 240


class LocalAdapterError(RuntimeError):
    pass


def _python_environment():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _invoke(source_path, compiler_root, python, output_dir, entry, arguments, log_name):
    """Use only the selected snapshot's public CLI and JSON response."""
    source_path = Path(source_path).resolve(strict=True)
    compiler_root = Path(compiler_root).resolve(strict=True)
    python = Path(python).absolute()
    output_dir = Path(output_dir).resolve()
    if not (compiler_root / "compiler/__main__.py").is_file():
        raise LocalAdapterError("Supply an explicit compiler snapshot containing compiler/__main__.py")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:::[A-Za-z][A-Za-z0-9_]*)?", entry):
        raise LocalAdapterError(f"Invalid generated entry name: {entry!r}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LocalAdapterError(f"Configured compiler Python is not executable: {python}")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _python_environment()
    environment["PYTHONPATH"] = str(compiler_root)
    command = [str(python), "-m", "compiler", "--input", str(source_path),
               "--opt-level", "1", "--fallback", "error", "--json", *arguments]
    try:
        result = subprocess.run(command, cwd=compiler_root, env=environment, text=True,
                                capture_output=True, check=False, timeout=COMPILER_PHASE_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as error:
        stdout, stderr = (value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
                          for value in (error.stdout, error.stderr))
        (output_dir / log_name).write_text(stdout + stderr)
        (output_dir / (log_name + ".timeout.json")).write_text(json.dumps({
            "argv": command, "cwd": str(compiler_root), "timeout_seconds": COMPILER_PHASE_TIMEOUT_SECONDS,
            "timed_out": True, "returncode": None}, indent=2) + "\n")
        raise LocalAdapterError(f"Compiler {entry} exceeded the {COMPILER_PHASE_TIMEOUT_SECONDS}-second phase limit; "
                                "no artifacts accepted") from error
    (output_dir / log_name).write_text(result.stdout + result.stderr)
    try:
        report = json.loads(result.stdout)
    except (ValueError, TypeError) as error:
        raise LocalAdapterError(
            "Compiler did not return a JSON result; refresh the explicit compiler snapshot: "
            + (result.stdout + result.stderr)[-4000:]) from error
    if result.returncode:
        reason = report.get("reason") if isinstance(report, dict) else None
        raise LocalAdapterError(f"Compiler rejected {entry}: {reason or (result.stdout + result.stderr)[-4000:]}")
    return report, command


def check(source_path: Path, compiler_root: Path, python: Path, entry: str, output_dir: Path) -> dict:
    """Return the selected entry's authoritative, strict eligibility record.

    A rejected candidate returns ``supported=False`` without emitting sources.
    Invalid tool responses and missing or ambiguous entries raise an adapter error.
    """
    records, _ = _invoke(source_path, compiler_root, python, output_dir, entry,
                         ["--list-candidates"], "check.log")
    if not isinstance(records, list):
        raise LocalAdapterError("Compiler candidate report must be a JSON array")
    selector = "qualified_name" if "::" in entry else "name"
    matches = [record for record in records if isinstance(record, dict)
               and str(record.get(selector, "")).lower() == entry.lower()]
    if len(matches) != 1:
        raise LocalAdapterError(f"Compiler candidate report must identify exactly one entry: {entry}")
    report, = matches
    if type(report.get("supported")) is not bool:
        raise LocalAdapterError("Compiler candidate report is missing its supported decision")
    (Path(output_dir) / "check.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
