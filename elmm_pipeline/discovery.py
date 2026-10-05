"""Run structural Fortran discovery in an explicitly selected parser environment."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Iterable
import fnmatch

from .common import python_environment


def discover(source_path: Path | Iterable[Path], project_root: Path, python: Path,
             output_dir: Path, defines: Iterable[str] = ("DPREC",),
             select: Iterable[str] = ()) -> dict:
    """Discover independently executable loop regions without application names.

    A file, directory, or explicit collection of files is accepted. Directories
    are scanned recursively for free-form Fortran. Discovery never changes an
    input file. ``select`` contains optional case-insensitive routine globs.
    """
    project_root = Path(project_root).resolve(strict=True)
    if isinstance(source_path, (str, Path)):
        path = Path(source_path).resolve(strict=True)
        sources = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".f90", ".f95", ".f03", ".f08"}) if path.is_dir() else [path]
    else:
        sources = [Path(path).resolve(strict=True) for path in source_path]
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request = {"sources": [str(path) for path in sources], "project_root": str(project_root),
               "output_dir": str(output_dir), "defines": list(defines), "select": list(select)}
    request_path = output_dir / "discovery-request.json"
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    worker = Path(__file__).with_name("tool_workers") / "discovery_worker.py"
    command = [str(Path(python).absolute()), str(worker), "--request", str(request_path)]
    with (output_dir / "discovery.log").open("w") as log:
        result = subprocess.run(command, cwd=output_dir, env=python_environment(), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Structural discovery failed; see {output_dir / 'discovery.log'}\n"
                           + (output_dir / "discovery.log").read_text()[-4000:])
    manifest = json.loads((output_dir / "discovery.json").read_text())
    manifest["command"] = command
    return manifest


def discover_project(project_root: Path, python: Path, output_dir: Path, *,
                     patterns: Iterable[str] = ("*.f90",), defines: Iterable[str] = ("DPREC",),
                     select: Iterable[str] = ()) -> dict:
    """Batch a project scan so imported modules are parsed once per worker."""
    project_root = Path(project_root).resolve(strict=True)
    patterns = list(patterns or ("*.f90",))
    patterns += [pattern[3:] for pattern in patterns if pattern.startswith("**/")]
    sources = [path for path in sorted(project_root.rglob("*")) if path.is_file()
               and path.suffix.lower() in {".f90", ".f95", ".f03", ".f08"}
               and any(fnmatch.fnmatchcase(str(path.relative_to(project_root)).lower(), pattern.lower())
                       or fnmatch.fnmatchcase(path.name.lower(), pattern.lower()) for pattern in patterns)]
    return discover(sources, project_root, python, output_dir, defines, select)
