"""Compile a discovered, self-contained region using an explicit compiler snapshot."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from .common import python_environment


class LocalAdapterError(RuntimeError):
    pass


def transform(source_path: Path, compiler_root: Path, python: Path,
              output_dir: Path, *, entry: str, context=None) -> dict:
    """Translate a normalized module; names and annotations carry no policy.

    Discovery owns capture analysis and array-bound normalization. This adapter
    invokes the selected compiler and namespaces its profiling symbols. It never
    rewrites the native application's declarations or arithmetic.
    """
    source_path = Path(source_path).resolve(strict=True)
    compiler_root = Path(compiler_root).resolve(strict=True)
    python = Path(python).absolute()
    output_dir = Path(output_dir).resolve()
    if not (compiler_root / "compiler/__main__.py").is_file():
        raise LocalAdapterError("Supply an explicit compiler snapshot containing compiler/__main__.py")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", entry):
        raise LocalAdapterError(f"Invalid generated entry name: {entry!r}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise LocalAdapterError(f"Configured compiler Python is not executable: {python}")
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = python_environment()
    environment["PYTHONPATH"] = str(compiler_root)
    command = [str(python), "-m", "compiler", "--input", str(source_path), "--kernel", entry,
               "--output-dir", str(output_dir), "--opt-level", "1", "--verbose",
               "--cpp-output", "region.cpp", "--fortran-output", "interface.f90",
               "--cuda-output", "region.cu", "--common-header", "runtime.hpp"]
    result = subprocess.run(command, cwd=compiler_root, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    (output_dir / "compiler.log").write_text(result.stdout)
    if result.returncode:
        raise LocalAdapterError(f"Compiler rejected {entry}: {result.stdout[-4000:]}")
    suffix = hashlib.sha256(entry.encode()).hexdigest()[:16]
    for filename in ("region.cpp", "region.cu", "interface.f90"):
        path = output_dir / filename
        if not path.is_file():
            raise LocalAdapterError(f"Compiler did not emit required file: {path}")
        text = path.read_text()
        for name in ("cpp_start_hot", "cpp_finish_hot"):
            text = re.sub(rf"\b{name}\b", f"{name}_{suffix}", text)
        # Enclosing native OpenMP loops already distribute these calls.
        if filename == "region.cpp" and (context or {}).get("nested_parallel"):
            text = "\n".join(line for line in text.splitlines()
                              if not re.match(r"\s*#\s*pragma\s+omp\b", line)) + "\n"
        path.write_text(text)
    metadata = {
        "tool": "local", "target": "cpp", "entry": entry, "changed": True,
        "context": context or {}, "opt_level": 1,
        "extra_sources": [str(output_dir / "interface.f90"), str(output_dir / "region.cpp")],
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "command": command,
        "policy": "compiler dependence analysis, scalar motion, loop fusion and locality scheduling",
    }
    (output_dir / "transformation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
