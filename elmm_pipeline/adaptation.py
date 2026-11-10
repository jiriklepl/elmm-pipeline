"""Apply generic discovery and tool passes while preserving unsupported native code."""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections import defaultdict
from fnmatch import fnmatchcase
from pathlib import Path

from .common import write_json
from .instrumentation import gpu_level_name, offload_contract, region_call


def check_generated_sources(sources, directory):
    """Reject unsupported generated syntax before changing any native source."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(sources):
        source = Path(source).resolve()
        if source.suffix.lower() in {".cpp", ".cc"}:
            command = [os.environ.get("CXX", "g++"), "-std=c++17", "-fopenmp", "-fsyntax-only", str(source)]
        else:
            command = [os.environ.get("FC", "gfortran"), "-fopenmp", "-ffree-line-length-none",
                       "-fsyntax-only", "-J" + str(directory), "-I" + str(directory), str(source)]
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True, check=False)
        (directory / f"syntax-{index}.log").write_text(result.stdout + result.stderr)
        if result.returncode:
            raise RuntimeError(f"Generated source rejected by native compiler ({source.name}): {result.stderr[-3000:]}")


def selected(region, include=(), exclude=(), marked_only=False):
    name = f"{region['module']}::{region['routine']}".lower()
    names = (name, region["routine"].lower(), f"{region['source']}::{name}".lower())
    matches = lambda patterns: any(fnmatchcase(n, p.lower()) for p in patterns for n in names)
    return (not include or matches(include)) and not matches(exclude) and (
        not marked_only or region.get("annotated", False))


def patch_regions(project_root, regions, *, coverage=True):
    """Validate every source span before publishing any changed native file."""
    grouped = defaultdict(list)
    for region in regions:
        grouped[region["source"]].append(region)
    patches = {}
    for relative, entries in grouped.items():
        path = (Path(project_root) / relative).resolve()
        if not path.is_relative_to(Path(project_root).resolve()):
            raise ValueError("Discovered source escapes the selected source tree")
        text = path.read_text()
        full_hash = hashlib.sha256(text.encode()).hexdigest()
        if any(region.get("source_sha256", full_hash) != full_hash for region in entries):
            raise ValueError(f"Source changed since discovery: {relative}")
        lines = text.splitlines(keepends=True)
        replacements = []
        imports = defaultdict(set)
        occupied = set()
        for region in entries:
            start, end = region["start_line"] - 1, region["end_line"]
            if start < 0 or end > len(lines) or end <= start:
                raise ValueError("Invalid discovered source span")
            if occupied.intersection(range(start, end)):
                raise ValueError("Overlapping discovered regions cannot both be outlined")
            occupied.update(range(start, end))
            original = "".join(lines[start:end])
            if region.get("original_text") is not None and original != region["original_text"]:
                raise ValueError(f"Source changed since discovery: {relative}:{start + 1}")
            expected = region.get("original_sha256")
            if expected and hashlib.sha256(original.encode()).hexdigest() != expected:
                raise ValueError(f"Source hash changed since discovery: {relative}:{start + 1}")
            if expected is None and region.get("original_text") is None:
                raise ValueError("Discovery must provide exact original source evidence")
            replacements.append((start, end, region_call(region, coverage=coverage)))
            # insertion_line is the one-based line before which USE is inserted.
            imported = [region["entry"]]
            query = offload_contract(region).get("native_fallback_query")
            if query:
                imported.append(query)
            imports[region["insertion_line"] - 1].add(
                f"  use {region['generated_module']}, only: {', '.join(imported)}\n")
            if region.get("gpu_scope"):
                imports[region["insertion_line"] - 1].add(
                    f"  use omp_lib, only: {gpu_level_name(region)} => omp_get_level\n")
        for at, statements in imports.items():
            if at in occupied or at < 0 or at > len(lines):
                raise ValueError("Invalid import insertion location")
            replacements.append((at, at, "".join(sorted(statements))))
        for start, end, value in sorted(replacements, reverse=True):
            lines[start:end] = [value]
        patches[path] = "".join(lines)
    for path, text in patches.items():
        path.write_text(text)
    return sorted(str(path.relative_to(Path(project_root).resolve())) for path in patches)


def adapt_project(tool, project_root, python, output_dir, *, compiler_root=None,
                  source_patterns=("**/*.f90",), include=(), exclude=(),
                  marked_only=False, defines=("DPREC",), coverage=True,
                  target="cpu", gpu_options=None, memory_model="scoped", scope_options=None):
    if tool != "local":
        raise ValueError(f"Unknown transformation tool: {tool}; expected local")
    if target not in {"cpu", "gpu"}:
        raise ValueError("Target must be cpu or gpu")
    if memory_model not in {"call", "scoped"}:
        raise ValueError("Memory model must be call or scoped")
    if memory_model == "scoped":
        if target != "gpu":
            raise ValueError("Scoped memory requires the local compiler and GPU target")
        if include or exclude or marked_only:
            raise ValueError("Scoped source inputs use explicit entry/source selection; call-region filters do not apply")
        from .scoped_adaptation import adapt_source_scopes
        return adapt_source_scopes(project_root, python, output_dir, compiler_root=compiler_root,
                                   options=scope_options or {}, defines=defines, gpu_options=gpu_options)
    if scope_options:
        raise ValueError("Scope options require memory_model='scoped'")
    raise ValueError("Ordinary-call adaptation was removed; use resident-build or checked source scopes")
