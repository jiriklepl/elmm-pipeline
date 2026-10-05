"""Apply generic discovery and tool passes while preserving unsupported native code."""
from __future__ import annotations

from collections import defaultdict
from fnmatch import fnmatchcase
import hashlib
import os
from pathlib import Path
import subprocess

from .common import write_json
from .instrumentation import region_call


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
        result = subprocess.run(command, cwd=directory, capture_output=True, text=True)
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
            imports[region["insertion_line"] - 1].add(
                f"  use {region['generated_module']}, only: {region['entry']}\n")
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
                  marked_only=False, defines=("DPREC",), coverage=True):
    from .discovery import discover_project

    project_root, output_dir = Path(project_root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    discovery = discover_project(project_root, python, output_dir / "discovery",
                                 patterns=source_patterns, defines=defines)
    accepted, rejected, extra_sources = [], list(discovery.get("rejected", [])), []
    for region in discovery["regions"]:
        if not selected(region, include, exclude, marked_only):
            rejected.append({"id": region["id"], "source": region["source"],
                             "routine": region["routine"], "reason": "excluded by selection hints"})
            continue
        target = output_dir / region["id"]
        try:
            if tool == "local":
                if region.get("context", {}).get("nested_parallel"):
                    uncertain = [symbol["name"] for symbol in region["symbols"]
                                 if symbol.get("rank", 0) and not symbol.get("contiguous_proven", False)]
                    if uncertain:
                        raise ValueError("Local contiguous-array interface inside native parallel context "
                                         "requires proved contiguous captures; possible copyback race: "
                                         + ", ".join(uncertain))
                from .local_adapter import transform
                metadata = transform(Path(region["input_path"]), compiler_root, python, target,
                                     entry=region["entry"], context=region.get("context"))
                sources = metadata["extra_sources"]
            else:
                from .recipes import transform
                metadata = transform(tool, Path(region["input_path"]), python, target,
                                     context=region.get("context"))
                sources = [metadata["generated_path"]]
            check_generated_sources(sources, target / "syntax-check")
            extra_sources.extend(sources)
            accepted.append({**region, "transformation": metadata})
        except (RuntimeError, ValueError) as error:
            rejected.append({"id": region["id"], "source": region["source"],
                             "routine": region["routine"], "reason": str(error),
                             "stage": tool})
    changed_files = patch_regions(project_root, accepted, coverage=coverage)
    metadata = {
        "tool": tool, "policy": "structural discovery, capture normalization, generic tool passes",
        "source_patterns": list(source_patterns), "selection": {"include": list(include),
        "exclude": list(exclude), "marked_only": marked_only}, "defines": list(defines),
        "coverage_enabled": coverage, "regions": accepted, "rejected": rejected,
        "discovered_regions": len(discovery["regions"]), "outlined_regions": len(accepted),
        "optimized_regions": sum(bool(r["transformation"].get("changed")) for r in accepted),
        "extra_sources": extra_sources, "changed_files": changed_files,
    }
    write_json(output_dir / "adaptation.json", metadata)
    return metadata
