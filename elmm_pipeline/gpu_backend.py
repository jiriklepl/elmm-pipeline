"""Reject the retired ordinary-call CUDA adapter before changing files.

Current application integration uses checked source graphs or resident-build.
Those callers supply canonical handles and own storage/publication boundaries.
"""
from pathlib import Path


def transform(tool: str, source_path: Path, python: Path, output_dir: Path, *,
              compiler_root: Path | None = None, entry: str,
              context: dict | None = None, gpu_options: dict | None = None) -> dict:
    if tool != "local":
        raise ValueError(f"Unknown GPU tool: {tool}; expected local")
    raise ValueError("Ordinary-call CUDA integration was removed; use resident-build or the source-graph adapter")
