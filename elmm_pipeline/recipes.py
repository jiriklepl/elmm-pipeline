"""Run isolated source-transformation workers on normalized Fortran modules."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from .common import python_environment


def transform(tool: str, source_path: Path, python: Path, output_dir: Path, *, context: dict | None = None) -> dict:
    """Process any standalone module and record actual pass activity.

    Input is immutable. Unsupported or unchanged regions remain valid native
    Fortran and are reported explicitly. Merely parsing/printing a module is not
    counted as an optimization. Tool interpreters are always explicit.
    """
    if tool not in {"loki", "psyclone"}:
        raise ValueError(f"Unknown transformation tool {tool!r}; expected loki or psyclone")
    source_path = Path(source_path).resolve(strict=True)
    python = Path(python).absolute()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    original = source_path.read_bytes()
    (output_dir / "module.original.f90").write_bytes(original)
    generated = output_dir / "module.generated.f90"
    metadata_path = output_dir / "worker.json"
    worker = Path(__file__).with_name("tool_workers") / f"{tool}_worker.py"
    command = [str(python), str(worker), "--source", str(source_path), "--output", str(generated),
               "--metadata", str(metadata_path), "--context", json.dumps(context or {})]
    with (output_dir / "transform.log").open("w") as log:
        result = subprocess.run(command, cwd=output_dir, env=python_environment(), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        tail = (output_dir / "transform.log").read_text()[-5000:]
        raise RuntimeError(f"{tool} transformation failed; see {output_dir / 'transform.log'}\n{tail}")
    metadata = json.loads(metadata_path.read_text())
    transformed = generated.read_bytes()
    if metadata["changed"] and transformed == original:
        raise RuntimeError(f"{tool} reported transformations but emitted unchanged source")
    metadata.update({"source_sha256": hashlib.sha256(original).hexdigest(),
                     "generated_sha256": hashlib.sha256(transformed).hexdigest(),
                     "generated_path": str(generated), "command": command})
    (output_dir / "transformation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
