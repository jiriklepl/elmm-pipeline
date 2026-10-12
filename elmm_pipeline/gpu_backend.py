"""Generate local CUDA backends without changing the GNU host Fortran ABI."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def transform(tool: str, source_path: Path, python: Path, output_dir: Path, *,
              compiler_root: Path | None = None, entry: str,
              context: dict | None = None, gpu_options: dict | None = None) -> dict:
    if tool != "local":
        raise ValueError(f"Unknown GPU tool: {tool}; expected local")
    source_path = Path(source_path).resolve(strict=True)
    python = Path(python).absolute()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from .local_adapter import transform as compile_local
    if compiler_root is None:
        raise ValueError("Local CUDA requires an explicit compiler root")
    # An outlined array result can leave halos untouched. Treat output
    # dummies as inout at the GPU boundary so the compiler's owned-device
    # allocation plan uploads those values before copying the array back.
    local_input = output_dir / "module.cuda-input.f90"
    local_input.write_text(re.sub(r"\bintent\s*\(\s*out\s*\)", "intent(inout)",
                                  source_path.read_text(), flags=re.IGNORECASE))
    metadata = compile_local(local_input, compiler_root, python, output_dir, entry=entry, context=context,
                             gpu_options=gpu_options)
    kernel_count = metadata["compiler_report"]["parallel_regions"]
    if kernel_count < 1:
        raise ValueError("Local CUDA compiler emitted no proven parallel regions")
    target, backend = "cuda", output_dir / "region.cu"
    metadata.update(target=target, kernel_regions=kernel_count, data_regions=1)
    proof = "CUDA checked kernel launches and FORT_RUNTIME_TRACE device events"
    passes = "compiler dependence analysis, CUDA kernels and one pooled workspace per region invocation"
    interface = output_dir / "interface.f90"
    units = [{"source": str(backend), "language": target, "tool": tool}]
    from .local_adapter import cuda_build_fields
    units[0].update(cuda_build_fields(metadata["compiler_report"], backend.name))
    metadata.update(tool=tool, entry=entry, changed=True, generated_path=str(interface),
                    extra_sources=[str(interface)], gpu_sources=units, compile_units=units,
                    source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    policy=passes, gpu_options=gpu_options or {})
    metadata["gpu"] = {"backend": target, "data_regions": metadata["data_regions"],
                       "kernel_regions": metadata["kernel_regions"],
                       "residency_scope": "one normalized entry invocation; all enclosed kernels share device allocations",
                       "written_array_policy": "copy in/out preserves unchanged halos",
                       "runtime_proof": proof, "proof_kernel_regions": 0,
                       "trace_environment": {"FORT_RUNTIME_TRACE": "1", "FORT_OFFLOAD_TRACE": "1"}}
    (output_dir / "transformation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
