"""Opt-in source scopes through capture extraction and public compiler artifacts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .common import python_environment, write_json
from .local_adapter import (
    LocalAdapterError,
    apply_scope_sources,
    remap_scope_facts,
    transform_scopes,
)


def scope_inputs(project_root, options):
    """Resolve an explicit bounded source selection inside the application copy."""
    root = Path(project_root).resolve(strict=True)
    if not options.get("entry") or not options.get("source") or options.get("capture_facts") is None:
        raise ValueError("Scoped execution requires entry, source and explicit capture facts")
    sources = []
    for value in [options["source"], *options.get("source_files", ())]:
        path = Path(value)
        path = (root / path).resolve(strict=True) if not path.is_absolute() else path.resolve(strict=True)
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Scope source inputs must stay inside the selected application source tree")
        if path not in sources:
            sources.append(path)
    return root, sources


def adapt_source_scopes(project_root, python, output_dir, *, compiler_root, options,
                        defines=(), gpu_options=None):
    """Extract available leaves; the compiler forms scopes and owns placement.

    Initialization, lifetime and participation facts are supplied explicitly.
    They are never inferred from array intent or application procedure names.
    """
    root, sources = scope_inputs(project_root, options)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    facts = options["capture_facts"]
    if isinstance(facts, (str, Path)):
        facts = json.loads(Path(facts).read_text())
    if not isinstance(facts, dict):
        raise LocalAdapterError("Scope capture facts must be a JSON object")
    facts = remap_scope_facts(facts, root)
    request = {"project_root": str(root), "source_files": [str(path) for path in sources],
               "entry": options["entry"], "defines": list(defines), "output_dir": str(output / "extraction")}
    request_path = output / "scope-extraction-request.json"
    write_json(request_path, request)
    worker = Path(__file__).with_name("tool_workers") / "scope_worker.py"
    command = [str(Path(python).absolute()), str(worker), "--request", str(request_path)]
    log_path = output / "scope-extraction.log"
    with log_path.open("w") as log:
        result = subprocess.run(command, cwd=output, env=python_environment(),
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"Scope capture extraction failed; see {log_path}\n" + log_path.read_text()[-4000:])
    extraction = json.loads((output / "extraction" / "scope-extraction.json").read_text())
    extraction["command"] = command
    metadata = transform_scopes(sources[0], compiler_root, python, output / "compiler",
                                entry=options["entry"], source_files=sources[1:], capture_facts=facts,
                                gpu_options=gpu_options, effect_contracts=options.get("effect_contracts"),
                                analysis_sources=Path(extraction["analysis_sources"]),
                                numerical_sources=Path(extraction["numerical_sources"]))
    manifest = metadata["scope_manifest"]
    # Application sources own their native module dependencies. Their generated
    # replacements are checked with the complete build, not as isolated modules.
    gpu_check = None
    if metadata["gpu_sources"]:
        from .gpu_toolchain import compile_gpu_sources
        gpu_check = compile_gpu_sources(metadata["gpu_sources"], output / "syntax-check",
                                        options=gpu_options, defines=list(defines),
                                        toolchain=gpu_options if (gpu_options or {}).get("versions") else None)
    changed = apply_scope_sources(metadata, {str(path): path for path in sources})
    regions = [{"id": scope["owner"], "entry": scope["owner"],
                "routine": options["entry"], "source": str(Path(scope["path"]).relative_to(root)),
                "start_line": scope["first_line"], "end_line": scope["last_line"],
                "loop_nests": len(scope["gpu_leaves"]), "transformation": metadata}
               for scope in manifest["scopes"]]
    adaptation = {"tool": "local", "target": "gpu", "memory_model": "scoped",
                  "policy": "compiler-owned source scopes, effects, coherence and execution",
                  "scope_entry": options["entry"], "scope_manifest": manifest,
                  "scope_transformation": metadata, "scope_extraction": extraction,
                  "capture_facts_sha256": manifest["capture_facts_sha256"],
                  "coverage_enabled": False,
                  "coverage_note": "Compiler source scopes do not emit adapter call markers; call coverage is unverified.",
                  "regions": regions, "rejected": manifest.get("boundaries", []),
                  "discovered_regions": manifest["scope_count"], "outlined_regions": len(regions),
                  "optimized_regions": len(regions), "extra_sources": metadata["extra_sources"],
                  "gpu_sources": metadata["gpu_sources"], "gpu_compile_check": gpu_check,
                  "changed_files": [str(path.relative_to(root)) for path in changed],
                  "native_success": not regions, "defines": list(defines),
                  "source_patterns": [str(path.relative_to(root)) for path in sources]}
    write_json(output / "adaptation.json", adaptation)
    return adaptation
