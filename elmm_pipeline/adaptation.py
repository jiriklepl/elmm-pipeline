"""Apply generic discovery and tool passes while preserving unsupported native code."""
from __future__ import annotations




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
