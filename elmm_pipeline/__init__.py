"""ELMM adapter over the Fortran-to-CUDA compiler.

The adapter imports the compiler as a library. The compiler checkout is the
one already importable, else ``ELMM_COMPILER_ROOT``, else the repository that
holds this directory as its ``elmm-pipeline`` submodule.
"""
import importlib.util
import os
import sys
from pathlib import Path


def use_compiler(root=None):
    """Put a compiler checkout on the import path and return its root."""
    explicit = root or os.environ.get("ELMM_COMPILER_ROOT")
    candidate = Path(explicit).resolve() if explicit else Path(__file__).resolve().parents[2]
    if not (candidate / "compiler/__main__.py").is_file():
        raise ValueError("Expected a compiler checkout containing compiler/__main__.py: " + str(candidate)
                         + ("" if explicit else "; set ELMM_COMPILER_ROOT or pass --compiler"))
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    return candidate


if importlib.util.find_spec("compiler") is None:
    use_compiler()
