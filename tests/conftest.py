"""Shared pytest setup for the pipeline test suite.

Tests exercise the pipeline against a compiler checkout. Use the explicit
``ELMM_COMPILER_ROOT`` when set; otherwise use the enclosing repository when
this checkout is its ``elmm-pipeline`` submodule.
"""

import os
import sys
from pathlib import Path

PIPELINE = Path(__file__).resolve().parents[1]
_compiler_root = Path(os.environ.get("ELMM_COMPILER_ROOT", PIPELINE.parent)).resolve()

for _path in (_compiler_root if (_compiler_root / "compiler" / "__main__.py").is_file() else None, PIPELINE):
    if _path is not None and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
