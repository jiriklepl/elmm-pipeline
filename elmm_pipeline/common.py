import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Compiler generation, verification and diagnostic relinks share this phase
# cap. Application processes retain their explicitly supplied runtime limits.
COMPILER_PHASE_TIMEOUT_SECONDS = 240


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest_tree(path):
    """Hash contents and names; ignore generated and VCS state."""
    path = Path(path)
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if any(part in {".git", "__pycache__", ".pytest_cache"} for part in p.relative_to(path).parts):
            continue
        if p.is_file() and p.suffix != ".pyc":
            h.update(str(p.relative_to(path)).encode())
            h.update(b"\0")
            h.update(p.read_bytes())
    return h.hexdigest()


def command(args, *, cwd=None, env=None, log=None, timeout=None):
    args = [str(a) for a in args]
    started = time.perf_counter()
    if log:
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open("w") as stream:
            result = subprocess.run(args, cwd=cwd, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
        output = Path(log).read_text(errors="replace")
    else:
        result = subprocess.run(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
        output = result.stdout
    elapsed = time.perf_counter() - started
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {args}\n{output[-6000:]}")
    return output, elapsed


def clean_environment(threads):
    env = dict(os.environ)
    # ELMM environment overrides must not silently replace the selected inputs.
    for key in list(env):
        if key.startswith("ELMM_"):
            del env[key]
    env.update(OMP_NUM_THREADS=str(threads), OMP_DYNAMIC="FALSE",
               OMP_SCHEDULE="static", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return env


def python_environment():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_NO_CACHE_DIR"] = "1"
    return env
