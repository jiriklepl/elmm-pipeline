"""Launch-only policy for the packaged solver's bounded NVRTC image cache.

This module never loads CUDA or compiles kernels. Cache lookup, compilation on
miss, and module loading remain inside the timed application process.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import stat

from ..build_receipts import file_identity
from ..gpu_pressure_build import combine_pressure_backends, require_pressure_backend
from .identity import digest, read
from .prepare import check, slug

CACHE_ENVIRONMENT_VARIABLE = "POISSON_SOLVER_KERNEL_CACHE_DIR"
_CACHE_DIRECTORY = ".poisson-kernel-cache"
_IDENTITY_FILE = "kernel-cache-identity.json"
_RECIPE_FILE = "kernel-cache-build-recipe.txt"


def _requests(adapter):
    if not isinstance(adapter, dict):
        raise ValueError("pressure cache requires a prepared adapter object")
    requests = [adapter.get("pressure_backend")]
    children = adapter.get("scope_packages", [])
    if not isinstance(children, list):
        raise ValueError("pressure cache requires a prepared package list")
    for child in children:
        requests.extend(_requests(child))
    return requests


def _installed_file(path, prefix, known):
    for part in (path, *path.parents):
        if part == prefix:
            break
        if part.is_symlink():
            raise ValueError("pressure cache identity must not use symlinks")
    try:
        identity = file_identity(path)
    except OSError as error:
        raise ValueError("pressure cache installed identity or recipe is missing") from error
    if known.get(identity["path"]) != identity["sha256"]:
        raise ValueError("pressure cache identity and recipe must be exact backend inputs")
    return identity


def _identity(request):
    # The compiler path is read from the hash-bound backend rather than an
    # inherited FC setting. The public validator still verifies its exact bytes
    # and every other backend field; no alternate manifest schema is accepted.
    manifest = Path(request["manifest"])
    if digest(manifest) != request["sha256"]:
        raise ValueError("Pressure backend manifest changed")
    serialized = read(manifest)
    try:
        compiler = serialized["fortran_compiler"]["path"]
    except (KeyError, TypeError) as error:
        raise ValueError("pressure cache requires the backend's exact compiler") from error
    backend = require_pressure_backend(request, fortran=compiler)
    library = Path(backend["provider_library"]["path"])
    prefix = library.parent.parent
    path = prefix / "share/PoissonSolver" / _IDENTITY_FILE
    known = {entry["path"]: entry["sha256"] for entry in backend["inputs"]}
    declared = [Path(name) for name in known if Path(name).name == _IDENTITY_FILE]
    if not path.exists() and not path.is_symlink() and not declared:
        return None
    if library.parent.name not in {"lib", "lib64"} or declared != [path]:
        raise ValueError("pressure cache installed identity is missing or unbound")
    identity = _installed_file(path, prefix, known)
    try:
        value = json.loads(path.read_text())
    except (ValueError, OSError) as error:
        raise ValueError("invalid pressure cache identity JSON") from error
    expected = {"schema_version": 1, "kind": "poisson-nvrtc-cubin-cache",
                "cache_format": "POISSON_CUBIN_1", "environment_variable": CACHE_ENVIRONMENT_VARIABLE,
                "max_image_bytes": 16 << 20, "max_cache_bytes": 64 << 20,
                "recipe_file": _RECIPE_FILE}
    if (not isinstance(value, dict) or set(value) != {*expected, "build_key", "nvrtc_builtins"}
            or any(value.get(k) != v or type(value[k]) is not type(v) for k, v in expected.items())
            or not isinstance(value.get("build_key"), str)
            or re.fullmatch(r"[0-9a-f]{64}", value["build_key"]) is None):
        raise ValueError("invalid pressure cache identity schema")
    builtins = value["nvrtc_builtins"]
    if (not isinstance(builtins, dict) or set(builtins) != {"path", "sha256"}
            or not isinstance(builtins["path"], str) or "\0" in builtins["path"]
            or not Path(builtins["path"]).is_absolute()
            or not isinstance(builtins["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", builtins["sha256"]) is None):
        raise ValueError("invalid pressure cache builtins identity")
    recipe = _installed_file(path.with_name(_RECIPE_FILE), prefix, known)
    if recipe["sha256"] != value["build_key"]:
        raise ValueError("pressure cache build key differs from its bound recipe")
    return {"identity": identity, "recipe": recipe, **value}


def _mutable_path(root, state, mode, request):
    base = root / _CACHE_DIRECTORY
    protected = [root / name for name in state["files"]]
    protected += [Path(name) for name in state["external_files"]]
    protected += [root / case["config"] for case in state["cases"]]
    for path in protected:
        path = Path(os.path.abspath(path))
        if path.is_relative_to(base) or base.is_relative_to(path):
            raise ValueError("pressure cache directory overlaps frozen campaign inputs")
    return base / slug(mode["name"]) / request["sha256"]


def _private_directory(root, destination):
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(root, flags)
    except OSError as error:
        raise ValueError("pressure cache campaign root must be a directory without symlinks") from error
    existed = True
    try:
        info = os.fstat(descriptor)
        if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) & 0o022:
            raise ValueError("pressure cache campaign root must be owned and not writable by others")
        for name in destination.relative_to(root).parts:
            try:
                os.mkdir(name, 0o700, dir_fd=descriptor)
                existed = False
            except FileExistsError:
                pass
            child = os.open(name, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            info = os.fstat(descriptor)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                raise ValueError("pressure cache directories must be owner-only mode0700")
    except OSError as error:
        raise ValueError("pressure cache path must contain safe directories without symlinks") from error
    finally:
        os.close(descriptor)
    return existed


def apply_pressure_cache(root, state, mode, environment):
    """Set one authenticated persistent cache path, or explicitly disable it."""
    environment.pop(CACHE_ENVIRONMENT_VARIABLE, None)
    record = {"schema_version": 1, "policy": "disabled", "reason": "no_pressure_backend",
              "environment_variable": CACHE_ENVIRONMENT_VARIABLE, "path": None,
              "precompiled_outside_application": False}
    if "adapter" not in mode:
        return record
    root = Path(os.path.abspath(root))
    request = combine_pressure_backends(_requests(read(root / mode["adapter"])))
    if request is None:
        return record
    # A mutable cache must never become an untracked modification inside a
    # frozen source/configuration tree. Recheck the campaign before choosing it.
    if check(root) != state:
        raise ValueError("pressure cache campaign state differs from its frozen manifest")
    selected = [item for item in state["modes"] if item["name"] == mode["name"]]
    if len(selected) != 1 or selected[0].get("adapter") != mode["adapter"]:
        raise ValueError("pressure cache mode differs from the frozen adapter selection")
    adapter_path = root / mode["adapter"]
    guarded = {str(root / name): value for name, value in state["files"].items()}
    guarded.update(state["external_files"])
    if guarded.get(str(adapter_path)) != digest(adapter_path):
        raise ValueError("pressure cache adapter is not a frozen campaign input")
    identity = _identity(request)
    record["backend"] = request
    if identity is None:
        record["reason"] = "legacy_backend_without_cache_identity"
        return record
    destination = _mutable_path(root, state, mode, request)
    reused = _private_directory(root, destination)
    environment[CACHE_ENVIRONMENT_VARIABLE] = str(destination)
    return {**record, "policy": "persistent_campaign_mode_backend", "reason": "bound_packaged_identity",
            "path": str(destination), "directory_reused": reused, "build_key": identity["build_key"],
            "identity": identity["identity"], "recipe": identity["recipe"],
            "cache_format": identity["cache_format"], "max_image_bytes": identity["max_image_bytes"],
            "max_cache_bytes": identity["max_cache_bytes"],
            "timing": "warmup and measurements include all application startup, cache lookup, compilation on miss and module loading"}
