"""Discover and compile CUDA objects for a GNU Fortran host program.

GPU entry points must use the C ABI (BIND(C), explicit-shape or assumed-size
arrays). Generated CUDA objects link into the original GNU host build.
Compiler installations are discovered on PATH or in normal system locations;
no sibling checkout or globally installed Python package is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .build_receipts import CommandRecorder

try:
    from typing import Literal, NotRequired, TypedDict  # type: ignore
except ImportError:
    try:
        from typing_extensions import Literal, NotRequired, TypedDict  # type: ignore
    except ImportError:
        from typing import Literal, TypedDict  # type: ignore

        class _NotRequired:
            def __getitem__(self, item):
                return item

        NotRequired = _NotRequired()  # type: ignore


class GPUSource(TypedDict):
    source: str | Path
    language: Literal["cuda"]
    tool: str
    numerical_contract: NotRequired[dict]


def validate_numerical_contract(contract: object) -> dict:
    """Validate a compiler-owned contract without interpreting numerical code."""
    keys = {"schema_version", "id", "identity", "required_cuda_options", "required_host_options"}
    if (not isinstance(contract, dict) or set(contract) != keys
            or type(contract.get("schema_version")) is not int or contract["schema_version"] != 1
            or contract.get("id") != "separate-arithmetic-v1"
            or contract.get("required_cuda_options") != [
                "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
            or contract.get("required_host_options") != ["-ffp-contract=off"]):
        raise ValueError("Unknown or conflicting required numerical build contract")
    payload = {key: value for key, value in contract.items() if key != "identity"}
    identity = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if contract["identity"] != identity:
        raise ValueError("Numerical build contract identity differs from its requirements")
    return {**contract, "required_cuda_options": list(contract["required_cuda_options"]),
            "required_host_options": list(contract["required_host_options"])}


def _executable(explicit: str | Path | None, names: list[str], candidates: list[Path]) -> str:
    choices = [str(explicit)] if explicit else [*names, *(str(path) for path in candidates)]
    for choice in choices:
        located = shutil.which(choice)
        if located:
            return str(Path(located).resolve())
    label = str(explicit) if explicit else " / ".join(names)
    raise RuntimeError(f"GPU compiler {label!r} was not found; install it or supply its explicit path.")


def _version(executable: str) -> str:
    result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Could not query GPU compiler {executable}: {result.stderr.strip()}")
    return (result.stdout + result.stderr).strip()


def probe_gpu(nvidia_smi: str | Path | None = None) -> dict:
    """Report visible NVIDIA devices without treating sandbox denial as absence.

    A failed probe records its diagnostic. Compilation is still possible with an
    explicit architecture; a successful query alone is not execution evidence.
    """
    command = shutil.which(str(nvidia_smi or "nvidia-smi"))
    if command is None:
        return {"available": False, "devices": [], "error": "nvidia-smi was not found"}
    argv = [command, "--query-gpu=name,driver_version,compute_cap", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(argv, text=True, capture_output=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "devices": [], "error": str(error), "argv": argv}
    devices = []
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 3:
                devices.append(dict(zip(("name", "driver_version", "compute_capability"), fields)))
    return {"available": bool(devices), "devices": devices, "argv": argv,
            "error": (result.stderr or result.stdout).strip() if not devices else None}


def _architecture(value: object) -> str:
    architecture = str(value).removeprefix("sm_").removeprefix("cc").replace(".", "")
    if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", architecture):
        raise ValueError(f"Invalid NVIDIA GPU architecture {value!r}; use a compute capability such as 86.")
    return architecture


def _cuda_runtime_directory(cuda_root: Path, compiler: str, explicit: str | Path | None) -> tuple[Path, dict]:
    if explicit is not None:
        directory = Path(explicit).resolve()
        if not (directory / "libcudart.so").is_file():
            raise RuntimeError("Explicit CUDA runtime directory has no libcudart.so: " + str(directory))
        return directory, {"method": "explicit", "directory": str(directory)}
    argv = [compiler, "-dumpmachine"]
    try:
        result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("Could not query CUDA host compiler target; supply cuda_runtime_dir explicitly") from error
    target = result.stdout.strip()
    if (result.returncode or len(target) > 128
            or not re.fullmatch(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_.]+)+", target)):
        raise RuntimeError("Invalid CUDA host compiler target; supply cuda_runtime_dir explicitly")
    parts = target.lower().split("-")
    # Red Hat's native GCC reports e.g. x86_64-redhat-linux, omitting the
    # GNU ABI suffix. It uses the same CUDA target as x86_64-linux-gnu.
    # Keep explicit x32/musl (and unknown ABI) targets out of this alias.
    linux_gnu = ("linux" in parts and parts[-1] == "gnu") or (
        len(parts) == 3 and parts[1:] == ["redhat", "linux"])
    targets = cuda_root / "targets"
    candidates = [targets / target / "lib"]
    # Architecture-only CUDA directories do not distinguish x32/musl ABIs.
    if linux_gnu:
        candidates.append(targets / (parts[0] + "-linux") / "lib")
    for directory in candidates:
        if (directory / "libcudart.so").is_file():
            return directory.resolve(), {"method": "host_target", "target": target, "argv": argv}
    # Older native toolkits have only lib64/lib. A cross compiler must not
    # silently pick the running host's runtime, nor a symlink to another target.
    if linux_gnu and parts[0] == platform.machine().lower() and platform.system() == "Linux":
        for directory in (cuda_root / "lib64", cuda_root / "lib"):
            resolved = directory.resolve()
            if not resolved.is_relative_to(targets.resolve()) and (resolved / "libcudart.so").is_file():
                return resolved, {"method": "native_flat_layout", "target": target, "argv": argv}
    raise RuntimeError(f"CUDA runtime libcudart.so for host target {target!r} was not found beneath "
                       f"{cuda_root}; supply cuda_runtime_dir explicitly")


def discover_toolchain(
    options: dict | None = None,
    languages: tuple[str, ...] | list[str] = ("cuda",),
    *,
    nvcc: str | Path | None = None,
    cuda_host: str | Path | None = None,
    arch: str | None = None,
    cuda_runtime_dir: str | Path | None = None,
) -> dict:
    """Return JSON-serializable paths, versions, architecture and device status.

    CUDA defaults to g++-14 when installed, because a newer system GNU compiler
    may exceed nvcc's supported host range. Supply ``cuda_host`` to override it.
    If device visibility is restricted, callers must provide ``arch`` explicitly.
    ``cuda_runtime_dir`` (also accepted in options) selects a custom runtime
    directory; otherwise the selected CUDA host compiler determines the target.
    """
    options = dict(options or {})
    unsupported = set(languages) - {"cuda"}
    if unsupported:
        raise ValueError(f"Unsupported GPU languages: {sorted(unsupported)}")
    device = probe_gpu(options.get("nvidia_smi"))
    architecture = arch or options.get("architecture") or options.get("arch")
    if architecture is None and device["devices"]:
        architecture = device["devices"][0]["compute_capability"]
    if architecture is None:
        raise RuntimeError("Cannot determine the GPU architecture from nvidia-smi; supply an explicit architecture (for example 86). "
                           + str(device["error"]))
    config: dict = {"architecture": _architecture(architecture), "device": device, "versions": {}}
    if "cuda" in languages:
        config["nvcc"] = _executable(nvcc or options.get("nvcc") or os.environ.get("NVCC"),
                                      ["nvcc"], [Path("/usr/local/cuda/bin/nvcc")])
        config["cuda_host_cxx"] = _executable(
            cuda_host or options.get("cuda_host_cxx") or options.get("cuda_host") or os.environ.get("CUDAHOSTCXX"),
            ["g++-14", "g++"], [])
        config["versions"]["nvcc"] = _version(config["nvcc"])
        config["versions"]["cuda_host_cxx"] = _version(config["cuda_host_cxx"])
        cuda_root = Path(config["nvcc"]).parent.parent
        override = cuda_runtime_dir if cuda_runtime_dir is not None else options.get(
            "cuda_runtime_dir", options.get("cuda_library_dir"))
        runtime_dir, discovery = _cuda_runtime_directory(cuda_root, config["cuda_host_cxx"], override)
        config["cuda_library_dir"] = str(runtime_dir)
        config["cuda_runtime_discovery"] = discovery
    return config


def validate_gpu_sources(sources: list[GPUSource] | list[dict]) -> list[dict]:
    normalized = []
    seen = set()
    for entry in sources:
        if not isinstance(entry, dict) or "source" not in entry or "language" not in entry:
            raise ValueError("GPU sources must be descriptors containing source and language")
        if "exact_window_build_contract" in entry:
            raise ValueError("Exact-window build contracts require complete CUDA and Fortran build attestation; "
                             "this application build consumer does not support them")
        source = Path(entry["source"]).resolve()
        language = entry["language"]
        suffixes = {"cuda": {".cu"}}
        if language not in suffixes or source.suffix.lower() not in suffixes[language] or not source.is_file():
            raise ValueError(f"Missing or unsupported GPU source ({language}): {source}")
        if source in seen:
            raise ValueError(f"Duplicate GPU source: {source}")
        seen.add(source)
        unit = {"source": str(source), "language": language, "tool": str(entry.get("tool", language)),
                "numerical_contract_status": "legacy_unspecified"}
        if "numerical_contract" in entry:
            if language != "cuda":
                raise ValueError("Numerical CUDA build contracts require a CUDA source")
            unit["numerical_contract"] = validate_numerical_contract(entry["numerical_contract"])
            unit["numerical_contract_status"] = "validated"
        normalized.append(unit)
    return normalized


def compile_gpu_sources(
    sources: list[GPUSource] | list[dict],
    build_dir: Path,
    options: dict | None = None,
    *,
    defines: list[str] | None = None,
    toolchain: dict | None = None,
    jobs: int = 1,
    application_macros: dict[str, str] | None = None,
) -> dict:
    """Compile self-contained GPU objects and return GNU linker inputs.

    Compiler diagnostics and exact commands are saved beside the objects.
    Independent CUDA units use at most ``jobs`` workers.
    """
    if type(jobs) is not int or jobs < 1:
        raise ValueError("jobs must be a positive integer")
    units = validate_gpu_sources(sources)
    if not units:
        return {"objects": [], "library_dirs": [], "libraries": [], "rpaths": [], "link_flags": [], "sources": []}
    compiler_environment = {name: os.environ.get(name) for name in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")}
    if any("numerical_contract" in unit for unit in units) and any(compiler_environment.values()):
        raise ValueError("Contracted CUDA builds reject hidden NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS overrides")
    languages = tuple(dict.fromkeys(unit["language"] for unit in units))
    config = toolchain or discover_toolchain(options, languages=languages)
    directory = build_dir.resolve() / "gpu"
    directory.mkdir(parents=True, exist_ok=True)
    receipt_directory = Path(tempfile.mkdtemp(prefix="command-receipts-", dir=directory))
    recorder = CommandRecorder(receipt_directory)
    commands = []
    objects = []
    manifest: dict = {"toolchain": config, "sources": units, "commands": commands, "objects": objects,
                      "jobs": jobs,
                      "command_receipts": str(receipt_directory),
                      "compiler_environment": compiler_environment,
                      "library_dirs": [], "libraries": [], "rpaths": [], "link_flags": []}
    log_path = directory / "gpu-build.log"
    plans = []
    macros = ["-D" + macro for macro in (defines or [])]
    for index, unit in enumerate(units):
        source = Path(unit["source"])
        target = directory / f"{index:03d}_{source.stem}.o"
        unit_log = directory / f"{index:03d}_{source.stem}.log"
        contract = unit.get("numerical_contract")
        numerical_options = (contract or {}).get("required_cuda_options", [])
        host_options = (contract or {}).get("required_host_options", [])
        command = [config["nvcc"], "-O3", "-std=c++17", "-arch=sm_" + config["architecture"],
                   "-ccbin", config["cuda_host_cxx"], "-Xcompiler=-fPIC,-fopenmp", *macros,
                   *numerical_options, *["-Xcompiler=" + option for option in host_options],
                   "-c", str(source), "-o", str(target)]
        if str(source) in (application_macros or {}):
            macro = application_macros[str(source)]
            if not re.fullmatch(r"FORT_EXACT_WINDOW_BUILD_[0-9a-f]{64}=1", macro):
                raise ValueError("Invalid verified application selection build macro")
            command.insert(-4, "-D" + macro)
        unit["resolved_numerical_options"] = {"cuda": numerical_options, "host": host_options}
        unit["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        unit["build_log"] = str(unit_log)
        commands.append(command)
        plans.append((index, command, target, unit_log))

    def compile_one(plan):
        _, command, target, unit_log = plan
        try:
            with unit_log.open("w", encoding="utf-8") as log:
                log.write(shlex.join(command) + "\n")
                log.flush()
                result = recorder.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode or not target.is_file():
                    return "failed", f"compiler exit {result.returncode}; object exists: {target.is_file()}"
        except Exception as error:  # noqa: BLE001 - retain worker failures after joining the entire batch
            return "failed", f"{type(error).__name__}: {error}"
        return "compiled", None

    outcomes = {}
    if jobs == 1:
        for plan in plans:
            outcomes[plan[0]] = compile_one(plan)
            if outcomes[plan[0]][0] == "failed":
                break
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(plans))) as executor:
            futures = {executor.submit(compile_one, plan): plan[0] for plan in plans}
            for future in as_completed(futures):
                position = futures[future]
                if future.cancelled():
                    outcomes[position] = ("cancelled", None)
                else:
                    outcomes[position] = future.result()
                    if outcomes[position][0] == "failed":
                        for pending in futures:
                            pending.cancel()

    # Workers have all joined. Only this parent assembles the public inventory
    # and aggregate diagnostics, in the original source order.
    with log_path.open("w", encoding="utf-8") as log:
        for position, command, target, unit_log in plans:
            status, error = outcomes.get(position, ("not_started", None))
            units[position]["build_status"] = status
            if error is not None:
                units[position]["build_error"] = error
            if status == "compiled":
                objects.append(str(target))
            if position in outcomes and status != "cancelled" and unit_log.is_file():
                with unit_log.open(encoding="utf-8", errors="replace") as diagnostics:
                    shutil.copyfileobj(diagnostics, log)
                if error is not None:
                    log.write(error + "\n")
            else:
                log.write(shlex.join(command) + f"\n{status}\n")
                if error is not None:
                    log.write(error + "\n")
    if any(unit["build_status"] != "compiled" for unit in units):
        (directory / "gpu-build-command.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"GPU compilation failed; inspect {log_path}")
    if "cuda" in languages:
        manifest["library_dirs"].append(config["cuda_library_dir"])
        manifest["rpaths"].append(config["cuda_library_dir"])
        manifest["libraries"].extend(["cudart", "stdc++"])
        cuda_host_cxx = config.get("cuda_host_cxx")
        if cuda_host_cxx:
            try:
                stdcxx_link = subprocess.check_output(
                    [cuda_host_cxx, "-print-file-name=libstdc++.so"],
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
                if os.path.isabs(stdcxx_link) and os.path.isfile(stdcxx_link):
                    stdcxx_real = os.path.realpath(stdcxx_link)
                    manifest["cuda_stdcxx_library"] = stdcxx_real
                    manifest["cuda_stdcxx_link_dir"] = os.path.dirname(stdcxx_link)
                    for d in (os.path.dirname(stdcxx_real), os.path.dirname(stdcxx_link)):
                        if d not in manifest["library_dirs"]:
                            manifest["library_dirs"].insert(0, d)
                        if d not in manifest["rpaths"]:
                            manifest["rpaths"].insert(0, d)
            except Exception:
                pass
    for key in ("objects", "library_dirs", "libraries", "rpaths", "link_flags"):
        manifest[key] = list(dict.fromkeys(manifest[key]))
    (directory / "gpu-build-command.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
