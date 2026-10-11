"""Discover and compile CUDA/OpenACC objects for a GNU Fortran host program.

GPU entry points must use the C ABI (BIND(C), explicit-shape or assumed-size
arrays). NVIDIA Fortran module files and descriptors never enter the GNU build.
Compiler installations are discovered on PATH or in normal system locations;
no sibling checkout or globally installed Python package is used.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
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
    language: Literal["cuda", "openacc"]
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


def discover_toolchain(
    options: dict | None = None,
    languages: tuple[str, ...] | list[str] = ("cuda", "openacc"),
    *,
    nvcc: str | Path | None = None,
    nvfortran: str | Path | None = None,
    cuda_host: str | Path | None = None,
    arch: str | None = None,
) -> dict:
    """Return JSON-serializable paths, versions, architecture and device status.

    CUDA defaults to g++-14 when installed, because a newer system GNU compiler
    may exceed nvcc's supported host range. Supply ``cuda_host`` to override it.
    If device visibility is restricted, callers must provide ``arch`` explicitly.
    """
    options = dict(options or {})
    unsupported = set(languages) - {"cuda", "openacc"}
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
        runtime_dirs = [cuda_root / "lib64", cuda_root / "targets" / "x86_64-linux" / "lib", cuda_root / "lib"]
        runtime_dir = next((path for path in runtime_dirs if (path / "libcudart.so").is_file()), None)
        if runtime_dir is None:
            raise RuntimeError(f"CUDA runtime libcudart.so was not found beneath {cuda_root}")
        config["cuda_library_dir"] = str(runtime_dir.resolve())
    if "openacc" in languages:
        installations = sorted(Path("/opt/nvidia/hpc_sdk").glob("*/*/compilers/bin/nvfortran"),
                               key=lambda path: tuple(int(number) for number in re.findall(r"\d+", path.parts[-4])), reverse=True)
        config["nvfortran"] = _executable(nvfortran or options.get("nvfortran") or os.environ.get("NVFORTRAN"),
                                           ["nvfortran"], installations)
        config["versions"]["nvfortran"] = _version(config["nvfortran"])
        config["nvfortran_library_dir"] = str(Path(config["nvfortran"]).parent.parent / "lib")
    return config


def validate_gpu_sources(sources: list[GPUSource] | list[dict]) -> list[dict]:
    normalized = []
    seen = set()
    for entry in sources:
        if not isinstance(entry, dict) or "source" not in entry or "language" not in entry:
            raise ValueError("GPU sources must be descriptors containing source and language")
        source = Path(entry["source"]).resolve()
        language = entry["language"]
        suffixes = {"cuda": {".cu"}, "openacc": {".f90", ".f"}}
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


def _openacc_link_configuration(config: dict, object_path: Path, directory: Path) -> dict:
    """Read the installed NVIDIA driver's link recipe, excluding its main/CRT.

    ``nordc`` embeds complete device images in each object, avoiding a second
    compiler's device-link step. Compiler-owned initialization objects remain
    necessary to select the GPU runtime. The final link is always GNU Fortran.
    """
    command = [config["nvfortran"], "-dryrun", "-Mnomain", "-acc=gpu",
               f"-gpu=cc{config['architecture']},nordc", str(object_path), "-o", str(directory / "link-probe")]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    output = result.stdout + result.stderr
    (directory / "nvfortran-link-dryrun.log").write_text(output, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"NVFortran runtime discovery failed; inspect {directory / 'nvfortran-link-dryrun.log'}")
    lines = [shlex.split(line) for line in output.splitlines() if " -lacchost" in line]
    if len(lines) != 1:
        raise RuntimeError("NVFortran did not report an unambiguous OpenACC runtime link recipe")
    tokens = lines[0]
    compiler_lib = Path(config["nvfortran_library_dir"]).resolve()
    library_dirs = []
    libraries = []
    objects = []
    rpaths = []
    link_flags = []
    for index, token in enumerate(tokens):
        if token.startswith("-L") and len(token) > 2:
            library_dirs.append(token[2:])
        elif token.startswith("-l") and len(token) > 2 and token not in {"-lgcc", "-lgcc_s", "-lc"}:
            libraries.append(token[2:])
        elif token == "-rpath" and index + 1 < len(tokens):
            rpaths.append(tokens[index + 1])
        elif token.endswith(".o") and Path(token).is_absolute() and Path(token).parent.resolve() == compiler_lib:
            objects.append(token)
        elif token == "-T" and index + 1 < len(tokens):
            link_flags.append("-Wl,-T," + tokens[index + 1])
        elif token.startswith("--dynamic-list="):
            link_flags.append("-Wl," + token)
    if not objects or "acchost" not in libraries:
        raise RuntimeError("NVFortran link recipe is missing OpenACC initialization objects or runtime")
    return {"objects": list(dict.fromkeys(objects)), "library_dirs": list(dict.fromkeys(library_dirs)),
            "libraries": list(dict.fromkeys(libraries)), "rpaths": list(dict.fromkeys(rpaths)), "link_flags": link_flags,
            "runtime_discovery_argv": command}


def compile_gpu_sources(
    sources: list[GPUSource] | list[dict],
    build_dir: Path,
    options: dict | None = None,
    *,
    defines: list[str] | None = None,
    toolchain: dict | None = None,
) -> dict:
    """Compile self-contained GPU objects and return GNU linker inputs.

    NVIDIA module output goes into its own directory, isolated from GNU module
    files. Compiler diagnostics and exact commands are saved beside the objects.
    """
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
    module_dir = directory / "nvfortran-modules"
    module_dir.mkdir(exist_ok=True)
    commands = []
    objects = []
    manifest: dict = {"toolchain": config, "sources": units, "commands": commands, "objects": objects,
                      "compiler_environment": compiler_environment,
                      "library_dirs": [], "libraries": [], "rpaths": [], "link_flags": []}
    log_path = directory / "gpu-build.log"
    with log_path.open("w", encoding="utf-8") as log:
        for index, unit in enumerate(units):
            source = Path(unit["source"])
            target = directory / f"{index:03d}_{source.stem}.o"
            macros = ["-D" + macro for macro in (defines or [])]
            if unit["language"] == "cuda":
                contract = unit.get("numerical_contract")
                numerical_options = (contract or {}).get("required_cuda_options", [])
                host_options = (contract or {}).get("required_host_options", [])
                command = [config["nvcc"], "-O3", "-std=c++17", "-arch=sm_" + config["architecture"],
                           "-ccbin", config["cuda_host_cxx"], "-Xcompiler=-fPIC,-fopenmp", *macros,
                           *numerical_options, *["-Xcompiler=" + option for option in host_options],
                           "-c", str(source), "-o", str(target)]
                unit["resolved_numerical_options"] = {"cuda": numerical_options, "host": host_options}
            else:
                command = [config["nvfortran"], "-O3", "-fPIC", "-acc=gpu", "-gpu=cc" + config["architecture"] + ",nordc",
                           "-Mfreeform", "-Mextend", "-Mpreprocess", "-module", str(module_dir), *macros,
                           "-c", str(source), "-o", str(target)]
            unit["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
            commands.append(command)
            log.write(shlex.join(command) + "\n")
            log.flush()
            result = subprocess.run(command, cwd=directory, stdout=log, stderr=subprocess.STDOUT, check=False)
            if result.returncode or not target.is_file():
                (directory / "gpu-build-command.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
                raise RuntimeError(f"GPU compilation failed; inspect {log_path}")
            objects.append(str(target))
            unit["build_status"] = "compiled"
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
    if "openacc" in languages:
        acc_object = Path(objects[next(index for index, unit in enumerate(units) if unit["language"] == "openacc")])
        runtime = _openacc_link_configuration(config, acc_object, directory)
        for key in ("objects", "library_dirs", "libraries", "rpaths", "link_flags"):
            manifest[key].extend(runtime[key])
        manifest["runtime_discovery_argv"] = runtime["runtime_discovery_argv"]
        # The public OpenACC API must precede GNU libgomp, which also exports
        # acc_* symbols. GNU libgomp must in turn precede NVIDIA libnvomp so
        # host GOMP_* calls use the compiler's own runtime. build() supplies
        # that ordered library prefix for the complete host application.
        manifest["openacc_library"] = str(Path(config["nvfortran_library_dir"]) / "libacchost.so")
        # libacchost references acc_register_library weakly. GNU libgomp's
        # implementation is an unsupported profiling stub; retain NVIDIA's
        # actual notification provider before GNU to satisfy that hook.
        manifest["openacc_notify_library"] = str(Path(config["nvfortran_library_dir"]) / "libaccnotify.so")
    for key in ("objects", "library_dirs", "libraries", "rpaths", "link_flags"):
        manifest[key] = list(dict.fromkeys(manifest[key]))
    (directory / "gpu-build-command.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
