"""Explicit optional pressure backend builds, separately bounded from ELMM.

Native/control builds never import or invoke this preparation path. Prepared
artifacts carry byte identities, not compiler source or placement authority.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

from .build_receipts import CommandRecorder, file_identity
from .gpu_pressure import PressureProblem, prepare_adapter

ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_BASE = {"scoped_runtime.h", "scoped_runtime.cu", "scoped_memory.f90"}
_RUNTIME_QUOTE = "scoped_source_quote.hpp"


def _runtime_requires_quote(source):
    return re.search(r'^\s*#\s*include\s*"scoped_source_quote\.hpp"', Path(source).read_text(), re.MULTILINE) is not None


def _runtime_inputs(directory):
    directory = Path(directory)
    names = _RUNTIME_BASE | ({_RUNTIME_QUOTE} if _runtime_requires_quote(directory / "scoped_runtime.cu") else set())
    return {name: file_identity(directory / name) for name in sorted(names)}


def _backend_request(request):
    if (not isinstance(request, dict) or set(request) != {"manifest", "sha256"}
            or not isinstance(request["manifest"], str) or not Path(request["manifest"]).is_absolute()
            or "\0" in request["manifest"] or not isinstance(request["sha256"], str)
            or re.fullmatch(r"[0-9a-f]{64}", request["sha256"]) is None):
        raise ValueError("Pressure backend requires a SHA-bound prepared manifest")
    return dict(request)


def pressure_runtime_cuda_binding(unit, original):
    """Match exact raw code or its exact validated numerical-contract banner.

    The actual generated source identity remains in the build receipt. No
    arbitrary comments, whitespace or executable code are normalized.
    """
    if file_identity(original["path"]) != original:
        raise ValueError("Compiler raw runtime changed from the prepared pressure backend")
    actual = file_identity(unit["source"])
    if actual["sha256"] == original["sha256"]:
        return {"source": actual, "raw_source": dict(original), "rendering": "exact raw source"}
    if "numerical_contract" not in unit:
        return None
    from .gpu_toolchain import validate_numerical_contract
    contract = validate_numerical_contract(unit["numerical_contract"])
    options = [*contract["required_cuda_options"], *("-Xcompiler=" + option for option in contract["required_host_options"])]
    banner = ("// Compiler numerical build contract: " + contract["id"] + " " + contract["identity"] + "\n"
              "// Required NVCC options: " + " ".join(options) + "\n"
              "// Consumers must enforce the public contract in the actual compile command.\n")
    if Path(unit["source"]).read_bytes() != banner.encode() + Path(original["path"]).read_bytes():
        return None
    return {"source": actual, "raw_source": dict(original), "rendering": "exact numerical contract banner plus raw source",
            "numerical_contract": contract}


def combine_pressure_backends(requests):
    """Combine explicit checked requests; never choose a provider by its name."""
    result = None
    for request in requests:
        if request is None:
            continue
        request = _backend_request(request)
        if result is not None and result != request:
            raise ValueError("Scope packages require one identical pressure backend")
        result = request
    return result


def pressure_backend_from_manifest(manifest):
    """Project build inputs only after checking the complete issued epoch.

    The duplicated source-facing external task list is not a build authority.
    Numerical/source/execution authority stays with the compiler's issued plan.
    """
    from .checked_execution_plan import validate_checked_execution_plans

    validate_checked_execution_plans(manifest)
    requests = []
    for scope in manifest.get("scopes", []):
        if not isinstance(scope, dict):
            continue
        plan = scope.get("checked_execution_plan", {})
        for member in plan.get("storage_epoch", {}).get("members", []):
            for task in member.get("device_tasks", []):
                request = task["provider"].get("pressure_backend")
                if request is not None:
                    requests.append(request)
    return combine_pressure_backends(requests)


def _json_identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verify_source_package(directory=None):
    lock = json.loads((ROOT / "pressure-dependencies.json").read_text())
    directory = Path(directory or ROOT / lock["provider"]["source"]).resolve(strict=True)
    origin_path, manifest_path = directory / "origin.json", directory / "source-manifest.json"
    if (file_identity(origin_path)["sha256"] != lock["provider"]["origin_sha256"]
            or file_identity(manifest_path)["sha256"] != lock["provider"]["source_manifest_sha256"]):
        raise ValueError("Pressure provider origin/source manifest differs from the pinned package")
    origin, manifest = json.loads(origin_path.read_text()), json.loads(manifest_path.read_text())
    controls = {"upstream.patch": "patch_sha256", "UPSTREAM-LICENSE-NOTICE.txt": "license_notice_sha256",
                "VkFFT-LICENSE": "VkFFT_license_sha256", "source-manifest.json": "source_manifest_sha256"}
    for name, key in controls.items():
        if file_identity(directory / name)["sha256"] != origin[key]:
            raise ValueError("Pressure provider package metadata changed: " + name)
    expected = set(controls) | {"origin.json"}
    for entry in manifest["files"]:
        name = entry["path"]
        original_path = directory / name
        path = original_path.resolve(strict=True)
        if not path.is_relative_to(directory) or original_path.is_symlink() or name in expected:
            raise ValueError("Invalid pressure source package path")
        if file_identity(path)["sha256"] != entry["sha256"] or stat.S_IMODE(path.stat().st_mode) != int(entry["mode"], 8):
            raise ValueError("Pressure provider source changed: " + name)
        expected.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError("Pressure provider package contains missing or additional files")
    return directory, manifest


def _check_vkfft(directory):
    directory = Path(directory).resolve(strict=True)
    pin = json.loads((ROOT / "pressure-dependencies.json").read_text())["VkFFT"]["revision"]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=directory,
                           capture_output=True, text=True, check=True)
    if result.stdout.strip() != pin or dirty.stdout or not (directory / "vkFFT/vkFFT.h").is_file():
        raise ValueError("VkFFT must be a clean checkout of the pinned dependency")
    return directory


def _phase(argv, directory, name, *, environment):
    command = {"argv": [str(x) for x in argv], "cwd": str(directory), "timeout_seconds": 240,
               "status": "running", "returncode": None}
    path = directory / (name + "-command.json")
    path.write_text(json.dumps(command, indent=2) + "\n")
    start = time.monotonic()
    try:
        with (directory / (name + ".log")).open("w") as log:
            result = subprocess.run(command["argv"], cwd=directory, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=240, check=False)
        command.update(status="passed" if result.returncode == 0 else "failed", returncode=result.returncode)
        if result.returncode:
            raise RuntimeError(f"Pressure backend {name} failed; inspect {directory / (name + '.log')}")
    except subprocess.TimeoutExpired:
        command.update(status="timed_out", timed_out=True)
        raise
    finally:
        command["elapsed_seconds"] = time.monotonic() - start
        path.write_text(json.dumps(command, indent=2) + "\n")


def prepare_pressure_backend(problem: PressureProblem, directory, *, runtime_directory, vkfft_source,
                             toolchain, fortran, jobs=2, package_directory=None, cpu_fallback="original"):
    """Build one immutable optional backend; no ELMM/source transformation runs.

    The VkFFT checkout is explicit and pinned. Dependency fetching, if needed,
    is a separate authorized operation. This function never uses a sibling
    development checkout or changes a tracked source/configuration.
    """
    if type(jobs) is not int or jobs < 1:
        raise ValueError("Pressure build jobs must be positive")
    if cpu_fallback not in ("original", "lazy_poisfft_v1"):
        raise ValueError("Unknown explicit pressure CPU fallback implementation")
    package, source_manifest = verify_source_package(package_directory)
    vkfft = _check_vkfft(vkfft_source)
    runtime = Path(runtime_directory).resolve(strict=True)
    runtime_inputs = _runtime_inputs(runtime)
    architecture = str(toolchain["architecture"])
    if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", architecture):
        raise ValueError("Explicit CUDA architecture required")
    cxx, nvcc, fc = (Path(value).resolve(strict=True) for value in
                     (toolchain["cuda_host_cxx"], toolchain["nvcc"], fortran))
    cuda_include = Path(toolchain.get("cuda_include_dir", Path(toolchain["cuda_library_dir"]).parent / "include"))
    if not (cuda_include / "cuda_runtime_api.h").is_file():
        raise ValueError("Explicit selected-toolkit CUDA include directory required")
    for name in ("CFLAGS", "CXXFLAGS", "FFLAGS", "FCFLAGS", "F90FLAGS", "LDFLAGS", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if os.environ.get(name):
            raise ValueError("Pressure backend rejects hidden compiler option override: " + name)
    directory = Path(directory).resolve()
    if directory == package or package in directory.parents:
        raise ValueError("Pressure build must stay outside the source package")
    directory.mkdir(parents=True, exist_ok=False)
    build, install = directory / "provider-build", directory / "provider-install"
    receipts = directory / "command-receipts"
    shutil.copyfile(Path(__file__).with_name("build_receipts.py"), directory / "build_receipts.py")
    launcher = directory / "compiler_launcher.py"
    launcher.write_text("from pathlib import Path\nimport sys\nfrom build_receipts import CommandRecorder\n"
                        "sys.exit(CommandRecorder(Path(__file__).parent/'command-receipts').run(sys.argv[1:]).returncode)\n")
    cmake = shutil.which("cmake")
    if not cmake:
        raise RuntimeError("CMake >=4 is required for the pressure backend")
    launch = str(Path(sys.executable).resolve()) + ";" + str(launcher)
    argv = [cmake, "-S", str(package), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
            "-DCMAKE_CXX_COMPILER=" + str(cxx), "-DCMAKE_CUDA_COMPILER=" + str(nvcc),
            "-DCMAKE_CUDA_HOST_COMPILER=" + str(cxx), "-DCMAKE_Fortran_COMPILER=" + str(fc),
            "-DCMAKE_CUDA_ARCHITECTURES=" + architecture, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DPOISSON_SOLVER_BUILD_FORTRAN=ON", "-DPOISSON_SOLVER_USE_MPI=OFF",
            "-DCMAKE_INSTALL_LIBDIR=lib", "-DCMAKE_INSTALL_INCLUDEDIR=include",
            "-DPOISSON_SOLVER_BUILD_TESTS=OFF", "-DPOISSON_SOLVER_BUILD_API_TESTS=OFF",
            "-DPOISSON_SOLVER_BUILD_EXAMPLES=OFF", "-DPOISSON_SOLVER_BUILD_DOCS=OFF",
            "-DFETCHCONTENT_SOURCE_DIR_VKFFT=" + str(vkfft)]
    environment = os.environ.copy()
    _phase(argv, directory, "configure", environment=environment)
    # Do not record ephemeral compiler-identification probes as production
    # inputs: install the launchers after language detection is complete.
    launcher_args = [cmake, "-S", str(package), "-B", str(build)]
    for language in ("CXX", "CUDA", "Fortran"):
        launcher_args += ["-DCMAKE_" + language + "_COMPILER_LAUNCHER=" + launch,
                          "-DCMAKE_" + language + "_LINKER_LAUNCHER=" + launch]
    _phase(launcher_args, directory, "record-compilers", environment=environment)
    _phase([cmake, "--build", str(build), "--parallel", str(jobs)], directory, "compile", environment=environment)
    _phase([cmake, "--install", str(build), "--prefix", str(install)], directory, "install", environment=environment)
    library = install / "lib/libpoisson-solver.so"
    if not library.is_file():
        raise RuntimeError("Pressure provider install did not produce its shared library")
    adapter = directory / "adapter"
    requirements = prepare_adapter(problem, adapter)
    # The optional ELMM backend retains the original CPU PoisFFT dependency.
    # This typed query is compiled by the application against those exact
    # native modules, never against a fabricated derived-type interface.
    query = Path(__file__).with_name("gpu_pressure_sources") / "gpu_pressure_poisfft.f90"
    query_identity = file_identity(query)
    shutil.copyfile(query, adapter / query.name)
    if file_identity(adapter / query.name)["sha256"] != query_identity["sha256"]:
        raise ValueError("Typed PoisFFT query changed while being copied")
    lazy_query = None
    if cpu_fallback == "lazy_poisfft_v1":
        lazy_query = query.with_name("gpu_pressure_lazy_poisfft.f90")
        lazy_identity = file_identity(lazy_query)
        shutil.copyfile(lazy_query, adapter / lazy_query.name)
        if file_identity(adapter / lazy_query.name) != {**lazy_identity, "path": str(adapter / lazy_query.name)}:
            raise ValueError("Lazy PoisFFT implementation changed while being copied")
        requirements["cpu_fallback"] = {
            "implementation": "lazy_poisfft_v1", "source": file_identity(adapter / lazy_query.name),
            "constructor": "operator metadata only; no CPU or GPU FFT plan",
            "fallback": "pinned CPU plan created once per receiver on first reached native Execute",
            "gpu_ownership": "existing checked caller-owned domain/provider slot"}
    shutil.copyfile(runtime / "scoped_runtime.h", adapter / "scoped_runtime.h")
    if _RUNTIME_QUOTE in runtime_inputs:
        shutil.copyfile(runtime / _RUNTIME_QUOTE, adapter / _RUNTIME_QUOTE)
    (adapter / "poisson-solver").mkdir()
    shutil.copyfile(install / "include/poisson-solver/device_api.h", adapter / "poisson-solver/device_api.h")
    recorder = CommandRecorder(receipts)
    objects = []
    for name, compiler in (("gpu_pressure", [str(cxx), "-O3", "-std=c++17", "-fPIC", "-ffp-contract=off"]),
                           ("gpu_pressure_halo", [str(nvcc), "-O3", "-std=c++17", "-arch=sm_" + architecture,
                                                  "-ccbin", str(cxx), "-Xcompiler=-fPIC", "--fmad=false",
                                                  "--ftz=false", "--prec-div=true", "--prec-sqrt=true"])):
        source = adapter / (name + (".cpp" if name == "gpu_pressure" else ".cu"))
        target = adapter / (name + ".o")
        command = [*compiler, "-I", str(adapter), "-I", str(cuda_include), "-c", str(source), "-o", str(target)]
        with (directory / (name + ".log")).open("w") as log:
            result = recorder.run(command, cwd=directory, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=240)
        if result.returncode:
            raise RuntimeError("Pressure adapter compilation failed: " + str(source))
        objects.append(str(target))
    verify_source_package(package)
    _check_vkfft(vkfft)
    if file_identity(query) != query_identity:
        raise ValueError("Typed PoisFFT query changed during pressure preparation")
    if lazy_query is not None and file_identity(lazy_query) != lazy_identity:
        raise ValueError("Lazy PoisFFT implementation changed during pressure preparation")
    for name, identity in runtime_inputs.items():
        if file_identity(runtime / name) != identity:
            raise ValueError("Compiler runtime changed during pressure preparation")
    inputs = [file_identity(package / entry["path"]) for entry in source_manifest["files"]]
    inputs += [file_identity(path) for path in sorted(vkfft.joinpath("vkFFT").rglob("*")) if path.is_file()]
    inputs += [file_identity(path) for path in sorted(adapter.rglob("*")) if path.is_file()]
    inputs += [file_identity(path) for path in sorted(install.rglob("*")) if path.is_file()]
    inputs += [file_identity(package / name) for name in ("origin.json", "source-manifest.json", "upstream.patch",
                                                         "UPSTREAM-LICENSE-NOTICE.txt", "VkFFT-LICENSE")]
    inputs += [*runtime_inputs.values(), file_identity(ROOT / "pressure-dependencies.json")]
    payload = {"schema_version": 1, "kind": "prepared_gpu_pressure_backend", "placement_authority": False,
               "requirements": requirements, "package": file_identity(package / "source-manifest.json"),
               "runtime": runtime_inputs, "fortran_compiler": file_identity(fc),
               "objects": objects, "library_dirs": [str(install / "lib")], "libraries": ["poisson-solver"],
               "rpaths": [str(install / "lib"), str(Path(toolchain["cuda_library_dir"]).resolve())],
               "extra_sources": [str(adapter / name) for name in ("gpu_pressure.f90", "gpu_pressure_poisfft.f90")]
                                + ([str(adapter / lazy_query.name)] if lazy_query is not None else []),
               "include_dirs": [str(install / "include/poisson-solver/fortran")],
               "command_receipts": str(receipts), "inputs": inputs, "provider_library": file_identity(library)}
    payload["identity"] = _json_identity(payload)
    path = directory / "pressure-backend.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return {"manifest": str(path), "sha256": file_identity(path)["sha256"]}


def require_pressure_backend(request, *, fortran):
    """Revalidate prepared bytes and compiler/module compatibility, no rebuild."""
    request = _backend_request(request)
    path = Path(request["manifest"]).resolve(strict=True)
    if file_identity(path)["sha256"] != request["sha256"]:
        raise ValueError("Pressure backend manifest changed")
    value = json.loads(path.read_text())
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1 or value.get("kind") != "prepared_gpu_pressure_backend"
            or value.get("placement_authority") is not False
            or value.get("identity") != _json_identity({key: item for key, item in value.items() if key != "identity"})):
        raise ValueError("Invalid prepared pressure backend identity")
    keys = {"schema_version", "kind", "placement_authority", "requirements", "package", "runtime",
            "fortran_compiler", "objects", "library_dirs", "libraries", "rpaths", "extra_sources",
            "include_dirs", "command_receipts", "inputs", "provider_library", "identity"}
    runtime = value.get("runtime")
    if (set(value) != keys or not isinstance(runtime, dict)
            or set(runtime) not in (_RUNTIME_BASE, _RUNTIME_BASE | {_RUNTIME_QUOTE})):
        raise ValueError("Prepared pressure backend schema is incomplete")
    if _runtime_requires_quote(runtime["scoped_runtime.cu"]["path"]) != (_RUNTIME_QUOTE in runtime):
        raise ValueError("Prepared pressure backend must bind its exact runtime quote header dependency")
    if (not value["objects"] or value["libraries"] != ["poisson-solver"]
            or value["requirements"].get("source_authority") is not False
            or value["requirements"].get("placement_authority") is not False):
        raise ValueError("Pressure backend cannot grant source/placement authority")
    known = {entry["path"]: entry["sha256"] for entry in value["inputs"]}
    for name in [*value["objects"], *value["extra_sources"], value["provider_library"]["path"]]:
        identity = file_identity(name)
        if known.get(identity["path"]) != identity["sha256"]:
            raise ValueError("Prepared pressure link/interface input is not bound")
    for name in value["include_dirs"]:
        if not Path(name).is_dir():
            raise ValueError("Prepared pressure module directory disappeared")
    lock = json.loads((ROOT / "pressure-dependencies.json").read_text())
    if (value["package"]["sha256"] != lock["provider"]["source_manifest_sha256"]
            or file_identity(value["package"]["path"]) != value["package"]):
        raise ValueError("Prepared pressure provider differs from the pinned package")
    if file_identity(fortran) != value["fortran_compiler"]:
        raise ValueError("Pressure Fortran modules require the exact selected host compiler")
    for identity in value["inputs"]:
        if file_identity(identity["path"]) != identity:
            raise ValueError("Prepared pressure backend input changed: " + identity["path"])
    for identity in value["runtime"].values():
        if file_identity(identity["path"]) != identity:
            raise ValueError("Compiler runtime differs from the prepared pressure backend")
    return value


def prepare_lazy_pressure_backend(request, directory, *, fortran):
    """Add the explicit lazy Fortran implementation to exact prepared code.

    This does not rebuild or relabel the provider's compiled artifacts. Their
    original paths, bytes, command receipts and runtime bindings are retained.
    The new manifest binds its base manifest and the additional source; fresh
    source/observability qualification and application compilation are needed.
    """
    backend = require_pressure_backend(request, fortran=fortran)
    if "cpu_fallback" in backend["requirements"]:
        raise ValueError("Lazy pressure preparation requires the original CPU implementation")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("gpu_pressure_sources") / "gpu_pressure_lazy_poisfft.f90"
    original = file_identity(source)
    target = directory / source.name
    shutil.copyfile(source, target)
    if file_identity(target)["sha256"] != original["sha256"]:
        raise ValueError("Lazy PoisFFT implementation changed while being copied")
    backend["requirements"]["cpu_fallback"] = {
        "implementation": "lazy_poisfft_v1", "source": file_identity(target),
        "constructor": "operator metadata only; no CPU or GPU FFT plan",
        "fallback": "pinned CPU plan created once per receiver on first reached native Execute",
        "gpu_ownership": "existing checked caller-owned domain/provider slot",
        "compiled_artifact_origin": dict(request), "fresh_compilation": False}
    backend["extra_sources"].append(str(target))
    backend["inputs"].extend([file_identity(request["manifest"]), file_identity(target)])
    backend["identity"] = _json_identity({key: value for key, value in backend.items() if key != "identity"})
    path = directory / "pressure-backend.json"
    path.write_text(json.dumps(backend, indent=2) + "\n")
    if file_identity(source) != original:
        raise ValueError("Lazy PoisFFT implementation changed during preparation")
    require_pressure_backend(request, fortran=fortran)
    result = {"manifest": str(path), "sha256": file_identity(path)["sha256"]}
    require_pressure_backend(result, fortran=fortran)
    return result


def prepare_poisfft_provider_manifest(request, path, *, fortran, qualification,
                                     original_body_sha256, task_contract_sha256):
    """Package an explicitly qualified inner Execute adapter, never infer one.

    The supplied qualification must cover the original receiver's observable
    effects and the exact physical boxes/configuration recipe. Its SHA and
    both compiler-computed scope identities are mandatory. Construction here
    supplies no source, placement or definition authority to the compiler.
    """
    backend = require_pressure_backend(request, fortran=fortran)
    if (not isinstance(qualification, dict) or set(qualification) != {"path", "sha256"}
            or not isinstance(qualification["path"], str) or not Path(qualification["path"]).is_absolute()
            or file_identity(qualification["path"])["sha256"] != qualification["sha256"]):
        raise ValueError("PoisFFT adapter requires its exact explicit qualification")
    for digest in (original_body_sha256, task_contract_sha256):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("PoisFFT qualification requires original body and task scope SHA256 identities")
    query_sources = [name for name in backend["extra_sources"] if Path(name).name == "gpu_pressure_poisfft.f90"]
    if len(query_sources) != 1:
        raise ValueError("Prepared backend does not bind the typed original PoisFFT query")
    artifacts = {name: file_identity(name)["sha256"] for name in
                 [request["manifest"], backend["provider_library"]["path"],
                  *backend["objects"], *backend["extra_sources"]]}
    numeric = {"qualification": dict(qualification), "original_bodies": [original_body_sha256],
               "task_contracts": [task_contract_sha256]}
    numeric["identity"] = _json_identity({"provider": artifacts, **numeric})
    record = {"version": 2, "kind": "external_device_provider",
              "fortran": {"module": "elmm_gpu_pressure", "config_type": "elmm_gpu_pressure_config_v1",
                          "create": "elmm_gpu_pressure_create_v1", "execute": "elmm_gpu_pressure_solve_v1",
                          "destroy": "elmm_gpu_pressure_destroy_v1"},
              "metadata_query": {"module": "elmm_gpu_pressure_poisfft",
                                 "procedure": "elmm_gpu_pressure_configure_poisfft_v1",
                                 "opaque_formals": ["d"], "configuration_argument": "config", "read_only": True,
                                 "configuration_components": [
                                     {"component": "solver%sizes", "kind": "int32", "count": 3},
                                     {"component": "solver%lengths", "kind": "real64", "count": 3},
                                     {"component": "solver%boundary_conditions", "kind": "int32", "count": 6},
                                     {"component": "solver%approximation", "kind": "int32", "count": 1}]},
              "artifacts": artifacts, "numerical_contract": numeric,
              "execution": {"stream": "supplied", "completion": "same_stream", "workspace": "domain_claim",
                            "fallback": "prework_only"}, "pressure_backend": dict(request)}
    lazy = backend["requirements"].get("cpu_fallback")
    if lazy is not None:
        sources = [name for name in backend["extra_sources"] if Path(name).name == "gpu_pressure_lazy_poisfft.f90"]
        if (not isinstance(lazy, dict) or lazy.get("implementation") != "lazy_poisfft_v1"
                or len(sources) != 1 or lazy.get("source") != file_identity(sources[0])):
            raise ValueError("Lazy PoisFFT provider requires its exact prepared implementation")
        record["metadata_query"].update(module="elmm_gpu_pressure_lazy_poisfft",
                                        procedure="elmm_gpu_pressure_configure_lazy_v1")
    path = Path(path).resolve()
    with path.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    return {"path": str(path), "sha256": file_identity(path)["sha256"]}
