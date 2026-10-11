"""Build ELMM and generated sources using only explicitly supplied paths.

The upstream SCons build expects ELMM and PoisFFT to be sibling directories.
No repository discovery, parent-directory imports, or global Python packages are
used here. Native compilers and system numerical libraries are prerequisites.
"""

from __future__ import annotations

import ctypes.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import venv

from .gpu_toolchain import GPUSource, compile_gpu_sources, validate_gpu_sources


SCONS_VERSION = "4.11.1"
NATIVE_LIBRARIES = ("fftw3", "fftw3f", "fftw3_omp", "fftw3f_omp", "blas", "lapack")



# These spellings and conditions match the pinned ELMM src/SConstruct. The
# pipeline controls GNU/OpenMP/double precision, and analysis consumes this same
# macro list. Other SCons switches must not silently select a different program.
PREPROCESSOR_SWITCHES = {
    "customSB": "CUSTOMSB",
    "customPB": "CUSTOMPB",
    "customPuff": "CUSTOM_PUFF_SOURCES",
    "customConfig": "CUSTOM_CONFIG",
    "customBoundaryConditions": "CUSTOM_BOUNDARY_CONDITIONS",
    "customInitialConditions": "CUSTOM_INITIAL_CONDITIONS",
    "customSurfaceTemperature": "CUSTOM_SURFACE_TEMPERATURE",
    "customSurfaceTemperatureFlux": "CUSTOM_SURFACE_TEMPERATURE_FLUX",
    "customTimeStepProcedure": "CUSTOM_TIMESTEP_PROCEDURE",
    "customOutput": "CUSTOM_OUTPUT",
    "customForcing": "CUSTOM_FORCING",
    "missing_pfft_r2r": "MISSING_PFFT_R2R",
}


def preprocessing_defines(extra_args: list[str]) -> list[str]:
    """Validate supported build options and return their exact active macros.

    Analysis and compilation both use GNU Fortran, -O3, -fopenmp and DPREC.
    Arbitrary compiler flags and compiler-family/ABI changes need a richer build
    description before they can be supported; reject them before discovery.
    """
    protected = {"dir", "prec", "mpi", "openmp", "comp", "cray", "ompp"}
    booleans = set(PREPROCESSOR_SWITCHES) | {"fast", "lto", "prof", "diag", "cgal", "aggressive", "sanitize"}
    paths_or_libraries = {"SB", "PB", "lapack_blas"}
    options = {}
    for argument in extra_args:
        key, separator, value = argument.partition("=")
        if not separator or not key or key in protected:
            raise ValueError(f"Unsupported build argument {argument!r}: the pipeline controls GNU/OpenMP, precision, MPI and output paths")
        if key in options:
            raise ValueError(f"Unsupported build argument {argument!r}: duplicate option {key!r}")
        if key == "extra":
            if value.strip():
                raise ValueError("Unsupported build argument 'extra': arbitrary compiler flags can change preprocessing or the Fortran ABI; use supported named --build-arg options")
        elif key in booleans:
            if value not in {"yes", "no"}:
                raise ValueError(f"Unsupported build argument {argument!r}: {key} must be yes or no")
            if key in {"aggressive", "sanitize"} and value != "no":
                raise ValueError(f"Unsupported build argument {argument!r}: these flags change predefined preprocessor macros and are not part of the analyzed build configuration")
        elif key not in paths_or_libraries:
            raise ValueError(f"Unsupported build argument {argument!r}: unknown option; use supported named SCons options")
        options[key] = value
    return ["DPREC", *[macro for key, macro in PREPROCESSOR_SWITCHES.items() if options.get(key) == "yes"]]


def _python_environment() -> dict[str, str]:
    """Keep private Python tools independent of caller package search paths."""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _library_cache() -> dict[str, Path]:
    """Read the host linker cache; absence is normal on non-Linux systems."""
    command = shutil.which("ldconfig")
    if command is None:
        for candidate in (Path("/sbin/ldconfig"), Path("/usr/sbin/ldconfig")):
            if candidate.is_file():
                command = str(candidate)
                break
    if command is None:
        return {}
    result = subprocess.run([command, "-p"], text=True, capture_output=True, check=False)
    libraries: dict[str, Path] = {}
    for line in result.stdout.splitlines():
        match = re.match(r"\s*(\S+)\s+\([^)]*\)\s+=>\s+(.*)$", line)
        if match and Path(match[2]).is_file():
            libraries.setdefault(match[1], Path(match[2]))
    return libraries


def _find_library(name: str, cache: dict[str, Path], compiler: str) -> Path:
    soname = ctypes.util.find_library(name)
    candidates = [f"lib{name}.so", f"lib{name}.dylib", f"lib{name}.a"]
    if soname:
        candidates.insert(0, soname)
    for candidate in candidates:
        if candidate in cache:
            return cache[candidate].resolve()
        result = subprocess.run(
            [compiler, f"-print-file-name={candidate}"],
            text=True, capture_output=True, check=False,
        )
        located = Path(result.stdout.strip())
        if result.returncode == 0 and located.is_absolute() and located.is_file():
            return located.resolve()
    raise RuntimeError(
        f"Native library {name!r} was not found. Install FFTW (double and single "
        "precision, including OpenMP), BLAS and LAPACK, or expose them through "
        "LIBRARY_PATH/your compiler's search path."
    )


def bootstrap_native(root: Path) -> dict[str, Path]:
    """Create a private SCons environment and detected native linker aliases.

    Re-running this function after moving the project recreates the environment
    if necessary and refreshes host-specific library links. Downloaded packages
    and links live under the caller's ignored cache, never in tracked sources.
    """
    root = root.resolve()
    compiler = shutil.which(os.environ.get("FC", "gfortran"))
    if not compiler:
        raise RuntimeError("A GNU Fortran compiler is required (gfortran, or FC).")
    for name in (os.environ.get("CC", "gcc"), os.environ.get("CXX", "g++")):
        if shutil.which(name) is None:
            raise RuntimeError(f"Required compiler {name!r} is not on PATH.")

    environment = root / ".cache" / "envs" / "build"
    python = environment / "bin" / "python"
    if not python.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    probe = subprocess.run(
        [str(python), "-c", "import SCons; print(SCons.__version__)"],
        capture_output=True, text=True, check=False, env=_python_environment(),
    )
    if probe.returncode != 0 or probe.stdout.strip() != SCONS_VERSION:
        subprocess.run(
            [str(python), "-m", "pip", "install", "--disable-pip-version-check",
             "--no-cache-dir", f"scons=={SCONS_VERSION}"],
            check=True, env=_python_environment(),
        )

    library_dir = root / ".cache" / "native" / "lib"
    library_dir.mkdir(parents=True, exist_ok=True)
    cache = _library_cache()
    provenance: dict[str, str] = {}
    for name in NATIVE_LIBRARIES:
        actual = _find_library(name, cache, compiler)
        extension = ".dylib" if actual.suffix == ".dylib" else ".so"
        alias = library_dir / f"lib{name}{extension}"
        if alias.is_symlink() or alias.exists():
            alias.unlink()
        alias.symlink_to(actual)
        provenance[name] = str(actual)
    (library_dir.parent / "libraries.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    return {"python": python, "library_dir": library_dir}


def build(
    elmm_root: Path,
    poisfft_root: Path,
    build_dir: Path,
    scons_python: Path,
    fftw_library_dir: Path,
    jobs: int,
    threads: int,
    extra_args: list[str],
    extra_sources: list[Path] | None = None,
    patched_sources: list[Path] | None = None,
    gpu_sources: list[GPUSource] | list[dict] | None = None,
    gpu_options: dict | None = None,
) -> Path:
    """Build double-precision ELMM and return the absolute executable path.

    ``elmm_root`` is a writable source snapshot. A separate SConstruct.pipeline
    file adds optional generated Fortran/C++ sources without editing upstream's
    build file. Fortran module dependencies are scanned by SCons; an explicit
    dependency also protects every changed native source's generated interface
    modules. Matching uses source paths, so upstream object names can differ.
    Optional CUDA/OpenACC units are compiled separately and linked into the GNU
    host through C entry points; their NVIDIA module files remain isolated.
    """
    if jobs < 1 or threads < 1:
        raise ValueError("jobs and threads must be positive")
    elmm_root = elmm_root.resolve()
    poisfft_root = poisfft_root.resolve()
    build_dir = build_dir.resolve()
    scons_python = scons_python.absolute()
    fftw_library_dir = fftw_library_dir.resolve()
    source_dir = elmm_root / "src"
    if not (source_dir / "SConstruct").is_file():
        raise ValueError(f"ELMM source tree has no src/SConstruct: {elmm_root}")
    expected_poisfft = (source_dir / "../../PoisFFT").resolve()
    if expected_poisfft != poisfft_root:
        raise ValueError(
            "ELMM's build requires the supplied PoisFFT tree at "
            f"{expected_poisfft}; stage ELMM and PoisFFT as siblings."
        )
    if not (poisfft_root / "src" / "poisfft.f90").is_file():
        raise ValueError(f"Invalid PoisFFT source tree: {poisfft_root}")
    defines = preprocessing_defines(extra_args)
    sources = [Path(path).resolve() for path in (extra_sources or [])]
    for path in sources:
        if not path.is_file() or path.suffix.lower() not in {".f90", ".f", ".cpp", ".cc", ".c"}:
            raise ValueError(f"Missing or unsupported generated source: {path}")
    patched = [Path(path).resolve() for path in (patched_sources or [])]
    for path in patched:
        if not path.is_file() or path.suffix.lower() not in {".f90", ".f"}:
            raise ValueError(f"Missing or unsupported patched Fortran source: {path}")
    gpu_units = validate_gpu_sources(gpu_sources or [])
    if {unit["source"] for unit in gpu_units} & {str(path) for path in sources}:
        raise ValueError("GPU sources must not also appear among GNU host extra_sources")

    build_dir.mkdir(parents=True, exist_ok=True)
    original = (source_dir / "SConstruct").read_text(encoding="utf-8")
    anchor = "t = env.Program("
    if original.count(anchor) != 1:
        raise RuntimeError("Unsupported ELMM SConstruct: program insertion point changed")
    gpu_build = compile_gpu_sources(gpu_units, build_dir, gpu_options, defines=defines) if gpu_units else None
    injection = [
        "# Additional sources supplied by elmm-pipeline.",
        "if os.environ.get('FC'):",
        "  env.Replace(LINK=os.environ['FC'])",
        "env.Replace(LIBPATH=[p for p in os.environ.get('LIBRARY_PATH', '').split(os.pathsep) if p])",
        "env.Append(CXXFLAGS=['-std=c++17'])",
        "env.Append(F90FLAGS=['-ffree-line-length-none'])",
        "env.Append(FORTRANFLAGS=['-ffree-line-length-none'])",
        "pipeline_extra = []",
        "pipeline_interfaces = []",
        f"pipeline_patched_sources = set({sorted({str(path) for path in patched})!r})",
    ]
    for index, source in enumerate(sources):
        target = build_dir / "extra" / f"{index:03d}_{source.stem}"
        injection.append(
            f"pipeline_object = env.Object(target={str(target)!r}, source={str(source)!r})"
        )
        injection.append("pipeline_extra += pipeline_object")
        if source.suffix.lower() in {".f90", ".f"}:
            injection.append("pipeline_interfaces += pipeline_object")
    if gpu_build:
        injection.extend([
            f"pipeline_extra += [env.File(p) for p in {gpu_build['objects']!r}]",
            f"env.AppendUnique(LIBPATH={gpu_build['library_dirs']!r})",
            f"env.AppendUnique(LIBS={gpu_build['libraries']!r})",
            f"env.AppendUnique(RPATH={gpu_build['rpaths']!r})",
            f"env.AppendUnique(LINKFLAGS={gpu_build['link_flags']!r})",
        ])
    # NVHPC ships libgomp.so -> libnvomp.so. Selecting that alias through an
    # inherited -L path replaces GNU host OpenMP with NVIDIA's compatibility
    # layer. Resolve GNU's own runtime explicitly for every build, including
    # the baseline. With OpenACC, NVIDIA's acc_* API must precede GNU libgomp's
    # competing acc_* API, while GNU's GOMP_* API must precede libnvomp.
    openacc_library = gpu_build.get("openacc_library") if gpu_build else None
    openacc_notify_library = gpu_build.get("openacc_notify_library") if gpu_build else None
    cuda_stdcxx_library = gpu_build.get("cuda_stdcxx_library") if gpu_build else None
    cuda_stdcxx_link_dir = gpu_build.get("cuda_stdcxx_link_dir") if gpu_build else None
    cuda_host_cxx = None
    if gpu_build and ("cuda" in [u.get("language") for u in gpu_units] or "stdc++" in gpu_build.get("libraries", [])):
        cuda_host_cxx = (
            gpu_build.get("toolchain", {}).get("cuda_host_cxx")
            or (gpu_options or {}).get("cuda_host_cxx")
            or (gpu_options or {}).get("cuda_host")
            or os.environ.get("CUDAHOSTCXX")
            or os.environ.get("CXX")
            or "g++"
        )
    injection.extend([
        "import subprocess as pipeline_subprocess, json as pipeline_json",
        "pipeline_host_fc = os.environ.get('FC', 'gfortran')",
        "pipeline_gomp_link = pipeline_subprocess.check_output([pipeline_host_fc, '-print-file-name=libgomp.so'], text=True).strip()",
        "if not os.path.isabs(pipeline_gomp_link) or not os.path.isfile(pipeline_gomp_link):",
        "  raise RuntimeError('The GNU Fortran compiler did not identify its own libgomp runtime')",
        "pipeline_gomp = os.path.realpath(pipeline_gomp_link)",
        "if 'nvomp' in os.path.basename(pipeline_gomp):",
        "  raise RuntimeError('GNU OpenMP runtime lookup resolved NVIDIA libnvomp; fix the GNU compiler configuration')",
        f"pipeline_openacc = {openacc_library!r}",
        f"pipeline_openacc_notify = {openacc_notify_library!r}",
        f"pipeline_cuda_stdcxx = {cuda_stdcxx_library!r}",
        f"pipeline_cuda_link_dir = {cuda_stdcxx_link_dir!r}",
        f"pipeline_cuda_cxx = {cuda_host_cxx!r}",
        "if not pipeline_cuda_stdcxx and pipeline_cuda_cxx:",
        "  try:",
        "    link_target = pipeline_subprocess.check_output([pipeline_cuda_cxx, '-print-file-name=libstdc++.so'], text=True).strip()",
        "    if os.path.isabs(link_target) and os.path.isfile(link_target):",
        "      pipeline_cuda_stdcxx = os.path.realpath(link_target)",
        "      pipeline_cuda_link_dir = os.path.dirname(link_target)",
        "  except Exception:",
        "    pass",
        "pipeline_runtime_prefix = ([pipeline_openacc, pipeline_openacc_notify] if pipeline_openacc else []) + [pipeline_gomp]",
        "pipeline_runtime_flags = []",
        "if pipeline_openacc:",
        "  pipeline_runtime_flags = [env.File(pipeline_openacc), '-Wl,--push-state,--no-as-needed', env.File(pipeline_openacc_notify), '-Wl,--pop-state']",
        "pipeline_runtime_flags += [env.File(pipeline_gomp)]",
        "if pipeline_cuda_stdcxx:",
        "  pipeline_runtime_prefix.append(pipeline_cuda_stdcxx)",
        "  pipeline_stdcxx_dir = os.path.dirname(pipeline_cuda_stdcxx)",
        "  dirs_to_prepend = [pipeline_stdcxx_dir] + ([pipeline_cuda_link_dir] if pipeline_cuda_link_dir else [])",
        "  env.PrependUnique(LIBPATH=dirs_to_prepend)",
        "  env.PrependUnique(RPATH=[pipeline_stdcxx_dir])",
        "  pipeline_runtime_flags += [env.File(pipeline_cuda_stdcxx)]",
        "env.Replace(PIPELINE_RUNTIME_LIBFLAGS=pipeline_runtime_flags)",
        "env.Replace(_LIBFLAGS='$PIPELINE_RUNTIME_LIBFLAGS ' + env['_LIBFLAGS'])",
        "env.PrependUnique(LIBPATH=[os.path.dirname(pipeline_gomp_link)])",
        "env.PrependUnique(RPATH=[os.path.dirname(pipeline_gomp)])",
        # DT_RPATH protects the chosen GNU SONAME from NVHPC libgomp aliases in
        # an inherited LD_LIBRARY_PATH, and applies to FFTW's dependencies too.
        "env.AppendUnique(LINKFLAGS=['-Wl,--disable-new-dtags'])",
        f"with open({str(build_dir / 'host-runtime.json')!r}, 'w') as pipeline_runtime_file:",
        "  pipeline_json.dump({'gnu_openmp': pipeline_gomp, 'ordered_prefix': pipeline_runtime_prefix}, pipeline_runtime_file, indent=2)",
    ])
    injection.extend([
        "if pipeline_interfaces:",
        "  for pipeline_native_object in objs:",
        "    if any(os.path.realpath(s.get_abspath()) in pipeline_patched_sources",
        "           for s in pipeline_native_object.sources):",
        "      Depends(pipeline_native_object, pipeline_interfaces)",
        "objs += [o for o in pipeline_extra if not str(o).endswith(('.mod', '.smod'))]",
        "",
    ])
    script = source_dir / "SConstruct.pipeline"
    script.write_text(original.replace(anchor, "\n".join(injection) + "\n" + anchor), encoding="utf-8")
    env = _python_environment()
    env["LIBRARY_PATH"] = str(fftw_library_dir) + (
        os.pathsep + env["LIBRARY_PATH"] if env.get("LIBRARY_PATH") else ""
    )
    # Upstream treats LD_LIBRARY_PATH as LIBRARY_PATH when it is present.
    env["LD_LIBRARY_PATH"] = str(fftw_library_dir) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )
    env["OMP_NUM_THREADS"] = str(threads)
    env["OMP_DYNAMIC"] = "FALSE"
    command = [
        str(scons_python), "-m", "SCons", "-f", script.name, "-j", str(jobs),
        "fast=yes", "aggressive=no", "lto=no", "openmp=yes", "prec=dp",
        f"dir={build_dir}", *extra_args,
    ]
    build_manifest = {"argv": command, "cwd": str(source_dir),
                      "native_library_dir": str(fftw_library_dir),
                      "preprocessing_defines": defines,
                      "extra_sources": [str(path) for path in sources],
                      "patched_sources": [str(path) for path in patched]}
    if gpu_build:
        build_manifest["gpu"] = gpu_build
    (build_dir / "build-command.json").write_text(
        json.dumps(build_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    with (build_dir / "build.log").open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=source_dir, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    binary = build_dir / "ELMM"
    if completed.returncode != 0 or not binary.is_file():
        raise RuntimeError(f"ELMM build failed; inspect {build_dir / 'build.log'}")
    runtime_manifest = build_dir / "host-runtime.json"
    if runtime_manifest.is_file():
        build_manifest["host_runtime"] = json.loads(runtime_manifest.read_text(encoding="utf-8"))
        (build_dir / "build-command.json").write_text(json.dumps(build_manifest, indent=2) + "\n", encoding="utf-8")
    return binary
