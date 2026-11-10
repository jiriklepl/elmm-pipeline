"""Build the primary ELMM GPU path from source, without historical artifacts.

One supported configuration: GNU Fortran, double precision, OpenMP, one CUDA
device, and the checked uniform-channel adapter. Other physical configurations
take the original native branches. Placement/transfer policy switches are not
part of this command. Numerical extraction and coherence belong to the compiler.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .resident_rk import identity

# The non-MPI, non-CGAL source set of the pinned ELMM SConstruct. Keep physical
# owners distinct: ELMM and PoisFFT both contain parameters.f90 and main.f90.
ELMM_UNITS = """tsteps dynamics subgrid advection advection_variable_z
alternate_ieee_isnan areasources arrayutilities boundaries buoyant_gases
diabatic_processes dummy_cgal_polyhedra dummy_par elevationmodels endianness
filters frames frames_common freeunit geometricshapes geometricshapes2d
immersedboundary initial interpolation large_scale_processes limiters
linearforcing linesources lists main momentum_diffusion multigrid multigrid2d
numerical_operators output_helpers outputs parameters pointsources poisson
pressure pthreads puffs radiation rng_par_zig scalar_advection scalar_diffusion
scalarboundaries scalars simplevtk solarradiation solidbodies sponge stop strings
subgrid_mixedtimescale surfacemapframes tiling turbinlet vertical_profiles
volumesources vtkframes wallmodels water_thermodynamics""".split()
POISFFT_UNITS = "customfftw3 precisions parameters fft poisfft main".split()
RK_PROCEDURES = """subgrid::sgs_sigma_uniform dynamics::convection dynamics::pressuregrad
momentumdiffusion::momentumdiffusion_nobranch_2ord
momentumdiffusion::momentumdiffusion_4ord_5point scalarboundaries::boundviscosity
arrayutilities::add_3d arrayutilities::set_3d arrayutilities::multiply_3d""".split()
ADVECTION_PROCEDURES = ["momentumadvection::" + n for n in
                        "cdudiv cdvdiv cdwdiv cduadv cdvadv cdwadv cd4divu cd4divv cd4divw".split()]
BOUNDARY_PROCEDURES = ["boundaries::bounduvw_do_bc", "boundaries::periodic_corners_edges"]


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n")


def deterministic_reference(text):
    """Approved change: serial TOP/BOTTOM plane sum, within the four-thread team."""
    old = """        !$omp do reduction(+:Phi_ref)
        do j = 1, Prny
          do i = 1, Prnx
            Phi_ref = Phi_ref + Pr(i,j,k_ref)
          end do
        end do
        !$omp end do

        !$omp single
        Phi_ref = Phi_ref / (Prnx*Prny)
        !$omp end single"""
    new = """        !$omp master
        do j = 1, Prny
          do i = 1, Prnx
            Phi_ref = Phi_ref + Pr(i,j,k_ref)
          end do
        end do
        Phi_ref = Phi_ref / (Prnx*Prny)
        !$omp end master
        !$omp barrier"""
    if text.count(new) == 1 and old not in text:
        return text
    if text.count(old) != 1:
        raise ValueError("Unsupported pressure-reference source; expected one non-MPI TOP/BOTTOM sum")
    return text.replace(old, new)


def snapshot_sources(elmm, poisfft, destination):
    records = []
    for name, root in (("elmm", elmm), ("PoisFFT", poisfft)):
        source = Path(root).resolve(strict=True) / "src"
        target = destination / name / "src"
        target.mkdir(parents=True)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".f90", ".f", ".c", ".h", ".cc", ".cpp"}:
                continue
            output = target / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            before = identity(path)
            shutil.copyfile(path, output)
            if identity(path) != before or identity(output)["sha256"] != before["sha256"]:
                raise ValueError("Source changed during snapshot: " + str(path))
            records.append({"original": before, "snapshot": identity(output)})
    pressure = destination / "elmm/src/pressure.f90"
    before = identity(pressure)
    pressure.write_text(deterministic_reference(pressure.read_text()))
    write(destination / "source-inputs.json", {"sources": records,
          "approved_pressure_reference": {"before": before, "after": identity(pressure)}})
    return destination / "elmm", destination / "PoisFFT"


def prepare(elmm, poisfft, compiler, directory, *, fortran, python):
    compiler = Path(compiler).resolve(strict=True)
    if not (compiler / "compiler/__main__.py").is_file():
        raise ValueError("Expected a compiler checkout containing compiler/__main__.py")
    sys.path.insert(0, str(compiler))
    from compiler.scopes.resident_entries import export_resident_entries
    from .resident_rk import (configured_sources, pointwise_advection, snapshot_filter,
                             guard_periodic_entries, value_resident_geometry, generate_entries,
                             materialize_entries, integrate_rk_epoch)
    from .gpu_hot_path import (materialize_pressure_stencils, prepare_pressure_device_entries,
                              materialize_resident_pressure, materialize_resident_pressure_tail)
    from .resident_time_loop import generate_time_entries, integrate_time_loop
    from .resident_walls import generate_wall_entries, integrate_wall_entries
    from .resident_workspace import integrate_workspace

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    elmm, poisfft = snapshot_sources(elmm, poisfft, directory / "application")
    paths = [elmm / "src" / (n + ".f90") for n in ELMM_UNITS]
    paths += [poisfft / "src" / (n + ".f90") for n in POISFFT_UNITS]
    procedures = RK_PROCEDURES + ADVECTION_PROCEDURES + BOUNDARY_PROCEDURES + ["filters::filter3ord"]
    configured = configured_sources(paths, procedures, directory / "configured",
                                    fortran_compiler=fortran, defines=["DPREC"])
    for name, transform in (("advection.f90", pointwise_advection), ("filters.f90", snapshot_filter)):
        record = next(r for r in configured["sources"] if Path(r["original"]["path"]).name == name)
        record["prepared"] = transform(record["prepared"]["path"], directory / (name + "-prepared"))["replacement"]
    write(directory / "configured.json", configured)
    print("Configured sources", len(paths), flush=True)
    extracted = export_resident_entries([r["prepared"]["path"] for r in configured["sources"]],
                                       procedures, directory / "extracted")
    write(directory / "extracted.json", extracted)
    packages = []
    for label, names, expected in (("rk", RK_PROCEDURES, 46),
                                   ("boundary", BOUNDARY_PROCEDURES[:1], 128),
                                   ("periodic", BOUNDARY_PROCEDURES[1:], 20),
                                   ("advection", ADVECTION_PROCEDURES, 15),
                                   ("filter", ["filters::filter3ord"], 4)):
        group = {**extracted, "entries": [e for e in extracted["entries"] if e["procedure"] in names],
                 "rejected": [e for e in extracted["rejected"] if e["procedure"] in names]}
        if label == "periodic":
            group = guard_periodic_entries(group, directory / "periodic-guards")
        group = value_resident_geometry(group, directory / (label + "-values"))
        package = generate_entries(group, compiler, python, directory / (label + "-entries"))
        covered = sum(len(item["region"]["source_loop_spans"]) for item in package["entries"])
        if covered < expected:
            raise ValueError(f"GPU source-loop coverage regressed for {label}: {covered}, minimum {expected}")
        package["source_loops_covered"] = covered
        if label == "boundary":
            # The primary boundary guard excludes unsupported boundary modes.
            # Preserve the rejection ledger, but do not publish inside an
            # unreachable native branch of the serial resident coordinator.
            package["excluded_boundary_rejections"] = package["rejected"]
            package["rejected"] = []
        packages.append(package)
        print("Generated", label, len(package["entries"]), "entries covering", covered, "source nests", flush=True)
    write(directory / "packages.json", packages)
    stencils = materialize_pressure_stencils(elmm / "src/pressure.f90", directory / "pressure-stencils")
    pressure_entries = prepare_pressure_device_entries(stencils, elmm / "src", compiler, python,
                                                       directory / "pressure-entries")
    pressure = materialize_resident_pressure(pressure_entries, elmm / "src/main.f90", directory / "pressure")
    pressure = materialize_resident_pressure_tail(pressure, elmm / "src/boundaries.f90", directory / "pressure-tail")
    numerical = materialize_entries(packages, directory / "numerical")
    rk = integrate_rk_epoch(configured, numerical, pressure, directory / "rk")
    time_entries = generate_time_entries(compiler, python, directory / "time-entries")
    integrated = integrate_time_loop(rk, configured, time_entries, directory / "time")
    wall_entries = generate_wall_entries(integrated, compiler, python, directory / "wall-entries")
    integrated = integrate_wall_entries(integrated, wall_entries, directory / "walls")
    integrated = integrate_workspace(integrated, directory / "integrated")
    entries = [e for p in packages for e in p["entries"]] + pressure_entries["entries"]
    entries += list(time_entries["entries"].values()) + list(wall_entries["entries"].values())
    result = {"version": 1, "elmm": str(elmm), "poisfft": str(poisfft),
              "compiler": str(compiler), "integration": integrated,
              "entry_directories": [e["directory"] for e in entries],
              "packages": str(directory / "packages.json"), "gpu_execution_established": False}
    write(directory / "prepared.json", result)
    print("Prepared", len(entries), "GPU entries", flush=True)
    return result


def diagnostic_main(text, *, resident):
    """Full allocation dumps at their original last observable points."""
    def block(fields, uses=""):
        lines = ["  block", uses, "    integer :: comparison_unit"]
        for label, actual in fields:
            lines += [f"    if (allocated({actual})) then",
                      f"      open(newunit=comparison_unit,file='fields-{label}.bin', &",
                      "           access='stream',form='unformatted',status='new')",
                      f"      write(comparison_unit) {actual}", "      close(comparison_unit)",
                      f"      open(newunit=comparison_unit,file='fields-{label}.shape',status='new')",
                      f"      write(comparison_unit,*) storage_size({actual}), shape({actual}), &",
                      f"           lbound({actual}), ubound({actual})", "      close(comparison_unit)", "    end if"]
        return "\n".join(lines + ["  end block", ""])
    from .gpu_hot_path import _once
    from .resident_time_loop import finalize_before_native
    intermediate = "U2 V2 W2 Ustar Vstar Wstar".split()
    fields = [(n, n) for n in "U V W Pr Viscosity".split()]
    fields += [(n, "comparison_" + n) for n in intermediate]
    fields += [(n, n) for n in "Temperature Moisture Scalar TDiff".split()]
    uses = "\n".join(f"    use Dynamics, only: comparison_{n} => {n}" for n in intermediate)
    anchor = "  call Dynamics_Deallocate"
    _once(text, anchor)
    text = text.replace(anchor, block(fields, uses) + anchor)
    names = "U_avg V_avg W_avg Pr_avg Scalar_avg".split()
    anchor = "  call DeallocateGlobals"
    _once(text, anchor)
    text = text.replace(anchor, block([(n, n) for n in names], "    use OutputData, only: " + ", ".join(names)) + anchor)
    return finalize_before_native(text) if resident else text


def _cmake_path(path):
    value = str(Path(path).resolve())
    if any(c in value for c in ('"', ';', '\n', '$')):
        raise ValueError("Unsupported CMake path: " + value)
    return '"' + value + '"'


def _preserve(path):
    """Keep failed attempts when resuming the same dependency-managed build."""
    if not path.exists():
        return
    index = 1
    while path.with_name(path.name + f".attempt-{index:02}").exists():
        index += 1
    shutil.copyfile(path, path.with_name(path.name + f".attempt-{index:02}"))


def _snapshot_directory(source, target):
    def files(root):
        return {str(p.relative_to(root)): identity(p)["sha256"] for p in root.rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".o", ".mod"}}
    if target.exists():
        if files(source) != files(target):
            raise ValueError("Build dependency changed; prepare a fresh build: " + str(source))
    else:
        shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.o", "*.mod"))


def _run(argv, directory, label, *, timeout):
    record = {"argv": list(map(str, argv)), "cwd": str(directory), "status": "running"}
    path = directory / (label + "-command.json")
    _preserve(path)
    _preserve(directory / (label + ".log"))
    write(path, record)
    start = time.monotonic()
    process = None
    try:
        with (directory / (label + ".log")).open("w") as log:
            process = subprocess.Popen(record["argv"], cwd=directory, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            returncode = process.wait(timeout=timeout)
        record.update(returncode=returncode, status="passed" if returncode == 0 else "failed")
        if returncode:
            raise RuntimeError(f"{label} failed; inspect {directory / (label + '.log')}")
    except BaseException as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        record["seconds"] = time.monotonic() - start
        write(path, record)


def build_prepared(prepared, directory, *, solver, vkfft, fortran, cxx, nvcc, architecture, jobs):
    """Fresh CMake build; neither objects nor modules from earlier builds enter."""
    from .gpu_pressure_build import _check_vkfft
    import re

    if jobs < 1 or not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", architecture):
        raise ValueError("Positive build jobs and explicit CUDA architecture are required")
    for key in ("CFLAGS", "CXXFLAGS", "CUDAFLAGS", "FFLAGS", "FCFLAGS", "F90FLAGS", "LDFLAGS",
                "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
        if os.environ.get(key):
            raise ValueError("Hidden compiler options are not supported: " + key)
    directory = Path(directory).resolve()
    compiler = Path(prepared["compiler"])
    from .poisson_source import snapshot_solver
    vkfft = _check_vkfft(vkfft)
    provider = directory / "provider-source"
    snapshot_solver(solver, provider, directory / "provider-inputs.json")
    runtime = directory / "runtime"
    _snapshot_directory(compiler / "compiler/runtime", runtime)
    adapter = directory / "pressure-adapter"
    _snapshot_directory(Path(__file__).with_name("gpu_pressure_sources"), adapter)
    elmm, poisfft = Path(prepared["elmm"]) / "src", Path(prepared["poisfft"]) / "src"
    overrides = {Path(r["path"]).name: Path(r["path"]) for r in prepared["integration"]["sources"]}
    for record in prepared["integration"]["sources"]:
        if identity(record["path"]) != record:
            raise ValueError("Prepared application source changed")
    common = [poisfft / (n + ".f90") for n in POISFFT_UNITS]
    common += [elmm / n for n in ("lapack.f", "rng_par_zig_c.c", "pthreads.c")]
    native = [elmm / (n + ".f90") for n in ELMM_UNITS if n != "main"] + common
    resident = [overrides.get(n + ".f90", elmm / (n + ".f90")) for n in ELMM_UNITS if n != "main"] + common
    resident += [overrides["resident_time_tasks.f90"]]
    resident += [runtime / n for n in ("scoped_memory.f90", "resident_domain.f90", "reductions.f90",
                                      "scoped_runtime.cu", "reductions.cu", "snapshots.f90", "snapshots.cu")]
    resident += [adapter / n for n in ("gpu_pressure.f90", "resident_pressure_state.f90", "gpu_pressure_tail.f90",
                                      "gpu_pressure.cpp", "gpu_pressure_halo.cu", "gpu_pressure_tail.cu")]
    for entry in prepared["entry_directories"]:
        resident += [Path(entry) / n for n in ("shared_interface.f90", "shared_entry.cu")]
    mains = {"ELMM": overrides["main.f90"], "ELMM-native": elmm / "main.f90"}
    for target, source in list(mains.items()):
        diagnostic = directory / (target + "-full20.f90")
        diagnostic.write_text(diagnostic_main(source.read_text(), resident=target == "ELMM"))
        mains[target + "-full20"] = diagnostic
    cmake = ["cmake_minimum_required(VERSION 4.0)",
             "project(ELMMResident LANGUAGES C CXX Fortran CUDA)",
             "set(CMAKE_EXPORT_COMPILE_COMMANDS ON)",
             "find_package(CUDAToolkit REQUIRED)", "find_package(OpenMP REQUIRED)",
             "set(POISSON_SOLVER_BUILD_FORTRAN ON CACHE BOOL \"\" FORCE)",
             "set(POISSON_SOLVER_USE_MPI OFF CACHE BOOL \"\" FORCE)",
             "set(BUILD_SHARED_LIBS ON CACHE BOOL \"\" FORCE)"]
    for option in ("TESTS", "API_TESTS", "EXAMPLES", "DOCS"):
        cmake.append(f'set(POISSON_SOLVER_BUILD_{option} OFF CACHE BOOL "" FORCE)')
    cmake += ['set(POISSON_SOLVER_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)',
              f'set(FETCHCONTENT_SOURCE_DIR_VKFFT {_cmake_path(vkfft)} CACHE PATH "" FORCE)',
              f"add_subdirectory({_cmake_path(provider)} provider)"]
    for lib in ("fftw3", "fftw3f", "fftw3_omp", "fftw3f_omp", "blas", "lapack"):
        cmake += [f"find_library(LIB_{lib} NAMES {lib} lib{lib}.so.3 lib{lib}.so.0 REQUIRED)"]
    cmake += ["add_library(native_libraries INTERFACE)",
              "target_link_libraries(native_libraries INTERFACE OpenMP::OpenMP_Fortran pthread " +
              " ".join("${LIB_" + n + "}" for n in ("fftw3_omp", "fftw3f_omp", "fftw3", "fftw3f", "blas", "lapack")) + ")"]
    for name, sources in (("native", native), ("resident", resident)):
        cmake += [f"add_library({name}_objects OBJECT\n  " + "\n  ".join(_cmake_path(p) for p in sources) + "\n)",
                  f'file(MAKE_DIRECTORY "${{CMAKE_BINARY_DIR}}/{name}-modules")',
                  f'set_target_properties({name}_objects PROPERTIES Fortran_MODULE_DIRECTORY "${{CMAKE_BINARY_DIR}}/{name}-modules")',
                  f'target_include_directories({name}_objects PUBLIC "${{CMAKE_BINARY_DIR}}/{name}-modules" ' +
                  " ".join(_cmake_path(p) for p in (elmm, poisfft, runtime, adapter)) + ")",
                  f"target_compile_definitions({name}_objects PRIVATE DPREC)",
                  f"target_compile_options({name}_objects PRIVATE "
                  '"$<$<COMPILE_LANGUAGE:Fortran>:-cpp;-O3;-fopenmp;-ffree-line-length-none;-ffp-contract=off;-fbacktrace;-g>" '
                  '"$<$<COMPILE_LANGUAGE:C,CXX>:-O3;-fopenmp;-ffp-contract=off;-g>" '
                  '"$<$<COMPILE_LANGUAGE:CUDA>:-O3;-std=c++17;--fmad=false;--ftz=false;--prec-div=true;--prec-sqrt=true;-Xcompiler=-fPIC,-fopenmp,-ffp-contract=off>" )',
                  f"target_link_libraries({name}_objects PUBLIC native_libraries)"]
        if name == "resident":
            cmake += ["target_link_libraries(resident_objects PUBLIC poisson-solver CUDA::cudart)"]
    cmake += [f"set_source_files_properties({_cmake_path(elmm / 'alternate_ieee_isnan.f90')} PROPERTIES COMPILE_OPTIONS -O0)"]
    for target, source in mains.items():
        objects = "native" if "native" in target else "resident"
        cmake += [f"add_executable({target} {_cmake_path(source)})",
                  f"target_link_libraries({target} PRIVATE {objects}_objects)",
                  f"target_compile_options({target} PRIVATE -cpp -O3 -fopenmp -DDPREC -ffree-line-length-none -ffp-contract=off -fbacktrace -g)",
                  f"set_target_properties({target} PROPERTIES LINKER_LANGUAGE Fortran)"]
    _preserve(directory / "CMakeLists.txt")
    (directory / "CMakeLists.txt").write_text("\n".join(cmake) + "\n")
    inputs = {str(p): identity(p)["sha256"] for p in set(native + resident + list(mains.values()))}
    for root in (runtime, adapter, provider, elmm, poisfft, *map(Path, prepared["entry_directories"])):
        inputs.update({str(p): identity(p)["sha256"] for p in root.rglob("*") if p.is_file()})
    inputs[str(directory / "CMakeLists.txt")] = identity(directory / "CMakeLists.txt")["sha256"]
    _preserve(directory / "build-inputs.json")
    write(directory / "build-inputs.json", inputs)
    output = directory / "build"
    _run(["cmake", "-S", directory, "-B", output, "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release",
          "-DCMAKE_Fortran_COMPILER=" + str(fortran.resolve()), "-DCMAKE_CXX_COMPILER=" + str(cxx.resolve()),
          "-DCMAKE_CUDA_COMPILER=" + str(nvcc.resolve()), "-DCMAKE_CUDA_HOST_COMPILER=" + str(cxx.resolve()),
          "-DCMAKE_CUDA_ARCHITECTURES=" + architecture], directory, "configure", timeout=240)
    _run(["cmake", "--build", output, "--parallel", str(jobs)], directory, "compile", timeout=2400)
    if any(identity(path)["sha256"] != sha for path, sha in inputs.items()):
        raise ValueError("Build source changed during compilation")
    result = {"status": "PASS", "binaries": {n: identity(output / n) for n in mains},
              "provider": identity(output / "lib/libpoisson-solver.so"),
              "gpu_execution_established": False, "performance_qualified": False}
    write(directory / "built.json", result)
    print("Built native and resident executables:", output, flush=True)
    return result


def add_arguments(parser):
    parser.add_argument("--elmm", type=Path, required=True)
    parser.add_argument("--poisfft", type=Path, required=True)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fortran", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--build-prepared", action="store_true")
    parser.add_argument("--solver", type=Path)
    parser.add_argument("--vkfft", type=Path)
    parser.add_argument("--cxx", type=Path)
    parser.add_argument("--nvcc", type=Path)
    parser.add_argument("--architecture")
    parser.add_argument("--jobs", type=int, default=2)


def execute(args):
    if args.jobs < 1:
        raise ValueError("Build jobs must be positive")
    if args.prepare_only and args.build_prepared:
        raise ValueError("Choose preparation or building prepared sources")
    if not args.prepare_only:
        if any(getattr(args, key) is None for key in ("solver", "vkfft", "cxx", "nvcc", "architecture")):
            raise ValueError("Building requires --solver, --vkfft, --cxx, --nvcc and --architecture")
    if args.build_prepared:
        prepared = json.loads((args.output / "prepared.json").read_text())
    else:
        prepared = prepare(args.elmm, args.poisfft, args.compiler.resolve(strict=True), args.output,
                           fortran=args.fortran, python=Path(sys.executable))
    if not args.prepare_only:
        build_prepared(prepared, args.output, solver=args.solver, vkfft=args.vkfft,
                       fortran=args.fortran, cxx=args.cxx, nvcc=args.nvcc,
                       architecture=args.architecture, jobs=args.jobs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
