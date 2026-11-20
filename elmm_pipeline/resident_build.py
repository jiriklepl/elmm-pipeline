"""Build the primary ELMM GPU path from source, without historical artifacts.

One supported configuration: GNU Fortran, double precision, OpenMP, one CUDA
device, and the checked uniform-channel adapter. Other physical configurations
take the original native branches. Placement/transfer policy switches are not
part of this command. Numerical extraction and coherence belong to the compiler.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil

from .ast_rewrites import parse_fortran, replace_once


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

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
BOUNDARY_PROCEDURES = ["boundaries::bounduvw_do_bc"]


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
    if text.count(old) == 1:
        return text.replace(old, new)

    already_pat = re.compile(
        r"(?im)^[ \t]*!\$omp\s+master\b[^\n]*\n"
        r"[ \t]*do\s+\w+\s*=\s*1\s*,\s*\w+\b[^\n]*\n"
        r"[ \t]*do\s+\w+\s*=\s*1\s*,\s*\w+\b[^\n]*\n"
        r"[ \t]*(\w+)\s*=\s*\1\s*\+\s*\w+\s*\([^\n]+\)[ \t]*\n"
        r"[ \t]*end\s*do\b[^\n]*\n"
        r"[ \t]*end\s*do\b[^\n]*\n"
        r"[ \t]*\1\s*=\s*\1\s*/[^\n]+\n"
        r"[ \t]*!\$omp\s+end\s+master\b[^\n]*\n"
        r"[ \t]*!\$omp\s+barrier\b[^\n]*"
    )
    reduction_pat = re.compile(
        r"(?im)^([ \t]*)!\$omp\s+do\s+reduction\s*\(\s*\+\s*:\s*(\w+)\s*\)[ \t]*(?:![^\n]*)?\n"
        r"([ \t]*)do\s+(\w+)\s*=\s*1\s*,\s*(\w+)[ \t]*(?:![^\n]*)?\n"
        r"([ \t]*)do\s+(\w+)\s*=\s*1\s*,\s*(\w+)[ \t]*(?:![^\n]*)?\n"
        r"([ \t]*)\2\s*=\s*\2\s*\+\s*(\w+)\s*\(\s*\7\s*,\s*\4\s*,\s*(\w+)\s*\)[ \t]*(?:![^\n]*)?\n"
        r"([ \t]*)end\s*do[ \t]*(?:![^\n]*)?\n"
        r"([ \t]*)end\s*do[ \t]*(?:![^\n]*)?\n"
        r"[ \t]*!\$omp\s+end\s+do[ \t]*(?:![^\n]*)?\n(?:\s*(?:![^\n]*)?\n)*"
        r"[ \t]*!\$omp\s+single[ \t]*(?:![^\n]*)?\n"
        r"([ \t]*)\2\s*=\s*\2\s*/\s*(?:\(\s*\8\s*\*\s*\5\s*\)|\(\s*\5\s*\*\s*\8\s*\))[ \t]*(?:![^\n]*)?\n"
        r"[ \t]*!\$omp\s+end\s+single\b[^\n]*"
    )
    matches = list(reduction_pat.finditer(text))
    if not matches and len(already_pat.findall(text)) == 1:
        return text
    if len(matches) != 1:
        raise ValueError("Unsupported pressure-reference source; expected one non-MPI TOP/BOTTOM sum")
    m = matches[0]
    indent = m.group(1)
    var, jvar, jmax, ivar, imax, arr, kref = (
        m.group(2), m.group(4), m.group(5), m.group(7), m.group(8), m.group(10), m.group(11)
    )
    replacement = (
        f"{indent}!$omp master\n"
        f"{indent}do {jvar} = 1, {jmax}\n"
        f"{indent}  do {ivar} = 1, {imax}\n"
        f"{indent}    {var} = {var} + {arr}({ivar},{jvar},{kref})\n"
        f"{indent}  end do\n"
        f"{indent}end do\n"
        f"{indent}{var} = {var} / ({imax}*{jmax})\n"
        f"{indent}!$omp end master\n"
        f"{indent}!$omp barrier"
    )
    return text[: m.start()] + replacement + text[m.end() :]


def _scons_disabled_branch(test_node):
    """Return True if an SConstruct ``if`` test belongs to MPI, CGAL, or missing_pfft."""
    names = {node.id for node in ast.walk(test_node) if isinstance(node, ast.Name)}
    if names & {"mpi", "cgal", "missing_pfft", "ompp"}:
        # Check for negated condition like ``cray != 'yes'`` (which is enabled).
        for node in ast.walk(test_node):
            if isinstance(node, ast.Compare) and any(isinstance(op, ast.NotEq) for op in node.ops):
                return False
        return True
    return False


# The halo copies of a periodic direction do not overlap from three cells on (ResidentRKSupported).
BOUNDARY_MINIMUM = " .and. ".join(f"{component}n{axis} >= 3" for component in "UVW" for axis in "xyz")
# The boundary types of the configuration that keeps its fields on the device (ResidentRKSupported).
# With them decided, the six faces of a boundary application are one entry instead of six.
CHANNEL_BOUNDARIES = (*(f"Btype({side}) == BC_PERIODIC" for side in ("We", "Ea", "So", "No")),
                      "Btype(Bo) == BC_NOSLIP", "Btype(To) == BC_NOSLIP")
# ELMM's primary-grid scalars that resident entries take by VALUE.
GRID_GEOMETRY = tuple("parameters::" + name for name in (
    "dxmin", "dymin", "dzmin", "unx", "uny", "unz", "vnx", "vny", "vnz", "wnx", "wny", "wnz", "prnx", "prny", "prnz"))


def discover_application_units(elmm_src: Path, poisfft_src: Path | None = None) -> tuple[list[str], list[str]]:
    """Discover active Fortran build units from ``SConstruct`` or directory scan."""
    elmm_src = Path(elmm_src).resolve(strict=True)
    sconstruct = elmm_src / "SConstruct"
    elmm_units: list[str] = []
    poisfft_units: list[str] = []

    if sconstruct.is_file():
        tree = ast.parse(sconstruct.read_text(encoding="utf-8"))

        def collect_objects(stmts):
            for stmt in stmts:
                if isinstance(stmt, ast.If):
                    if _scons_disabled_branch(stmt.test):
                        collect_objects(stmt.orelse)
                    else:
                        collect_objects(stmt.body)
                        collect_objects(stmt.orelse)
                    continue
                for node in ast.walk(stmt):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "Object"
                    ):
                        src_val = None
                        for kw in node.keywords:
                            if kw.arg == "source" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                                src_val = kw.value.value
                        if src_val and src_val.lower().endswith(".f90"):
                            p = Path(src_val)
                            stem = p.stem
                            if "PoisFFT" in p.parts:
                                if stem not in poisfft_units:
                                    poisfft_units.append(stem)
                            elif stem not in elmm_units:
                                elmm_units.append(stem)

        collect_objects(tree.body)

    if not elmm_units:
        excluded_suffixes = ("-inc.f90", "-inc-def.f90", "-inc-proc.f90", "_mpi.f90")
        excluded_names = {
            "cgal_polyhedra.f90",
            "frames_parallelio.f90",
            "custom_mpi.f90",
            "custom_ieee_isnan.f90",
            "customvolscalsource.f90",
        }
        for path in sorted(elmm_src.glob("*.f90")):
            low = path.name.lower()
            if (
                low in excluded_names
                or "-inc" in low
                or low.endswith(excluded_suffixes)
                or low.startswith(("domains_bc_", "exchange_mpi"))
            ):
                continue
            if path.stem not in elmm_units:
                elmm_units.append(path.stem)

    if not poisfft_units and poisfft_src is not None and Path(poisfft_src).is_dir():
        for path in sorted(Path(poisfft_src).glob("*.f90")):
            low = path.name.lower()
            if low.endswith("-inc.f90") or low in {"custompfft.f90", "loop_nest_3d.f90"}:
                continue
            if path.stem not in poisfft_units:
                poisfft_units.append(path.stem)

    default_poisfft = list(POISFFT_UNITS) if (sconstruct.is_file() or poisfft_src is not None) else []
    return elmm_units or list(ELMM_UNITS), poisfft_units or default_poisfft


def discover_resident_procedure_groups(paths: list[Path]) -> dict[str, list[str]]:
    """Discover numerical procedure groups from Fortran source modules."""
    from fparser.two.utils import walk

    target_modules = {
        "momentumadvection",
        "filters",
        "boundaries",
        "subgrid",
        "dynamics",
        "momentumdiffusion",
        "scalarboundaries",
        "arrayutilities",
        "timesteps",
    }
    available: dict[str, tuple[bool, set[str]]] = {}
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        mod_matches = re.findall(r"(?im)^\s*module\s+([a-z_][a-z0-9_]*)\b", text)
        if not any(m.lower() in target_modules for m in mod_matches):
            continue
        try:
            tree = parse_fortran(text)
        except Exception:
            continue
        for module in (n for n in walk(tree) if type(n).__name__ == "Module"):
            hdr = next((n for n in module.content if type(n).__name__ == "Module_Stmt"), None)
            if hdr is None:
                continue
            mod_name = str(hdr.items[1]).lower()
            if mod_name not in target_modules:
                continue
            for routine in (n for n in walk(module) if type(n).__name__ == "Subroutine_Subprogram"):
                if type(getattr(routine, "parent", None)).__name__ == "Internal_Subprogram_Part":
                    continue
                stmt = next((n for n in routine.content if type(n).__name__ == "Subroutine_Stmt"), None)
                if stmt is None:
                    continue
                qname = f"{mod_name}::{str(stmt.items[1]).lower()}"
                has_loops = any(type(n).__name__ == "Block_Nonlabel_Do_Construct" for n in walk(routine))
                callees = {
                    str(n.items[0]).lower()
                    for n in walk(routine)
                    if type(n).__name__ == "Call_Stmt" and len(n.items) > 0
                }
                available[qname] = (has_loops, callees)

    advection = [
        name for name, (has_loops, _) in available.items()
        if name.startswith("momentumadvection::") and has_loops
    ] or list(ADVECTION_PROCEDURES)

    filter_procs = [
        name for name in ("filters::filter3ord",) if name in available
    ] or [
        name for name, (has_loops, _) in available.items()
        if name.startswith("filters::") and has_loops
    ]

    boundary = [name for name in ("boundaries::bounduvw_do_bc",) if name in available] or [BOUNDARY_PROCEDURES[0]]

    rk = [name for name in RK_PROCEDURES if name in available]
    rk_modules = {"subgrid", "momentumdiffusion"}
    rk_roots = available.get("timesteps::tmarchrk3", (False, set()))[1] | available.get(
        "dynamics::subgridstresses", (False, set())
    )[1]
    for name, (has_loops, _) in available.items():
        mod, _, sub = name.partition("::")
        if mod in rk_modules and has_loops and name not in rk and sub in rk_roots:
            rk.append(name)
    if not rk:
        rk = list(RK_PROCEDURES)

    return {
        "rk": rk,
        "boundary": boundary,
        "advection": advection,
        "filter": filter_procs,
    }


def snapshot_sources(elmm, poisfft, destination):
    records = []
    for name, root in (("elmm", elmm), ("PoisFFT", poisfft)):
        source = Path(root).resolve(strict=True) / "src"
        target = destination / name / "src"
        target.mkdir(parents=True)
        for path in sorted(source.rglob("*")):
            if not path.is_file() or (
                path.suffix.lower() not in {".f90", ".f", ".c", ".h", ".cc", ".cpp"}
                and path.name != "SConstruct"
            ):
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


# The ELMM sources whose arrays an entry or a residency block binds.
ADDRESSABLE = """parameters arrayutilities subgrid dynamics momentum_diffusion advection scalarboundaries boundaries
tsteps linearforcing volumesources sponge tiling filters wallmodels advection_variable_z water_thermodynamics
scalar_diffusion pressure main outputs frames_common vtkframes frames surfacemapframes""".split()
# Source loop nests that must end up compiled, by selection. Fewer is a coverage regression.
# The divergence check of the pressure selection and the diagnostics are loops that only accumulate
# scalars; each of those counts as its three compiled nests.
MINIMUM_NESTS = {"rk": 82, "boundary": 218, "advection": 15, "filter": 4, "pressure": 35,
                 "diagnostics": 21}
PRESSURE_PROCEDURES = ["pressure::prepoisson", "pressure::postpoisson"]
# What a step reads of the flow without changing it: the CFL number and the mean friction velocities.
DIAGNOSTIC_PROCEDURES = ["dynamics::timesteplength::get_u_dx_max", "wallmodels::groundustar",
                         "wallmodels::groundustarflat"]
# Host statements in the selected procedures that touch arrays the entries keep on the device, as the
# compiler's audit listed them when this file was recorded. A statement that is not listed stops the build.
KNOWN_HOST_ACCESSES = Path(__file__).with_name("host_accesses.json")


def prepare(elmm, poisfft, compiler, directory, *, fortran, precision="dp", audit=True):
    """Write the resident variant of ELMM's sources and compile its entries.

    With ``audit`` off, host statements that are not in the recorded list are
    only written to ``host-accesses.json`` instead of stopping the build.
    """
    from compiler.resident.project import ResidentProject, Selection
    from compiler.resident.project import prepare as prepare_application

    from .gpu_hot_path import HELD, integrate_pressure, integrate_pressure_tail
    from .resident_rk import DOMAIN, integrate_rk_epoch
    from .resident_time_loop import generate_time_entries, integrate_time_loop
    from .resident_walls import generate_wall_entries, integrate_wall_entries, name_ground_lists
    from .resident_workspace import integrate_workspace

    if precision not in {"dp", "sp"}:
        raise ValueError(f"Unsupported precision {precision!r}: expected 'dp' or 'sp'")
    compiler = Path(compiler).resolve(strict=True)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    elmm, poisfft = snapshot_sources(elmm, poisfft, directory / "application")
    source = elmm / "src"
    elmm_units, poisfft_units = discover_application_units(source, poisfft / "src")
    paths = [source / (n + ".f90") for n in elmm_units]
    paths += [poisfft / "src" / (n + ".f90") for n in poisfft_units]
    groups = {**discover_resident_procedure_groups(paths), "pressure": list(PRESSURE_PROCEDURES),
              "diagnostics": list(DIAGNOSTIC_PROCEDURES)}
    kernels = []

    def edit(texts):
        """ELMM's own residency code, added to its sources after the entry calls are in place."""
        owned = {path.name: path for path in texts if path.parent == source}
        sources = {name: texts[path] for name, path in owned.items()}
        integrate_pressure(sources)
        integrate_pressure_tail(sources)
        integrate_rk_epoch(sources)
        time_entries = generate_time_entries(directory / "time-entries", precision=precision)
        integrate_time_loop(sources, time_entries, precision=precision)
        wall_entries = generate_wall_entries(sources, directory / "wall-entries", precision=precision)
        integrate_wall_entries(sources, wall_entries, precision=precision)
        integrate_workspace(sources)
        kernels.extend(item["directory"] for entries in (time_entries, wall_entries)
                       for item in entries["entries"].values())
        return {owned.get(name, name): text for name, text in sources.items()}

    selections = tuple(
        Selection(label, tuple(groups[label]), guard=BOUNDARY_MINIMUM if label == "boundary" else None,
                  # The primary-grid counts and spacings are set before the epoch is entered.
                  value_scalars=GRID_GEOMETRY, minimum_nests=MINIMUM_NESTS[label],
                  # The activation condition excludes the boundary modes whose nests are rejected.
                  native_rejections=label != "boundary",
                  assume=CHANNEL_BOUNDARIES if label == "boundary" else (),
                  # The corner and edge copies of the three components then stand next to the faces.
                  inline=("boundaries::periodic_corners_edges",) if label == "boundary" else ())
        for label in MINIMUM_NESTS)
    project = ResidentProject(
        sources=tuple(paths), selections=selections, defines=("DPREC",) if precision == "dp" else (), domain=DOMAIN,
        rewrites={"advection.f90": ("tile_scratch",), "filters.f90": ("line_snapshot",),
                  "wallmodels.f90": (name_ground_lists,)},
        edit=edit, held=HELD,
        addressable=tuple(source / (name + ".f90") for name in ADDRESSABLE),
        known_host_accesses=frozenset(item["id"] for item in json.loads(KNOWN_HOST_ACCESSES.read_text())),
        strict_audit=audit)
    prepared = prepare_application(project, directory / "resident", fortran=fortran)
    write(directory / "host-accesses.json", [{key: item[key] for key in ("id", "procedure", "arrays", "text")}
                                             for item in prepared["audit"]])
    for row in prepared["selections"]:
        print("Generated", row["label"], row["entries"], "entries covering", row["nests"], "source nests", flush=True)
    # The container runner matches the runtime's trace against the compiler's entry records.
    write(directory / "packages.json", [
        json.loads((directory / "resident" / (label + "-entries") / "device-entries.json").read_text())
        for label in MINIMUM_NESTS])
    result = {"version": 2, "precision": precision, "elmm": str(elmm), "poisfft": str(poisfft),
              "elmm_units": elmm_units, "poisfft_units": poisfft_units, "procedure_groups": groups,
              "compiler": str(compiler),
              "sources": [{**row, "sha256": identity(row["resident"])["sha256"]} for row in prepared["sources"]],
              "entry_directories": prepared["entries"] + kernels,
              "packages": str(directory / "packages.json"), "gpu_execution_established": False}
    write(directory / "prepared.json", result)
    print("Prepared", len(result["entry_directories"]), "GPU entries", flush=True)
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
    from .resident_time_loop import finalize_before_native
    intermediate = "U2 V2 W2 Ustar Vstar Wstar".split()
    fields = [(n, n) for n in "U V W Pr Viscosity".split()]
    fields += [(n, "comparison_" + n) for n in intermediate]
    fields += [(n, n) for n in "Temperature Moisture Scalar TDiff".split()]
    uses = "\n".join(f"    use Dynamics, only: comparison_{n} => {n}" for n in intermediate)
    text = replace_once(text, "  call Dynamics_Deallocate", block(fields, uses), insert=True)
    names = "U_avg V_avg W_avg Pr_avg Scalar_avg".split()
    text = replace_once(
        text,
        "  call DeallocateGlobals",
        block([(n, n) for n in names], "    use OutputData, only: " + ", ".join(names)),
        insert=True,
    )
    return finalize_before_native(text) if resident else text


def build_prepared(prepared, directory, *, solver, vkfft, fortran, cxx, nvcc, architecture, jobs,
                   strict_fp=True):
    """Fresh CMake build; neither objects nor modules from earlier builds enter."""
    from compiler.resident.build import (Executable, ObjectLibrary, cmake_lists, cmake_path,
                                         configure_and_compile, snapshot_directory)
    from compiler.runtime import RESIDENT_SOURCES

    from .poisson_source import check_vkfft, snapshot_solver

    if not re.fullmatch(r"[1-9][0-9]{1,2}[af]?", architecture):
        raise ValueError("An explicit CUDA architecture is required")
    directory = Path(directory).resolve()
    compiler = Path(prepared["compiler"])
    vkfft = check_vkfft(vkfft)
    provider = directory / "provider-source"
    snapshot_solver(solver, provider, directory / "provider-inputs.json")
    runtime = snapshot_directory(compiler / "compiler/runtime", directory / "runtime")
    adapter = snapshot_directory(Path(__file__).with_name("gpu_pressure_sources"), directory / "pressure-adapter")
    elmm, poisfft = Path(prepared["elmm"]) / "src", Path(prepared["poisfft"]) / "src"
    for row in prepared["sources"]:
        if identity(row["resident"])["sha256"] != row["sha256"]:
            raise ValueError("Prepared application source changed: " + row["resident"])
    overrides = {Path(row["original"]).name: Path(row["resident"]) for row in prepared["sources"]
                 if row["original"] is not None and Path(row["original"]).parent == elmm}
    added = [Path(row["resident"]) for row in prepared["sources"] if row["original"] is None]
    common = [poisfft / (n + ".f90") for n in prepared["poisfft_units"]]
    common += [elmm / n for n in ("lapack.f", "rng_par_zig_c.c", "pthreads.c") if (elmm / n).is_file()]
    units = [n for n in prepared["elmm_units"] if n != "main"]
    native = [elmm / (n + ".f90") for n in units] + common
    resident = [overrides[n + ".f90"] for n in units] + common + added
    resident += [runtime / n for n in RESIDENT_SOURCES]
    resident += [adapter / n for n in ("gpu_pressure.f90", "resident_pressure_state.f90", "gpu_pressure_tail.f90",
                                      "gpu_pressure.cpp", "gpu_pressure_halo.cu", "gpu_pressure_tail.cu")]
    for entry in prepared["entry_directories"]:
        resident += [Path(entry) / n for n in ("shared_interface.f90", "shared_entry.cu")]
    mains = {"ELMM": overrides["main.f90"], "ELMM-native": elmm / "main.f90"}
    for target, source in list(mains.items()):
        diagnostic = directory / (target + "-full20.f90")
        diagnostic.write_text(diagnostic_main(source.read_text(), resident=target == "ELMM"))
        mains[target + "-full20"] = diagnostic
    prologue = ['set(POISSON_SOLVER_BUILD_FORTRAN ON CACHE BOOL "" FORCE)',
                'set(POISSON_SOLVER_USE_MPI OFF CACHE BOOL "" FORCE)',
                'set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)']
    prologue += [f'set(POISSON_SOLVER_BUILD_{option} OFF CACHE BOOL "" FORCE)'
                 for option in ("TESTS", "API_TESTS", "EXAMPLES", "DOCS")]
    prologue += ['set(POISSON_SOLVER_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)',
                 f'set(FETCHCONTENT_SOURCE_DIR_VKFFT {cmake_path(vkfft)} CACHE PATH "" FORCE)',
                 f"add_subdirectory({cmake_path(provider)} provider)"]
    libraries = ("fftw3_omp", "fftw3f_omp", "fftw3", "fftw3f", "blas", "lapack")
    prologue += [f"find_library(LIB_{lib} NAMES {lib} lib{lib}.so.3 lib{lib}.so.0 REQUIRED)" for lib in libraries]
    prologue += ["add_library(native_libraries INTERFACE)",
                 "target_link_libraries(native_libraries INTERFACE OpenMP::OpenMP_Fortran pthread "
                 + " ".join("${LIB_" + lib + "}" for lib in libraries) + ")"]
    text = cmake_lists(
        "ELMMResident",
        [ObjectLibrary("native", tuple(native), link=("native_libraries",)),
         ObjectLibrary("resident", tuple(resident), link=("native_libraries", "poisson-solver", "CUDA::cudart"))],
        [Executable(target, source, "native" if "native" in target else "resident") for target, source in mains.items()],
        include_directories=(elmm, poisfft, runtime, adapter),
        definitions=("DPREC",) if prepared["precision"] == "dp" else (),
        fortran_options=("-O3", "-fopenmp", "-ffree-line-length-none", "-fbacktrace", "-g"),
        prologue=prologue, source_options={elmm / "alternate_ieee_isnan.f90": ("-O0",)},
        strict_fp=strict_fp)
    (directory / "CMakeLists.txt").write_text(text)
    inputs = {str(p): identity(p)["sha256"] for p in set(native + resident + list(mains.values()))}
    for root in (runtime, adapter, provider, elmm, poisfft, *map(Path, prepared["entry_directories"])):
        inputs.update({str(p): identity(p)["sha256"] for p in root.rglob("*") if p.is_file()})
    output = configure_and_compile(directory, fortran=fortran, cxx=cxx, nvcc=nvcc, architecture=architecture, jobs=jobs)
    if any(identity(path)["sha256"] != sha for path, sha in inputs.items()):
        raise ValueError("Build source changed during compilation")
    result = {"status": "PASS", "strict_fp": strict_fp, "binaries": {n: identity(output / n) for n in mains},
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
    parser.add_argument("--record-host-accesses", action="store_true",
                        help="List new host accesses in host-accesses.json instead of stopping on them")
    parser.add_argument("--precision", choices=("dp", "sp"), default="dp",
                        help="Floating-point precision: dp (double, default) or sp (single)")
    parser.add_argument("--fast-fp", action="store_true",
                        help="Opt in to FMA contraction and fast CUDA div/sqrt instead of strict no-FMA flags")


def execute(args):
    precision = getattr(args, "precision", "dp")
    strict_fp = not getattr(args, "fast_fp", False)
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
                           fortran=args.fortran, precision=precision, audit=not args.record_host_accesses)
    if not args.prepare_only:
        build_prepared(prepared, args.output, solver=args.solver, vkfft=args.vkfft,
                       fortran=args.fortran, cxx=args.cxx, nvcc=args.nvcc,
                       architecture=args.architecture, jobs=args.jobs, strict_fp=strict_fp)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
