"""Source preparation for a resident ELMM timestep.

Keep dispatch, boundary handling and saved state in their original routines,
but expose the pressure stencil bodies as ordinary module procedures. These
are numerical compiler inputs, not hand-written CUDA replacements. Native
execution follows the same branches and evaluates the same arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def _identity(path: Path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _routine(text: str, name: str):
    start = list(re.finditer(r"(?im)^\s*subroutine\s+" + re.escape(name) + r"\s*\(", text))
    end = list(re.finditer(r"(?im)^\s*end\s+subroutine\s+" + re.escape(name) + r"\b[^\n]*", text))
    if len(start) != 1 or len(end) != 1 or start[0].start() >= end[0].start():
        raise ValueError(f"Expected one complete original {name} procedure")
    return start[0].start(), end[0].end()


def _once(text: str, anchor: str):
    if text.count(anchor) != 1:
        raise ValueError(f"Pressure outlining requires one original anchor: {anchor!r}")
    return text.index(anchor)


def _divergence_expression(source: Path):
    """Read the existing scalar PURE stencil instead of duplicating its formula."""
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    tree = ParserFactory().create(std="f2008")(FortranStringReader(source.read_text()))
    matches = []
    for node in walk(tree):
        if type(node).__name__ != "Function_Subprogram":
            continue
        header = next(x for x in node.content if type(x).__name__ == "Function_Stmt")
        if str(header.items[1]).lower() == "divergence4uniform":
            matches.append((node, header))
    if len(matches) != 1:
        raise ValueError("Expected the original scalar Divergence4Uniform function")
    function, header = matches[0]
    if ("pure" not in str(header.items[0]).lower()
            or [str(x).lower() for x in header.items[2].items] !=
            ["i", "j", "k", "u", "v", "w", "dx", "dy", "dz"]):
        raise ValueError("Unsupported Divergence4Uniform signature")
    execution = next(x for x in function.content if type(x).__name__ == "Execution_Part")
    statements = [x for x in execution.content if type(x).__name__ != "Comment"]
    if (len(statements) != 1 or type(statements[0]).__name__ != "Assignment_Stmt"
            or str(statements[0].items[0]).lower() != "divergence4uniform"):
        raise ValueError("Divergence4Uniform must remain one scalar result expression")
    constants = [x for x in walk(function) if type(x).__name__ == "Type_Declaration_Stmt"
                 and "parameter" in str(x.items[1]).lower()]
    names = {str(item.items[0]).lower() for declaration in constants for item in declaration.items[2].items}
    if names != {"c1", "c3"}:
        raise ValueError("Unexpected constants in the original divergence stencil")
    expression = str(statements[0].items[2])
    for name in ("dx", "dy", "dz"):
        expression = re.sub(r"\b" + name + r"\b", name + "min", expression, flags=re.I)
    return expression, "".join("    " + str(x) + "\n" for x in constants)


def materialize_pressure_stencils(source: str | Path, directory: str | Path):
    """Expose the original RHS, velocity correction and pressure update loops.

    The three replacements are calls at the original control positions. The
    moved bodies keep their arithmetic; the PURE divergence expression is
    expanded from its original source. Phi/RHS allocation, solver
    selection, mass flux, pressure reference, diagnostics and halos remain in
    their original frame. All source variants, including FD2/FD4 and Q, remain
    represented; this preparation does not silently specialize the physics.

    GPU placement is a subsequent compiler step. A successful materialization
    alone says nothing about backend coverage or performance.
    """
    source = Path(source).resolve(strict=True)
    original = _identity(source)
    text = source.read_bytes().decode("utf-8")
    operators = source.with_name("numerical_operators.f90")
    operator_identity = _identity(operators)
    divergence, divergence_constants = _divergence_expression(operators)
    helper_names = ("BuildPoissonRHS", "CorrectPoissonVelocity", "UpdatePoissonPressure")
    if any(re.search(r"(?i)\b" + name + r"\b", text) for name in helper_names):
        raise ValueError("Pressure stencil helper names already occur in the input")

    edits, helpers = [], []

    def outline(owner, name, start_anchor, end_anchor, arguments, declarations):
        lower, upper = _routine(text, owner)
        body = text[lower:upper]
        begin = _once(body, start_anchor)
        end = _once(body, end_anchor)
        if begin >= end:
            raise ValueError(f"Reversed stencil outline in {owner}")
        original_body = body[begin:end]
        moved = original_body
        if name == "BuildPoissonRHS":
            actual = "Divergence4Uniform(i, j, k, U, V, W, dxmin, dymin, dzmin)"
            _once(moved, actual)
            moved = moved.replace(actual, "(" + divergence + ")")
            declarations += divergence_constants
        call = "    call " + name + "(" + arguments + ")\n\n"
        helper = ("\n  subroutine " + name + "(" + arguments + ")\n"
                  + declarations + "\n" + moved + "  end subroutine " + name + "\n")
        edits.append({"owner": owner, "procedure": "pressure::" + name.lower(),
                      "start": lower + begin, "end": lower + end,
                      "original_body": original_body, "replacement": call,
                      "body_sha256": hashlib.sha256(original_body.encode()).hexdigest(),
                      "outlined_body_sha256": hashlib.sha256(moved.encode()).hexdigest()})
        helpers.append(helper)

    velocity_inputs = "    real(knd), contiguous, intent(in) :: U(-2:,-2:,-2:), V(-2:,-2:,-2:), W(-2:,-2:,-2:)\n"
    velocity_outputs = velocity_inputs.replace("intent(in)", "intent(inout)")
    outline("PrePoisson", "BuildPoissonRHS",
            "    if (discretization_order == 4) then", "  end subroutine PrePoisson",
            "U, V, W, Q, RHS, dt2_rec",
            velocity_inputs
            + "    real(knd), allocatable, intent(in) :: Q(:,:,:)\n"
            + "    real(knd), contiguous, intent(out) :: RHS(0:,0:,0:)\n"
            + "    real(knd), intent(in) :: dt2_rec\n"
            + "    integer :: i, j, k\n")
    outline("PostPoisson", "CorrectPoissonVelocity",
            "    !$omp parallel private (i,j,k)", "    if (update_pressure) then",
            "U, V, W, Phi, Au, Av, Aw, dt2",
            velocity_outputs
            + "    real(knd), intent(in) :: Phi(-1:,-1:,-1:)\n"
            + "    real(knd), intent(in) :: Au, Av, Aw, dt2\n"
            + "    real(knd), parameter :: C1 = 9._knd / 8, C3 = 1._knd / (8*3)\n"
            + "    integer :: i, j, k\n")
    outline("PostPoisson", "UpdatePoissonPressure",
            "      if (pressure_solution%projection_method==PROJECTION_METHOD_PRESSURE) then",
            "      call FixPressureToReference(Pr)",
            "Pr, Phi, dt3, dxmin2, dymin2, dzmin2",
            "    real(knd), contiguous, intent(inout) :: Pr(-1:,-1:,-1:)\n"
            + "    real(knd), intent(in) :: Phi(-1:,-1:,-1:)\n"
            + "    real(knd), intent(in) :: dt3, dxmin2, dymin2, dzmin2\n"
            + "    integer :: i, j, k\n")

    # Apply disjoint edits against original coordinates, then insert helpers
    # in the same module so all existing USE/host bindings stay available.
    edits.sort(key=lambda edit: edit["start"])
    if any(a["end"] > b["start"] for a, b in zip(edits, edits[1:])):
        raise ValueError("Overlapping pressure stencil outlines")
    replacement = text
    for edit in reversed(edits):
        replacement = replacement[:edit["start"]] + edit["replacement"] + replacement[edit["end"]:]
    endings = list(re.finditer(r"(?im)^\s*end\s+module(?:\s+pressure)?\s*$", replacement))
    if len(endings) != 1:
        raise ValueError("Expected one original pressure module end")
    at = endings[0].start()
    replacement = replacement[:at] + "\n".join(helpers) + "\n" + replacement[at:]
    if _identity(source) != original or _identity(operators) != operator_identity:
        raise ValueError("Pressure source changed during outlining")

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / source.name
    target.write_bytes(replacement.encode("utf-8"))
    record = {"version": 1, "kind": "elmm_resident_pressure_stencil_preparation",
              "original": original, "replacement": _identity(target),
              "procedures": [edit["procedure"] for edit in edits], "edits": edits,
              "saved_state": "unchanged original PressureCorrection frame",
              "body_policy": "original control positions and arithmetic; scalar PURE divergence expression expanded from its source",
              "inlined_function": {"source": operator_identity, "procedure": "numericaloperators::divergence4uniform",
                                   "expression": divergence, "constants": divergence_constants},
              "next_stage": "fresh configured-source export and compiler GPU region construction",
              "gpu_execution_established": False}
    (directory / "pressure-stencils.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def prepare_pressure_device_entries(prepared, project_root, compiler_root, python, directory):
    """Generate shared-domain entries for every pressure stencil branch.

    Fail on an uncovered body instead of treating partial GPU coverage as a
    successful pressure implementation. The caller still has to compose these
    entries with boundary, solver and reference tasks in its residency domain.
    """
    from .discovery import discover
    from .local_adapter import transform

    source = Path(prepared["replacement"]["path"])
    if _identity(source) != prepared["replacement"]:
        raise ValueError("Prepared pressure source changed before device generation")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    names = [name.split("::", 1)[1] for name in prepared["procedures"]]
    discovery = discover(source, Path(project_root), Path(python), directory / "discovery",
                         select=names, backend="gpu", compiler_root=Path(compiler_root),
                         compiler_python=Path(python))
    if discovery["rejected"] or {r["routine"].lower() for r in discovery["regions"]} != set(names):
        raise ValueError("Incomplete pressure GPU coverage; see " + str(directory / "discovery/discovery.json"))
    entries = []
    for region in discovery["regions"]:
        output = directory / "entries" / region["id"]
        generated = transform(Path(region["input_path"]), Path(compiler_root), Path(python), output,
                              entry=region["entry"], context=region["context"],
                              gpu_options={"gpu_policy": "sections", "memory_model": "scoped",
                                           "scope_transfers": "direct", "host_threads": 4})
        entries.append({"region": region, "directory": str(output), "entry": generated["shared_entry"],
                        "compiler_report": str(output / "compiler-report.json")})
    record = {"version": 1, "kind": "elmm_pressure_shared_device_entries",
              "prepared_source": prepared, "entries": entries,
              "execution_contract": "one caller-owned domain; canonical handles passed to shared entries",
              "integration_remaining": ["velocity boundaries and mass flux", "poisson-solver task",
                                        "Phi halos", "pressure reference and pressure halos"],
              "gpu_execution_established": False}
    (directory / "pressure-device-entries.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def materialize_resident_pressure(entries, main_source, directory):
    """Wire compiled stencils and the device solver into the original frame.

    This integration slice owns a persistent pressure domain, with saved scratch
    and solver plans. Borrowed application fields are published and unregistered
    before the remaining host reference/boundary work and unmanaged return.
    It deliberately does not claim timestep-wide residency yet.
    """
    prepared = entries["prepared_source"]
    source = Path(prepared["replacement"]["path"])
    if _identity(source) != prepared["replacement"]:
        raise ValueError("Pressure input changed after numerical generation")
    original = source.read_text()
    lines = original.splitlines(keepends=True)
    gpu_lines = list(lines)
    uses = []
    spans = []
    for item in sorted(entries["entries"], key=lambda item: item["region"]["start_line"], reverse=True):
        region, entry = item["region"], item["entry"]
        start, end = region["start_line"] - 1, region["end_line"]
        body = "".join(lines[start:end])
        if hashlib.sha256(body.encode()).hexdigest() != region["source_span_sha256"]:
            raise ValueError("Pressure numerical source span changed: " + region["id"])
        spans.append((start, end))
        alias = "pressure_gpu_" + region["id"]
        uses.append("  use " + entry["fortran_module"] + ", only: " + alias + " => run\n")
        actual = dict(zip(region["args"], region["actual_args"], strict=True))
        arrays = {parameter["name"]: parameter for parameter in entry["array_parameters"]}
        bindings = []
        for name, parameter in arrays.items():
            if name.lower() in ("phi", "rhs"):
                actual[name] = "resident_pressure%" + name.lower()
            else:
                handle = "resident_" + name.lower()
                origin = region["module"].lower() + "::" + region["routine"].lower() + "::" + name.lower()
                bindings.append("      " + handle + " = resident_pressure%bind" + str(parameter["rank"])
                                + "(" + actual[name] + ", lbound(" + actual[name] + "), .true., &\n"
                                + "        origin='" + origin + "')\n")
                actual[name] = handle
        args = ["resident_pressure%context", "FORT_SCOPE_GPU"] + [actual[name] for name in entry["argument_order"][2:]]
        handles = ["resident_" + name.lower() for name in arrays if name.lower() not in ("phi", "rhs")]
        replacement = "    block\n"
        if handles:
            replacement += "      integer(c_int64_t) :: " + ", ".join(handles) + "\n"
        replacement += "".join(bindings)
        replacement += "      call pressure_require(" + alias + "( &\n        " + ", &\n        ".join(args) + "))\n    end block\n"
        gpu_lines[start:end] = [replacement]
    spans.sort()
    if any(a[1] > b[0] for a, b in zip(spans, spans[1:])):
        raise ValueError("Overlapping pressure numerical entries")
    gpu_text = "".join(gpu_lines)

    replacement = original
    helpers = []
    for name in ("BuildPoissonRHS", "CorrectPoissonVelocity", "UpdatePoissonPressure"):
        lo, hi = _routine(gpu_text, name)
        body = gpu_text[lo:hi]
        body = re.sub(r"(?im)^\s*!\$omp[^\n]*\n", "\n", body)
        body = re.sub(r"\b" + name + r"\b", name + "GPU", body, flags=re.I)
        helpers.append(body)
        lo, hi = _routine(replacement, name)
        native = replacement[lo:hi]
        signature = re.search(r"(?i)subroutine\s+" + name + r"\(([^)]*)\)", native)
        anchor = {"BuildPoissonRHS": "    if (discretization_order == 4) then",
                  "CorrectPoissonVelocity": "    !$omp parallel private (i,j,k)",
                  "UpdatePoissonPressure": "      if (pressure_solution%projection_method==PROJECTION_METHOD_PRESSURE) then"}[name]
        at = _once(native, anchor)
        dispatch = ("    if (resident_pressure%active) then\n      call " + name + "GPU(" + signature[1]
                    + ")\n      return\n    endif\n\n")
        replacement = replacement[:lo] + native[:at] + dispatch + native[at:] + replacement[hi:]

    replacement = replacement.replace("  use Parameters\n", "  use Parameters\n"
        "  use, intrinsic :: iso_c_binding\n  use fort_scoped_memory, only: FORT_SCOPE_GPU\n"
        "  use elmm_gpu_pressure, only: elmm_gpu_pressure_config_v1\n"
        "  use elmm_resident_pressure_state, only: resident_pressure => resident_flow, pressure_require\n" + "".join(uses), 1)
    replacement = replacement.replace("  type(pressure_solution_control) :: pressure_solution\n",
        "  type(pressure_solution_control) :: pressure_solution\n"
        "  public :: FinalizeResidentPressure\n", 1)
    replacement = replacement.replace("    call PrePoisson(U,V,W,Q,RHS,dt2,mass_flux)",
        "    call StartResidentPressure(Phi,RHS)\n\n    call PrePoisson(U,V,W,Q,RHS,dt2,mass_flux)", 1)
    replacement = replacement.replace("          call Poiss_PoisFFT(Phi,RHS)",
        "          if (resident_pressure%active) then\n            call resident_pressure%solve()\n"
        "          else\n            call Poiss_PoisFFT(Phi,RHS)\n          endif", 1)
    replacement = replacement.replace("    call Bound_Phi(Phi)",
        "    if (resident_pressure%active) then\n      call resident_pressure%halo()\n"
        "    else\n      call Bound_Phi(Phi)\n    endif", 1)
    replacement = replacement.replace("      call Poiss_Residue(Phi, RHS, res)",
        "      if (resident_pressure%active) call resident_pressure%observe_scratch()\n"
        "      call Poiss_Residue(Phi, RHS, res)", 1)
    replacement = replacement.replace("      call FixPressureToReference(Pr)",
        "      if (resident_pressure%active) call resident_pressure%publish()\n"
        "      call FixPressureToReference(Pr)", 1)
    replacement = replacement.replace("    end if !update_pressure", "    end if !update_pressure\n"
        "    if (resident_pressure%active .and. .not. resident_pressure%joined) call resident_pressure%publish()", 1)
    replacement = replacement.replace("  end subroutine PressureCorrection", "    if (.not. resident_pressure%joined) resident_pressure%active = .false.\n"
        "  end subroutine PressureCorrection", 1)

    state_helpers = '''
  subroutine StartResidentPressure(Phi,RHS)
    real(knd), contiguous, target, intent(in) :: Phi(-1:,-1:,-1:), RHS(0:,0:,0:)
    type(elmm_gpu_pressure_config_v1) :: config
    logical :: supported
    supported = .false.
#if defined(DPREC) && !defined(PAR)
    supported = gridtype == GRID_UNIFORM .and. explicit_diffusion .and. &
                pressure_solution%poisson_solver == POISSON_SOLVER_POISFFT .and. &
                (discretization_order == 2 .or. discretization_order == 4) .and. &
                min(Prnx,Prny,Prnz) >= 2
    supported = supported .and. all(PoissonBtype == 0 .or. PoissonBtype == 4)
    supported = supported .and. all(PoissonBtype(1:6:2) == PoissonBtype(2:6:2))
#endif
    if (.not. supported) then
      call resident_pressure%finish()
      return
    endif
    config%solver%sizes = [Prnx,Prny,Prnz]
    config%solver%lengths = [gxmax-gxmin,gymax-gymin,gzmax-gzmin]
    config%solver%boundary_conditions = PoissonBtype
    config%solver%approximation = discretization_order
    config%solver%device_budget_bytes = 1073741824_c_int64_t
    call resident_pressure%start(config)
    resident_pressure%phi = resident_pressure%bind3(Phi,lbound(Phi),.true.,.true.,origin='pressure::phi')
    resident_pressure%rhs = resident_pressure%bind3(RHS,lbound(RHS),.false.,.true.,origin='pressure::rhs')
  end subroutine

  subroutine FinalizeResidentPressure()
    call resident_pressure%finish()
  end subroutine
'''
    ending = re.search(r"(?im)^\s*end\s+module(?:\s+pressure)?\s*$", replacement)
    replacement = replacement[:ending.start()] + "\n".join(helpers) + state_helpers + replacement[ending.start():]
    # C-interoperable addresses must outlive each numerical child. All borrowed
    # registrations end within PressureCorrection, and saved scratch has TARGET.
    names = {"pressurecorrection", "prepoisson", "postpoisson", "buildpoissonrhs", "correctpoissonvelocity",
             "updatepoissonpressure", "buildpoissonrhsgpu", "correctpoissonvelocitygpu", "updatepoissonpressuregpu"}
    for name in names:
        lo, hi = _routine(replacement, name)
        body = replacement[lo:hi]
        body = re.sub(r"(?im)^(\s*real\(knd\)[^\n]*?)(\s*::[^\n]*\([^\n]*:[^\n]*)$",
                      lambda match: match[0] if re.search(r"\btarget\b", match[1], re.I)
                      else match[1].rstrip() + ", target " + match[2], body)
        # Explicit DIMENSION attributes put the colons before '::'.
        body = re.sub(r"(?im)^(\s*real\(knd\)[^\n]*dimension\([^\n]*:[^\n]*?)(\s*::[^\n]*)$",
                      lambda match: match[0] if re.search(r"\btarget\b", match[1], re.I)
                      else match[1].rstrip() + ", target " + match[2], body)
        replacement = replacement[:lo] + body + replacement[hi:]

    main_source = Path(main_source).resolve(strict=True)
    main = main_source.read_text()
    anchor = "  call DeallocateGlobals"
    _once(main, anchor)
    main = main.replace(anchor, "  block\n    use Pressure, only: FinalizeResidentPressure\n"
                        "    call FinalizeResidentPressure()\n  end block\n\n" + anchor)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    pressure_path, main_path = directory / source.name, directory / main_source.name
    pressure_path.write_text(replacement)
    main_path.write_text(main)
    record = {"version": 1, "kind": "resident_pressure_source_integration", "pressure": _identity(pressure_path),
              "main": _identity(main_path), "original_main": _identity(main_source),
              "entries": [{"directory": item["directory"], "entry": item["entry"]["entry"]} for item in entries["entries"]],
              "state_source": _identity(Path(__file__).parent / "gpu_pressure_sources/resident_pressure_state.f90"),
              "host_boundaries": ["PrePoisson velocity boundaries and mass flux", "pressure reference and Bound_Pr",
                                  "optional residue/divergence diagnostics", "return to unmanaged RK stage"],
              "gpu_execution_established": False}
    (directory / "resident-pressure.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def materialize_resident_pressure_tail(integrated, boundaries_source, directory):
    """Replace the approved reference/Pr-face pair with one external GPU task.

    This explicit numerical adapter is bound to the original source routines.
    Other reference choices retain the original host pair after publication.
    """
    pressure = Path(integrated["pressure"]["path"])
    if _identity(pressure) != integrated["pressure"]:
        raise ValueError("Integrated pressure source changed")
    text = pressure.read_text()
    boundaries_source = Path(boundaries_source).resolve(strict=True)

    def routine_sha(source, name):
        start = re.search(r"(?im)^  subroutine " + name + r"\(", source)
        if start is None:
            raise ValueError("Missing source routine for pressure tail: " + name)
        end = re.search(r"(?im)^  end subroutine[^\n]*", source[start.start():])
        if end is None:
            raise ValueError("Unterminated pressure-tail source routine")
        return hashlib.sha256(source[start.start():start.start() + end.end()].encode()).hexdigest()

    reference_sha = routine_sha(text, "FixPressureToReference")
    faces_sha = routine_sha(boundaries_source.read_text(), "Bound_Pr")
    if (reference_sha != "74de3c6757cfba541cf1b58d464596f464af134e982f136a591ac04730eb1a3c"
            or faces_sha != "cceb68be8061dda1ef5f697e5f42e1227492caa3567be06cf241905e65b0922d"):
        raise ValueError("GPU pressure tail requires the approved reference and original Bound_Pr formulas")
    anchor = ("      if (resident_pressure%active) call resident_pressure%publish()\n"
              "      call FixPressureToReference(Pr)\n    \n      call Bound_Pr(Pr)")
    _once(text, anchor)
    text = text.replace(anchor, '''      if (ResidentPressureTailSupported()) then
        call ResidentPressureTail(Pr)
      else
        if (resident_pressure%active) call resident_pressure%publish()
        call FixPressureToReference(Pr)
        call Bound_Pr(Pr)
      endif''')
    helpers = '''
  logical function ResidentPressureTailSupported() result(supported)
    supported = resident_pressure%active .and. min(Prnx,Prny,Prnz) >= 3
    supported = supported .and. (pressure_solution%reference%type == PRESSURE_REFERENCE_NONE .or. &
      (pressure_solution%reference%type == PRESSURE_REFERENCE_BOUNDARY .and. &
       (pressure_solution%reference%boundary_index == Bo .or. pressure_solution%reference%boundary_index == To)))
    ! The solver's periodic/Neumann pair must describe the actual Pr faces.
    supported = supported .and. all((PrBtype == BC_PERIODIC) .eqv. (PoissonBtype == 0))
    supported = supported .and. .not. any(PrBtype == BC_DIRICHLET)
    supported = supported .and. .not. any(PrBtype >= BC_MPI_BOUNDS_MIN .and. PrBtype <= BC_MPI_BOUNDS_MAX)
  end function

  subroutine ResidentPressureTail(Pr)
    use elmm_gpu_pressure_tail, only: elmm_gpu_pressure_tail_v1
    use fort_scoped_memory, only: fort_scope_external_report_v1
    real(knd), contiguous, target, intent(inout) :: Pr(-1:,-1:,-1:)
    integer(c_int64_t) :: pressure_handle
    integer(c_int) :: reference
    type(fort_scope_external_report_v1) :: report
    reference = 0
    if (pressure_solution%reference%type == PRESSURE_REFERENCE_BOUNDARY) then
      reference = 1
      if (pressure_solution%reference%boundary_index == To) reference = 2
    endif
    pressure_handle = resident_pressure%bind3(Pr,lbound(Pr),.true.,origin='pressure::tail::pr')
    call pressure_require(elmm_gpu_pressure_tail_v1(resident_pressure%context,pressure_handle, &
      int([Prnx,Prny,Prnz],c_int32_t),int(PoissonBtype,c_int32_t),reference,report))
  end subroutine
'''
    end = re.search(r"(?im)^\s*end\s+module(?:\s+pressure)?\s*$", text)
    text = text[:end.start()] + helpers + text[end.start():]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / "pressure.f90"
    target.write_text(text)
    record = {**integrated, "pressure": _identity(target), "tail_source_contract": {
        "reference_sha256": reference_sha, "faces_sha256": faces_sha,
        "boundaries": _identity(boundaries_source), "reference": "original serial TOP/BOTTOM order or NONE"},
        "host_boundaries": ["PrePoisson velocity boundaries and mass flux", "unsupported reference modes",
                            "optional residue/divergence diagnostics", "return to unmanaged RK stage"],
        "gpu_execution_established": False}
    (directory / "resident-pressure.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
