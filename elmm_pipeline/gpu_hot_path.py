"""ELMM's pressure step inside the resident epoch.

The stencil loops of ``PrePoisson`` and ``PostPoisson`` are ordinary compiled
entries. This module adds what is specific to ELMM around them: the device
Poisson solver in place of the host one, the solver's two scratch arrays as
handles the domain keeps, and the pressure reference and face update as one
device task.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from .ast_rewrites import (
    expand_pure_calls_in_text,
    find_call_span,
    find_routine_span,
    find_unique_span,
    inline_pure_scalar_function,
    parse_fortran,
    replace_once,
)

# The solver's scratch arrays are bound once per pressure step and passed to entries by handle.
HELD = {("pressure::prepoisson", "rhs"): "resident_flow%rhs", ("pressure::postpoisson", "phi"): "resident_flow%phi"}


def expand_divergence(operators):
    """A rewrite of pressure.f90 that puts the fourth-order divergence formula where it is called.

    The compiler does not yet inline a PURE function whose array arguments
    have lower bounds, so the one call inside ``PrePoisson`` is expanded here
    from the function's own source.
    """
    info = inline_pure_scalar_function(Path(operators).read_text(), "Divergence4Uniform")

    def rewrite(text):
        lower, upper = find_routine_span(text, "PrePoisson")
        body, count = expand_pure_calls_in_text(text[lower:upper], info)
        if count != 1:
            raise ValueError("Expected one Divergence4Uniform call in PrePoisson")
        body = replace_once(body, "    dt2_rec = 1._knd / dt2", info["constants_source"], insert=True)
        return text[:lower] + body + text[upper:]

    return rewrite


def _insert_after_update_pressure(text: str, addition: str) -> str:
    """Insert ``addition`` immediately after the ``if (update_pressure)`` block."""
    try:
        return replace_once(text, "    end if !update_pressure", addition, after=True, insert=True)
    except ValueError:
        pass
    from fparser.two.utils import walk

    lo, hi = find_routine_span(text, "PressureCorrection")
    body = text[lo:hi]
    tree = parse_fortran(body)
    for node in walk(tree):
        if type(node).__name__ != "If_Construct":
            continue
        then_stmt = next((c for c in node.content if type(c).__name__ == "If_Then_Stmt"), None)
        end_stmt = next((c for c in reversed(node.content) if type(c).__name__ == "End_If_Stmt"), None)
        if then_stmt is None or end_stmt is None:
            continue
        if re.search(r"\bupdate_pressure\b", str(then_stmt), re.IGNORECASE):
            lines = body.splitlines(keepends=True)
            end_line = end_stmt.item.span[1]
            pos = sum(len(l) for l in lines[:end_line])
            updated = body[:pos] + addition.lstrip("\n") + "\n" + body[pos:]
            return text[:lo] + updated + text[hi:]
    raise ValueError("Pressure outlining requires update_pressure block in PressureCorrection")


def _verify_pressure_tail_routines(pressure_text: str, boundaries_text: str) -> tuple[str, str]:
    """Structurally verify FixPressureToReference and Bound_Pr without rigid SHA-256 locks."""
    from fparser.two.utils import walk

    try:
        r_lo, r_hi = find_routine_span(pressure_text, "FixPressureToReference")
        f_lo, f_hi = find_routine_span(boundaries_text, "Bound_Pr")
    except ValueError as error:
        raise ValueError("Missing source routine for pressure tail") from error
    ref_body = pressure_text[r_lo:r_hi]
    face_body = boundaries_text[f_lo:f_hi]
    ref_tree = parse_fortran(ref_body)
    face_tree = parse_fortran(face_body)
    ref_assignments = [
        re.sub(r"\s+", "", str(n)).lower()
        for n in walk(ref_tree)
        if type(n).__name__ == "Assignment_Stmt"
    ]
    ref_names = {str(n).lower() for n in walk(ref_tree) if type(n).__name__ == "Name"}
    if (
        "pr" not in ref_names
        or "pressure_reference_none" not in ref_names
        or "pressure_reference_boundary" not in ref_names
        or not any("pr(i,j,k)=pr(i,j,k)-" in stmt for stmt in ref_assignments)
    ):
        raise ValueError("GPU pressure tail requires the approved reference and original Bound_Pr formulas")
    face_names = {str(n).lower() for n in walk(face_tree) if type(n).__name__ == "Name"}
    if not {"pr", "prbtype", "bc_periodic", "bc_dirichlet"}.issubset(face_names):
        raise ValueError("GPU pressure tail requires the approved reference and original Bound_Pr formulas")
    return (
        hashlib.sha256(ref_body.encode()).hexdigest(),
        hashlib.sha256(face_body.encode()).hexdigest(),
    )


def integrate_pressure(sources):
    """Connect the compiled stencils to the device solver in the original pressure frame.

    ``sources`` maps ELMM file names to their text and is changed in place.
    Borrowed application fields are published before the remaining host
    reference and boundary work when the pressure step runs on its own.
    """
    replacement = sources["pressure.f90"]
    replacement = replace_once(
        replacement,
        "  use Parameters\n",
        "  use, intrinsic :: iso_c_binding\n  use fort_scoped_memory, only: FORT_SCOPE_GPU\n"
        "  use elmm_gpu_pressure, only: elmm_gpu_pressure_config_v1\n"
        "  use elmm_resident_pressure_state, only: resident_pressure => resident_flow, pressure_require\n",
        after=True,
        insert=True,
    )
    replacement = replace_once(
        replacement,
        "  type(pressure_solution_control) :: pressure_solution\n",
        "  public :: FinalizeResidentPressure\n",
        after=True,
        insert=True,
    )
    replacement = replace_once(
        replacement,
        "    call PrePoisson(U,V,W,Q,RHS,dt2,mass_flux)",
        "    call StartResidentPressure(Phi,RHS)\n\n",
        after=False,
        insert=True,
    )
    poiss_lo, poiss_hi, poiss_raw, _ = find_call_span(replacement, "Poiss_PoisFFT", within_routine="PressureCorrection")
    replacement = (
        replacement[:poiss_lo]
        + "          if (resident_pressure%active) then\n            call resident_pressure%solve()\n"
        + "          else\n"
        + poiss_raw
        + "\n          endif"
        + replacement[poiss_hi:]
    )
    bound_lo, bound_hi, bound_raw, _ = find_call_span(replacement, "Bound_Phi", within_routine="PressureCorrection")
    replacement = (
        replacement[:bound_lo]
        + "    if (resident_pressure%active) then\n      call resident_pressure%halo()\n"
        + "    else\n"
        + bound_raw
        + "\n    endif"
        + replacement[bound_hi:]
    )
    res_lo, res_hi, res_raw, _ = find_call_span(replacement, "Poiss_Residue", within_routine="PressureCorrection")
    replacement = (
        replacement[:res_lo]
        + "      if (resident_pressure%active) call resident_pressure%observe_scratch()\n"
        + res_raw
        + replacement[res_hi:]
    )
    fix_lo, fix_hi, fix_raw, _ = find_call_span(replacement, "FixPressureToReference", within_routine="PostPoisson")
    replacement = (
        replacement[:fix_lo]
        + "      if (resident_pressure%active) call resident_pressure%publish()\n"
        + fix_raw
        + replacement[fix_hi:]
    )
    replacement = _insert_after_update_pressure(
        replacement,
        "\n    if (resident_pressure%active .and. .not. resident_pressure%joined) call resident_pressure%publish()",
    )
    replacement = replace_once(
        replacement,
        "  end subroutine PressureCorrection",
        "    if (.not. resident_pressure%joined) resident_pressure%active = .false.\n",
        after=False,
        insert=True,
    )

    state_helpers = '''
  subroutine StartResidentPressure(Phi,RHS)
    real(knd), contiguous, target, intent(in) :: Phi(-1:,-1:,-1:), RHS(0:,0:,0:)
    type(elmm_gpu_pressure_config_v1) :: config
    logical :: supported
    supported = .false.
#if !defined(PAR)
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
    config%solver%lengths = [real(gxmax-gxmin,c_double),real(gymax-gymin,c_double),real(gzmax-gzmin,c_double)]
    config%solver%boundary_conditions = PoissonBtype
    config%solver%approximation = discretization_order
#ifdef DPREC
    config%solver%flags = 0
#else
    config%solver%flags = 1
#endif
    config%solver%device_budget_bytes = 1073741824_c_int64_t
    call resident_pressure%start(config)
    resident_pressure%phi = resident_pressure%bind3(Phi,lbound(Phi),.true.,.true.,origin='pressure::phi')
    resident_pressure%rhs = resident_pressure%bind3(RHS,lbound(RHS),.false.,.true.,origin='pressure::rhs')
  end subroutine

  subroutine FinalizeResidentPressure()
    call resident_pressure%finish()
  end subroutine
'''
    ending = re.search(r"(?im)^\s*end\s*module(?:\s+pressure)?\s*(?:!.*)?$", replacement)
    replacement = replacement[:ending.start()] + state_helpers + replacement[ending.start():]
    sources["pressure.f90"] = replacement
    sources["main.f90"] = replace_once(
        sources["main.f90"],
        "  call DeallocateGlobals",
        "  block\n    use Pressure, only: FinalizeResidentPressure\n"
        "    call FinalizeResidentPressure()\n  end block\n\n",
        after=False,
        insert=True,
    )


def integrate_pressure_tail(sources):
    """Replace the approved reference/Pr-face pair with one external GPU task.

    This explicit numerical adapter is bound to the original source routines.
    Other reference choices retain the original host pair after publication.
    """
    text = sources["pressure.f90"]
    _verify_pressure_tail_routines(text, sources["boundaries.f90"])
    anchor = ("      if (resident_pressure%active) call resident_pressure%publish()\n"
              "      call FixPressureToReference(Pr)\n    \n      call Bound_Pr(Pr)")
    s_span, e_span = find_unique_span(text, anchor)
    text = text[:s_span] + '''      if (ResidentPressureTailSupported()) then
        call ResidentPressureTail(Pr)
      else
        if (resident_pressure%active) call resident_pressure%publish()
        call FixPressureToReference(Pr)
        call Bound_Pr(Pr)
      endif''' + text[e_span:]
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
    use elmm_gpu_pressure_tail, only: elmm_gpu_pressure_tail_real32_v1, elmm_gpu_pressure_tail_v1
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
#ifdef DPREC
    call pressure_require(elmm_gpu_pressure_tail_v1(resident_pressure%context,pressure_handle, &
      int([Prnx,Prny,Prnz],c_int32_t),int(PoissonBtype,c_int32_t),reference,report))
#else
    call pressure_require(elmm_gpu_pressure_tail_real32_v1(resident_pressure%context,pressure_handle, &
      int([Prnx,Prny,Prnz],c_int32_t),int(PoissonBtype,c_int32_t),reference,report))
#endif
  end subroutine
'''
    end = re.search(r"(?im)^\s*end\s*module(?:\s+pressure)?\s*(?:!.*)?$", text)
    text = text[:end.start()] + helpers + text[end.start():]
    sources["pressure.f90"] = text
