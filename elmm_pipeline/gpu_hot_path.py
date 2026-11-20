"""ELMM's pressure step inside the resident epoch.

The stencil loops of ``PrePoisson`` and ``PostPoisson`` are ordinary compiled
entries. This module adds what is specific to ELMM around them: the device
Poisson solver in place of the host one, and the solver's two scratch arrays
as handles the domain keeps. The pressure reference and the pressure halo are
compiled from ELMM's own procedures, like the stencils.
"""
from __future__ import annotations

import re

from .ast_rewrites import find_call_span, find_routine_span, parse_fortran, replace_once

# The solver's scratch arrays are bound once per pressure step and passed to entries by handle.
HELD = {("pressure::prepoisson", "rhs"): "resident_flow%rhs", ("pressure::postpoisson", "phi"): "resident_flow%phi"}


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
        # The reference at a point is interpolated by host statements; the other references are entries.
        + "      if (resident_pressure%active .and. pressure_solution%reference%type == PRESSURE_REFERENCE_POINT) &\n"
        + "        call resident_pressure%publish()\n"
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
