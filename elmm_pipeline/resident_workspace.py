"""Storage for the Runge-Kutta constants that outlives one call."""
from .ast_rewrites import find_routine_span, replace_once


def integrate_workspace(sources):
    """Give the RK constants storage that outlives one call, so entries can keep them on the device.

    ``sources`` maps ELMM file names to their text and is changed in place.
    """
    text = sources["tsteps.f90"]
    lo, hi = find_routine_span(text, "TMarchRK3")
    body = text[lo:hi]
    body = replace_once(
        body,
        "    integer :: RK_stage",
        """    ! Stable caller-owned inputs; original PARAMETER expressions are unchanged.
    real(knd), save, target :: resident_beta(size(RK_beta))=RK_beta
    real(knd), save, target :: resident_rho(size(RK_rho))=RK_rho
""",
        after=False,
        insert=True,
    )
    body = replace_once(
        body,
        "                      RK_beta, RK_rho, RK_stage, time_stepping%dt)",
        "                      resident_beta, resident_rho, RK_stage, time_stepping%dt)",
    )
    sources["tsteps.f90"] = text[:lo] + body + text[hi:]
