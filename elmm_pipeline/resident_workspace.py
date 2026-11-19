"""Give checked RK constants and private snapshots their proved owner lifetime."""
import hashlib
import re

from .ast_rewrites import (
    find_routine_span,
    replace_once,
)


def integrate_workspace(sources):
    """Give the RK constants and the filter snapshot storage that outlives one call.

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

    text = sources["dynamics.f90"]
    release_block = (
        "    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
        "      call resident_flow%release_real(beta)\n      call resident_flow%release_real(rho)\n    end block\n"
    )
    sources["dynamics.f90"] = replace_once(text, release_block, "")

    text = sources["filters.f90"]
    lo, hi = find_routine_span(text, "Filter3ord")
    body = text[lo:hi]
    header_match = re.match(r"^[ \t]*subroutine\s+Filter3ord\s*\([^)]*\)[ \t]*\n", body, re.IGNORECASE)
    if header_match is None:
        raise ValueError("Missing Filter3ord header")
    site = int.from_bytes(hashlib.sha256(b"ELMM::Filters::Filter3ord::complete_snapshot::v1").digest()[:8], "big") & ((1 << 63) - 1)
    body = (
        body[: header_match.end()]
        + "    use, intrinsic :: iso_c_binding\n"
        "    use elmm_resident_pressure_state, only: resident_flow\n"
        + body[header_match.end() :]
    )
    tmp_decl_replacement = (
        "real(knd), pointer, contiguous :: tmp(:,:,:)\n"
        f"    integer(c_int64_t), parameter :: work_site={site}_c_int64_t\n"
        "    logical :: managed_workspace"
    )
    try:
        body = replace_once(
            body,
            "REAL(KIND = knd), ALLOCATABLE, TARGET :: tmp(:, :, :)",
            tmp_decl_replacement,
        )
    except ValueError:
        body, n_sub = re.subn(
            r"(?im)^[ \t]*real\b[^\n]*\ballocatable\b[^\n]*::[ \t]*tmp\s*\(\s*:\s*,\s*:\s*,\s*:\s*\)",
            "    " + tmp_decl_replacement,
            body,
            count=1,
        )
        if n_sub != 1:
            raise ValueError("Missing 3D allocatable tmp declaration in Filter3ord")
    body = replace_once(
        body,
        "    allocate(tmp(-2:ubound(U,1),-2:ubound(U,2),-2:ubound(U,3)))",
        """    managed_workspace=resident_flow%active
    if (managed_workspace) then
      call resident_flow%acquire_work_real_3(work_site,lbound(U),ubound(U),tmp)
    else
      allocate(tmp(-2:ubound(U,1),-2:ubound(U,2),-2:ubound(U,3)))
    endif""",
    )
    bind_pat = re.compile(
        r"(?im)rk_handle_tmp\s*=\s*resident_flow%bind_real(?:64|32)?\s*\(\s*tmp\s*,\s*lbound\s*\(\s*tmp\s*\)\s*,\s*\.true\.\s*,\s*&\s*\n\s*origin='filters::filter3ord::tmp'\s*\)"
    )
    body, count = bind_pat.subn(
        "rk_handle_tmp=resident_flow%bind_real(tmp,lbound(tmp),.false.,persistent=.true., &\n"
        "          origin='filters::filter3ord::tmp')",
        body,
    )
    if count != 4:
        raise ValueError("Unexpected snapshot bindings: " + str(count))
    body = replace_once(
        body,
        "    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
        "      call resident_flow%release_real(tmp)\n    end block\n",
        """    if (managed_workspace) then
      call resident_flow%release_work_real_3(work_site,tmp)
    else
      deallocate(tmp)
    endif
""",
    )
    sources["filters.f90"] = text[:lo] + body + text[hi:]
