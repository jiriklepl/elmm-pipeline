"""Give checked RK constants and private snapshots their proved owner lifetime."""
from pathlib import Path
import hashlib
import json

from .gpu_hot_path import _once, _routine
from .resident_rk import identity


def integrate_workspace(integration, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {Path(r["path"]).name: Path(r["path"]).read_text() for r in integration["sources"]}

    text = sources["tsteps.f90"]
    lo, hi = _routine(text, "TMarchRK3")
    body = text[lo:hi]
    anchor = "    integer :: RK_stage"
    _once(body, anchor)
    body = body.replace(anchor, """    ! Stable caller-owned inputs; original PARAMETER expressions are unchanged.
    real(knd), save, target :: resident_beta(size(RK_beta))=RK_beta
    real(knd), save, target :: resident_rho(size(RK_rho))=RK_rho
""" + anchor)
    anchor = "                      RK_beta, RK_rho, RK_stage, time_stepping%dt)"
    _once(body, anchor)
    body = body.replace(anchor, "                      resident_beta, resident_rho, RK_stage, time_stepping%dt)")
    sources["tsteps.f90"] = text[:lo] + body + text[hi:]

    text = sources["dynamics.f90"]
    anchor = "    block\n      use elmm_resident_pressure_state, only: resident_flow\n" \
             "      call resident_flow%release_real64(beta)\n      call resident_flow%release_real64(rho)\n    end block\n"
    _once(text, anchor)
    sources["dynamics.f90"] = text.replace(anchor, "")

    text = sources["filters.f90"]
    lo, hi = _routine(text, "Filter3ord")
    body = text[lo:hi]
    header = "  subroutine Filter3ord(U, Utype, dir)\n"
    _once(body, header)
    site = int.from_bytes(hashlib.sha256(b"ELMM::Filters::Filter3ord::complete_snapshot::v1").digest()[:8], "big") & ((1 << 63) - 1)
    body = body.replace(header, header + "    use, intrinsic :: iso_c_binding\n"
                        "    use elmm_resident_pressure_state, only: resident_flow\n")
    declaration = "REAL(KIND = knd), ALLOCATABLE, TARGET :: tmp(:, :, :)"
    _once(body, declaration)
    body = body.replace(declaration, "real(knd), pointer, contiguous :: tmp(:,:,:)\n"
                        f"    integer(c_int64_t), parameter :: work_site={site}_c_int64_t\n"
                        "    logical :: managed_workspace")
    anchor = "    allocate(tmp(-2:ubound(U,1),-2:ubound(U,2),-2:ubound(U,3)))"
    _once(body, anchor)
    body = body.replace(anchor, """    managed_workspace=resident_flow%active
    if (managed_workspace) then
      call resident_flow%acquire_work_real64_3(work_site,lbound(U),ubound(U),tmp)
    else
      allocate(tmp(-2:ubound(U,1),-2:ubound(U,2),-2:ubound(U,3)))
    endif""")
    origin = "origin='filters::filter3ord::tmp')"
    anchor = "rk_handle_tmp=resident_flow%bind_real64(tmp,lbound(tmp),.true., &\n          " + origin
    count = body.count(anchor)
    if count != 4:
        raise ValueError("Unexpected snapshot bindings: " + str(count))
    body = body.replace(anchor, "rk_handle_tmp=resident_flow%bind_real64(tmp,lbound(tmp),.false.,persistent=.true., &\n"
              "          " + origin)
    anchor = "    block\n      use elmm_resident_pressure_state, only: resident_flow\n" \
             "      call resident_flow%release_real64(tmp)\n    end block\n"
    _once(body, anchor)
    body = body.replace(anchor, """    if (managed_workspace) then
      call resident_flow%release_work_real64_3(work_site,tmp)
    else
      deallocate(tmp)
    endif
""")
    sources["filters.f90"] = text[:lo] + body + text[hi:]
    records = []
    for name, text in sources.items():
        path = directory / name
        path.write_text(text)
        records.append(identity(path))
    result = {"version": 1, "sources": records, "basis": integration,
              "work_site": site, "snapshot_lifetime": "domain-owned; exclusive source borrow; discard dead values on reuse",
              "coefficient_lifetime": "caller SAVE TARGET arrays initialized by the unchanged RK PARAMETER expressions",
              "gpu_execution_established": False}
    (directory / "workspace-integration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
