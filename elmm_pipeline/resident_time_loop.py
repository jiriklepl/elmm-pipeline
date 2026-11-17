"""Checked primary-channel timestep lifetime and numerical source preparation.

This adapter owns the ELMM control region, not coherence. Ordinary compiler
entries implement the per-cell arithmetic; the generic runtime reduces CFL.
Unsupported output work closes the epoch before executing its original source.
Primary VTK output uses bounded immutable snapshots and joins before teardown.
"""
from pathlib import Path
import json

from .ast_rewrites import (
    find_call_span,
    find_routine_span,
    find_unique_span,
    replace_once,
)
from .resident_rk import addressable_arrays, identity


def generate_time_entries(compiler_root, python, directory, *, precision="dp"):
    from .local_adapter import transform

    real_type = "real(8)" if precision == "dp" else "real(4)"
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {
        "cfl": f"""module resident_cfl_source
implicit none
contains
subroutine values(u,v,w,cfl,nx,ny,nz,dx,dy,dz)
{real_type}, intent(in) :: u(:,:,:),v(:,:,:),w(:,:,:)
{real_type}, intent(out) :: cfl(:,:,:)
integer, value, intent(in) :: nx,ny,nz
{real_type}, value, intent(in) :: dx,dy,dz
integer :: i,j,k
{real_type} :: p
do k=1,nz
 do j=1,ny
  do i=1,nx
   p = max(abs(u(i+3,j+3,k+3)),abs(u(i+2,j+3,k+3))) / dx
   p = p + max(abs(v(i+3,j+3,k+3)),abs(v(i+3,j+2,k+3))) / dy
   p = p + max(abs(w(i+3,j+3,k+3)),abs(w(i+3,j+3,k+2))) / dz
   cfl(i,j,k)=p
  enddo
 enddo
enddo
end subroutine
end module
""",
        "average": f"""module resident_average_source
implicit none
contains
subroutine accumulate(source,average,nx,ny,nz,ox,oy,oz,weight)
{real_type}, intent(in) :: source(:,:,:)
{real_type}, intent(inout) :: average(:,:,:)
integer, value, intent(in) :: nx,ny,nz,ox,oy,oz
{real_type}, value, intent(in) :: weight
integer :: i,j,k
do k=1,nz
 do j=1,ny
  do i=1,nx
   average(i,j,k)=average(i,j,k)+source(i+ox,j+oy,k+oz)*weight
  enddo
 enddo
enddo
end subroutine
end module
"""}
    result = {"version": 1, "entries": {}}
    for name, text in sources.items():
        source = directory / (name + ".f90")
        source.write_text(text)
        entry = "resident_cfl_source::values" if name == "cfl" else "resident_average_source::accumulate"
        target = directory / (name + "-entry")
        transformed = transform(source, Path(compiler_root), Path(python), target, entry=entry,
                                gpu_options={"gpu_policy": "sections", "memory_model": "scoped",
                                             "scope_transfers": "direct", "host_threads": 4})
        shared = transformed["shared_entry"]
        if not shared["region_execution"] or not all(r["gpu_available"] for r in shared["region_execution"]):
            raise ValueError("Timestep numerical entry lacks GPU execution: " + name)
        result["entries"][name] = {"source": identity(source), "directory": str(target), "entry": shared}
    (directory / "time-entries.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _tasks(entries, *, precision="dp"):
    cfl, avg = (entries["entries"][n]["entry"] for n in ("cfl", "average"))
    c_real = "c_double" if precision == "dp" else "c_float"
    real_type_c = f"real({c_real})"
    zero_literal = "0._c_double" if precision == "dp" else "0._c_float"
    max_fn = "fort_scope_max_real64_v1" if precision == "dp" else "fort_scope_max_real32_v1"
    def invoke(entry, arguments):
        values = ["resident_flow%context", "FORT_SCOPE_GPU"] + [arguments[n] for n in entry["argument_order"][2:]]
        return "call pressure_require(" + ("cfl_run" if entry is cfl else "average_run") + "( &\n      " + ", &\n      ".join(values) + "))"
    cfl_call = invoke(cfl, dict(u="handles(1)", v="handles(2)", w="handles(3)", cfl="handle",
                                nx="nx", ny="ny", nz="nz", dx="dx", dy="dy", dz="dz"))
    avg_call = invoke(avg, dict(source="source_handle", average="average_handle", nx="size(average,1)",
                                ny="size(average,2)", nz="size(average,3)", ox="offset(1)",
                                oy="offset(2)", oz="offset(3)", weight="weight"))
    return f"""module elmm_resident_time_tasks
  use, intrinsic :: iso_c_binding
  use fort_scoped_memory
  use fort_resident_reductions
  use elmm_resident_pressure_state, only: resident_flow, pressure_require
  use {cfl['fortran_module']}, only: cfl_run=>run
  use {avg['fortran_module']}, only: average_run=>run
  implicit none
  private
  public :: ResidentCFL, ResidentAverage
  {real_type_c}, allocatable, target, save :: cfl_values(:,:,:)
contains
  subroutine ResidentCFL(U,V,W,nx,ny,nz,dx,dy,dz,maximum,nan)
    {real_type_c}, contiguous, target, intent(in) :: U(:,:,:),V(:,:,:),W(:,:,:)
    integer, intent(in) :: nx,ny,nz
    {real_type_c}, intent(in) :: dx,dy,dz
    {real_type_c}, intent(out) :: maximum
    logical, intent(out) :: nan
    integer(c_int64_t) :: handles(3),handle
    integer(c_int32_t) :: any_nan
    type(fort_scope_external_report_v1) :: report
    if (min(nx,ny,nz)<1) error stop 'invalid CFL grid'
    if (.not.allocated(cfl_values)) allocate(cfl_values(nx,ny,nz))
    if (any(shape(cfl_values)/=[nx,ny,nz])) error stop 'CFL storage epoch shape changed'
    if (any(shape(U)<[nx,ny,nz]+3) .or. any(shape(V)<[nx,ny,nz]+3) .or. &
        any(shape(W)<[nx,ny,nz]+3)) error stop 'CFL velocity geometry changed'
    handles(1)=resident_flow%bind_real(U,[-2,-2,-2],.true.,origin='time::cfl::u')
    handles(2)=resident_flow%bind_real(V,[-2,-2,-2],.true.,origin='time::cfl::v')
    handles(3)=resident_flow%bind_real(W,[-2,-2,-2],.true.,origin='time::cfl::w')
    handle=resident_flow%bind_real(cfl_values,[1,1,1],.false.,persistent=.true.,origin='time::cfl::values')
    {cfl_call}
    call pressure_require({max_fn}(resident_flow%context,handle,3_c_int32_t, &
      size(cfl_values,kind=c_size_t),{zero_literal},maximum,any_nan,report))
    nan=any_nan/=0
  end subroutine
  subroutine ResidentAverage(source,average,source_lower,average_lower,offset,weight)
    {real_type_c}, contiguous, target, intent(in) :: source(:,:,:)
    {real_type_c}, contiguous, target, intent(inout) :: average(:,:,:)
    integer, intent(in) :: source_lower(3),average_lower(3),offset(3)
    {real_type_c}, intent(in) :: weight
    integer(c_int64_t) :: source_handle,average_handle
    if (any(offset<0) .or. any(shape(average)+offset>shape(source))) error stop 'invalid average view'
    source_handle=resident_flow%bind_real(source,source_lower,.true.,origin='time::average::source')
    average_handle=resident_flow%bind_real(average,average_lower,.true.,origin='time::average::result')
    {avg_call}
  end subroutine
end module
"""


def finalize_before_native(text):
    """Publish while all fields (including diagnostic RK scratch) still exist."""
    finalizer = "  block\n    use Pressure, only: FinalizeResidentPressure\n    call FinalizeResidentPressure()\n  end block\n"
    if "    use Frames_common, only: FinalizeResidentOutput\n" in text:
        finalizer = finalizer.replace("    use Pressure", "    use Frames_common, only: FinalizeResidentOutput\n    use Pressure")
        finalizer = finalizer.replace("    call FinalizeResidentPressure()", "    call FinalizeResidentOutput()\n    call FinalizeResidentPressure()")
    text = replace_once(text, finalizer, "")
    diagnostic = "  block\n    use Dynamics, only: comparison_U2 => U2"
    anchor = diagnostic if diagnostic in text else "  call Dynamics_Deallocate"
    return replace_once(text, anchor, finalizer, after=False, insert=True)


def integrate_time_loop(integration, configured, entries, directory, *, precision="dp"):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {Path(r["path"]).name: Path(r["path"]).read_text() for r in integration["sources"]}
    originals = {Path(r["original"]["path"]).name: Path(r["prepared"]["path"]) for r in configured["sources"]}
    cut = "\n    if (resident_flow%joined) then\n      call resident_flow%end_epoch()\n      resident_flow%joined=.false.\n    endif\n"
    sources["tsteps.f90"] = replace_once(sources["tsteps.f90"], cut, "\n")

    # First-step setup remains native. Subsequent reached CFL work sees the
    # same domain as RK and output, including swapped physical allocations.
    text = sources["dynamics.f90"]
    lo, hi = find_routine_span(text, "TimeStepLength")
    body = text[lo:hi]
    anchor = "    subroutine get_U_dx_max\n"
    body = replace_once(
        body,
        anchor,
        """      use elmm_resident_pressure_state, only: resident_flow
      use elmm_resident_time_tasks, only: ResidentCFL
      if (resident_flow%active) then
        if (gridtype/=GRID_UNIFORM) error stop 'resident CFL requires uniform grid'
        call ResidentCFL(U,V,W,Prnx,Prny,Prnz,dxmin,dymin,dzmin,U_dx_max,nan)
        return
      endif
""",
        after=True,
        insert=True,
    )
    sources["dynamics.f90"] = text[:lo] + body + text[hi:]

    text = originals["outputs.f90"].read_text()
    lo, hi = find_routine_span(text, "OutTStep")
    body = text[lo:hi]
    body = replace_once(
        body,
        "    use Wallmodels, only: ComputeViscsWM\n",
        "    use elmm_resident_pressure_state, only: resident_flow\n"
        "    use elmm_resident_time_tasks, only: ResidentAverage\n",
        after=True,
        insert=True,
    )
    body = replace_once(
        body,
        "    if (.not.called) then",
        """    if (resident_flow%active) then
      if (size(probes)>0 .or. enable_profiles .or. store%probes_fluxes==1 .or. &
          store%tke==1 .or. store%avg_UU_prime>0 .or. Btype(To)==BC_AUTOMATIC_FLUX .or. &
          enable_buoyancy .or. enable_moisture .or. num_of_scalars>0) then
        call resident_flow%end_epoch()
        resident_flow%joined=.false.
      endif
    endif

""",
        after=False,
        insert=True,
    )
    for field, offset in (("U", "0,0,0"), ("V", "0,0,0"), ("W", "0,0,0"), ("Pr", "2,2,2")):
        original = f"{field}_avg = {field}_avg + {field}" + ("(1:Prnx,1:Prny,1:Prnz)" if field == "Pr" else "") + " * time_weight"
        s_lo, s_hi = find_unique_span(body, original)
        matched = body[s_lo:s_hi]
        body = (
            body[:s_lo]
            + f"""if (resident_flow%active) then
          call ResidentAverage({field},{field}_avg,lbound({field}),lbound({field}_avg),[{offset}],time_weight)
        else
          {matched}
        endif"""
            + body[s_hi:]
        )
    sources["outputs.f90"] = text[:lo] + body + text[hi:]

    text = originals["frames_common.f90"].read_text()
    f_lo, f_hi, f_call, _ = find_call_span(text, "D%Fill")
    text = (
        text[:f_lo]
        + """        block
          use, intrinsic :: iso_c_binding
          use elmm_resident_pressure_state, only: resident_flow
          integer(c_int64_t) :: frame_reads(5)
          if (resident_flow%active) then
            frame_reads(1)=resident_flow%bind_real(U,lbound(U),.true.,origin='frames::u')
            frame_reads(2)=resident_flow%bind_real(V,lbound(V),.true.,origin='frames::v')
            frame_reads(3)=resident_flow%bind_real(W,lbound(W),.true.,origin='frames::w')
            frame_reads(4)=resident_flow%bind_real(Pr,lbound(Pr),.true.,origin='frames::pr')
            frame_reads(5)=resident_flow%bind_real(Viscosity,lbound(Viscosity),.true.,origin='frames::viscosity')
            call resident_flow%observe(frame_reads)
          endif
        end block
"""
        + f_call
        + text[f_hi:]
    )
    sources["frames_common.f90"] = text
    # Their derived-type bindings import the changed frame declaration.
    for name in ("vtkframes.f90", "frames.f90", "surfacemapframes.f90"):
        sources[name] = originals[name].read_text()
    sources["main.f90"] = finalize_before_native(sources["main.f90"])
    from .resident_output import asynchronous_frames
    sources["frames_common.f90"], sources["vtkframes.f90"], sources["main.f90"] = asynchronous_frames(
        sources["frames_common.f90"], sources["vtkframes.f90"], sources["main.f90"])
    sources["resident_time_tasks.f90"] = _tasks(entries, precision=precision)
    records = []
    for name, text in sources.items():
        path = directory / name
        path.write_text(addressable_arrays(text))
        records.append(identity(path))
    result = {"version": 1, "sources": records, "basis": integration,
              "epoch": "primary adaptive time loop; native output fallback closes before access",
              "publication": "two immutable VTK snapshot slots; unsupported frames read only; writer join before field dumps and deallocation",
              "gpu_execution_established": False}
    (directory / "time-integration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
