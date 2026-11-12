"""Checked primary-channel timestep lifetime and numerical source preparation.

This adapter owns the ELMM control region, not coherence. Ordinary compiler
entries implement the per-cell arithmetic; the generic runtime reduces CFL.
Unsupported output work closes the epoch before executing its original source.
Primary VTK output uses bounded immutable snapshots and joins before teardown.
"""
from pathlib import Path
import json
import re

from .gpu_hot_path import _once, _routine
from .resident_rk import addressable_arrays, identity


def generate_time_entries(compiler_root, python, directory):
    from .local_adapter import transform

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {
        "cfl": """module resident_cfl_source
implicit none
contains
subroutine values(u,v,w,cfl,nx,ny,nz,dx,dy,dz)
real(8), intent(in) :: u(:,:,:),v(:,:,:),w(:,:,:)
real(8), intent(out) :: cfl(:,:,:)
integer, value, intent(in) :: nx,ny,nz
real(8), value, intent(in) :: dx,dy,dz
integer :: i,j,k
real(8) :: p
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
        "average": """module resident_average_source
implicit none
contains
subroutine accumulate(source,average,nx,ny,nz,ox,oy,oz,weight)
real(8), intent(in) :: source(:,:,:)
real(8), intent(inout) :: average(:,:,:)
integer, value, intent(in) :: nx,ny,nz,ox,oy,oz
real(8), value, intent(in) :: weight
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


def _tasks(entries):
    cfl, avg = (entries["entries"][n]["entry"] for n in ("cfl", "average"))
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
  real(c_double), allocatable, target, save :: cfl_values(:,:,:)
contains
  subroutine ResidentCFL(U,V,W,nx,ny,nz,dx,dy,dz,maximum,nan)
    real(c_double), contiguous, target, intent(in) :: U(:,:,:),V(:,:,:),W(:,:,:)
    integer, intent(in) :: nx,ny,nz
    real(c_double), intent(in) :: dx,dy,dz
    real(c_double), intent(out) :: maximum
    logical, intent(out) :: nan
    integer(c_int64_t) :: handles(3),handle
    integer(c_int32_t) :: any_nan
    type(fort_scope_external_report_v1) :: report
    if (min(nx,ny,nz)<1) error stop 'invalid CFL grid'
    if (.not.allocated(cfl_values)) allocate(cfl_values(nx,ny,nz))
    if (any(shape(cfl_values)/=[nx,ny,nz])) error stop 'CFL storage epoch shape changed'
    if (any(shape(U)<[nx,ny,nz]+3) .or. any(shape(V)<[nx,ny,nz]+3) .or. &
        any(shape(W)<[nx,ny,nz]+3)) error stop 'CFL velocity geometry changed'
    handles(1)=resident_flow%bind_real64(U,[-2,-2,-2],.true.,origin='time::cfl::u')
    handles(2)=resident_flow%bind_real64(V,[-2,-2,-2],.true.,origin='time::cfl::v')
    handles(3)=resident_flow%bind_real64(W,[-2,-2,-2],.true.,origin='time::cfl::w')
    handle=resident_flow%bind_real64(cfl_values,[1,1,1],.false.,persistent=.true.,origin='time::cfl::values')
    {cfl_call}
    call pressure_require(fort_scope_max_real64_v1(resident_flow%context,handle,3_c_int32_t, &
      size(cfl_values,kind=c_size_t),0._c_double,maximum,any_nan,report))
    nan=any_nan/=0
  end subroutine
  subroutine ResidentAverage(source,average,source_lower,average_lower,offset,weight)
    real(c_double), contiguous, target, intent(in) :: source(:,:,:)
    real(c_double), contiguous, target, intent(inout) :: average(:,:,:)
    integer, intent(in) :: source_lower(3),average_lower(3),offset(3)
    real(c_double), intent(in) :: weight
    integer(c_int64_t) :: source_handle,average_handle
    if (any(offset<0) .or. any(shape(average)+offset>shape(source))) error stop 'invalid average view'
    source_handle=resident_flow%bind_real64(source,source_lower,.true.,origin='time::average::source')
    average_handle=resident_flow%bind_real64(average,average_lower,.true.,origin='time::average::result')
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
    _once(text, finalizer)
    text = text.replace(finalizer, "")
    diagnostic = "  block\n    use Dynamics, only: comparison_U2 => U2"
    anchor = diagnostic if diagnostic in text else "  call Dynamics_Deallocate"
    position = _once(text, anchor)
    return text[:position] + finalizer + text[position:]


def integrate_time_loop(integration, configured, entries, directory):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {Path(r["path"]).name: Path(r["path"]).read_text() for r in integration["sources"]}
    originals = {Path(r["original"]["path"]).name: Path(r["prepared"]["path"]) for r in configured["sources"]}
    cut = "\n    if (resident_flow%joined) then\n      call resident_flow%end_epoch()\n      resident_flow%joined=.false.\n    endif\n"
    _once(sources["tsteps.f90"], cut)
    sources["tsteps.f90"] = sources["tsteps.f90"].replace(cut, "\n")

    # First-step setup remains native. Subsequent reached CFL work sees the
    # same domain as RK and output, including swapped physical allocations.
    text = sources["dynamics.f90"]
    lo, hi = _routine(text, "TimeStepLength")
    body = text[lo:hi]
    anchor = "    subroutine get_U_dx_max\n"
    _once(body, anchor)
    body = body.replace(anchor, anchor + """      use elmm_resident_pressure_state, only: resident_flow
      use elmm_resident_time_tasks, only: ResidentCFL
      if (resident_flow%active) then
        if (gridtype/=GRID_UNIFORM) error stop 'resident CFL requires uniform grid'
        call ResidentCFL(U,V,W,Prnx,Prny,Prnz,dxmin,dymin,dzmin,U_dx_max,nan)
        return
      endif
""")
    sources["dynamics.f90"] = text[:lo] + body + text[hi:]

    text = originals["outputs.f90"].read_text()
    lo, hi = _routine(text, "OutTStep")
    body = text[lo:hi]
    anchor = "    use Wallmodels, only: ComputeViscsWM\n"
    _once(body, anchor)
    body = body.replace(anchor, anchor + "    use elmm_resident_pressure_state, only: resident_flow\n"
                        "    use elmm_resident_time_tasks, only: ResidentAverage\n")
    anchor = "    if (.not.called) then"
    pos = _once(body, anchor)
    body = body[:pos] + """    if (resident_flow%active) then
      if (size(probes)>0 .or. enable_profiles .or. store%probes_fluxes==1 .or. &
          store%tke==1 .or. store%avg_UU_prime>0 .or. Btype(To)==BC_AUTOMATIC_FLUX .or. &
          enable_buoyancy .or. enable_moisture .or. num_of_scalars>0) then
        call resident_flow%end_epoch()
        resident_flow%joined=.false.
      endif
    endif

""" + body[pos:]
    for field, offset in (("U", "0,0,0"), ("V", "0,0,0"), ("W", "0,0,0"), ("Pr", "2,2,2")):
        original = f"{field}_avg = {field}_avg + {field}" + ("(1:Prnx,1:Prny,1:Prnz)" if field == "Pr" else "") + " * time_weight"
        _once(body, original)
        body = body.replace(original, f"""if (resident_flow%active) then
          call ResidentAverage({field},{field}_avg,lbound({field}),lbound({field}_avg),[{offset}],time_weight)
        else
          {original}
        endif""")
    sources["outputs.f90"] = text[:lo] + body + text[hi:]

    text = originals["frames_common.f90"].read_text()
    anchor = "        call D%Fill(U, V, W, Pr, Viscosity, Temperature, Moisture, Scalar)"
    pos = _once(text, anchor)
    text = text[:pos] + """        block
          use, intrinsic :: iso_c_binding
          use elmm_resident_pressure_state, only: resident_flow
          integer(c_int64_t) :: frame_reads(5)
          if (resident_flow%active) then
            frame_reads(1)=resident_flow%bind_real64(U,lbound(U),.true.,origin='frames::u')
            frame_reads(2)=resident_flow%bind_real64(V,lbound(V),.true.,origin='frames::v')
            frame_reads(3)=resident_flow%bind_real64(W,lbound(W),.true.,origin='frames::w')
            frame_reads(4)=resident_flow%bind_real64(Pr,lbound(Pr),.true.,origin='frames::pr')
            frame_reads(5)=resident_flow%bind_real64(Viscosity,lbound(Viscosity),.true.,origin='frames::viscosity')
            call resident_flow%observe(frame_reads)
          endif
        end block
""" + text[pos:]
    sources["frames_common.f90"] = text
    # Their derived-type bindings import the changed frame declaration.
    for name in ("vtkframes.f90", "frames.f90", "surfacemapframes.f90"):
        sources[name] = originals[name].read_text()
    sources["main.f90"] = finalize_before_native(sources["main.f90"])
    from .resident_output import asynchronous_frames
    sources["frames_common.f90"], sources["vtkframes.f90"], sources["main.f90"] = asynchronous_frames(
        sources["frames_common.f90"], sources["vtkframes.f90"], sources["main.f90"])
    sources["resident_time_tasks.f90"] = _tasks(entries)
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
