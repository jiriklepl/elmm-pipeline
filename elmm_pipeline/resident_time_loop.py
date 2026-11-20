"""Checked primary-channel timestep lifetime and numerical source preparation.

This adapter owns the ELMM control region, not coherence. Ordinary compiler
entries implement the per-cell arithmetic; the CFL loop is compiled from ELMM's own source.
Unsupported output work closes the epoch before executing its original source.
Primary VTK output uses bounded immutable snapshots and joins before teardown.
"""
from pathlib import Path

from .ast_rewrites import (
    find_call_span,
    find_routine_span,
    find_unique_span,
    replace_once,
)


def generate_time_entries(directory, *, precision="dp"):
    from compiler.scopes.resident_entries import compile_resident_entry

    real_type = "real(8)" if precision == "dp" else "real(4)"
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {
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
        target = directory / (name + "-entry")
        shared = compile_resident_entry(source, "resident_average_source::accumulate", target)["scoped"]
        if not shared["region_execution"] or not all(r["gpu_available"] for r in shared["region_execution"]):
            raise ValueError("Timestep numerical entry lacks GPU execution: " + name)
        result["entries"][name] = {"directory": str(target), "entry": shared}
    return result


def _tasks(entries, *, precision="dp"):
    avg = entries["entries"]["average"]["entry"]
    real_type_c = "real(c_double)" if precision == "dp" else "real(c_float)"
    arguments = dict(source="source_handle", average="average_handle", nx="size(average,1)",
                     ny="size(average,2)", nz="size(average,3)", ox="offset(1)",
                     oy="offset(2)", oz="offset(3)", weight="weight")
    values = ["resident_flow%context", "FORT_SCOPE_GPU"] + [arguments[n] for n in avg["argument_order"][2:]]
    avg_call = "call pressure_require(average_run( &\n      " + ", &\n      ".join(values) + "))"
    return f"""module elmm_resident_time_tasks
  use, intrinsic :: iso_c_binding
  use fort_scoped_memory
  use elmm_resident_pressure_state, only: resident_flow, pressure_require
  use {avg['fortran_module']}, only: average_run=>run
  implicit none
  private
  public :: ResidentAverage
contains
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


def integrate_time_loop(sources, entries, *, precision="dp"):
    """Keep the epoch open across time steps: averaging and frame output read device values.

    ``sources`` maps ELMM file names to their text and is changed in place.
    """

    text = sources["outputs.f90"]
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

    text = sources["frames_common.f90"]
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
    sources["main.f90"] = finalize_before_native(sources["main.f90"])
    from .resident_output import asynchronous_frames
    sources["frames_common.f90"], sources["vtkframes.f90"], sources["main.f90"] = asynchronous_frames(
        sources["frames_common.f90"], sources["vtkframes.f90"], sources["main.f90"])
    sources["resident_time_tasks.f90"] = _tasks(entries, precision=precision)
