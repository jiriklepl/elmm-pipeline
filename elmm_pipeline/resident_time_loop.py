"""The epoch across ELMM's time steps.

The per-cell arithmetic of a step, the CFL loop and the running averages are
entries compiled from ELMM's own source. This module keeps the fields on the
device from one step to the next: output work that this adapter does not cover
ends the epoch before its original source runs, and VTK output takes device
snapshots and is joined before teardown.
"""
from .ast_rewrites import find_call_span, find_routine_span, replace_once


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


def integrate_time_loop(sources):
    """Keep the epoch open across time steps: frame output reads device values.

    ``sources`` maps ELMM file names to their text and is changed in place.
    """

    text = sources["outputs.f90"]
    lo, hi = find_routine_span(text, "OutTStep")
    body = text[lo:hi]
    body = replace_once(
        body,
        "    use Wallmodels, only: ComputeViscsWM\n",
        "    use elmm_resident_pressure_state, only: resident_flow\n",
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
      endif
    endif

""",
        after=False,
        insert=True,
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
