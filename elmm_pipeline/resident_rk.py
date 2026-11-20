"""ELMM's Runge-Kutta stages inside the resident epoch.

The compiler extracts, compiles and calls the numerical loop nests. This
module places the epoch around the RK stages, says when ELMM's configuration
supports it, and keeps the remaining host code coherent with the device.
"""
from __future__ import annotations

import re

from compiler.scopes.resident_splice import ResidentDomain

from .ast_rewrites import find_call_span, find_routine_span, parse_fortran, replace_once

DOMAIN = ResidentDomain("elmm_resident_pressure_state", "resident_flow", "pressure_require", prefix="rk")


def verify_resident_execution(packages, trace, required_procedures):
    """Require submitted GPU kernels, and reject observed generated-CPU work.

    A generated CUDA file or a GPU placement request does not establish actual
    execution. Match the runtime ledger to compiler-issued implementation IDs.
    This check is for traced milestones, outside untraced performance runs.
    """
    from collections import Counter

    implementations = {}
    for package in packages:
        for entry in package["entries"]:
            for record in entry["entry"]["runtime_provenance"]["records"]:
                if record["kind"] == "implementation":
                    implementations[record["id"]] = (entry["region"]["procedure"], record["backend"])
    kernels, native = Counter(), Counter()
    for line in trace.splitlines():
        event = re.match(r"FORT_SCOPED (\w+)", line)
        identity = re.search(r" implementation=([a-f0-9]{64})\b", line)
        if event is None or identity is None or identity[1] not in implementations:
            continue
        procedure, backend = implementations[identity[1]]
        if backend == "generated_cpu":
            native[procedure] += 1
        elif backend == "cuda" and event[1] in {"launch", "graph_kernel"}:
            kernels[procedure] += 1
    if native:
        raise ValueError("Resident entries executed on CPU: " + repr(dict(native)))
    missing = sorted(set(required_procedures) - kernels.keys())
    if missing:
        raise ValueError("No GPU kernel execution recorded for: " + ", ".join(missing))
    return {"gpu_kernels_by_procedure": dict(sorted(kernels.items())), "generated_cpu_events": dict(native),
            "evidence": "runtime launch/graph-kernel ledger matched to compiler implementation identities"}


def _decl_shape_lower_bounds(node, entity_shape):
    attrs_str = re.sub(r"\s+", "", str(node.items[1] or "")).lower()
    if entity_shape is not None:
        return re.sub(r"\s+", "", str(entity_shape)).lower()
    m = re.search(r"dimension\(([^()]+)\)", attrs_str)
    return m.group(1) if m else ""


def contiguous_filter_arguments(text):
    from fparser.two.utils import walk

    for routine in ("Filter3ord", "Filter"):
        start, end = find_routine_span(text, routine)
        body = text[start:end]
        tree = parse_fortran(body)
        real_decls = []
        int_decls = []
        for decl in walk(tree):
            if type(decl).__name__ != "Type_Declaration_Stmt":
                continue
            dtype_str = re.sub(r"\s+", "", str(decl.items[0])).lower()
            attrs_str = re.sub(r"\s+", "", str(decl.items[1] or "")).lower()
            shapes = [_decl_shape_lower_bounds(decl, ent.items[1]) for ent in decl.items[2].items]
            if "-2:,-2:,-2:" not in shapes:
                continue
            if dtype_str.startswith("real") and "intent(inout)" in attrs_str:
                real_decls.append((decl.item.span, "contiguous" in attrs_str))
            elif dtype_str.startswith("integer") and "intent(in)" in attrs_str and "intent(inout)" not in attrs_str:
                int_decls.append((decl.item.span, "contiguous" in attrs_str))
        if len(real_decls) != 1 or len(int_decls) != 1:
            raise ValueError("Unsupported filter array declaration in " + routine)
        lines = body.splitlines(keepends=True)
        for (first_line, last_line), has_contig in sorted(real_decls + int_decls, key=lambda x: x[0][0], reverse=True):
            if not has_contig:
                chunk = "".join(lines[first_line - 1 : last_line])
                before, after = chunk.split("::", 1)
                lines[first_line - 1 : last_line] = [before.rstrip() + ", contiguous ::" + after]
        body = "".join(lines)
        text = text[:start] + body + text[end:]
    return text


def integrate_rk_epoch(output):
    """Connect the compiled primary RK path and the pressure child to one epoch.

    ``output`` maps ELMM file names to their text and is changed in place.
    Uniform DP, Sigma SGS, explicit FD2 diffusion and periodic/no-slip channel
    boundaries are the initial supported execution configuration. Other settings
    retain their native path. Physical allocations survive the proved three
    MOVE_ALLOC exchanges; the RK caller return publishes before unmanaged work.
    """
    output["filters.f90"] = contiguous_filter_arguments(output["filters.f90"])

    arrays = output["arrayutilities.f90"]
    arr_lo, arr_hi = find_routine_span(arrays, "set_int_3D")
    # Conversion is scalar and exact under the original assignment rules. One
    # implementation owns the fill, including ordering against queued readers.
    arrays = arrays[:arr_lo] + '''    subroutine set_int_3D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:)
      integer,intent(in) :: from
      call set_3D(to,real(from,knd))
    end subroutine''' + arrays[arr_hi:]
    output["arrayutilities.f90"] = arrays

    def at_once(text, anchor, addition, after=False):
        return replace_once(text, anchor, addition, after=after, insert=True)

    def publish_block(indent="    "):
        # Host code that follows reads or writes fields the epoch keeps on the device.
        return indent + "!$fort publish\n"

    tsteps = output["tsteps.f90"]
    lo, hi = find_routine_span(tsteps, "TMarchRK3")
    body = tsteps[lo:hi]
    body = at_once(body, "    use RungeKutta_coefficients\n", "    use, intrinsic :: iso_c_binding\n"
                   "    use elmm_resident_pressure_state, only: resident_flow\n", after=True)
    sub_lo, _, _, _ = find_call_span(body, "SubgridStresses")
    begin_epoch_code = (
        "      if (.not. resident_flow%joined) then\n"
        "        if (ResidentRKSupported()) then\n"
        "          call resident_flow%begin_epoch(0_c_int)\n"
        "          resident_flow%joined=.true.\n        endif\n      endif\n\n"
    )
    # The epoch stays open across RK stages and time steps; output and unsupported work close it.
    body = body[:sub_lo] + begin_epoch_code + body[sub_lo:]
    # ELMM applies the velocity boundaries before and after IBMomentum. Without immersed-boundary
    # points IBMomentum changes nothing, and the supported boundaries only copy interior values,
    # so the second application would rewrite what the first one wrote.
    body, repeated = re.subn(
        r"(call IBMomentum\((\w+), (\w+), (\w+)\)\n(?:[ \t]*\n)*)([ \t]*)call BoundUVW\(\2, \3, \4\)\n",
        lambda m: (f"{m[1]}{m[5]}block\n{m[5]}  use ImmersedBoundary, only: UIBPoints, VIBPoints, WIBPoints\n"
                   f"{m[5]}  if (.not. resident_flow%active .or. size(UIBPoints)+size(VIBPoints)+size(WIBPoints)>0) &\n"
                   f"{m[5]}    call BoundUVW({m[2]}, {m[3]}, {m[4]})\n{m[5]}end block\n"),
        body)
    if repeated != 2:
        raise ValueError(f"Expected two boundary applications after IBMomentum in TMarchRK3, found {repeated}")
    tsteps = tsteps[:lo] + body + tsteps[hi:]
    lo, hi = find_routine_span(tsteps, "IBMassSources")
    body = tsteps[lo:hi]
    # In the configured serial application an empty point set leaves the
    # complete Q allocation at +0. Bound_Q only adds those zero halo values.
    # Keep the reached GPU fill and avoid constructing a false host consumer.
    try:
        body = replace_once(
            body,
            "    call set(Q, 0)",
            "    call set(Q, 0._knd)\n"
            "    if (size(Up)+size(Vp)+size(Wp)==0) return\n" + publish_block(),
        )
    except ValueError:
        s_lo, s_hi, s_call, _ = find_call_span(body, "set")
        body = (
            body[:s_lo]
            + s_call
            + "\n    if (size(Up)+size(Vp)+size(Wp)==0) return\n"
            + publish_block()
            + body[s_hi:]
        )
    tsteps = tsteps[:lo] + body + tsteps[hi:]
    # These unported immersed-boundary operations are reached only when their
    # point arrays contain work. Empty primary-channel paths keep all replicas.
    tsteps = at_once(tsteps, "    if (size(Up) + size(Vp) + size(Wp)>0) then\n", publish_block("      "), after=True)
    lo, hi = find_routine_span(tsteps, "NullInterior")
    body = tsteps[lo:hi]
    body = at_once(body, "    !$omp parallel private(i)",
                   "    if (nUnull+nVnull+nWnull>0) then\n" + publish_block("      ") + "    endif\n")
    tsteps = tsteps[:lo] + body + tsteps[hi:]
    supported = '''
  logical function ResidentRKSupported() result(supported)
    use Subgrid, only: sgstype, SigmaModel
    use Filters, only: filtertype
    use ImmersedBoundary, only: ScalFlIBPoints
    use BuoyantGases, only: enable_buoyant_scalars
    supported = gridtype==GRID_UNIFORM .and. explicit_diffusion .and. discretization_order==2
    supported = supported .and. sgstype==SigmaModel .and. (filtertype==0 .or. filtertype==1)
    supported = supported .and. (advection_method==2 .or. advection_method==4 .or. advection_method==5 .or. advection_method==6)
    supported = supported .and. min(Unx,Uny,Unz,Vnx,Vny,Vnz,Wnx,Wny,Wnz)>=3
    supported = supported .and. all(Btype([We,Ea,So,No])==BC_PERIODIC) .and. all(Btype([Bo,To])==BC_NOSLIP)
    supported = supported .and. .not. enable_buoyancy .and. .not. enable_moisture .and. .not. enable_liquid
    supported = supported .and. num_of_scalars==0 .and. abs(Coriolis_parameter)<=tiny(1._knd)
    ! Buoyant gases change W on the host even when no scalar is transported.
    supported = supported .and. .not. enable_buoyant_scalars
    supported = supported .and. .not. enable_top_sponge .and. .not. enable_out_sponge_x .and. .not. enable_out_sponge_y
    supported = supported .and. .not. enable_fixed_flow_rate
    supported = supported .and. debuglevel==0 .and. steady/=1 .and. size(ScalFlIBPoints)==0
    supported = supported .and. .not. pressure_solution%check_divergence
    supported = supported .and. .not. any(pressure_solution%correct_mass_flux)
  end function
'''
    end = re.search(r"(?im)^\s*end\s*module(?:\s+TimeSteps)?\b", tsteps)
    if end is None:
        raise ValueError("Missing timestep module end")
    # Match the TimeSteps module end specifically when multiple modules exist.
    ts_end = re.search(r"(?im)^\s*end\s*module\s+TimeSteps\b", tsteps)
    end_mod = ts_end if ts_end is not None else end
    output["tsteps.f90"] = tsteps[:end_mod.start()] + supported + tsteps[end_mod.start():]

    dynamics = output["dynamics.f90"]
    lo, hi = find_routine_span(dynamics, "SubgridStresses")
    body = dynamics[lo:hi]
    # Wall-model records still live on the host. Their field inputs are reads:
    # observing the current allocation must preserve every device replica.
    # The optional pressure gradient is read only by wall model 2. The primary
    # configurations exclude thermal/moisture fields through the RK guard.
    wall_observation = '''    block
      use, intrinsic :: iso_c_binding
      use elmm_resident_pressure_state, only: resident_flow
      integer(c_int64_t) :: wall_reads(4)
      integer :: wall_count
      if (resident_flow%active .and. wallmodeltype>0) then
        wall_reads(1)=resident_flow%bind_real(U,lbound(U),.true.,origin='subgrid::walls::u')
        wall_reads(2)=resident_flow%bind_real(V,lbound(V),.true.,origin='subgrid::walls::v')
        wall_reads(3)=resident_flow%bind_real(W,lbound(W),.true.,origin='subgrid::walls::w')
        wall_count=3
        if (wallmodeltype==2) then
          wall_reads(4)=resident_flow%bind_real(Pr,lbound(Pr),.true.,origin='subgrid::walls::pr')
          wall_count=4
        endif
        call resident_flow%observe(wall_reads(:wall_count))
      endif
    end block

'''
    wall_if = re.search(r"(?im)^[ \t]*if\s*\(\s*wallmodeltype\s*>\s*0\s*\)\s*then\b", body)
    if wall_if is None:
        raise ValueError("Missing wallmodeltype branch in SubgridStresses")
    first_wall = wall_if.start()
    body = body[:first_wall] + wall_observation + body[first_wall:]
    wm_lo, wm_hi, wm_call, _ = find_call_span(body, "ComputeViscsWM")
    body = body[:wm_lo] + '''      block
        use, intrinsic :: iso_c_binding
        use elmm_resident_pressure_state, only: resident_flow
        integer(c_int64_t) :: wall_write
        wall_write=0
        if (resident_flow%active) then
          if (size(WMPoints)>0) then
            wall_write=resident_flow%host_box_real(Viscosity,lbound(Viscosity), &
              [minval(WMPoints%xi),minval(WMPoints%yj),minval(WMPoints%zk)], &
              [maxval(WMPoints%xi),maxval(WMPoints%yj),maxval(WMPoints%zk)],.true.)
          endif
        endif
''' + wm_call + '''
        if (wall_write/=0) call resident_flow%host_end(wall_write)
      end block''' + body[wm_hi:]
    output["dynamics.f90"] = dynamics[:lo] + body + dynamics[hi:]

    # Optional forcing keeps its existing condition and native implementation.
    # Enter coherence only when it actually has work; these calls are empty in
    # both primary channel configurations.
    linear = output["linearforcing.f90"]
    l_lo, l_hi = find_routine_span(linear, "linear_forcing")
    before, body, after_lin = linear[:l_lo], linear[l_lo:l_hi], linear[l_hi:]
    body = at_once(body, "    if (enable_linear_forcing) then\n", publish_block("      "), after=True)
    output["linearforcing.f90"] = before + body + after_lin
    volumes = output["volumesources.f90"]
    lo, hi = find_routine_span(volumes, "ResistanceForce")
    body = volumes[lo:hi]
    body = at_once(body, "           if (allocated(src)) then\n",
                   "             if (size(src)>0) then\n" + publish_block("               ") + "             endif\n", after=True)
    output["volumesources.f90"] = volumes[:lo] + body + volumes[hi:]
    sponge = output["sponge.f90"]
    s_lo, s_hi = find_routine_span(sponge, "rayleigh_damping_uvw")
    s_body = sponge[s_lo:s_hi]
    r_lo, r_hi, r_call, _ = find_call_span(s_body, "rayleigh_damping_top_uvw")
    s_if_match = re.search(
        r"(?im)^[ \t]*if\s*\(\s*enable_top_rayleigh_damping\s*\)\s*",
        s_body[:r_lo],
    )
    if s_if_match is not None and s_if_match.end() >= r_lo - 2:
        span_lo = s_if_match.start()
        s_body = (
            s_body[:span_lo]
            + "    if (enable_top_rayleigh_damping) then\n"
            + publish_block("      ")
            + "      " + r_call.lstrip() + "\n    endif"
            + s_body[r_hi:]
        )
    else:
        s_body = at_once(s_body, "    if (enable_top_rayleigh_damping) call rayleigh_damping_top_uvw(U2, V2, W2, U, V, W)", "")
    output["sponge.f90"] = sponge[:s_lo] + s_body + sponge[s_hi:]
