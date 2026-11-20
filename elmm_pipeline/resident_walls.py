"""Primary planar-wall source adapter over compiler-generated numerical entries.

The packed planes are ordinary Fortran allocations registered with the existing
domain. They contain wall state, not replica/coherence metadata. Coordinates are
validated as complete, unique rectangular planes before replacing indirect loops.
"""
from pathlib import Path
import hashlib
import re

from .ast_rewrites import (
    find_call_span,
    find_routine_span,
    find_unique_span,
    parse_fortran,
    replace_once,
)


def _verify_wall_helpers(source):
    """Structurally verify required wall-model routines without rigid SHA locks."""
    from fparser.two.utils import walk

    required = (
        "WMFlatUstar",
        "WMFlatStress",
        "WMFlatVisc",
        "vel_and_dist",
        "ComputeViscsWM",
        "ComputeUVWFluxesWM",
    )
    tree = parse_fortran(source)
    routines = {}
    for node in walk(tree):
        if type(node).__name__ != "Subroutine_Subprogram":
            continue
        stmt = next((c for c in node.content if type(c).__name__ == "Subroutine_Stmt"), None)
        if stmt is None:
            continue
        rname = str(stmt.items[1]).lower()
        if rname not in routines or type(getattr(node, "parent", None)).__name__ != "Internal_Subprogram_Part":
            routines[rname] = node

    digests = {}
    for name in required:
        key = name.lower()
        node = routines.get(key)
        if node is None:
            raise ValueError("Missing original wall helper: " + name)
        execution = next((c for c in node.content if type(c).__name__ == "Execution_Part"), None)
        if execution is None:
            raise ValueError("Empty wall helper body: " + name)
        r_lo, r_hi = find_routine_span(source, name)
        routine_text = source[r_lo:r_hi]
        callees = {
            str(n.items[0]).lower()
            for n in walk(node)
            if type(n).__name__ == "Call_Stmt" and len(n.items) > 0
        }
        names = {str(n).lower() for n in walk(node) if type(n).__name__ == "Name"}
        if key == "wmflatustar" and (
            not any(type(n).__name__ == "Block_Nonlabel_Do_Construct" for n in walk(node))
            or not {"ustar", "vel", "dist"}.issubset(names)
        ):
            raise ValueError("Wall helper changed; requalify the scalarization: " + name)
        if key == "wmflatstress" and not {"vel_and_dist", "wmflatustar"}.issubset(callees):
            raise ValueError("Wall helper changed; requalify the scalarization: " + name)
        if key == "wmflatvisc" and "wmflatstress" not in callees:
            raise ValueError("Wall helper changed; requalify the scalarization: " + name)
        if key == "vel_and_dist" and not {"dist", "tan_vect"}.issubset(names):
            raise ValueError("Wall helper changed; requalify the scalarization: " + name)
        if key == "computeviscswm" and not {"wmpoints", "viscosity"}.issubset(names):
            raise ValueError("Wall helper changed; requalify the scalarization: " + name)
        if key == "computeuvwfluxeswm" and "fluxes" not in callees:
            raise ValueError("Wall helper changed; requalify the scalarization: " + name)
        canonical = re.sub(r"\s+", "", re.sub(r"!.*", "", routine_text)).lower()
        digests[name] = hashlib.sha256(canonical.encode()).hexdigest()
    return digests


def _norm3(result, values, kind_suffix="_8"):
    """The pinned GNU Fortran NORM2 lowering, including scaling and order."""
    lines = [f"norm_scale=1.{kind_suffix}", f"norm_value=0.{kind_suffix}"]
    for value in values:
        lines += [f"if ({value}/=0.{kind_suffix}) then", f"  norm_abs=abs({value})",
                  "  if (norm_abs>norm_scale) then", "    norm_ratio=norm_scale/norm_abs",
                  f"    norm_value=(norm_ratio*norm_ratio)*norm_value+1.{kind_suffix}", "    norm_scale=norm_abs",
                  "  else", "    norm_ratio=norm_abs/norm_scale",
                  "    norm_value=norm_ratio*norm_ratio+norm_value", "  endif", "endif"]
    return "\n".join(lines + [f"{result}=sqrt(norm_value)*norm_scale"])


def _shift_arrays(text, arrays):
    for name, shift in arrays.items():
        # Prepared scalar coordinates have no nested function calls.
        pattern = r"\b" + name + r"\(([^()]*)\)"
        def shifted(match):
            args = match[1].split(",")
            if len(args) != 3:
                raise ValueError("Unexpected wall array reference: " + match[0])
            return name + "(" + ",".join("(" + a.strip() + ")+" + str(shift) for a in args) + ")"
        text = re.sub(pattern, shifted, text, flags=re.I)
    return text


def _stress_source(original, component, *, precision="dp"):
    real_type = "real(8)" if precision == "dp" else "real(4)"
    kind_suffix = "_8" if precision == "dp" else "_4"
    if component == 0:
        velocities = f"""v1=(U(i,j,k)+U(i-1,j,k))/2.{kind_suffix}
v2=(V(i,j,k)+V(i,j-1,k))/2.{kind_suffix}
v3=(W(i,j,k)+W(i,j,k-1))/2.{kind_suffix}"""
    else:
        try:
            lv_lo, lv_hi = find_routine_span(original, "local_velocity", kind="function")
        except ValueError as error:
            raise ValueError("Missing original wall velocity interpolation") from error
        body = original[lv_lo:lv_hi]
        case_pat = (
            re.compile(rf"(?im)^[ \t]*case\s*\(\s*{component}\s*\)[ \t]*(?:!.*)?$")
            if component < 3
            else re.compile(r"(?im)^[ \t]*case\s+default[ \t]*(?:!.*)?$")
        )
        m_case = case_pat.search(body)
        if m_case is None:
            raise ValueError(f"Missing local_velocity case {component}")
        body = body[m_case.end() :]
        stop = re.search(r"(?im)^\s*(case\b|end\s*select)", body)
        velocities = body[:stop.start()]
        for old, new in (("xi", "i"), ("yj", "j"), ("zk", "k")):
            velocities = re.sub(r"\b" + old + r"\b", new, velocities, flags=re.I)
        velocities = re.sub(r"vel\((\d)\)", r"v\1", velocities, flags=re.I).replace("_knd", kind_suffix)
    velocities = _shift_arrays(velocities, dict(U=3, V=3, W=3))
    # A bounded active flag preserves the original convergence/50-iteration
    # ordering while keeping the point loop expressible by ordinary lowering.
    flux = "" if component == 0 else f"""
{_norm3('mag', ('v1', 'v2', 'v3'), kind_suffix)}
if (mag>sqrt(epsilon(1.{kind_suffix}))) then
  fval=(v{component}/mag)*star**2*reciprocal
  if (direction==5) fval=-fval
else
  fval=0
endif
fluxp(q)=fval
fluxm(q)=fval
"""
    args = ("u,v,w,xi,yj,zk,distz,ustar," + ("fluxp,fluxm," if component else "") + "n,viscosity"
            + (",reciprocal,direction" if component else ""))
    return f"""module resident_wall_stress_{component}
implicit none
contains
subroutine stress({args})
{real_type}, intent(in) :: u(:,:,:),v(:,:,:),w(:,:,:)
integer, intent(in) :: xi(:),yj(:),zk(:)
{real_type}, intent(in) :: distz(:)
{real_type}, intent(inout) :: ustar(:)
{f"{real_type}, intent(out) :: fluxp(:),fluxm(:)" if component else ""}
integer, value, intent(in) :: n
{real_type}, value, intent(in) :: viscosity
{f"{real_type}, value, intent(in) :: reciprocal" if component else ""}
{"integer, value, intent(in) :: direction" if component else ""}
integer :: q,i,j,k,iteration,done
{real_type} :: distance_z,v1,v2,v3,dot,dist,vel,mag,star,star2,lam,zpl,fval
{real_type} :: norm_scale,norm_value,norm_ratio,norm_abs
do q=1,n
  i=xi(q)
  j=yj(q)
  k=zk(q)
  distance_z=distz(q)
  {velocities}
  {_norm3('dist', (f'0.{kind_suffix}', f'0.{kind_suffix}', 'distance_z'), kind_suffix)}
  v1=v1-0.{kind_suffix}
  v2=v2-0.{kind_suffix}
  v3=v3-0.{kind_suffix}
  dot=((0.{kind_suffix}+v1*0.{kind_suffix})+v2*0.{kind_suffix})+v3*distance_z
  v1=v1-dot*0.{kind_suffix}/dist**2
  v2=v2-dot*0.{kind_suffix}/dist**2
  v3=v3-dot*distance_z/dist**2
  {_norm3('vel', ('v1', 'v2', 'v3'), kind_suffix)}
  star=ustar(q)
  if (vel/=0) then
    lam=sqrt(viscosity*vel/dist)
    if (dist*lam/viscosity<5.{kind_suffix}) then
      star=lam
    else
      done=0
      do iteration=1,50
        if (done==0) then
          star2=star
          zpl=dist*star2/viscosity
          if (zpl<=5.{kind_suffix}) then
            star=sqrt(viscosity*vel/dist)
          else if (zpl<30.{kind_suffix}) then
            star=vel/(5*log(zpl)-3.05)
          else
            star=vel/(log(zpl)/0.4{kind_suffix}+5.5{kind_suffix})
          endif
          if (abs(star-star2)/abs(star)<1e-4{kind_suffix}) then
            done=1
          else if (iteration>=50) then
            star=0
            done=1
          endif
        endif
      enddo
    endif
  endif
  if (star<0) star=0
  ustar(q)=star
  {flux}
enddo
end subroutine
end module
"""


def generate_wall_entries(sources, directory, *, precision="dp"):
    """Compile the stress kernels: one loop over a list of wall points per velocity component."""
    from compiler.scopes.resident_entries import compile_resident_entry

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    original = sources["wallmodels.f90"]
    _verify_wall_helpers(original)
    # The stress kernels carry the pinned GNU Fortran 15 scaled NORM2 with its accumulation order.
    result = {"entries": {}}
    for component in range(4):
        name = "stress" + str(component)
        path = directory / (name + ".f90")
        path.write_text(_stress_source(original, component, precision=precision))
        target = directory / (name + "-entry")
        shared = compile_resident_entry(path, f"resident_wall_stress_{component}::stress", target)["scoped"]
        if not shared["region_execution"] or not all(r["gpu_available"] for r in shared["region_execution"]):
            raise ValueError("Wall numerical entry lacks GPU execution: " + name)
        result["entries"][name] = {"directory": str(target), "entry": shared}
    return result


def _driver(entries, *, precision="dp"):
    """Fortran for module Wallmodels: when the lists can run resident, and the calls of the stress kernels.

    The friction velocity and the fluxes stay fields of ELMM's own wall-point
    lists. The domain keeps them on the device between steps; the corrections
    in the momentum diffusion read them there through the compiler's entries.
    """
    kind = "FORT_SCOPE_REAL64" if precision == "dp" else "FORT_SCOPE_REAL32"
    uses, dispatch = [], []

    def field(member, code, writing, label):
        return (f"resident_flow%bind_field(c_loc(points(lbound(points,1))%{member}),stride,count,{code}, &\n"
                f"          {writing},origin='walls::{label}::{member}')")

    for name, item in entries["entries"].items():
        entry, component = item["entry"], int(name[-1])
        uses.append("    use " + entry["fortran_module"] + ", only: " + name + "_run=>run")
        actual = dict(u="handles(1)", v="handles(2)", w="handles(3)", xi="fields(1)", yj="fields(2)", zk="fields(3)",
                      distz="fields(4)", ustar="fields(5)", fluxp="fields(6)", fluxm="fields(7)", n="size(points)",
                      viscosity="molecular_viscosity", reciprocal="1._knd/dzmin", direction="side")
        args = ["resident_flow%context", "FORT_SCOPE_GPU"] + [actual[n] for n in entry["argument_order"][2:]]
        call = f"call pressure_require({name}_run( &\n          " + ", &\n          ".join(args) + "))"
        if component:
            dispatch.append(f"        case({component})\n          {call}")
        else:
            pressure_call = call
    geometry = "\n".join(f"      fields({index})={field(member, code, '.false.', '@LABEL@')}"
                         for index, (member, code) in enumerate(
                             (("xi", "FORT_SCOPE_INTEGER32"), ("yj", "FORT_SCOPE_INTEGER32"),
                              ("zk", "FORT_SCOPE_INTEGER32"), ("distz", kind)), 1))
    declarations = """
  integer(kind=8), save :: resident_wall_epoch=-1
  public :: ResidentWallsPrepare, ResidentPrStress, ResidentUVWStress, resident_wall_epoch
"""
    procedures = f"""
  logical function ResidentWallsPrepare() result(ok)
    use elmm_resident_pressure_state, only: resident_flow
    integer :: side,component
    type(WMPointUVW), pointer :: points(:)
    ok=.false.
    if (.not.resident_flow%active) return
    if (resident_wall_epoch==resident_flow%generation) then
      ok=.true.
      return
    endif
    if (wallmodeltype/=1 .or. molecular_viscosity<=0 .or. gridtype/=GRID_UNIFORM) return
    if (enable_buoyancy .or. enable_moisture .or. num_of_scalars>0) return
    if (.not.allocated(WMPoints)) return
    if (size(WMPoints)==0) return
    if (any(WMPoints%z0>0) .or. any(WMPoints%prescribed_ustar)) return
    if (any(WMPoints%distx/=0) .or. any(WMPoints%disty/=0) .or. any(WMPoints%distz==0)) return
    if (any(WMPoints%wallu/=0) .or. any(WMPoints%wallv/=0) .or. any(WMPoints%wallw/=0)) return
    do component=1,3
      do side=1,6
        if (.not.associated(WMPointsUVW(side,component)%points)) return
        points=>WMPointsUVW(side,component)%points
        ! The kernels implement flat walls normal to z only.
        if (side<5 .and. size(points)/=0) return
        if (size(points)==0) cycle
        if (any(points%z0>0) .or. any(points%prescribed_ustar)) return
        if (any(points%distx/=0) .or. any(points%disty/=0) .or. any(points%distz==0)) return
        if (any(points%wallu/=0) .or. any(points%wallv/=0) .or. any(points%wallw/=0)) return
      enddo
    enddo
    resident_wall_epoch=resident_flow%generation
    ok=.true.
  end function

  subroutine ResidentPrStress(U,V,W)
    use, intrinsic :: iso_c_binding
    use fort_scoped_memory, only: FORT_SCOPE_GPU, FORT_SCOPE_INTEGER32, {kind}
    use elmm_resident_pressure_state, only: resident_flow,pressure_require
@USES@
    real(knd),contiguous,target,intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    integer(c_int64_t) :: handles(3),fields(5)
    integer(c_size_t) :: stride,count
    if (resident_wall_epoch/=resident_flow%generation) error stop 'wall epoch not prepared'
    handles(1)=resident_flow%bind_real(U,lbound(U),.true.,origin='walls::stress::u')
    handles(2)=resident_flow%bind_real(V,lbound(V),.true.,origin='walls::stress::v')
    handles(3)=resident_flow%bind_real(W,lbound(W),.true.,origin='walls::stress::w')
    associate(points=>WMPoints)
      stride=int(storage_size(points)/8,c_size_t)
      count=size(points,kind=c_size_t)
{geometry.replace("@LABEL@", "points")}
      fields(5)={field("ustar", kind, ".true.", "points")}
      {pressure_call}
    end associate
  end subroutine

  subroutine ResidentUVWStress(U,V,W)
    use, intrinsic :: iso_c_binding
    use fort_scoped_memory, only: FORT_SCOPE_GPU, FORT_SCOPE_INTEGER32, {kind}
    use elmm_resident_pressure_state, only: resident_flow,pressure_require
@USES@
    real(knd),contiguous,target,intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    integer(c_int64_t) :: handles(3),fields(7)
    integer(c_size_t) :: stride,count
    integer :: component,side
    type(WMPointUVW), pointer :: points(:)
    if (resident_wall_epoch/=resident_flow%generation) error stop 'wall epoch not prepared'
    handles(1)=resident_flow%bind_real(U,lbound(U),.true.,origin='walls::stress::u')
    handles(2)=resident_flow%bind_real(V,lbound(V),.true.,origin='walls::stress::v')
    handles(3)=resident_flow%bind_real(W,lbound(W),.true.,origin='walls::stress::w')
    do component=1,3
      do side=5,6
        points=>WMPointsUVW(side,component)%points
        if (size(points)==0) cycle
        stride=int(storage_size(points)/8,c_size_t)
        count=size(points,kind=c_size_t)
{geometry.replace("@LABEL@", "momentum").replace("      fields", "        fields")}
        fields(5)={field("ustar", kind, ".true.", "momentum")}
        fields(6)={field("fluxp", kind, ".true.", "momentum")}
        fields(7)={field("fluxm", kind, ".true.", "momentum")}
        select case(component)
@STRESS@
        end select
      enddo
    enddo
  end subroutine
"""
    return declarations, procedures.replace("@USES@", "\n".join(uses)).replace("@STRESS@", "\n".join(dispatch))


# The lists of the walls below and above, in the order ELMM's ground diagnostic visits them.
_FLAT_LISTS = ("UzmWMpoints", "UzpWMpoints", "VzmWMpoints", "VzpWMpoints")
_SIDE_LISTS = tuple(component + side + "WMpoints" for component in "UV" for side in ("xm", "xp", "ym", "yp"))


def name_ground_lists(text):
    """Let the mean friction velocity of the U and V wall points be accumulated over named lists.

    ELMM visits the lists through an array of pointers, which no compiled
    loop can follow. With points only on the walls below and above, four
    module arrays hold all of them; a procedure of its own accumulates over
    those, in ELMM's order, and can run on the device. Other geometries keep
    ELMM's loop.
    """
    lo, hi = find_routine_span(text, "GroundUstarUVW", kind="function")
    body = text[lo:hi]
    # ELMM's own loop, with its OpenMP region, stays whole in the other branch.
    start = body.index("      !$omp parallel reduction(+:res,n)\n")
    stop = body.index("      !$omp end parallel\n", start) + len("      !$omp end parallel\n")
    flat = " + ".join(f"size({name})" for name in _SIDE_LISTS)
    body = (body[:start] + f"      if ({flat} == 0) then\n        call GroundUstarFlat(res, n)\n      else\n"
            + body[start:stop] + "      end if\n" + body[stop:])
    loops = "".join(f"""    !$fort reduction(+: res, n)
    do k = 1, size({name})
      if ({name}(k)%zk==1) then
        res = res + {name}(k)%ustar
        n = n + 1
      end if
    end do
""" for name in _FLAT_LISTS)
    procedure = f"""  subroutine GroundUstarFlat(res, n)
    real(knd), intent(inout) :: res
    integer, intent(inout) :: n
    integer :: k
{loops}  end subroutine GroundUstarFlat


"""
    return text[:lo] + procedure + body + text[hi:]


def integrate_wall_entries(sources, entries, *, precision="dp"):
    """Run the planar wall model on the device while the epoch is active.

    ``sources`` maps ELMM file names to their text and is changed in place.
    The stress kernels replace the two host routines that compute the friction
    velocity and the wall fluxes. The corrections in the momentum diffusion
    are compiled from ELMM's own loops and need no edit here.
    """
    declarations, procedures = _driver(entries, precision=precision)
    wall = sources["wallmodels.f90"]
    start = re.search(r"(?im)^[ \t]*module\s+Wallmodels\b", wall).start()
    contains = re.search(r"(?im)^[ \t]*contains\b[^\n]*$", wall[start:]).start() + start
    wall = wall[:contains] + declarations + wall[contains:]
    end = re.search(r"(?im)^\s*end\s*module\s+Wallmodels\b", wall)
    if end is None:
        raise ValueError("Missing wall model module end")
    wall = wall[:end.start()] + procedures + wall[end.start():]
    sources["wallmodels.f90"] = wall

    text = sources["dynamics.f90"]
    lo, hi = find_routine_span(text, "SubgridStresses")
    body = text[lo:hi]
    # Replace the prior whole-field observation and native viscosity transaction
    # together. Primary Sigma SGS overwrites every interior viscosity value.
    start = find_unique_span(body, "    block\n      use, intrinsic :: iso_c_binding")[0]
    sub_lo, _, _, _ = find_call_span(body[start:], "SubgridModel")
    stop = start + sub_lo
    old = body[start:stop]
    # Retain the old path verbatim for unsupported configurations.
    new = """    block
      use elmm_resident_pressure_state, only: resident_flow
      if (resident_flow%active) then
        if (.not.ResidentWallsPrepare()) then
          call resident_flow%end_epoch()
          resident_flow%joined=.false.
        endif
      endif
      if (resident_flow%active) then
        call ResidentPrStress(U,V,W)
      else
@OLD@
      endif
    end block

""".replace("@OLD@", old.rstrip())
    body = body[:start] + new + body[stop:]
    flux_lo, flux_hi, flux_call, _ = find_call_span(body, "ComputeUVWFluxesWM")
    body = body[:flux_lo] + """      block
        use elmm_resident_pressure_state, only: resident_flow
        if (resident_flow%active) then
          call ResidentUVWStress(U,V,W)
        else
""" + flux_call + """
        endif
      end block""" + body[flux_hi:]
    sources["dynamics.f90"] = text[:lo] + body + text[hi:]
