"""Primary planar-wall source adapter over compiler-generated numerical entries.

The packed planes are ordinary Fortran allocations registered with the existing
domain. They contain wall state, not replica/coherence metadata. Coordinates are
validated as complete, unique rectangular planes before replacing indirect loops.
"""
from pathlib import Path
import hashlib
import json
import re

from .ast_rewrites import (
    find_call_span,
    find_routine_span,
    find_unique_span,
    parse_fortran,
    replace_once,
)
from .resident_rk import identity


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
flux(i,j)=fval
"""
    args = "u,v,w,ustar," + ("flux," if component else "") + "nx,ny,k,distance_z,viscosity" + (",reciprocal,direction" if component else "")
    return f"""module resident_wall_stress_{component}
implicit none
contains
subroutine stress({args})
{real_type}, intent(in) :: u(:,:,:),v(:,:,:),w(:,:,:)
{real_type}, intent(inout) :: ustar(:,:)
{f"{real_type}, intent(out) :: flux(:,:)" if component else ""}
integer, value, intent(in) :: nx,ny,k
{real_type}, value, intent(in) :: distance_z,viscosity
{f"{real_type}, value, intent(in) :: reciprocal" if component else ""}
{"integer, value, intent(in) :: direction" if component else ""}
integer :: i,j,iteration,done
{real_type} :: v1,v2,v3,dot,dist,vel,mag,star,star2,lam,zpl,fval
{real_type} :: norm_scale,norm_value,norm_ratio,norm_abs
do j=1,ny
 do i=1,nx
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
  star=ustar(i,j)
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
  ustar(i,j)=star
  {flux}
 enddo
enddo
end subroutine
end module
"""


def _corrections(integration):
    text = next(Path(r["path"]).read_text() for r in integration["sources"] if Path(r["path"]).name == "momentum_diffusion.f90")
    lo, hi = find_routine_span(text, "MomentumDiffusion_nobranch_2ord")
    body = text[lo:hi]
    pattern = r"(?ms)^    block\n(?:(?!^    end block).)*?wall_handles\(3\)(?:(?!^    end block).)*?^    end block\n"
    groups = {}
    for match in re.finditer(pattern, body):
        block = match[0]
        point = re.search(r"do i = 1, size\(([UVW][xyz][mp]WMpoints)\)", block, re.I)[1]
        begin = re.search(r"(?im)^\s*do i = 1, size\(", block).end()
        begin = block.index("\n", begin) + 1
        end = block.index("      end do", begin)
        groups.setdefault(point.lower(), []).append({"block": block, "statements": block[begin:end]})
    if len(groups) != 18 or any(len(v) != 2 for v in groups.values()):
        raise ValueError("Unexpected wall correction topology")
    return groups


def generate_wall_entries(integration, directory, *, precision="dp"):
    from compiler.scopes.resident_entries import compile_resident_entry

    real_type = "real(8)" if precision == "dp" else "real(4)"
    kind_suffix = "_8" if precision == "dp" else "_4"
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    wall = next(r for r in integration["sources"] if Path(r["path"]).name == "wallmodels.f90")
    original = Path(wall["path"]).read_text()
    helpers = _verify_wall_helpers(original)
    sources = {"stress" + str(c): (_stress_source(original, c, precision=precision), f"resident_wall_stress_{c}::stress") for c in range(4)}
    groups = _corrections(integration)
    for component, letter in enumerate("uvw", 1):
        for side in "mp":
            point = letter + "z" + side + "wmpoints"
            statements = "\n".join(v["statements"] for v in groups[point])
            for field, coordinate in (("xi", "i"), ("yj", "j"), ("zk", "k")):
                statements = re.sub(point + r"\(i\)%" + field, coordinate, statements, flags=re.I)
            statements = re.sub(point + r"\(i\)%fluxp", "flux(i,j)", statements, flags=re.I)
            # Remove redundant parentheses from the generated include's affine
            # coordinates before physical-origin normalization.
            statements = re.sub(r"\(([ijk](?:[+-]\d+)?)\)", r"\1", statements, flags=re.I)
            statements = re.sub(r"\b" + letter + "2" + r"\b", "result", statements, flags=re.I)
            statements = re.sub(r"\b" + letter + r"\b", "velocity", statements, flags=re.I)
            statements = _shift_arrays(statements, dict(result=3, velocity=3, nu=2))
            name = "correction" + str(component) + side
            source = f"""module resident_wall_{name}
implicit none
contains
subroutine correction(result,velocity,nu,flux,nx,ny,k,recdzmin2)
{real_type}, intent(inout) :: result(:,:,:)
{real_type}, intent(in) :: velocity(:,:,:),nu(:,:,:),flux(:,:)
integer, value, intent(in) :: nx,ny,k
{real_type}, value, intent(in) :: recdzmin2
integer :: i,j
do j=1,ny
 do i=1,nx
{statements.replace('_knd', kind_suffix)}
 enddo
enddo
end subroutine
end module
"""
            sources[name] = (source, f"resident_wall_{name}::correction")
    result = {"version": 1, "original_wall": wall, "helper_contract": helpers,
              "norm_lowering": "pinned GNU Fortran 15 scaled NORM2, accumulation order preserved",
              "entries": {}, "correction_groups": groups}
    for name, (source, entry) in sources.items():
        path = directory / (name + ".f90")
        path.write_text(source)
        target = directory / (name + "-entry")
        shared = compile_resident_entry(path, entry, target)["scoped"]
        if not shared["region_execution"] or not all(r["gpu_available"] for r in shared["region_execution"]):
            raise ValueError("Wall numerical entry lacks GPU execution: " + name)
        result["entries"][name] = {"source": identity(path), "directory": str(target), "entry": shared}
        (directory / "wall-entries.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _driver(entries):
    uses, dispatch_stress, dispatch_correction = [], [], []
    for name, item in entries["entries"].items():
        entry = item["entry"]
        uses.append("    use " + entry["fortran_module"] + ", only: " + name + "_run=>run")
        if name.startswith("stress"):
            component = int(name[-1])
            actual = dict(u="handles(1)", v="handles(2)", w="handles(3)", ustar="star_handle",
                          flux="flux_handle", nx="p%nx", ny="p%ny", k="p%k", distance_z="p%distz",
                          viscosity="molecular_viscosity", reciprocal="1._knd/dzmin", direction="p%direction")
            target = dispatch_stress
            case = component
        else:
            component, side = int(name[-2]), name[-1]
            actual = dict(result="handles(1)", velocity="handles(2)", nu="handles(3)", flux="flux_handle",
                          nx="p%nx", ny="p%ny", k="p%k", recdzmin2="1._knd/dzmin**2")
            target = dispatch_correction
            case = component * 10 + (1 if side == "m" else 2)
        args = ["resident_flow%context", "FORT_SCOPE_GPU"] + [actual[n] for n in entry["argument_order"][2:]]
        target.append(f"      case({case})\n        call pressure_require({name}_run( &\n          " + ", &\n          ".join(args) + "))")
    declarations = """
  type resident_wall_plane
    integer :: nx=0,ny=0,k=0,direction=0
    real(knd) :: distz=0
    integer, allocatable :: record(:,:)
    real(knd), allocatable :: star(:,:),flux(:,:)
  end type
  type(resident_wall_plane), target, save :: resident_planes(2,0:3)
  integer(kind=8), save :: resident_wall_epoch=-1
  public :: ResidentWallsPrepare, ResidentPrStress, ResidentUVWStress, ResidentWallCorrection
  public :: ResidentWallsPublish
"""
    procedures = """
  logical function ResidentWallsPrepare() result(ok)
    use elmm_resident_pressure_state, only: resident_flow
    integer :: side,component,q,i,j,k,nx,ny,nz,n(3,3)
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
    if (size(WMPoints)/=2*Prnx*Prny) return
    if (any(WMPoints%z0>0) .or. any(WMPoints%prescribed_ustar)) return
    if (any(WMPoints%distx/=0) .or. any(WMPoints%disty/=0)) return
    if (any(WMPoints%wallu/=0) .or. any(WMPoints%wallv/=0) .or. any(WMPoints%wallw/=0)) return
    do component=1,3
      do side=1,4
        if (.not.associated(WMPointsUVW(side,component)%points)) return
        if (size(WMPointsUVW(side,component)%points)/=0) return
      enddo
    enddo
    n(:,1)=[Unx,Uny,Unz];n(:,2)=[Vnx,Vny,Vnz];n(:,3)=[Wnx,Wny,Wnz]
    do component=0,3
      if (component==0) then
        nx=Prnx;ny=Prny;nz=Prnz
      else
        nx=n(1,component);ny=n(2,component);nz=n(3,component)
      endif
      do side=1,2
        associate(p=>resident_planes(side,component))
          if (allocated(p%star)) deallocate(p%star,p%flux,p%record)
          p%nx=nx;p%ny=ny;p%k=merge(1,nz,side==1);p%direction=side+4
          allocate(p%star(nx,ny),p%flux(nx,ny),p%record(nx,ny))
          p%record=0;p%star=0;p%flux=0
          if (component==0) then
            do q=1,size(WMPoints)
              if (WMPoints(q)%zk/=p%k) cycle
              i=WMPoints(q)%xi;j=WMPoints(q)%yj
              if (i<1.or.i>nx.or.j<1.or.j>ny) return
              if (p%record(i,j)/=0) return
              p%record(i,j)=q;p%star(i,j)=WMPoints(q)%ustar
              p%distz=WMPoints(q)%distz
            enddo
            if (any(p%record==0) .or. p%distz==0) return
            do j=1,ny
              do i=1,nx
                if (WMPoints(p%record(i,j))%distz/=p%distz) return
              enddo
            enddo
          else
            if (.not.associated(WMPointsUVW(side+4,component)%points)) return
            points=>WMPointsUVW(side+4,component)%points
            if (size(points)==0) then
              p%nx=0;p%ny=0
              cycle
            endif
            if (size(points)/=nx*ny) return
            if (any(points%z0>0) .or. any(points%prescribed_ustar)) return
            if (any(points%distx/=0) .or. any(points%disty/=0)) return
            if (any(points%wallu/=0) .or. any(points%wallv/=0) .or. any(points%wallw/=0)) return
            p%distz=points(1)%distz
            if (p%distz==0.or.any(points%distz/=p%distz).or.any(points%zk/=p%k)) return
            do q=1,size(points)
              i=points(q)%xi;j=points(q)%yj
              if (i<1.or.i>nx.or.j<1.or.j>ny) return
              if (p%record(i,j)/=0) return
              p%record(i,j)=q;p%star(i,j)=points(q)%ustar;p%flux(i,j)=points(q)%fluxp
            enddo
            if (any(p%record==0)) return
          endif
        end associate
      enddo
    enddo
    resident_wall_epoch=resident_flow%generation
    ok=.true.
  end function

  subroutine ResidentPrStress(U,V,W)
    real(knd),contiguous,target,intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    call ResidentWallStress(U,V,W,0)
  end subroutine

  subroutine ResidentUVWStress(U,V,W)
    real(knd),contiguous,target,intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    integer :: component
    do component=1,3
      call ResidentWallStress(U,V,W,component)
    enddo
  end subroutine

  subroutine ResidentWallStress(U,V,W,component)
    use, intrinsic :: iso_c_binding
    use fort_scoped_memory, only: FORT_SCOPE_GPU
    use elmm_resident_pressure_state, only: resident_flow,pressure_require
@USES@
    real(knd),contiguous,target,intent(in) :: U(-2:,-2:,-2:),V(-2:,-2:,-2:),W(-2:,-2:,-2:)
    integer,intent(in) :: component
    integer(c_int64_t) :: handles(3),star_handle,flux_handle
    integer :: side
    if (resident_wall_epoch/=resident_flow%generation) error stop 'wall epoch not prepared'
    handles(1)=resident_flow%bind_real(U,lbound(U),.true.,origin='walls::stress::u')
    handles(2)=resident_flow%bind_real(V,lbound(V),.true.,origin='walls::stress::v')
    handles(3)=resident_flow%bind_real(W,lbound(W),.true.,origin='walls::stress::w')
    do side=1,2
      associate(p=>resident_planes(side,component))
        if (p%nx==0) cycle
        star_handle=resident_flow%bind_real(p%star,[1,1],.true.,origin='walls::stress::star')
        flux_handle=0
        if (component>0) flux_handle=resident_flow%bind_real(p%flux,[1,1],.false.,origin='walls::stress::flux')
        select case(component)
@STRESS@
        end select
      end associate
    enddo
  end subroutine

  subroutine ResidentWallCorrection(result,velocity,nu,component,side)
    use, intrinsic :: iso_c_binding
    use fort_scoped_memory, only: FORT_SCOPE_GPU
    use elmm_resident_pressure_state, only: resident_flow,pressure_require
@USES@
    real(knd),contiguous,target,intent(inout) :: result(-2:,-2:,-2:)
    real(knd),contiguous,target,intent(in) :: velocity(-2:,-2:,-2:),nu(-1:,-1:,-1:)
    integer,intent(in) :: component,side
    integer(c_int64_t) :: handles(3),flux_handle
    if (resident_wall_epoch/=resident_flow%generation) error stop 'wall correction outside prepared epoch'
    associate(p=>resident_planes(side,component))
      if (p%nx==0) return
      handles(1)=resident_flow%bind_real(result,lbound(result),.true.,origin='walls::correction::result')
      handles(2)=resident_flow%bind_real(velocity,lbound(velocity),.true.,origin='walls::correction::velocity')
      handles(3)=resident_flow%bind_real(nu,lbound(nu),.true.,origin='walls::correction::nu')
      flux_handle=resident_flow%bind_real(p%flux,[1,1],.false.,origin='walls::correction::flux')
      select case(component*10+side)
@CORRECTION@
      end select
    end associate
  end subroutine

  subroutine ResidentWallsPublish(diagnostics_only)
    use, intrinsic :: iso_c_binding
    use elmm_resident_pressure_state, only: resident_flow
    logical,optional,intent(in) :: diagnostics_only
    logical :: diagnostics
    integer :: side,component,i,j,q,last_side,last_component
    integer(c_int64_t) :: handle
    type(WMPointUVW),pointer :: points(:)
    if (.not.resident_flow%active.or.resident_wall_epoch/=resident_flow%generation) return
    diagnostics=.false.
    if (present(diagnostics_only)) diagnostics=diagnostics_only
    last_side=2;last_component=3
    if (diagnostics) then
      last_side=1;last_component=2
    endif
    do component=0,last_component
      do side=1,last_side
        associate(p=>resident_planes(side,component))
          if (p%nx==0) cycle
          handle=resident_flow%bind_real(p%star,[1,1],.true.,origin='walls::publish::star')
          call resident_flow%observe([handle])
          if (component==0) then
            do j=1,p%ny
              do i=1,p%nx
                WMPoints(p%record(i,j))%ustar=p%star(i,j)
              enddo
            enddo
          else
            points=>WMPointsUVW(side+4,component)%points
            if (.not.diagnostics) then
              handle=resident_flow%bind_real(p%flux,[1,1],.false.,origin='walls::publish::flux')
              call resident_flow%observe([handle])
            endif
            do j=1,p%ny
              do i=1,p%nx
                q=p%record(i,j);points(q)%ustar=p%star(i,j)
                if (.not.diagnostics) then
                  points(q)%fluxp=p%flux(i,j);points(q)%fluxm=p%flux(i,j)
                endif
              enddo
            enddo
          endif
        end associate
      enddo
    enddo
  end subroutine
"""
    return declarations, procedures.replace("@USES@", "\n".join(uses)).replace("@STRESS@", "\n".join(dispatch_stress)).replace("@CORRECTION@", "\n".join(dispatch_correction))


def integrate_wall_entries(integration, entries, directory):
    from compiler.scopes.resident_splice import addressable_arrays

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    sources = {Path(r["path"]).name: Path(r["path"]).read_text() for r in integration["sources"]}
    declarations, procedures = _driver(entries)
    wall = sources["wallmodels.f90"]
    start = re.search(r"(?im)^[ \t]*module\s+Wallmodels\b", wall).start()
    contains = re.search(r"(?im)^[ \t]*contains\b[^\n]*$", wall[start:]).start() + start
    wall = wall[:contains] + declarations + wall[contains:]
    end = re.search(r"(?im)^\s*end\s*module\s+Wallmodels\b", wall)
    if end is None:
        raise ValueError("Missing wall model module end")
    wall = wall[:end.start()] + procedures + wall[end.start():]
    for name, anchor in (("GroundUstar", "    if (offset_to_global_z==0) then"),
                         ("GroundUstarUVW", "    if (any(WMPoints%zk == 1)) then")):
        f_lo, f_hi = find_routine_span(wall, name, kind="function")
        f_body = wall[f_lo:f_hi]
        f_body = replace_once(f_body, anchor, "    call ResidentWallsPublish(.true.)\n", after=False, insert=True)
        wall = wall[:f_lo] + f_body + wall[f_hi:]
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
          call ResidentWallsPublish()
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

    text = sources["momentum_diffusion.f90"]
    for point, group in entries["correction_groups"].items():
        component = "uvw".index(point[0]) + 1
        for index, record in enumerate(group):
            block = record["block"]
            find_unique_span(text, block)
            call = ""
            if point[1] == "z" and index == 0:
                side = 1 if point[2] == "m" else 2
                field = point[0].upper()
                call = f"        call ResidentWallCorrection({field}2,{field},nu,{component},{side})\n"
            # The validated primary topology has no x/y wall records. The two
            # z correction loops are fused per plane, without changing either
            # point's accumulation order or the ordering between planes.
            replacement = "    block\n      use elmm_resident_pressure_state, only: resident_flow\n      if (resident_flow%active) then\n" + call + "      else\n" + block + "      endif\n    end block\n"
            text = text.replace(block, replacement, 1)
    sources["momentum_diffusion.f90"] = text

    text = sources["outputs.f90"]
    text = replace_once(text, "    use Wallmodels, only: ComputeViscsWM\n", "    use Wallmodels, only: ComputeViscsWM, ResidentWallsPublish\n")
    text = replace_once(text, "        call resident_flow%end_epoch()", "        call ResidentWallsPublish()\n", after=False, insert=True)
    sources["outputs.f90"] = text
    text = sources["pressure.f90"]
    text = replace_once(
        text,
        "  subroutine FinalizeResidentPressure()\n",
        "    use Wallmodels, only: ResidentWallsPublish\n    call ResidentWallsPublish()\n",
        after=True,
        insert=True,
    )
    records = []
    for name, text in sources.items():
        path = directory / name
        path.write_text(addressable_arrays(text))
        records.append(identity(path))
    result = {"version": 1, "sources": records, "basis": integration,
              "wall_contract": "uniform flat stationary complete z planes; checked topology; no x/y wall records",
              "dead_store": "pre-SGS point viscosity overwritten at every interior cell by primary Sigma SGS",
              "gpu_execution_established": False}
    (directory / "wall-integration.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
