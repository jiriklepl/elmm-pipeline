"""Configured source preparation for the resident RK implementation.

The application adapter owns the configured source and the RK lifetime. The
compiler owns numerical extraction, dependence analysis and shared CUDA entry
generation. There is no second buffer-validity or placement implementation here.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .ast_rewrites import (
    find_all_calls,
    find_call_span,
    find_routine_span,
    inline_tile_scratch_nests,
    parse_fortran,
    promote_line_snapshots,
    replace_once,
)


def identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def configured_sources(paths, procedures, directory, *, fortran_compiler, defines):
    """Expand configured includes and serialize selected numerical coordinators.

    The original application is untouched. OpenMP is removed only in the selected
    procedures of the generated variant: its kernels will be dispatched by the
    serial resident coordinator. All statements, branches and arithmetic remain.
    Numerical dependence checks subsequently run on that complete serial source.
    This is not permission to execute uncovered native effects on stale buffers.
    """
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    parser = ParserFactory().create(std="f2008")
    selected = {name.lower() for name in procedures}
    selected_modules = {name.split("::", 1)[0] for name in selected if "::" in name}
    records = []
    for filename in paths:
        source = Path(filename).resolve(strict=True)
        before = identity(source)
        command = [str(fortran_compiler), "-cpp", "-E", "-P", "-O3", "-ffree-line-length-none", "-fopenmp",
                   *("-D" + value for value in defines), "-I" + str(source.parent), str(source)]
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        if result.returncode:
            raise RuntimeError("RK preprocessing failed: " + str(source) + "\n" + result.stderr)
        lines = result.stdout.splitlines(keepends=True)
        removed = []
        modules_in_file = {m.group(1).lower() for m in re.finditer(r"(?im)^\s*module\s+([a-z_]\w*)\b", result.stdout)}
        if modules_in_file & selected_modules:
            tree = parser(FortranStringReader(result.stdout, ignore_comments=False))
            for module in (node for node in walk(tree) if type(node).__name__ == "Module"):
                header = next(node for node in module.content if type(node).__name__ == "Module_Stmt")
                module_name = str(header.items[1]).lower()
                for routine in (node for node in walk(module) if type(node).__name__ == "Subroutine_Subprogram"):
                    statement = next(node for node in routine.content if type(node).__name__ == "Subroutine_Stmt")
                    name = module_name + "::" + str(statement.items[1]).lower()
                    if name not in selected:
                        continue
                    ending = routine.content[-1]
                    first, last = statement.item.span[0], ending.item.span[1]
                    for index in range(first - 1, last):
                        if re.match(r"\s*!\$omp\b", lines[index], re.I):
                            removed.append({"line": index + 1, "directive": lines[index].rstrip(), "procedure": name})
                            lines[index] = "\n"
        target = directory / (hashlib.sha256(str(source).encode()).hexdigest()[:12] + "_" + source.name)
        target.write_text("".join(lines))
        if identity(source) != before:
            raise ValueError("RK source changed while preparing: " + str(source))
        records.append({"original": before, "prepared": identity(target), "command": command,
                        "serialized_directives": removed})
    record = {"version": 1, "kind": "configured_resident_rk_sources", "defines": list(defines),
              "procedures": sorted(selected), "sources": records,
              "execution_authorized": False,
              "contract": "serial coordinator source; uncovered native effects need explicit coherence transactions"}
    (directory / "configured-sources.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


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


def guard_periodic_entries(extracted, directory):
    """Keep halo copies independent under the adapter's three-cell minimum.

    The added predicate is also a reached-call guard. Small grids use the native
    implementation. The compiler proves these integer predicates rather than
    accepting an assertion that a potentially overlapping copy is independent.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    for original in extracted["entries"]:
        if original["procedure"] != "boundaries::periodic_corners_edges":
            continue
        source = Path(original["input_path"]).read_text()
        assignments = list(re.finditer(r"(?im)^([ \t]*)u\(", source))
        if len(assignments) != len(original["source_loop_spans"]):
            raise ValueError("Periodic corner entry must have one assignment per original nest")
        guard = "nx >= 3 .and. ny >= 3 .and. nz >= 3"
        prepared = source
        for assignment in reversed(assignments):
            end = re.search(r"(?im)^[ \t]*end do\b", source[assignment.start():])
            if end is None:
                raise ValueError("Periodic corner entry has no enclosing point loop")
            stop = assignment.start() + end.start()
            prepared = (prepared[:assignment.start()] + "      if (" + guard + ") then\n"
                        + prepared[assignment.start():stop] + "      endif\n" + prepared[stop:])
        path = directory / Path(original["input_path"]).name
        path.write_text(prepared)
        entries.append({**original, "input_path": str(path),
                        "runtime_guards": [*original["runtime_guards"], guard],
                        "adapter_guard": {"condition": guard, "original": identity(original["input_path"]),
                                          "prepared": identity(path), "false_path": "original native procedure"}})
    return {**extracted, "entries": entries, "rejected": []}


def _native_wall_correction(body):
    """Retain the original correction with conservative, bounded host access.

    This adapter covers the original second-order no-branch correction includes.
    Their stencil radius is at most two around the named wall-point coordinates.
    Empty point sets evaluate no field descriptor or coordinate reduction.
    """
    points = set(re.findall(r"\b([UVW][xyz][mp]WMpoints)\s*\(i\)\s*%", body, re.I))
    writes = set(re.findall(r"(?im)^\s*([UVW]2)\s*\([^\n]*", body))
    if len(points) != 1 or len(writes) != 1:
        raise ValueError("Unknown native wall-correction body")
    points, target = next(iter(points)), next(iter(writes))
    fields = [target, target[0], "nu"]
    lines = ["    block", "      use, intrinsic :: iso_c_binding",
             "      use elmm_resident_pressure_state, only: resident_flow",
             "      integer(c_int64_t) :: wall_handles(3)", "      integer :: wall_first(3),wall_last(3),wall_index",
             "      wall_handles=0", "      if (resident_flow%active) then", "        if (size(" + points + ")>0) then",
             "          wall_first=[" + ",".join("minval(" + points + "%" + axis + ")" for axis in ("xi", "yj", "zk")) + "]-2",
             "          wall_last=[" + ",".join("maxval(" + points + "%" + axis + ")" for axis in ("xi", "yj", "zk")) + "]+2"]
    for index, field in enumerate(fields, 1):
        lines.append(f"          wall_handles({index})=resident_flow%host_box_real({field},lbound({field}), &")
        lines.append("            wall_first,wall_last," + (".true." if index == 1 else ".false.") + ")")
    lines += ["        endif", "      endif", body.rstrip(),
              "      do wall_index=1,3", "        if (wall_handles(wall_index)/=0) call resident_flow%host_end(wall_handles(wall_index))",
              "      enddo", "    end block"]
    return "\n".join(lines) + "\n"


def materialize_entries(packages, directory):
    """Splice the compiled entries into ELMM's sources through its resident flow domain.

    Activation belongs to the outer RK adapter. Unsupported reached loops
    publish before native execution, except the wall-point corrections, which
    acquire smaller exact geometric supersets.
    """
    from compiler.scopes.resident_splice import ResidentDomain, splice_resident_entries

    def native(region, original):
        if region["procedure"] == "momentumdiffusion::momentumdiffusion_nobranch_2ord":
            return _native_wall_correction(original)
        return None

    domain = ResidentDomain("elmm_resident_pressure_state", "resident_flow", "pressure_require", prefix="rk")
    return splice_resident_entries(packages, directory, domain=domain, native=native)


def pointwise_advection(source, directory):
    """Inline private tile flux arrays into their original consumers.

    Read formulas from the source. Each private flux has one point definition;
    a consumer offset selects that definition at the corresponding global point.
    Keep the directional updates and their original arithmetic order.
    This removes CPU cache tiles, without changing the advection discretization.
    """
    source = Path(source).resolve(strict=True)
    text = source.read_text()
    updated, changes = inline_tile_scratch_nests(text)
    if not changes:
        raise ValueError("Incomplete fourth-order advection preparation")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / source.name
    target.write_text(updated)
    record = {
        "version": 1,
        "kind": "pointwise_advection",
        "original": identity(source),
        "replacement": identity(target),
        "directional_updates": len(changes),
        "formula_source": "original private flux definitions and consumer expressions",
    }
    (directory / "pointwise-advection.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def snapshot_filter(source, directory):
    """Lift each line snapshot to a complete snapshot for pointwise filtering.

    The original filter reads a private copy of each input line before writing
    that line. A complete input snapshot preserves those reads and the original
    seven-term accumulation order while exposing independent output points.
    Retire the temporary through coherence before Fortran deallocates it.
    """
    source = Path(source).resolve(strict=True)
    text = source.read_text()
    updated, edits = promote_line_snapshots(text)
    if not edits:
        raise ValueError("Expected at least one line-snapshot filter routine")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / source.name
    target.write_text(updated)
    record = {
        "version": 1,
        "kind": "snapshot_filter",
        "original": identity(source),
        "replacement": identity(target),
        "arithmetic": "original Filter3ord weights, conditions and accumulation order",
        "lifetime": "retire snapshot after queued readers before automatic deallocation",
    }
    (directory / "snapshot-filter.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def _decl_shape_lower_bounds(node, entity_shape):
    attrs_str = re.sub(r"\s+", "", str(node.items[1] or "")).lower()
    if entity_shape is not None:
        return re.sub(r"\s+", "", str(entity_shape)).lower()
    m = re.search(r"dimension\(([^()]+)\)", attrs_str)
    return m.group(1) if m else ""


def contiguous_periodic_boundary(text):
    from fparser.two.utils import walk

    b_lo, b_hi = find_routine_span(text, "BoundUVW_do_BC")
    owner = text[b_lo:b_hi]
    all_calls = find_all_calls(text, "periodic_corners_edges")
    outside = [c for c in all_calls if not (b_lo <= c[0] and c[1] <= b_hi)]
    inside = [c for c in all_calls if b_lo <= c[0] and c[1] <= b_hi]
    normalized_calls = sorted(re.sub(r"\s+", "", ",".join(c[3])).lower() for c in inside)
    if outside or normalized_calls != ["u,1", "v,2", "w,3"]:
        raise ValueError("Unsupported periodic boundary callers")

    owner_tree = parse_fortran(owner)
    found_owner_arrays: dict[str, bool] = {}
    for decl in walk(owner_tree):
        if type(decl).__name__ != "Type_Declaration_Stmt":
            continue
        dtype_str = re.sub(r"\s+", "", str(decl.items[0])).lower()
        attrs_str = re.sub(r"\s+", "", str(decl.items[1] or "")).lower()
        if not dtype_str.startswith("real"):
            continue
        if "intent(inout)" not in attrs_str or "contiguous" not in attrs_str:
            continue
        for ent in decl.items[2].items:
            vname = str(ent.items[0]).lower()
            if vname in {"u", "v", "w"} and _decl_shape_lower_bounds(decl, ent.items[1]) == "-2:,-2:,-2:":
                found_owner_arrays[vname] = True
    if set(found_owner_arrays) != {"u", "v", "w"}:
        raise ValueError("BoundUVW_do_BC must declare contiguous U, V, W arrays")

    p_lo, p_hi = find_routine_span(text, "periodic_corners_edges")
    body = text[p_lo:p_hi]
    body_tree = parse_fortran(body)
    target_spans = []
    for decl in walk(body_tree):
        if type(decl).__name__ != "Type_Declaration_Stmt":
            continue
        dtype_str = re.sub(r"\s+", "", str(decl.items[0])).lower()
        attrs_str = re.sub(r"\s+", "", str(decl.items[1] or "")).lower()
        if not dtype_str.startswith("real") or "intent(inout)" not in attrs_str:
            continue
        for ent in decl.items[2].items:
            if str(ent.items[0]).lower() == "u" and _decl_shape_lower_bounds(decl, ent.items[1]) == "-2:,-2:,-2:":
                target_spans.append((decl.item.span, "contiguous" in attrs_str))
    if len(target_spans) != 1:
        raise ValueError("Unsupported U declaration in periodic_corners_edges")
    (first_line, last_line), has_contig = target_spans[0]
    if not has_contig:
        lines = body.splitlines(keepends=True)
        chunk = "".join(lines[first_line - 1 : last_line])
        before, after = chunk.split("::", 1)
        lines[first_line - 1 : last_line] = [before.rstrip() + ", contiguous ::" + after]
        body = "".join(lines)
    return text[:p_lo] + body + text[p_hi:]


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


def integrate_rk_epoch(configured, numerical, pressure, directory):
    """Connect the compiled primary RK path and the pressure child to one epoch.

    Uniform DP, Sigma SGS, explicit FD2 diffusion and periodic/no-slip channel
    boundaries are the initial supported execution configuration. Other settings
    retain their native path. Physical allocations survive the proved three
    MOVE_ALLOC exchanges; the RK caller return publishes before unmanaged work.
    """
    from compiler.scopes.resident_splice import addressable_arrays


    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    replacements = {item["original"]["path"]: item["replacement"]["path"] for item in numerical["sources"]}
    sources = {Path(item["original"]["path"]).name: Path(replacements.get(item["prepared"]["path"], item["prepared"]["path"]))
               for item in configured["sources"] if "poisfft" not in item["original"]["path"].lower()}
    # Only modules whose numerical code or address contracts are changed need
    # rebuilding. No source graph owner or legacy placement wrapper is retained.
    names = {"parameters.f90", "arrayutilities.f90", "subgrid.f90", "dynamics.f90", "momentum_diffusion.f90",
             "advection.f90", "scalarboundaries.f90", "boundaries.f90", "tsteps.f90", "linearforcing.f90",
             "volumesources.f90", "sponge.f90", "tiling.f90", "filters.f90", "wallmodels.f90",
             "advection_variable_z.f90", "water_thermodynamics.f90", "scalar_diffusion.f90"}
    output = {name: sources[name].read_text() for name in names}
    output["pressure.f90"] = Path(pressure["pressure"]["path"]).read_text()
    output["main.f90"] = Path(pressure["main"]["path"]).read_text()
    output["boundaries.f90"] = contiguous_periodic_boundary(output["boundaries.f90"])
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
        return (indent + "block\n" + indent + "  use elmm_resident_pressure_state, only: resident_flow\n"
                + indent + "  if (resident_flow%active) call resident_flow%publish()\n" + indent + "end block\n")

    tsteps = output["tsteps.f90"]
    lo, hi = find_routine_span(tsteps, "TMarchRK3")
    body = tsteps[lo:hi]
    body = at_once(body, "    use RungeKutta_coefficients\n", "    use, intrinsic :: iso_c_binding\n"
                   "    use elmm_resident_pressure_state, only: resident_flow\n", after=True)
    sub_lo, _, _, _ = find_call_span(body, "SubgridStresses")
    begin_epoch_code = (
        "      if (.not. resident_flow%joined) then\n"
        "        if (ResidentRKSupported()) then\n"
        "          call resident_flow%begin_epoch(0_c_int,2147483648_c_size_t)\n"
        "          resident_flow%joined=.true.\n        endif\n      endif\n\n"
    )
    body = body[:sub_lo] + begin_epoch_code + body[sub_lo:]
    end_do_matches = list(re.finditer(r"(?im)^[ \t]*end\s*do\b[^\n]*", body))
    if not end_do_matches:
        raise ValueError("Missing RK stage loop")
    end_pos = end_do_matches[-1].end()
    body = body[:end_pos] + ("\n    if (resident_flow%joined) then\n"
                             "      call resident_flow%end_epoch()\n      resident_flow%joined=.false.\n    endif\n") + body[end_pos:]
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
    supported = gridtype==GRID_UNIFORM .and. explicit_diffusion .and. discretization_order==2
    supported = supported .and. sgstype==SigmaModel .and. (filtertype==0 .or. filtertype==1)
    supported = supported .and. (advection_method==2 .or. advection_method==4 .or. advection_method==5 .or. advection_method==6)
    supported = supported .and. min(Unx,Uny,Unz,Vnx,Vny,Vnz,Wnx,Wny,Wnz)>=3
    supported = supported .and. all(Btype([We,Ea,So,No])==BC_PERIODIC) .and. all(Btype([Bo,To])==BC_NOSLIP)
    supported = supported .and. .not. enable_buoyancy .and. .not. enable_moisture .and. .not. enable_liquid
    supported = supported .and. num_of_scalars==0 .and. abs(Coriolis_parameter)<=tiny(1._knd)
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
    lo, hi = find_routine_span(dynamics, "Convection")
    body = dynamics[lo:hi]
    body = at_once(body, "  contains", "    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
                   "      call resident_flow%release_real(beta)\n      call resident_flow%release_real(rho)\n    end block\n\n")
    output["dynamics.f90"] = dynamics[:lo] + body + dynamics[hi:]

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

    records = []
    for name, text in output.items():
        target = directory / name
        target.write_text(addressable_arrays(text))
        records.append(identity(target))
    record = {"version": 1, "kind": "resident_rk_epoch_integration", "sources": records,
              "epoch": "all reached RK stages in one invocation; pressure borrows caller domain",
              "allocation_exchange": "original exchange_alloc_3D: A->tmp, B->A, tmp->B with MOVE_ALLOC",
              "unmanaged_return": "batch observe then retire borrowed handles at RK caller return",
              "gpu_execution_established": False}
    (directory / "rk-integration.json").write_text(json.dumps(record, indent=2) + "\n")
    return record
