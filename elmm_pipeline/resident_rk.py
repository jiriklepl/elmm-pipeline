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
        tree = parser(FortranStringReader(result.stdout, ignore_comments=False))
        removed = []
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


def generate_entries(extracted, compiler_root, python, directory):
    from .local_adapter import transform

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    entries, rejected = [], list(extracted["rejected"])
    for region in extracted["entries"]:
        target = directory / region["source_identity"][:16]
        try:
            result = transform(Path(region["input_path"]), Path(compiler_root), Path(python), target,
                               entry=region["entry"],
                               gpu_options={"gpu_policy": "sections", "memory_model": "scoped",
                                            "scope_transfers": "direct", "host_threads": 4})
        except RuntimeError as error:
            rejected.append({"procedure": region["procedure"], "first_line": region["first_line"],
                             "last_line": region["last_line"], "reason": str(error)})
            continue
        execution = result["shared_entry"]["region_execution"]
        unavailable = [item for item in execution if not item["gpu_available"]]
        if unavailable:
            rejected.append({"procedure": region["procedure"], "first_line": region["first_line"],
                             "last_line": region["last_line"],
                             "reason": "Resident GPU entry contains native-only regions", "regions": unavailable})
            continue
        entries.append({"region": region, "directory": str(target), "entry": result["shared_entry"]})
    record = {"version": 1, "kind": "resident_rk_device_entries", "source_inputs": extracted["source_inputs"],
              "entries": entries, "rejected": rejected, "gpu_execution_established": False}
    (directory / "device-entries.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def value_resident_geometry(extracted, directory):
    """Pass reached descriptor origins and validated grid geometry by VALUE.

    Lower-bound actuals are evaluated by the caller after binding the reached
    full allocation. Primary-grid counts and spacings are initialized before the
    RK adapter is entered. Other scalars retain their guarded reference ABI.
    VALUE is a language-level input contract, not a naming exemption in the
    compiler's conditional-input analysis.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    grid = {"parameters::" + name for name in ("dxmin", "dymin", "dzmin", "unx", "uny", "unz",
            "vnx", "vny", "vnz", "wnx", "wny", "wnz")}
    entries = []
    for original in extracted["entries"]:
        values = [p["name"] for p in original["parameters"]
                  if p["lower_bound_dimension"] is not None or p["resource"] in grid]
        text = Path(original["input_path"]).read_text()
        for name in values:
            pattern = r"(?im)^(integer|real\(8\)), intent\(in\) :: " + re.escape(name) + r"$"
            text, count = re.subn(pattern, r"\1, value, intent(in) :: " + name, text)
            if count != 1:
                raise ValueError("Unexpected resident scalar declaration: " + name)
        path = directory / Path(original["input_path"]).name
        path.write_text(text)
        entries.append({**original, "input_path": str(path), "value_inputs": values,
                        "value_contract": "reached descriptor origins and initialized primary-grid geometry",
                        "original_numerical_input": identity(original["input_path"])})
    return {**extracted, "entries": entries}


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
        lines.append(f"          wall_handles({index})=resident_flow%host_box_real64({field},lbound({field}), &")
        lines.append("            wall_first,wall_last," + (".true." if index == 1 else ".false.") + ")")
    lines += ["        endif", "      endif", body.rstrip(),
              "      do wall_index=1,3", "        if (wall_handles(wall_index)/=0) call resident_flow%host_end(wall_handles(wall_index))",
              "      enddo", "    end block"]
    return "\n".join(lines) + "\n"


def materialize_entries(packages, directory):
    """Replace proved numerical spans and preserve explicit native boundaries.

    Activation belongs to the outer RK adapter. Every unsupported reached loop
    publishes before native execution; the wall-point adapter uses smaller exact
    geometric supersets. This function never infers effects for other calls.
    """
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    edits, expected = {}, {}
    for package in packages:
        expected.update(package["source_inputs"])
        for item in package["entries"]:
            region, entry = item["region"], item["entry"]
            bindings = {binding["resource"]: binding for binding in region["resources"]}
            actual, registration, handles = {}, [], []
            for parameter in region["parameters"]:
                binding = bindings[parameter["resource"]]
                name = binding["name"]
                if parameter["lower_bound_dimension"] is not None:
                    actual[parameter["name"]] = f"lbound({name},{parameter['lower_bound_dimension']})"
                elif parameter["rank"]:
                    handle = "rk_handle_" + parameter["name"]
                    actual[parameter["name"]] = handle
                    method = {("real", 8): "bind_real64", ("integer", 4): "bind_integer32"}.get((binding["type"], binding["kind"]))
                    if method is None:
                        raise ValueError("Unsupported resident array ABI: " + str(binding))
                    handles.append(handle)
                    origin = region["procedure"] + "::" + name
                    registration.append(f"        {handle}=resident_flow%{method}({name},lbound({name}),.true., &\n"
                                        f"          origin='{origin}')")
                else:
                    actual[parameter["name"]] = name
            arguments = ["resident_flow%context", "FORT_SCOPE_GPU"] + [actual[name] for name in entry["argument_order"][2:]]
            code = ["    block", "      use, intrinsic :: iso_c_binding",
                    "      use fort_scoped_memory, only: FORT_SCOPE_GPU",
                    "      use elmm_resident_pressure_state, only: resident_flow, pressure_require",
                    "      use " + entry["fortran_module"] + ", only: rk_device_run => run",
                    "      integer(c_int64_t) :: " + ",".join(handles),
                    "      if (resident_flow%active) then"]
            for guard in region["runtime_guards"]:
                code.append("        if (.not. (" + guard + ")) error stop 'resident numerical geometry changed'")
            code += registration + ["        call pressure_require(rk_device_run( &", "          " + ", &\n          ".join(arguments) + "))",
                                    "      else", "@ORIGINAL@", "      endif", "    end block"]
            edits.setdefault(region["source"], []).append((region["first_line"], region["last_line"], "\n".join(code) + "\n"))
        for region in package["rejected"]:
            # Extraction rejections carry procedure identity; find that source
            # in successful entries from the same procedure, when available.
            sources = {entry["region"]["source"] for entry in package["entries"]
                       if entry["region"]["procedure"] == region["procedure"]}
            if not sources:
                continue  # Entirely native routine; outer call boundary owns it.
            if len(sources) != 1:
                raise ValueError("Ambiguous resident native source")
            source = next(iter(sources))
            wall = region["procedure"] == "momentumdiffusion::momentumdiffusion_nobranch_2ord"
            code = "@WALL@" if wall else ("    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
                    "      if (resident_flow%active) call resident_flow%publish()\n@ORIGINAL@\n    end block\n")
            edits.setdefault(source, []).append((region["first_line"], region["last_line"], code))
    records = []
    for source, replacements in edits.items():
        path = Path(source)
        if identity(path)["sha256"] != expected[source]:
            raise ValueError("Resident numerical source changed: " + source)
        lines = path.read_text().splitlines(keepends=True)
        replacements.sort()
        if any(a[1] >= b[0] for a, b in zip(replacements, replacements[1:])):
            raise ValueError("Overlapping resident source replacements")
        for first, last, code in reversed(replacements):
            original = "".join(lines[first - 1:last])
            replacement = _native_wall_correction(original) if code == "@WALL@" else code.replace("@ORIGINAL@", original.rstrip())
            lines[first - 1:last] = [replacement]
        target = directory / path.name
        target.write_text("".join(lines))
        records.append({"original": identity(path), "replacement": identity(target), "region_count": len(replacements)})
    result = {"version": 1, "kind": "resident_rk_numerical_source_edits", "sources": records,
              "activation_authority": "outer RK adapter; native call effects are not inferred here"}
    (directory / "source-edits.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def pointwise_advection(source, directory):
    """Inline ELMM's private tile flux arrays into their original consumers.

    Read formulas from the source. Each private flux has one point definition;
    a consumer offset selects that definition at the corresponding global point.
    Keep the three directional updates and their original arithmetic order.
    This removes CPU cache tiles, without changing the advection discretization.
    """
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    source = Path(source).resolve(strict=True)
    text = source.read_text()
    lines = text.splitlines(keepends=True)
    tree = ParserFactory().create(std="f2008")(FortranStringReader(text, ignore_comments=False))
    changes = []

    def statements(node):
        return [item for item in node.content if type(item).__name__ not in {"Comment", "Nonlabel_Do_Stmt", "End_Do_Stmt"}]

    def inner(loop, variable):
        header = next(item for item in loop.content if type(item).__name__ == "Nonlabel_Do_Stmt")
        if str(header.items[1].items[1][0]).lower() != variable:
            raise ValueError("Unexpected advection tile iterator")
        children = statements(loop)
        if len(children) != 1 or type(children[0]).__name__ != "Block_Nonlabel_Do_Construct":
            raise ValueError("Unexpected advection tile nesting")
        return children[0]

    def shift(expression, offsets):
        return re.sub(r"\b[ijk]\b", lambda match: "(" + match[0] + "+(" + str(offsets["ijk".index(match[0].lower())]) + "))", expression, flags=re.I)

    for routine in (node for node in walk(tree) if type(node).__name__ == "Subroutine_Subprogram"):
        header = next(node for node in routine.content if type(node).__name__ == "Subroutine_Stmt")
        name = str(header.items[1]).lower()
        if name not in {"cd4divu", "cd4divv", "cd4divw"}:
            continue
        component = name[-1].upper()
        mask = next((node for node in walk(routine) if type(node).__name__ == "Type_Declaration_Stmt"
                     and re.search(r"\bi_mask4\b", str(node), re.I)), None)
        if mask is None or "[-1,9,9,-1]" not in re.sub(r"\s+", "", str(mask)).lower():
            raise ValueError("Unsupported fourth-order interpolation mask")
        execution = next(node for node in routine.content if type(node).__name__ == "Execution_Part")
        tiles = [node for node in execution.content if type(node).__name__ == "Block_Nonlabel_Do_Construct"]
        if len(tiles) != 3:
            raise ValueError("Expected three directional advection tiles")
        for tile in tiles:
            bi = inner(inner(tile, "bk"), "bj")
            pair = statements(bi)
            if len(pair) != 2 or any(type(node).__name__ != "Block_Nonlabel_Do_Construct" for node in pair):
                raise ValueError("Advection tile must have one producer and one consumer nest")
            producer, consumer = pair
            producer_leaf = inner(inner(producer, "k"), "j")
            consumer_leaf = inner(inner(consumer, "k"), "j")
            definitions = {}
            for statement in statements(producer_leaf):
                if type(statement).__name__ != "Assignment_Stmt":
                    raise ValueError("Advection flux producer is not straight-line arithmetic")
                lhs, _, rhs = statement.items
                key = str(lhs).replace(" ", "").lower()
                if key in {"li", "lj", "lk"}:
                    continue
                expression = str(rhs)
                for local, value in definitions.items():
                    if "(" not in local:
                        expression = re.sub(r"\b" + re.escape(local) + r"\b", "(" + value + ")", expression, flags=re.I)
                definitions[key] = expression
            if set(key for key in definitions if "(" in key) != {"uv1(li,lj,lk)", "uv3(li,lj,lk)"}:
                raise ValueError("Advection producer has unexpected scratch storage")
            body = []
            for statement in statements(consumer_leaf):
                if type(statement).__name__ == "Assignment_Stmt" and str(statement.items[0]).lower() in {"li", "lj", "lk"}:
                    continue
                rendered = str(statement)
                for reference in reversed(list(walk(statement))):
                    if type(reference).__name__ != "Part_Ref" or str(reference.items[0]).lower() not in {"uv1", "uv3"}:
                        continue
                    offsets = []
                    for axis, index in zip(("li", "lj", "lk"), reference.items[1].items, strict=True):
                        match = re.fullmatch(axis + r"(?:([+-])(\d+))?", re.sub(r"\s+", "", str(index)).lower())
                        if match is None:
                            raise ValueError("Nonconstant advection scratch offset")
                        offsets.append(0 if match[1] is None else int(match[1] + match[2]))
                    value = definitions[str(reference.items[0]).lower() + "(li,lj,lk)"]
                    rendered = rendered.replace(str(reference), "(" + shift(value, offsets) + ")")
                body.append(rendered)
            replacement = (f"    do k=1,{component}nz\n      do j=1,{component}ny\n        do i=1,{component}nx\n"
                           + "\n".join(body) + "\n        enddo\n      enddo\n    enddo\n")
            first = next(node for node in tile.content if type(node).__name__ == "Nonlabel_Do_Stmt").item.span[0]
            last = tile.content[-1].item.span[1]
            changes.append((first, last, replacement))
    if len(changes) != 9:
        raise ValueError("Incomplete fourth-order advection preparation")
    for first, last, replacement in sorted(changes, reverse=True):
        lines[first-1:last] = [replacement]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / source.name
    target.write_text("".join(lines))
    record = {"version": 1, "kind": "pointwise_advection", "original": identity(source), "replacement": identity(target),
              "directional_updates": len(changes), "formula_source": "original private flux definitions and consumer expressions"}
    (directory / "pointwise-advection.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def snapshot_filter(source, directory):
    """Lift each line snapshot to a complete snapshot for pointwise filtering.

    The original filter reads a private copy of each input line before writing
    that line. A complete input snapshot preserves those reads and the original
    seven-term accumulation order while exposing independent output points.
    Retire the temporary through coherence before Fortran deallocates it.
    """
    from .gpu_hot_path import _once, _routine

    source = Path(source).resolve(strict=True)
    text = source.read_text()
    lo, hi = _routine(text, "Filter3ord")
    body = text[lo:hi]
    declaration = "    real(knd) :: tmp(-2:ubound(U,dir))"
    _once(body, declaration)
    body = body.replace(declaration, "    real(knd), allocatable :: tmp(:,:,:)\n")
    anchor = "    mini = lbound(U,1) + 3"
    _once(body, anchor)
    body = body.replace(anchor, "    allocate(tmp(-2:ubound(U,1),-2:ubound(U,2),-2:ubound(U,3)))\n"
                        "    do k=-2,ubound(U,3)\n      do j=-2,ubound(U,2)\n        do i=-2,ubound(U,1)\n"
                        "          tmp(i,j,k)=U(i,j,k)\n        enddo\n      enddo\n    enddo\n\n" + anchor)
    for assignment in ("tmp = U(:,j,k)", "tmp = U(i,:,k)", "tmp = U(i,j,:)"):
        _once(body, assignment)
        body = body.replace(assignment, "! Input line is in the complete snapshot.")
    for index, coordinates in (("ii", "ii,j,k"), ("jj", "i,jj,k"), ("kk", "i,j,kk")):
        _once(body, "tmp(" + index + ")")
        body = body.replace("tmp(" + index + ")", "tmp(" + coordinates + ")")
    # Express the seven fixed stencil offsets directly. The compiler can then
    # lower the small private weight array without a variable indexing loop.
    for axis, index in (("i", "ii"), ("j", "jj"), ("k", "kk")):
        loop = re.search(r"(?im)^\s*do " + index + r" = " + axis + r"-3, " + axis + r"\+3\b.*?^\s*end do\b",
                         body, re.S)
        if loop is None:
            raise ValueError("Missing original seven-point filter accumulation")
        original = loop[0]
        shifted = re.sub(r"\b" + index + r"\b", "(" + axis + "+" + index + ")", original)
        shifted = shifted.replace("(" + axis + "+" + index + ")-" + axis, index)
        shifted = shifted.replace("do (" + axis + "+" + index + ") = " + axis + "-3, " + axis + "+3",
                                  "do " + index + " = -3, 3")
        body = body[:loop.start()] + shifted + body[loop.end():]
    ending = re.search(r"(?im)^\s*end subroutine Filter3ord", body)
    if ending is None:
        raise ValueError("Missing filter procedure end")
    body = body[:ending.start()] + ("\n    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
            "      call resident_flow%release_real64(tmp)\n    end block\n") + body[ending.start():]
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    target = directory / source.name
    target.write_text(text[:lo] + body + text[hi:])
    record = {"version": 1, "kind": "snapshot_filter", "original": identity(source), "replacement": identity(target),
              "arithmetic": "original Filter3ord weights, conditions and accumulation order",
              "lifetime": "retire snapshot after queued readers before automatic deallocation"}
    (directory / "snapshot-filter.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def addressable_arrays(text):
    """Give borrowed primitive allocations and dummy arrays stable C addresses."""
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    tree = ParserFactory().create(std="f2008")(FortranStringReader(text, ignore_comments=False))
    lines = text.splitlines(keepends=True)
    changes = []
    for node in walk(tree):
        if type(node).__name__ != "Type_Declaration_Stmt":
            continue
        dtype, attributes, entities = node.items
        attrs = str(attributes).lower()
        if not re.match(r"(?i)^(real|integer|logical)\b", str(dtype)) or re.search(r"\b(target|parameter|pointer)\b", attrs):
            continue
        is_array = "dimension" in attrs or any(entity.items[1] is not None for entity in entities.items)
        parent = node.parent
        while parent is not None and type(parent).__name__ not in {"Module", "Subroutine_Subprogram", "Function_Subprogram", "Derived_Type_Def", "Main_Program"}:
            parent = getattr(parent, "parent", None)
        if not is_array or parent is None or type(parent).__name__ == "Derived_Type_Def":
            continue
        if type(parent).__name__ not in {"Module", "Main_Program"} and "intent" not in attrs and "allocatable" not in attrs:
            continue
        first, last = node.item.span
        declaration = str(node)
        before, after = declaration.split("::", 1)
        changes.append((first, last, before.rstrip() + ", TARGET ::" + after + "\n"))
    for first, last, replacement in sorted(changes, reverse=True):
        lines[first-1:last] = [replacement]
    return "".join(lines)


def contiguous_periodic_boundary(text):
    from .gpu_hot_path import _once, _routine

    start, end = _routine(text, "BoundUVW_do_BC")
    owner = text[start:end]
    call_pattern = r"(?im)^\s*call\s+periodic_corners_edges\s*\(([^)]*)\)"
    calls = re.findall(call_pattern, owner)
    if (sorted(re.sub(r"\s+", "", call).lower() for call in calls) != ["u,1", "v,2", "w,3"]
            or re.search(call_pattern, text[:start] + text[end:])):
        raise ValueError("Unsupported periodic boundary callers")
    _once(owner, "real(knd), intent(inout), contiguous :: U(-2:,-2:,-2:), "
                 "V(-2:,-2:,-2:), W(-2:,-2:,-2:)")
    start, end = _routine(text, "periodic_corners_edges")
    body = text[start:end]
    declaration = "real(knd), intent(inout) :: U(-2:,-2:,-2:)"
    _once(body, declaration)
    body = body.replace(declaration, "real(knd), intent(inout), contiguous :: U(-2:,-2:,-2:)")
    return text[:start] + body + text[end:]


def contiguous_filter_arguments(text):
    from .gpu_hot_path import _routine

    for routine in ("Filter3ord", "Filter"):
        start, end = _routine(text, routine)
        body = text[start:end]
        for declaration in (
            "real(knd), dimension(-2:,-2:,-2:), intent(inout)",
            "integer, dimension(-2:,-2:,-2:), intent(in)",
        ):
            if body.count(declaration) != 1:
                raise ValueError("Unsupported filter array declaration in " + routine)
            body = body.replace(declaration, declaration + ", contiguous")
        text = text[:start] + body + text[end:]
    return text


def integrate_rk_epoch(configured, numerical, pressure, directory):
    """Connect the compiled primary RK path and the pressure child to one epoch.

    Uniform DP, Sigma SGS, explicit FD2 diffusion and periodic/no-slip channel
    boundaries are the initial supported execution configuration. Other settings
    retain their native path. Physical allocations survive the proved three
    MOVE_ALLOC exchanges; the RK caller return publishes before unmanaged work.
    """
    from .gpu_hot_path import _once, _routine

    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    replacements = {item["original"]["path"]: item["replacement"]["path"] for item in numerical["sources"]}
    sources = {Path(item["original"]["path"]).name: Path(replacements.get(item["prepared"]["path"], item["prepared"]["path"]))
               for item in configured["sources"] if "/elmm/src/" in item["original"]["path"]}
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
    begin = re.search(r"(?im)^\s*subroutine set_int_3D\(", arrays)
    if begin is None:
        raise ValueError("Missing integer-valued REAL64 fill overload")
    end = re.search(r"(?im)^\s*end subroutine[^\n]*", arrays[begin.start():])
    stop = begin.start() + end.end()
    # Conversion is scalar and exact under the original assignment rules. One
    # implementation owns the fill, including ordering against queued readers.
    arrays = arrays[:begin.start()] + '''
    subroutine set_int_3D(to,from)
      real(knd),contiguous,intent(out) :: to(:,:,:)
      integer,intent(in) :: from
      call set_3D(to,real(from,knd))
    end subroutine
''' + arrays[stop:]
    output["arrayutilities.f90"] = arrays

    def at_once(text, anchor, addition, after=False):
        position = _once(text, anchor) + (len(anchor) if after else 0)
        return text[:position] + addition + text[position:]

    def publish_block(indent="    "):
        return (indent + "block\n" + indent + "  use elmm_resident_pressure_state, only: resident_flow\n"
                + indent + "  if (resident_flow%active) call resident_flow%publish()\n" + indent + "end block\n")

    tsteps = output["tsteps.f90"]
    lo, hi = _routine(tsteps, "TMarchRK3")
    body = tsteps[lo:hi]
    body = at_once(body, "    use RungeKutta_coefficients\n", "    use, intrinsic :: iso_c_binding\n"
                   "    use elmm_resident_pressure_state, only: resident_flow\n", after=True)
    body = at_once(body, "      call SubgridStresses(U, V, W, Pr, Temperature, Moisture)",
                   "      if (.not. resident_flow%joined) then\n"
                   "        if (ResidentRKSupported()) then\n"
                   "          call resident_flow%begin_epoch(0_c_int,2147483648_c_size_t)\n"
                   "          resident_flow%joined=.true.\n        endif\n      endif\n\n")
    end = body.rfind("    end do")
    if end < 0:
        raise ValueError("Missing RK stage loop")
    end += len("    end do")
    body = body[:end] + ("\n    if (resident_flow%joined) then\n"
                         "      call resident_flow%end_epoch()\n      resident_flow%joined=.false.\n    endif\n") + body[end:]
    tsteps = tsteps[:lo] + body + tsteps[hi:]
    lo, hi = _routine(tsteps, "IBMassSources")
    body = tsteps[lo:hi]
    # In the configured serial application an empty point set leaves the
    # complete Q allocation at +0. Bound_Q only adds those zero halo values.
    # Keep the reached GPU fill and avoid constructing a false host consumer.
    _once(body, "    call set(Q, 0)")
    body = body.replace("    call set(Q, 0)", "    call set(Q, 0._knd)\n"
                        "    if (size(Up)+size(Vp)+size(Wp)==0) return\n" + publish_block())
    tsteps = tsteps[:lo] + body + tsteps[hi:]
    # These unported immersed-boundary operations are reached only when their
    # point arrays contain work. Empty primary-channel paths keep all replicas.
    tsteps = at_once(tsteps, "    if (size(Up) + size(Vp) + size(Wp)>0) then\n", publish_block("      "), after=True)
    lo, hi = _routine(tsteps, "NullInterior")
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
    end = re.search(r"(?im)^\s*end module TimeSteps\b", tsteps)
    if end is None:
        raise ValueError("Missing timestep module end")
    output["tsteps.f90"] = tsteps[:end.start()] + supported + tsteps[end.start():]

    dynamics = output["dynamics.f90"]
    lo, hi = _routine(dynamics, "Convection")
    body = dynamics[lo:hi]
    body = at_once(body, "  contains", "    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
                   "      call resident_flow%release_real64(beta)\n      call resident_flow%release_real64(rho)\n    end block\n\n")
    output["dynamics.f90"] = dynamics[:lo] + body + dynamics[hi:]

    dynamics = output["dynamics.f90"]
    lo, hi = _routine(dynamics, "SubgridStresses")
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
        wall_reads(1)=resident_flow%bind_real64(U,lbound(U),.true.,origin='subgrid::walls::u')
        wall_reads(2)=resident_flow%bind_real64(V,lbound(V),.true.,origin='subgrid::walls::v')
        wall_reads(3)=resident_flow%bind_real64(W,lbound(W),.true.,origin='subgrid::walls::w')
        wall_count=3
        if (wallmodeltype==2) then
          wall_reads(4)=resident_flow%bind_real64(Pr,lbound(Pr),.true.,origin='subgrid::walls::pr')
          wall_count=4
        endif
        call resident_flow%observe(wall_reads(:wall_count))
      endif
    end block

'''
    first_wall = body.index("    if (wallmodeltype>0) then")
    body = body[:first_wall] + wall_observation + body[first_wall:]
    original = "                      call ComputeViscsWM(U,V,W,Pr,Temperature,Moisture)"
    _once(body, original)
    body = body.replace(original, '''      block
        use, intrinsic :: iso_c_binding
        use elmm_resident_pressure_state, only: resident_flow
        integer(c_int64_t) :: wall_write
        wall_write=0
        if (resident_flow%active) then
          if (size(WMPoints)>0) then
            wall_write=resident_flow%host_box_real64(Viscosity,lbound(Viscosity), &
              [minval(WMPoints%xi),minval(WMPoints%yj),minval(WMPoints%zk)], &
              [maxval(WMPoints%xi),maxval(WMPoints%yj),maxval(WMPoints%zk)],.true.)
          endif
        endif
        call ComputeViscsWM(U,V,W,Pr,Temperature,Moisture)
        if (wall_write/=0) call resident_flow%host_end(wall_write)
      end block''')
    output["dynamics.f90"] = dynamics[:lo] + body + dynamics[hi:]

    # Optional forcing keeps its existing condition and native implementation.
    # Enter coherence only when it actually has work; these calls are empty in
    # both primary channel configurations.
    linear = output["linearforcing.f90"]
    start = re.search(r"(?im)^\s*subroutine\s+linear_forcing\(", linear).start()
    before, body = linear[:start], linear[start:]
    body = at_once(body, "    if (enable_linear_forcing) then\n", publish_block("      "), after=True)
    output["linearforcing.f90"] = before + body
    volumes = output["volumesources.f90"]
    lo, hi = _routine(volumes, "ResistanceForce")
    body = volumes[lo:hi]
    body = at_once(body, "           if (allocated(src)) then\n",
                   "             if (size(src)>0) then\n" + publish_block("               ") + "             endif\n", after=True)
    output["volumesources.f90"] = volumes[:lo] + body + volumes[hi:]
    sponge = output["sponge.f90"]
    anchor = "    if (enable_top_rayleigh_damping) call rayleigh_damping_top_uvw(U2, V2, W2, U, V, W)"
    _once(sponge, anchor)
    output["sponge.f90"] = sponge.replace(anchor, "    if (enable_top_rayleigh_damping) then\n" + publish_block("      ")
                                          + "      call rayleigh_damping_top_uvw(U2, V2, W2, U, V, W)\n    endif")

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
