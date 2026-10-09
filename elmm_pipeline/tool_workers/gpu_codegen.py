"""Portable GNU/NVHPC bridge for a normalized numeric Fortran entry.

Only ordinary explicit-shape arrays cross BIND(C). Compiler-private assumed
shape descriptors never leave the GNU host module. Extents are 64-bit values.
"""
from __future__ import annotations

import hashlib
import re


def wrap_fortran(source: str, limit: int = 120) -> str:
    """Continue long numeric expressions and directives without splitting tokens.

    fparser's AST printer joins continuation lines; NVIDIA rejects source lines
    beyond its finite free-form limit. The normalized grammar has no long
    character values, but the scanner still respects quoted strings.
    """
    result = []
    for original in source.splitlines():
        line = original.rstrip()
        directive = bool(re.match(r"\s*!\$acc\b", line, re.IGNORECASE))
        if line.lstrip().startswith("!") and not directive:
            result.append(line)
            continue
        while len(line) > limit:
            quote, breaks, index = None, [], 0
            while index < min(len(line), limit - 2):
                char = line[index]
                if quote:
                    if char == quote:
                        if index + 1 < len(line) and line[index + 1] == quote:
                            index += 1
                        else:
                            quote = None
                elif char in "\"'":
                    quote = char
                elif char in " ," and index > 8:
                    breaks.append(index + (char == ","))
                index += 1
            if not breaks:
                raise ValueError("Cannot safely continue generated Fortran line")
            at = breaks[-1]
            result.append(line[:at].rstrip() + " &")
            line = ("!$acc& " if directive else "  & ") + line[at:].lstrip()
        result.append(line)
    return "\n".join(result) + "\n"


def bridge(source: str, entry: str) -> tuple[str, str, dict]:
    from fparser.common.readfortran import FortranStringReader
    from fparser.two.parser import ParserFactory
    from fparser.two.utils import walk

    tree = ParserFactory().create(std="f2008")(FortranStringReader(source, ignore_comments=False))
    matches = [node for node in walk(tree) if type(node).__name__ == "Subroutine_Subprogram"
               and str(node.content[0].items[1]).lower() == entry.lower()]
    if len(matches) != 1:
        raise ValueError("GPU bridge requires exactly one selected subroutine: " + entry)
    routine = matches[0]
    module = routine.parent
    while module is not None and type(module).__name__ != "Module":
        module = module.parent
    if module is None:
        raise ValueError("GPU bridge requires a standalone module")
    module_name = str(module.content[0].items[1])
    args = [str(arg).lower() for arg in routine.content[0].items[2].items]
    suffix = hashlib.sha256((module_name.lower() + "::" + entry.lower()).encode()).hexdigest()[:16]
    prefix = "elmm_gpu_" + suffix
    c_entry, c_probe = prefix + "_entry", prefix + "_probe"
    spec = next(n for n in routine.content if type(n).__name__ == "Specification_Part")
    body = next(n for n in routine.content if type(n).__name__ == "Execution_Part")
    module_spec = next((str(n) for n in module.content if type(n).__name__ == "Specification_Part"), "")
    declarations, locals_, leading = {}, [], []
    statements = [child for statement in spec.content
                  for child in (statement.content if type(statement).__name__ == "Implicit_Part" else [statement])]
    for statement in statements:
        if type(statement).__name__ == "Comment" and re.match(r"\s*!\$acc\b", str(statement), re.IGNORECASE):
            leading.append(str(statement))
            continue
        if type(statement).__name__ != "Type_Declaration_Stmt":
            if type(statement).__name__ in {"Implicit_Part", "Implicit_Stmt", "Comment"}:
                locals_.append(str(statement))
                continue
            raise ValueError("Unsupported normalized GPU specification: " + str(statement))
        dtype, attrs, entities = statement.items
        attributes = list(attrs.items) if attrs is not None else []
        attr_shape = next((attr.items[1] for attr in attributes
                           if type(attr).__name__ == "Dimension_Attr_Spec"), None)
        for entity in entities.items:
            name = str(entity.items[0]).lower()
            shape = entity.items[1] if entity.items[1] is not None else attr_shape
            rank = len(shape.items) if shape is not None else 0
            intent = next((str(attr.items[1]).lower() for attr in attributes
                           if type(attr).__name__ == "Intent_Attr_Spec"), "inout")
            kept = [str(attr) for attr in attributes if type(attr).__name__ != "Dimension_Attr_Spec"]
            original = str(dtype) + (", " + ", ".join(kept) if kept else "") + " :: " + str(entity)
            if attr_shape is not None and entity.items[1] is None:
                original = original.replace(" :: " + str(entity), " :: " + name + "(" + str(shape) + ")")
            if name not in args:
                locals_.append(original)
                continue
            if str(dtype.items[0]).lower() not in {"real", "integer", "double precision"}:
                raise ValueError("GPU C bridge supports real and integer numeric arguments: " + name)
            if any(str(attr).lower() in {"pointer", "allocatable", "optional", "target", "volatile"} for attr in attributes):
                raise ValueError("Unsupported GPU C ABI attribute for " + name)
            dims = [f"{prefix}_d{args.index(name)}_{axis}" for axis in range(rank)]
            bounds = []
            for dim, extent in zip(shape.items if shape is not None else [], dims):
                lower = str(dim.items[0]) if dim.items[0] is not None else "1"
                bounds.append(f"({lower}):({lower})+{extent}-1")
            # Copy all written arrays into the device, including INTENT(OUT),
            # because a normalized region may write only an interior subset.
            boundary_intent = "in" if intent == "in" else "inout"
            explicit = f"{dtype}, intent({boundary_intent})"
            if not rank and intent == "in":
                explicit += ", value"
            explicit += " :: " + name + ("(" + ", ".join(bounds) + ")" if rank else "")
            if rank and intent == "out":
                original = re.sub(r"intent\s*\(\s*out\s*\)", "intent(inout)", original, flags=re.IGNORECASE)
            declarations[name] = {"name": name, "rank": rank, "intent": boundary_intent, "original": original,
                                  "explicit": explicit, "dimensions": dims, "dtype": str(dtype)}
    if set(args) != set(declarations):
        raise ValueError("Missing explicit numeric declarations in GPU bridge")
    extents = [dim for name in args for dim in declarations[name]["dimensions"]]
    abi_args = extents + args
    signature = ", &\n    ".join(abi_args)
    dimension_decl = ["integer(kind=8), value, intent(in) :: " + name for name in extents]
    argument_decl = [declarations[name]["explicit"] for name in args if not declarations[name]["rank"]]
    argument_decl += [declarations[name]["explicit"] for name in args if declarations[name]["rank"]]
    abi_decls = dimension_decl + argument_decl
    call_args = [f"size({name}, {axis + 1}, kind=8)" for name in args
                 for axis in range(declarations[name]["rank"])] + args
    interface = [f"module {module_name}", module_spec, "contains", f"subroutine {entry}(&",
                 "    " + ", &\n    ".join(args) + ")"]
    implicit = [line for line in locals_ if re.match(r"\s*implicit\b", line, re.IGNORECASE)]
    local_declarations = [line for line in locals_ if line not in implicit]
    interface += implicit + local_declarations + [declarations[name]["original"] for name in args]
    interface += ["logical, save :: " + prefix + "_proved = .false.", "interface",
                  f"subroutine {c_entry}(&", "    " + signature + f") bind(C, name='{c_entry}')",
                  "import", *abi_decls, f"end subroutine {c_entry}",
                  f"subroutine {c_probe}() bind(C, name='{c_probe}')", f"end subroutine {c_probe}", "end interface",
                  f"!$omp critical ({prefix}_proof)", f"if (.not. {prefix}_proved) then",
                  f"call {c_probe}()", f"{prefix}_proved = .true.", "end if",
                  f"!$omp end critical ({prefix}_proof)",
                  f"call {c_entry}(&", "    " + ", &\n    ".join(call_args) + ")",
                  f"end subroutine {entry}", f"end module {module_name}"]
    # Backend module names differ, so NVHPC .mod files cannot shadow GNU ones.
    backend_module_spec = re.sub(r"(?im)^\s*(public|private)\s*::[^\n]*$", "", module_spec)
    backend = [f"module {prefix}_module", backend_module_spec, "contains",
               f"subroutine {c_entry}(&", "    " + signature + f") bind(C, name='{c_entry}')",
               *implicit, *local_declarations, *abi_decls, *leading, str(body), f"end subroutine {c_entry}",
               f"subroutine {c_probe}() bind(C, name='{c_probe}')",
               "use openacc, only: acc_on_device, acc_device_nvidia, acc_set_device_type",
               "integer :: on_gpu", "call acc_set_device_type(acc_device_nvidia)", "on_gpu = 0",
               "!$acc serial copy(on_gpu)", "on_gpu = merge(1, 0, acc_on_device(acc_device_nvidia))",
               "!$acc end serial", "if (on_gpu /= 1) error stop 'OpenACC did not execute on NVIDIA GPU'",
               f"write(*,'(a)') 'ELMM_PIPELINE_GPU_DEVICE {entry} NVIDIA'",
               f"end subroutine {c_probe}", f"end module {prefix}_module"]
    metadata = {"module": module_name, "entry": entry, "c_entry": c_entry, "proof_entry": c_probe,
                "arguments": list(declarations.values()), "abi": "BIND(C), explicit-shape arrays, 64-bit value extents",
                "proof": "first-call acc_on_device(acc_device_nvidia), guarded by GNU OpenMP critical"}
    return wrap_fortran("\n".join(interface)), wrap_fortran("\n".join(backend)), metadata
