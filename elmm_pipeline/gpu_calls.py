"""Capture-safe call sequences, checked through the compiler's public CLI.

Only direct leaf module subroutines and whole-variable arguments are admitted.
Calls remain calls in normalized Fortran: inlining, dependence proofs, transfer
analysis and execution policy belong to the selected compiler.
"""
from __future__ import annotations

import re

from . import structural as s


def _visible_name(scope, symbol):
    """Find an existing binding; do not silently import a helper's private state."""
    candidates = {symbol.name.lower()}
    current = scope
    while current is not None:
        candidates.update(current.symbols)
        for _, aliases in current.imports:
            candidates.update(aliases or {})
        current = current.parent
    for name in sorted(candidates):
        if scope.resolve(name) is symbol:
            return name
    raise s.Unsupported(f"helper capture is not visible in caller: {symbol.name}")


def _helper(call, scope):
    from .gpu_discovery import OpenMP, directives, lifted_nest, live_after

    name, values = call.items
    if s.kind(name) != "Name":
        raise s.Unsupported("only direct module subroutine calls can be grouped")
    symbol = scope.resolve(str(name))
    resolved = scope.project.procedures.get(id(symbol))
    if resolved is None:
        raise s.Unsupported(f"unresolved or non-module helper {name}")
    parsed, routine = resolved
    own, module, dummies, insertion = s.original_scope(parsed, routine, scope.project)
    header = next(node for node in s.content(routine) if s.kind(node) == "Subroutine_Stmt")
    formals = [str(value).lower() for value in getattr(header.items[2], "items", ())]
    actuals = list(getattr(values, "items", ()))
    if len(formals) != len(actuals) or any(s.kind(value) != "Name" for value in actuals):
        raise s.Unsupported("grouped calls require positional whole-variable arguments")
    bindings = dict(zip(formals, map(lambda value: str(value).lower(), actuals)))
    for formal, actual in bindings.items():
        expected, supplied = own.resolve(formal), scope.resolve(actual)
        if expected is None or supplied is None:
            raise s.Unsupported("unresolved helper argument")
        s.symbol_checks(expected, own, dummies)
        if expected.rank != supplied.rank or s.type_spec(expected, logical=True) != s.type_spec(supplied, logical=True):
            raise s.Unsupported("helper argument type or rank mismatch")
        if expected.rank and s.kind(expected.shape) != "Assumed_Shape_Spec_List":
            raise s.Unsupported("grouped helper arrays require assumed-shape dummies")
    execution = next((node for node in s.content(routine) if s.kind(node) == "Execution_Part"), None)
    nodes = s.code_children(execution)
    if not nodes:
        raise s.Unsupported("empty helper")
    omp = OpenMP(parsed, routine)
    if omp.parallel or omp.unsupported:
        raise s.Unsupported("helper requires unsupported OpenMP dispatch")
    consumed, private, nests, ranks = set(), set(), [], []
    for node in nodes:
        if s.kind(node) == "Block_Nonlabel_Do_Construct":
            context, clauses, own_private = omp.describe(node)
            if context["dispatch"] != "serial":
                raise s.Unsupported("helper requires collective dispatch")
            consumed.update(value["start_line"] for value in context["consumed_directives"])
            private.update(own_private)
            try:
                nest, tiles, removed, rank = lifted_nest(node, own, structured=True)
            except s.Unsupported:
                nest, tiles, removed = s.Nest(node, [], [node]), [], set()
                current, rank = node, 0
                while s.kind(current) == "Block_Nonlabel_Do_Construct":
                    rank += 1
                    children = s.code_children(current)[1:-1]
                    current = children[0] if len(children) == 1 else None
            if any(int(clause.get("collapse", "1")) > rank for clause in clauses):
                raise s.Unsupported("helper OpenMP collapse exceeds complete loop nest")
            ranks.append(len(nest.loops) if nest.loops else rank)
            if tiles:
                for name in removed:
                    value = own.resolve(name)
                    if (value is None or value.scope is not own or name in dummies or value.rank
                            or "save" in value.attributes or value.initializer is not None):
                        raise s.Unsupported("helper tile iterator has externally visible state")
                    s.symbol_checks(value, own, dummies)
                _, end = parsed.origin_span(node)
                iterators = {s.loop_info(value)[0].lower() for value in s.walk(node)
                             if s.kind(value) == "Block_Nonlabel_Do_Construct"}
                if live_after(routine, end, iterators, parsed, node):
                    raise s.Unsupported("helper loop iterator remains live after domain lifting")
                # A nonpositive tile stride gives an empty inner unit-stride
                # range MIN(tile+stride-1, upper). Keep that case empty, and
                # evaluate the test here, after the original scalar setup.
                # Bare defined scalars/constants cannot trap when read before
                # a possibly empty outer range. Do not hoist arbitrary bounds.
                steps = [s.F.Level_2_Expr(tile["step"]) for tile in tiles]
                for step in steps:
                    if s.kind(step) != "Name":
                        s.integer_constant(step, own)
                condition = " .and. ".join(f"({step}) > 0" for step in steps)
                body = s.block_tree([nest], own)
                wrapped = s.block_tree([s.Nest(node, [], [f"if ({condition}) then", body, "end if"])], own)
                nest = s.Nest(node, [], s.content(wrapped))
            nests.append(nest)
        else:
            nests.append(s.Nest(node, [], [node]))
    if any(item.start not in consumed for item in omp.items):
        raise s.Unsupported("unconsumed OpenMP directive in helper")
    # Retain scalar preparation, conditions and loop coordinates in place.
    # In particular, do not evaluate helper-local bounds at the caller.
    region = s.make_block_region(nests, own, module, dummies, routine, parsed, insertion, 0,
                                 liveness=live_after, statement_span=True)
    region.update(loop_nests=len(ranks), loop_ranks=ranks)
    captures = {value["name"].lower() for value in region["symbols"]}
    if captures & private:
        raise s.Unsupported("helper capture is OpenMP private/firstprivate")
    for source in list(scope.project.files.values()):
        if isinstance(source, s.Parsed):
            threadprivate = {name for item in directives(source) if item.text.startswith("threadprivate")
                             for name in re.findall(r"\b[a-z]\w*\b", item.text)[1:]}
            if any(name in threadprivate or own.resolve(name).name.lower() in threadprivate for name in captures):
                raise s.Unsupported("helper capture is OpenMP threadprivate")
    return region, own, bindings


def make_call_region(nodes, scope, module, dummies, routine, parsed, insertion, omp, ordinal):
    """Outline adjacent leaf calls from a serial caller, without crossing control flow."""
    if len(nodes) < 2:
        raise s.Unsupported("a call group requires at least two adjacent calls")
    if omp.items:
        raise s.Unsupported("call grouping currently requires a serial caller")
    parent = getattr(nodes[0], "parent", None)
    spans = [parsed.origin_span(node) for node in nodes]
    for index, node in enumerate(nodes):
        s.scope_checks(node)
        if s.kind(node) != "Call_Stmt" or getattr(node, "parent", None) is not parent:
            raise s.Unsupported("call group crosses a control-flow boundary")
        first, last = spans[index]
        lines = parsed.text.splitlines()[first - 1:last]
        if not re.match(r"\s*call\b", lines[0], re.I) or any(";" in line or line.lstrip().startswith("#") for line in lines):
            raise s.Unsupported("call group lacks exact original statement spans")
        if index and not omp.blank_between(spans[index - 1][1], first):
            raise s.Unsupported("call group crosses a statement or synchronization boundary")

    helpers = [_helper(node, scope) for node in nodes]
    arrays, captured, locals_, writes = {}, {}, {}, set()
    calls, constants, guards = [], [], []
    occupied = set(scope.symbols)

    def temporary(value):
        name = "call_value_" + str(len(constants))
        while name in occupied:
            name += "_"
        occupied.add(name)
        locals_[name] = s.Symbol(name, "integer", None, None, set(), None, None, scope)
        constants.append((name, value))
        return name

    for helper, own, bindings in helpers:
        arguments, mapped = [], {}
        for metadata in helper["symbols"]:
            name = metadata["name"].lower()
            symbol = own.resolve(name)
            if "parameter" in symbol.attributes and not symbol.rank:
                # A helper-local integer constant is not caller storage.
                if s.type_spec(symbol) != "integer":
                    raise s.Unsupported("unsupported helper-local constant type")
                mapped[name] = temporary(str(s.integer_constant(s.F.Name(name), own)))
                continue
            actual = bindings[name] if name in bindings else _visible_name(scope, symbol)
            supplied = s.visible_symbol(scope, actual)
            s.symbol_checks(supplied, scope, dummies, stable_allocatable=True)
            (arrays if supplied.rank else captured)[actual] = supplied
            mapped[name] = actual
            if metadata["intent"] == "inout":
                if supplied.intent == "in":
                    raise s.Unsupported("helper writes an INTENT(IN) capture")
                writes.add(actual)
        for argument, actual in zip(helper["args"], helper["actual_args"]):
            if argument.lower() in mapped:
                arguments.append(mapped[argument.lower()])
                continue
            bound = re.fullmatch(r"lbound\((\w+), (\d+)\)", actual, re.I)
            if bound is None:
                raise s.Unsupported("unsupported normalized helper argument")
            name, dimension = bound[1].lower(), int(bound[2]) - 1
            if name in bindings:
                shape = own.resolve(name).shape.items[dimension]
                lower = shape.items[0]
                value = 1 if lower is None else s.integer_constant(lower, own)
                arguments.append(temporary(str(value)))
                # Empty assumed-shape dimensions have LBOUND=1, regardless of
                # declared lower bounds. Keep that case on the native path.
                guards.append(f"size({mapped[name]}, {dimension + 1}) > 0")
            else:
                arguments.append((mapped[name], dimension))
        array_actuals = [mapped[value["name"].lower()] for value in helper["symbols"] if value["rank"]]
        if any(array_actuals.count(actual) > 1 and actual in writes for actual in array_actuals):
            raise s.Unsupported("aliased writable helper arguments")
        calls.append((helper["entry"], arguments))

    identities = {}
    for name in arrays:
        identities.setdefault(id(scope.resolve(name)), set()).add(name)
    if any(len(names) > 1 and names & writes for names in identities.values()):
        raise s.Unsupported("multiple visible aliases of written call-group storage")
    allocated = sorted(value.name for value in arrays.values() if "allocatable" in value.attributes)
    if not scope.intrinsic_unshadowed("lbound") or not scope.intrinsic_unshadowed("size"):
        raise s.Unsupported("descriptor intrinsic binding cannot be proved")
    if allocated and not scope.intrinsic_unshadowed("allocated"):
        raise s.Unsupported("ALLOCATED intrinsic binding cannot be proved")

    def body(lower_names):
        lines = [f"    {name} = {value}" for name, value in constants]
        for entry, arguments in calls:
            values = [lower_names[value[0]][value[1]] if isinstance(value, tuple) else value for value in arguments]
            lines.extend([f"    call {entry}( &", ", &\n".join("      " + value for value in values) + ")"])
        return lines

    region = s.render_region([s.Nest(node, [], []) for node in nodes], scope, module, dummies, routine,
                             parsed, insertion, set(), ordinal, arrays, captured, locals_, writes,
                             structured=True, statement_span=True, body_builder=body)
    definitions = {helper["entry"]: helper["module_source"].split("contains\n", 1)[1].rsplit("end module", 1)[0]
                   for helper, _, _ in helpers}
    region["module_source"] = region["module_source"].replace(
        f"end module {region['generated_module']}", "\n".join(definitions.values()) + f"end module {region['generated_module']}")
    region["context"] = {"nested_parallel": False, "gpu_dispatch": "serial"}
    ranks = [rank for helper, _, _ in helpers for rank in helper["loop_ranks"]]
    region.update(loop_nests=len(ranks), loop_ranks=ranks)
    region["gpu_scope"] = {"kind": "call_sequence", "dispatch": "serial", "expected_omp_level": 0,
                           "native_fallback": True, "allocated_arrays": allocated,
                           "guard": " .and. ".join(dict.fromkeys(guards)) or None,
                           "grouped_calls": len(nodes), "grouped_nests": len(ranks), "tiles": [],
                           "consumed_directives": [], "barriers": None,
                           "member_spans": [{"start_line": first, "end_line": last} for first, last in spans]}
    region["legality"] += "; direct leaf calls retained for compiler-owned inlining"
    return region
