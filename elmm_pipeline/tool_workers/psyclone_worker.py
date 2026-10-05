"""Generic PSyclone inlining, motion and fusion with explicit legality checks."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import re

from generic_analysis import Access, Assignment as CheckedAssignment, Loop as CheckedLoop, Scope, Unsupported
from generic_analysis import analyze_nest, contains_unsupported, may_move_before, prove_fusion


def contiguous_contract(source):
    contracts, routine = {}, None
    # Normalized inputs use declaration statements with ordinary named entities.
    # Joining continuations also supports hand-written generic worker inputs.
    source = re.sub(r"&\s*\n\s*&?", "", source)
    for line in source.splitlines():
        start = re.match(r"\s*(?:(?:pure|recursive|elemental)\s+)*subroutine\s+(\w+)\b", line, re.I)
        if start:
            routine = start[1].lower()
            contracts[routine] = set()
        if routine and "::" in line and re.search(r"\bcontiguous\b", line.split("::", 1)[0], re.I):
            entities = line.split("::", 1)[1]
            depth, chunks, begin = 0, [], 0
            for index, char in enumerate(entities):
                depth += (char == "(") - (char == ")")
                if char == "," and depth == 0:
                    chunks.append(entities[begin:index]); begin = index + 1
            chunks.append(entities[begin:])
            for chunk in chunks:
                match = re.match(r"\s*(\w+)", chunk)
                if match:
                    contracts[routine].add(match[1].lower())
        if re.match(r"\s*end\s*subroutine\b", line, re.I):
            routine = None
    return contracts


def restore_contiguous(source, contracts):
    lines, routine, restored = [], None, {}
    for line in source.splitlines():
        start = re.match(r"\s*(?:(?:pure|recursive|elemental)\s+)*subroutine\s+(\w+)\b", line, re.I)
        if start:
            routine = start[1].lower()
            restored[routine] = set()
        if routine and "::" in line:
            spec, entities = line.split("::", 1)
            name = re.match(r"\s*(\w+)", entities)
            if name and name[1].lower() in contracts.get(routine, set()):
                line = spec.rstrip() + ", contiguous ::" + entities
                restored[routine].add(name[1].lower())
        lines.append(line)
        if re.match(r"\s*end\s*subroutine\b", line, re.I):
            routine = None
    for routine, names in contracts.items():
        if restored.get(routine, set()) != names:
            raise ValueError("Failed to restore CONTIGUOUS contract for " + routine)
    return "\n".join(lines) + "\n"


def transform(source: str, context: dict | None = None) -> tuple[str, dict]:
    from psyclone.line_length import FortLineLength
    from psyclone.psyir.backend.fortran import FortranWriter
    from psyclone.psyir.frontend.fortran import FortranReader
    from psyclone.psyir.nodes import ArrayReference, Assignment, Call, CodeBlock, Container, IntrinsicCall, Literal, Loop, Reference, Routine
    from psyclone.psyir.symbols import ArrayType, ArgumentInterface, DataSymbol, ScalarType, StaticInterface, UnsupportedFortranType
    from psyclone.psyir.transformations import InlineTrans, LoopFuseTrans, MoveTrans

    contracts = contiguous_contract(source)
    tree = FortranReader().psyir_from_source(re.sub(r",\s*contiguous\b", "", source, flags=re.I))
    writer = FortranWriter()
    metadata = {"tool": "psyclone", "version": importlib.metadata.version("psyclone"),
                "changed": False, "inlined_calls": 0, "loop_fusions": 0,
                "moved_assignments": 0, "routines": [], "parallelism": "preserved; no new parallel region",
                "context": context or {}, "passes": ["same-module inlining", "independent scalar motion", "proved loop fusion"],
                "compatibility": "CONTIGUOUS attributes restored per declared dummy"}

    if re.search(r"^\s*!\$(?:omp|acc)\b", source, re.I | re.M):
        metadata["diagnostics"] = ["native parallel directives retained; transform directive-free outlined regions"]
        return source, metadata

    def access(ref):
        return Access(ref.name.lower(), tuple(writer(index) for index in ref.indices) if isinstance(ref, ArrayReference) else None)

    def checked(node):
        if isinstance(node, Assignment):
            if not isinstance(node.lhs, Reference):
                return Unsupported("non-reference assignment target")
            refs = node.rhs.walk(Reference)
            if isinstance(node.lhs, ArrayReference):
                for index in node.lhs.indices:
                    refs += index.walk(Reference)
            calls = [call.intrinsic.name.lower() if isinstance(call, IntrinsicCall) else "procedure:" + call.routine.name.lower()
                     for call in node.rhs.walk(Call)]
            return CheckedAssignment(access(node.lhs), [access(ref) for ref in refs if isinstance(ref.symbol, DataSymbol)], calls)
        if isinstance(node, Loop):
            return CheckedLoop(node.variable.name.lower(),
                               tuple(writer(expr).replace(" ", "").lower() for expr in
                                     (node.start_expr, node.stop_expr, node.step_expr)),
                               [checked(child) for child in node.loop_body.children])
        return Unsupported(type(node).__name__)

    def scope_for(routine):
        symbols = {symbol.name.lower(): symbol for symbol in routine.symbol_table.datasymbols}
        arguments = {symbol.name.lower() for symbol in routine.symbol_table.argument_list}
        arrays = {name for name, symbol in symbols.items() if isinstance(symbol.datatype, ArrayType)}
        readonly = {name for name, symbol in symbols.items() if symbol.is_constant or
                    (symbol.is_argument and symbol.interface.access == ArgumentInterface.Access.READ)}
        unsafe = {name for name, symbol in symbols.items() if symbol.is_import or symbol.is_unresolved
                  or isinstance(symbol.datatype, UnsupportedFortranType)
                  or (isinstance(symbol.interface, StaticInterface) and not symbol.is_constant)}
        for ref in routine.walk(Reference):
            if not isinstance(ref.symbol, DataSymbol):
                continue
            # Routine symbols attached to intrinsic calls are not data accesses.
            if ref.name.lower() not in symbols and not getattr(ref.symbol, "is_constant", False):
                unsafe.add(ref.name.lower())
        return Scope(arguments, arrays, readonly, set(symbols) - arguments - readonly - unsafe, unsafe)

    for routine in tree.walk(Routine):
        report = {"name": routine.name, "module": routine.parent.name if hasattr(routine.parent, "name") else None,
                  "changed": False, "inlined_calls": 0, "loop_fusions": 0, "moved_assignments": 0, "diagnostics": []}
        siblings = {node.name.lower(): node for node in routine.parent.children if isinstance(node, Routine)}
        for call in list(routine.walk(Call)):
            if isinstance(call, IntrinsicCall):
                continue
            helper = siblings.get(call.routine.name.lower())
            if helper is None or helper is routine:
                report["diagnostics"].append("call retained: unavailable or recursive callee " + call.routine.name)
                continue
            scope = scope_for(helper)
            failure = None
            if not helper.children or helper.walk(CodeBlock) or any(not isinstance(node, (Assignment, Loop)) for node in helper.children):
                failure = "callee contains unsupported statements or calls"
            for node in helper.children:
                if isinstance(node, Loop):
                    _, reason = analyze_nest(checked(node), scope)
                    failure = failure or reason
                elif isinstance(node, Assignment):
                    item = checked(node)
                    if not isinstance(item, CheckedAssignment) or item.target.indices is not None or item.target.name not in scope.locals or item.calls:
                        failure = "callee has observable or unsupported scalar setup"
            if len(call.arguments) != len(helper.symbol_table.argument_list) or any(name for name in call.argument_names):
                failure = "keyword or nonmatching arguments"
            actuals = []
            for actual in call.arguments:
                if type(actual) is not Reference:
                    failure = "only whole-variable actual arguments are supported"
                else:
                    actuals.append(actual.name.lower())
            if len(actuals) != len(set(actuals)):
                failure = "aliased actual arguments"
            if failure:
                report["diagnostics"].append("call retained: " + call.routine.name + ": " + failure)
                continue
            try:
                InlineTrans().apply(call)
                report["inlined_calls"] += 1
            except Exception as exc:
                report["diagnostics"].append("inline rejected: " + str(exc))
        # Inlining introduces LBOUND(actual,dim). For an ordinary assumed-shape
        # dummy its lower bound is declared locally (one when omitted), independent
        # of its caller's allocation. Resolve only this declaration-proved case.
        for call in list(routine.walk(IntrinsicCall)):
            if call.intrinsic != IntrinsicCall.Intrinsic.LBOUND or len(call.arguments) != 2:
                continue
            array, dim = call.arguments
            if type(array) is not Reference or not isinstance(dim, Literal):
                continue
            symbol = array.symbol
            if not symbol.is_argument or not isinstance(symbol.datatype, ArrayType):
                continue
            try:
                shape = symbol.datatype.shape[int(dim.value) - 1]
            except (ValueError, IndexError):
                continue
            if shape == ArrayType.Extent.ATTRIBUTE:
                call.replace_with(Literal("1", ScalarType.integer_type()))
            elif isinstance(shape, ArrayType.ArrayBounds) and isinstance(shape.lower, Literal) and shape.lower.value == "1":
                # LBOUND of a zero-extent dimension is one, even with another
                # declared lower bound. Do not fold nonunit lower bounds.
                call.replace_with(shape.lower.copy())
        scope = scope_for(routine)
        moved = True
        while moved:
            moved = False
            for loop, setup in zip(list(routine.children), list(routine.children)[1:]):
                if isinstance(loop, Loop) and isinstance(setup, Assignment) and may_move_before(checked(setup), checked(loop), scope):
                    try:
                        MoveTrans().apply(setup, loop)
                        report["moved_assignments"] += 1
                        moved = True
                        break
                    except Exception as exc:
                        report["diagnostics"].append("motion rejected: " + str(exc))
        # Fuse full nests one level at a time, after proving the complete nests.
        # PSyclone adds its own dependency analysis at every transformed level.
        while True:
            pair = None
            for left, right in zip(list(routine.children), list(routine.children)[1:]):
                if isinstance(left, Loop) and isinstance(right, Loop):
                    induction = {node.variable.name.lower() for loop in (left, right) for node in loop.walk(Loop)}
                    outside_nodes = [node for node in routine.children if node is not left and node is not right]
                    outside = {ref.name.lower() for node in outside_nodes for ref in node.walk(Reference)}
                    opaque = any(contains_unsupported(checked(node)) for node in outside_nodes)
                    legal, reason = prove_fusion(checked(left), checked(right), scope, allow_renaming=not opaque and not bool(induction & outside))
                    if legal:
                        pair = left, right
                        break
                    report["diagnostics"].append("fusion retained: " + reason)
            if pair is None:
                break
            left, right = pair
            rank = len(analyze_nest(checked(left), scope)[0]["variables"])
            # Validate the outer pair before modifying either nest.
            try:
                LoopFuseTrans().validate(left, right)
                LoopFuseTrans().apply(left, right)
                report["loop_fusions"] += 1
                for _ in range(rank - 1):
                    children = list(left.loop_body.children)
                    if len(children) != 2 or not all(isinstance(node, Loop) for node in children):
                        raise ValueError("unexpected structure after outer-loop fusion")
                    LoopFuseTrans().apply(*children)
                    report["loop_fusions"] += 1
                    left = children[0]
            except Exception as exc:
                report["diagnostics"].append("fusion rejected by PSyclone: " + str(exc))
                break
        for node in routine.children:
            if isinstance(node, Loop):
                _, reason = analyze_nest(checked(node), scope)
                if reason:
                    report["diagnostics"].append("loop retained: " + reason)
            elif isinstance(node, (Call, CodeBlock)):
                report["diagnostics"].append("statement retained: " + type(node).__name__)
        if not any(report[key] for key in ("inlined_calls", "loop_fusions", "moved_assignments")) and not report["diagnostics"]:
            report["diagnostics"].append("no applicable optimization; original loop structure retained")
        report["diagnostics"] = list(dict.fromkeys(report["diagnostics"]))
        report["changed"] = any(report[key] for key in ("inlined_calls", "loop_fusions", "moved_assignments"))
        for key in ("inlined_calls", "loop_fusions", "moved_assignments"):
            metadata[key] += report[key]
        metadata["routines"].append(report)
    metadata["changed"] = any(report["changed"] for report in metadata["routines"])
    generated = FortLineLength().process(restore_contiguous(writer(tree), contracts)) if metadata["changed"] else source
    return generated, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--context", default="{}")
    args = parser.parse_args()
    generated, metadata = transform(args.source.read_text(), json.loads(args.context))
    args.output.write_text(generated)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n")
