"""Generic, conservatively proved transformations using Loki's real IR."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path

from generic_analysis import Access, Assignment as CheckedAssignment, Loop as CheckedLoop, Scope, Unsupported
from generic_analysis import PURE_INTRINSICS, analyze_nest, contains_unsupported, may_move_before, prove_fusion


def transform(source: str, context: dict | None = None) -> tuple[str, dict]:
    from loki import (FP, Assignment, CallStatement, Comment, CommentBlock, FindNodes,
                     FindVariables, FindInlineCalls, Loop, Pragma, Sourcefile, Transformer)
    from loki.transformations.inline.procedures import inline_subroutine_calls
    from loki.transformations.transform_loop import do_loop_fusion

    tree = Sourcefile.from_source(source, frontend=FP)
    metadata = {"tool": "loki", "version": importlib.metadata.version("loki"),
                "changed": False, "inlined_calls": 0, "loop_fusions": 0,
                "moved_assignments": 0, "routines": [], "parallelism": "preserved; no new parallel region",
                "context": context or {}, "passes": ["same-module inlining", "independent scalar motion", "proved loop fusion"]}

    if re.search(r"^\s*!\$(?:omp|acc)\b", source, re.I | re.M):
        metadata["diagnostics"] = ["native parallel directives retained; transform directive-free outlined regions"]
        return source, metadata

    def meaningful(body):
        return [node for node in body if not isinstance(node, (Comment, CommentBlock))]

    def access(variable):
        dimensions = getattr(variable, "dimensions", None)
        return Access(variable.name.lower(), tuple(str(dim) for dim in dimensions) if dimensions else None)

    def checked(node):
        if isinstance(node, Assignment):
            # Index expressions are reads too; this records bounds captured as
            # ordinary scalar arguments without hard-coding axis names.
            refs = list(FindVariables().visit(node.rhs))
            for dim in getattr(node.lhs, "dimensions", ()) or ():
                refs.extend(FindVariables().visit(dim))
            return CheckedAssignment(access(node.lhs), [access(var) for var in refs],
                                     [("" if getattr(call.function.type.dtype, "is_intrinsic", False) else "procedure:") + str(call.function).lower()
                                      for call in FindInlineCalls().visit(node.rhs)])
        if isinstance(node, Loop):
            return CheckedLoop(node.variable.name.lower(),
                               tuple(str(value).replace(" ", "").lower() for value in
                                     (node.bounds.start, node.bounds.stop, node.bounds.step or 1)),
                               [checked(child) for child in meaningful(node.body)])
        return Unsupported(type(node).__name__)

    def scope_for(routine, module):
        arguments = {var.name.lower() for var in routine.arguments}
        variables = {var.name.lower(): var for var in (*module.variables, *routine.variables)}
        arrays = {name for name, var in variables.items() if var.type.shape}
        readonly = {name for name, var in variables.items() if var.type.intent == "in" or var.type.parameter}
        unsafe = {name for name, var in variables.items()
                  if any(getattr(var.type, attr, False) for attr in ("pointer", "target", "volatile", "allocatable", "save"))
                  or (getattr(var.type, "initial", None) is not None and not var.type.parameter)
                  or (name not in {v.name.lower() for v in routine.variables} and not var.type.parameter)}
        for var in FindVariables().visit(routine.body):
            if var.name.lower() not in variables:
                unsafe.add(var.name.lower())
        return Scope(arguments, arrays, readonly, set(variables) - arguments - readonly - unsafe, unsafe)

    for module in tree.modules:
        routines = {routine.name.lower(): routine for routine in module.subroutines}
        for routine in module.subroutines:
            report = {"name": routine.name, "module": module.name, "changed": False,
                      "inlined_calls": 0, "loop_fusions": 0, "moved_assignments": 0, "diagnostics": []}
            routine.enrich(module.subroutines)
            # Inlining is deliberately limited to whole variables, with no
            # aliasing writable actuals and a fully checked callee body.
            for call in list(FindNodes(CallStatement).visit(routine.body)):
                helper = routines.get(str(call.name).lower())
                if helper is None or helper is routine:
                    report["diagnostics"].append("call retained: unavailable or recursive callee " + str(call.name))
                    continue
                helper_scope = scope_for(helper, module)
                body = meaningful(helper.body.body)
                failure = None
                if not body or any(not isinstance(node, (Assignment, Loop)) for node in body):
                    failure = "callee contains unsupported statements or calls"
                for node in body:
                    if isinstance(node, Loop):
                        _, reason = analyze_nest(checked(node), helper_scope)
                        failure = failure or reason
                    elif isinstance(node, Assignment):
                        item = checked(node)
                        if item.target.indices is not None or item.target.name not in helper_scope.locals or item.calls:
                            failure = "callee has observable or unsupported scalar setup"
                if call.kwarguments or len(call.arguments) != len(helper.arguments):
                    failure = "keyword or nonmatching arguments"
                actuals = []
                for actual in call.arguments:
                    if not hasattr(actual, "name") or getattr(actual, "dimensions", None):
                        failure = "only whole-variable actual arguments are supported"
                    else:
                        actuals.append(actual.name.lower())
                if len(actuals) != len(set(actuals)):
                    failure = "aliased actual arguments"
                if failure:
                    report["diagnostics"].append("call retained: " + str(call.name) + ": " + failure)
                    continue
                try:
                    inline_subroutine_calls(routine, [call], helper)
                    report["inlined_calls"] += 1
                except Exception as exc:
                    # The tool's own validator remains authoritative as well.
                    report["diagnostics"].append("inline rejected: " + str(exc))
            scope = scope_for(routine, module)
            body = list(routine.body.body)
            # Move a scalar setup only across an adjacent independently proved
            # loop. Repeat so several independent coefficients can move.
            moved = True
            while moved:
                moved = False
                significant = [(i, node) for i, node in enumerate(body) if not isinstance(node, (Comment, CommentBlock))]
                for (left, loop), (right, setup) in zip(significant, significant[1:]):
                    if isinstance(loop, Loop) and isinstance(setup, Assignment) and may_move_before(checked(setup), checked(loop), scope):
                        body.insert(left, body.pop(right))
                        report["moved_assignments"] += 1
                        moved = True
                        break
            routine.body._update(body=tuple(body))
            while True:
                body = meaningful(routine.body.body)
                pair = None
                for left, right in zip(body, body[1:]):
                    if isinstance(left, Loop) and isinstance(right, Loop):
                        induction = {node.variable.name.lower() for loop in (left, right) for node in FindNodes(Loop).visit(loop)}
                        outside_nodes = [node for node in meaningful(routine.body.body) if node is not left and node is not right]
                        outside = {var.name.lower() for var in FindVariables().visit(tuple(outside_nodes))}
                        opaque = any(contains_unsupported(checked(node)) for node in outside_nodes)
                        legal, reason = prove_fusion(checked(left), checked(right), scope, allow_renaming=not opaque and not bool(induction & outside))
                        if legal:
                            pair = (left, right)
                            break
                        report["diagnostics"].append("fusion retained: " + reason)
                if pair is None:
                    break
                rank = len(analyze_nest(checked(pair[0]), scope)[0]["variables"])
                routine.body = Transformer({loop: (Pragma(keyword="loki", content=f"loop-fusion group(generic) collapse({rank})"), loop)
                                            for loop in pair}).visit(routine.body)
                before = len(FindNodes(Loop).visit(routine.body))
                do_loop_fusion(routine)
                after = len(FindNodes(Loop).visit(routine.body))
                if after >= before:
                    raise RuntimeError("Loki accepted fusion request but did not reduce loop count")
                report["loop_fusions"] += before - after
            for loop in [node for node in routine.body.body if isinstance(node, Loop)]:
                _, reason = analyze_nest(checked(loop), scope)
                if reason:
                    report["diagnostics"].append("loop retained: " + reason)
            if not any(report[key] for key in ("inlined_calls", "loop_fusions", "moved_assignments")) and not report["diagnostics"]:
                report["diagnostics"].append("no applicable optimization; original loop structure retained")
            report["diagnostics"] = list(dict.fromkeys(report["diagnostics"]))
            report["changed"] = any(report[key] for key in ("inlined_calls", "loop_fusions", "moved_assignments"))
            for key in ("inlined_calls", "loop_fusions", "moved_assignments"):
                metadata[key] += report[key]
            metadata["routines"].append(report)
    metadata["changed"] = any(report["changed"] for report in metadata["routines"])
    return (tree.to_fortran() + "\n" if metadata["changed"] else source), metadata


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
