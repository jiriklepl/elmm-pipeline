"""Generic GPU passes over real Loki/PSyclone IR, then a portable C bridge."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from generic_analysis import Access, Scope, Unsupported, analyze_nest
from generic_analysis import Assignment as CheckedAssignment
from generic_analysis import Loop as CheckedLoop
from gpu_codegen import bridge


def prove_gpu_loops(loops, setups, scope):
    """Parallel induction/private values must not escape their compute region."""
    analyses = []
    for loop in loops:
        analysis, reason = analyze_nest(loop, scope)
        if reason:
            raise ValueError("Unsafe GPU parallelism: " + reason)
        analyses.append(analysis)
    outside_reads = {read.name for setup in setups for read in setup.reads}
    for index, analysis in enumerate(analyses):
        private = set(analysis["variables"]) | analysis["scalar_writes"]
        if private & outside_reads:
            raise ValueError("GPU induction or private scalar is observed outside its loop")
        for other_index, other in enumerate(analyses):
            if index == other_index:
                continue
            bound_reads = set(re.findall(r"[a-z_][a-z0-9_]*", " ".join(" ".join(b) for b in other["bounds"])))
            if private & (other["incoming"] | bound_reads):
                raise ValueError("GPU induction or private scalar is observed in another loop")
    return analyses


def loki_gpu(source, entry, context):
    from loki import (
        FP,
        Assignment,
        Comment,
        CommentBlock,
        FindInlineCalls,
        FindVariables,
        Loop,
        Pragma,
        Sourcefile,
        Transformer,
    )
    from loki_worker import transform

    generated, metadata = transform(source, context)
    tree = Sourcefile.from_source(generated, frontend=FP)
    routine = next(r for m in tree.modules for r in m.subroutines if r.name.lower() == entry.lower())
    arguments = {arg.name.lower() for arg in routine.arguments}
    variables = {var.name.lower(): var for var in routine.variables}
    arrays = {name for name, var in variables.items() if var.type.shape}
    readonly = {name for name, var in variables.items() if var.type.intent == "in" or var.type.parameter}
    scope = Scope(arguments, arrays, readonly, set(variables) - arguments - readonly)
    if arrays - arguments:
        raise ValueError("GPU arrays must be explicit dummy arguments")

    def meaningful(body):
        return [n for n in body if not isinstance(n, (Comment, CommentBlock, Pragma))]

    def access(var):
        dims = getattr(var, "dimensions", None)
        return Access(var.name.lower(), tuple(str(dim) for dim in dims) if dims else None)

    def checked(node):
        if isinstance(node, Assignment):
            refs = list(FindVariables().visit(node.rhs))
            for dim in getattr(node.lhs, "dimensions", ()) or ():
                refs.extend(FindVariables().visit(dim))
            return CheckedAssignment(access(node.lhs), [access(var) for var in refs],
                                     [("" if getattr(call.function.type.dtype, "is_intrinsic", False) else "procedure:")
                                      + str(call.function).lower() for call in FindInlineCalls().visit(node.rhs)])
        if isinstance(node, Loop):
            return CheckedLoop(node.variable.name.lower(),
                               tuple(str(value).replace(" ", "").lower() for value in
                                     (node.bounds.start, node.bounds.stop, node.bounds.step or 1)),
                               [checked(child) for child in meaningful(node.body)])
        return Unsupported(type(node).__name__)

    loops = [node for node in meaningful(routine.body.body) if isinstance(node, Loop)]
    if not loops:
        raise ValueError("No eligible GPU loop in selected entry")
    if any(not isinstance(node, (Loop, Assignment)) for node in meaningful(routine.body.body)):
        raise ValueError("Unsupported host statement in GPU region")
    for node in meaningful(routine.body.body):
        if isinstance(node, Assignment) and any(var.name.lower() in arrays for var in FindVariables().visit(node)):
            raise ValueError("Host array accesses inside GPU data region require synchronization")
    analyses = prove_gpu_loops([checked(loop) for loop in loops],
                               [checked(node) for node in meaningful(routine.body.body) if isinstance(node, Assignment)], scope)
    loop_map = {}
    for loop, analysis in zip(loops, analyses):
        clauses = f"parallel loop gang vector collapse({len(analysis['variables'])})"
        if arrays:
            clauses += " present(" + ", ".join(sorted(arrays)) + ")"
        if analysis["scalar_writes"]:
            clauses += " private(" + ", ".join(sorted(analysis["scalar_writes"])) + ")"
        loop_map[loop] = (Pragma(keyword="acc", content=clauses), loop,
                          Pragma(keyword="acc", content="end parallel loop"))
    routine.body = Transformer(loop_map).visit(routine.body)
    clauses = "data"
    inputs = sorted(arrays & readonly)
    written = sorted(arrays - readonly)
    if inputs:
        clauses += " copyin(" + ", ".join(inputs) + ")"
    if written:
        clauses += " copy(" + ", ".join(written) + ")"
    routine.body.prepend(Pragma(keyword="acc", content=clauses))
    routine.body.append(Pragma(keyword="acc", content="end data"))
    metadata["passes"] += ["Loki IR OpenACC data region", "proved parallel loop pragmas"]
    metadata.update(changed=True, kernel_regions=len(loops), data_regions=1)
    return tree.to_fortran() + "\n", metadata


def psyclone_gpu(source, entry, context):
    from psyclone.psyir.backend.fortran import FortranWriter
    from psyclone.psyir.frontend.fortran import FortranReader
    from psyclone.psyir.nodes import (
        ACCDataDirective,
        ACCLoopDirective,
        ArrayReference,
        Assignment,
        Call,
        IntrinsicCall,
        Loop,
        Reference,
        Routine,
    )
    from psyclone.psyir.symbols import ArgumentInterface, ArrayType, DataSymbol
    from psyclone.psyir.transformations import ACCLoopTrans
    from psyclone.transformations import ACCDataTrans, ACCParallelTrans
    from psyclone_worker import contiguous_contract, restore_contiguous, transform

    generated, metadata = transform(source, context)
    contracts = contiguous_contract(generated)
    tree = FortranReader().psyir_from_source(re.sub(r",\s*contiguous\b", "", generated, flags=re.IGNORECASE))
    writer = FortranWriter()
    routine = next(r for r in tree.walk(Routine) if r.name.lower() == entry.lower())
    symbols = {s.name.lower(): s for s in routine.symbol_table.datasymbols}
    arguments = {s.name.lower() for s in routine.symbol_table.argument_list}
    arrays = {n for n, s in symbols.items() if isinstance(s.datatype, ArrayType)}
    readonly = {n for n, s in symbols.items() if s.is_constant or
                (s.is_argument and s.interface.access == ArgumentInterface.Access.READ)}
    scope = Scope(arguments, arrays, readonly, set(symbols) - arguments - readonly)
    if arrays - arguments:
        raise ValueError("GPU arrays must be explicit dummy arguments")

    def access(ref):
        return Access(ref.name.lower(), tuple(writer(index) for index in ref.indices)
                      if isinstance(ref, ArrayReference) else None)

    def checked(node):
        if isinstance(node, Assignment):
            refs = node.rhs.walk(Reference)
            if isinstance(node.lhs, ArrayReference):
                for index in node.lhs.indices:
                    refs += index.walk(Reference)
            return CheckedAssignment(access(node.lhs), [access(ref) for ref in refs if isinstance(ref.symbol, DataSymbol)],
                                     [call.intrinsic.name.lower() if isinstance(call, IntrinsicCall)
                                      else "procedure:" + call.routine.name.lower() for call in node.rhs.walk(Call)])
        if isinstance(node, Loop):
            return CheckedLoop(node.variable.name.lower(),
                               tuple(writer(expr).replace(" ", "").lower() for expr in
                                     (node.start_expr, node.stop_expr, node.step_expr)),
                               [checked(child) for child in node.loop_body.children])
        return Unsupported(type(node).__name__)

    class HaloPreservingData(ACCDataDirective):
        """A data-region policy that preserves unwritten parts of every array.

        PSyclone's default output-only clause loses halo values when a region
        writes only its interior. Access is derived from declared dummy intent;
        compiler-private scalars are intentionally absent from the data region.
        """
        def create_data_movement_deep_copy_refs(self):
            reads, readwrites = {}, {}
            if not self.children:
                return reads, {}, readwrites
            for name in sorted(arrays):
                (reads if name in readonly else readwrites)[name] = Reference(symbols[name])
            return reads, {}, readwrites

    class PrivateLoop(ACCLoopDirective):
        def begin_string(self, leading_acc=True):
            result = super().begin_string(leading_acc)
            names = getattr(self, "gpu_private", ())
            return result + (" private(" + ", ".join(names) + ")" if names else "")

    loops = [node for node in routine.children if isinstance(node, Loop)]
    if not loops:
        raise ValueError("No eligible GPU loop in selected entry")
    if any(not isinstance(node, (Loop, Assignment)) for node in routine.children):
        raise ValueError("Unsupported host statement in GPU region")
    for node in routine.children:
        if isinstance(node, Assignment) and any(ref.name.lower() in arrays for ref in node.walk(Reference)):
            raise ValueError("Host array accesses inside GPU data region require synchronization")
    analyses = prove_gpu_loops([checked(loop) for loop in loops],
                               [checked(node) for node in routine.children if isinstance(node, Assignment)], scope)
    for loop, analysis in zip(loops, analyses):
        ACCLoopTrans().apply(loop, {"collapse": len(analysis["variables"]), "gang": True, "vector": True})
        directive = loop.parent.parent
        private = PrivateLoop(children=[n.detach() for n in list(directive.dir_body.children)],
                              collapse=directive.collapse, gang=directive.gang, vector=directive.vector)
        private.gpu_private = sorted(analysis["scalar_writes"])
        directive.replace_with(private)
        ACCParallelTrans().apply(private)
    ACCDataTrans().apply(list(routine.children))
    data = routine.children[0]
    data.replace_with(HaloPreservingData(children=[n.detach() for n in list(data.dir_body.children)]))
    generated = restore_contiguous(writer(tree), contracts)
    metadata["passes"] += ["ACCLoopTrans", "ACCParallelTrans", "ACCDataTrans", "halo-preserving data-clause IR policy"]
    metadata.update(changed=True, kernel_regions=len(loops), data_regions=1)
    return generated, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", choices=("loki", "psyclone"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--entry", required=True)
    parser.add_argument("--context", default="{}")
    args = parser.parse_args()
    source = args.source.read_text()
    if re.search(r"^\s*!\$(?:omp|acc)\b", source, re.IGNORECASE | re.MULTILINE):
        raise ValueError("GPU input must be normalized without native parallel directives")
    generated, metadata = (loki_gpu if args.tool == "loki" else psyclone_gpu)(source, args.entry, json.loads(args.context))
    host, backend, abi = bridge(generated, args.entry)
    (args.output_dir / "module.acc.f90").write_text(generated)
    (args.output_dir / "interface.f90").write_text(host)
    (args.output_dir / "backend.acc.f90").write_text(backend)
    metadata["bridge"] = abi
    (args.output_dir / "worker.json").write_text(json.dumps(metadata, indent=2) + "\n")
