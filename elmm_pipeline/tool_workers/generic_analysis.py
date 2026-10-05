"""Backend-neutral, conservative dependence checks for source transformations.

The backend supplies typed references. No procedure, field, rank, or axis names
are part of this contract. Unproved cases are diagnostics, never forced passes.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass, field
from fractions import Fraction
import re


@dataclass(frozen=True)
class Access:
    name: str
    indices: tuple[str, ...] | None = None


@dataclass
class Assignment:
    target: Access
    reads: list[Access]
    calls: list[str] = field(default_factory=list)


@dataclass
class Loop:
    variable: str
    bounds: tuple[str, str, str]
    body: list


@dataclass
class Unsupported:
    reason: str


@dataclass
class Scope:
    arguments: set[str]
    arrays: set[str]
    readonly: set[str]
    locals: set[str]
    unsafe: set[str] = field(default_factory=set)


PURE_INTRINSICS = {
    "abs", "min", "max", "sqrt", "exp", "log", "log10", "sin", "cos", "tan",
    "asin", "acos", "atan", "atan2", "sinh", "cosh", "tanh", "real", "int",
    "nint", "floor", "ceiling", "mod", "modulo", "sign", "dim", "dble",
    "kind", "size", "lbound", "ubound", "epsilon", "tiny", "huge", "merge",
}


def affine(expression: str) -> tuple[tuple[str, Fraction], ...] | None:
    """Canonicalize integer affine arithmetic, rejecting opaque expressions."""
    expression = re.sub(r"\b(\d+)_[a-zA-Z0-9_]+\b", r"\1", expression.lower())
    try:
        node = ast.parse(expression.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None

    def visit(n):
        if isinstance(n, ast.Name):
            return {n.id: Fraction(1)}
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return {"": Fraction(n.value)}
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            sign = -1 if isinstance(n.op, ast.USub) else 1
            return {k: sign * v for k, v in visit(n.operand).items()}
        if isinstance(n, ast.BinOp):
            a, b = visit(n.left), visit(n.right)
            if isinstance(n.op, (ast.Add, ast.Sub)):
                sign = -1 if isinstance(n.op, ast.Sub) else 1
                return {k: a.get(k, 0) + sign * b.get(k, 0) for k in a.keys() | b.keys()}
            if isinstance(n.op, ast.Mult):
                if set(a) <= {""}:
                    return {k: a.get("", 0) * v for k, v in b.items()}
                if set(b) <= {""}:
                    return {k: b.get("", 0) * v for k, v in a.items()}
        raise ValueError("non-affine expression")

    try:
        return tuple(sorted((key, value) for key, value in visit(node).items() if value))
    except ValueError:
        return None


def index_key(access):
    if access.indices is None:
        return None
    values = tuple(affine(value) for value in access.indices)
    return None if any(value is None for value in values) else values


def injective(indices, variables):
    """A full column-rank affine subscript map cannot alias two iterations."""
    matrix = [[dict(index).get(var, Fraction(0)) for var in variables] for index in indices]
    rank = 0
    for col in range(len(variables)):
        pivot = next((i for i in range(rank, len(matrix)) if matrix[i][col]), None)
        if pivot is None:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        factor = matrix[rank][col]
        matrix[rank] = [v / factor for v in matrix[rank]]
        for i in range(len(matrix)):
            if i != rank:
                factor = matrix[i][col]
                matrix[i] = [a - factor * b for a, b in zip(matrix[i], matrix[rank])]
        rank += 1
    return rank == len(variables)


def analyze_nest(loop, scope):
    variables, bounds = [], []
    current = loop
    while isinstance(current, Loop):
        variables.append(current.variable)
        bounds.append(current.bounds)
        if len(current.body) == 1 and isinstance(current.body[0], Loop):
            current = current.body[0]
        else:
            break
    if any(bound[2] != "1" for bound in bounds):
        return None, "nonunit stride requires a stride-preserving backend transformation"
    if any(re.search(r"\b" + re.escape(variable) + r"\b", " ".join(bound[:2]))
           for bound in bounds for variable in variables):
        return None, "nonrectangular loop bounds require additional proof"
    if len(set(variables)) != len(variables):
        return None, "reused induction variable in nested loops"
    if any(variable not in scope.locals or variable in scope.unsafe for variable in variables):
        return None, "observable or unsafe induction variable"
    body = current.body
    if not body or any(not isinstance(node, Assignment) for node in body):
        return None, "requires a perfect loop nest with straight-line assignments"
    writes, accesses, incoming, scalar_writes = {}, {}, set(), set()
    defined = set(variables)
    for node in body:
        if any(call.lower() not in PURE_INTRINSICS for call in node.calls):
            return None, "unknown or side-effecting call in loop"
        for access in node.reads:
            if access.name in scope.unsafe:
                return None, "pointer, target, volatile or unresolved symbol: " + access.name
            if access.indices is None:
                if access.name not in defined:
                    incoming.add(access.name)
            else:
                accesses.setdefault(access.name, []).append(index_key(access))
        target = node.target
        if target.name in scope.unsafe or target.name in scope.readonly:
            return None, "write to unsafe or read-only symbol: " + target.name
        if target.indices is None:
            if target.name not in scope.locals or target.name in variables:
                return None, "observable scalar write: " + target.name
            scalar_writes.add(target.name)
            defined.add(target.name)
        else:
            key = index_key(target)
            if key is None or not injective(key, variables):
                return None, "array write is not provably injective affine indexing: " + target.name
            if target.name in writes and writes[target.name] != key:
                return None, "multiple index maps written to array: " + target.name
            writes[target.name] = key
            accesses.setdefault(target.name, []).append(key)
    # An affine expression is injective only when every non-induction symbol
    # in its index map is invariant across the nest. A local assigned earlier
    # in each iteration is not invariant merely because it is initialized.
    # For example t=-i; a(i+t+1)=... always writes a(1).
    for node in body:
        for access in (node.target, *node.reads):
            if access.indices is not None and any(
                re.search(r"\b" + re.escape(name) + r"\b", expression, re.I)
                for expression in access.indices for name in scalar_writes
            ):
                return None, "array subscript depends on a loop-written scalar: " + access.name
    if scalar_writes & incoming:
        return None, "loop-carried scalar dependence: " + ", ".join(sorted(scalar_writes & incoming))
    for name, key in writes.items():
        if any(access != key for access in accesses[name]):
            return None, "loop-carried array dependence or unproved subscript: " + name
    if not writes:
        return None, "no array output"
    for variable, bound in zip(variables, bounds):
        if any(re.search(r"\b" + re.escape(name) + r"\b", " ".join(bound), re.I)
               for name in scalar_writes):
            return None, "loop bound depends on a written scalar"
    return {"variables": variables, "bounds": bounds, "writes": writes,
            "accesses": accesses, "scalar_writes": scalar_writes, "incoming": incoming}, None


def prove_fusion(first, second, scope, *, allow_renaming=False):
    left, reason = analyze_nest(first, scope)
    if reason:
        return False, reason
    right, reason = analyze_nest(second, scope)
    if reason:
        return False, reason
    if left["bounds"] != right["bounds"]:
        return False, "iteration spaces differ"
    if left["variables"] != right["variables"]:
        if not allow_renaming or len(left["variables"]) != len(right["variables"]):
            return False, "induction variables differ and have not been proved dead outside the nests"
        if set(left["variables"]) & right["incoming"] or set(right["variables"]) & left["incoming"]:
            return False, "loop nest reads the other nest's induction variable"
        # The caller has established that removed induction variables are not
        # observable outside these nests. Compare subscripts modulo alpha-renaming.
        for analysis in (left, right):
            rename = {var: "@iteration" + str(index) for index, var in enumerate(analysis["variables"])}
            def renamed(key):
                if key is None:
                    return None
                return tuple(tuple(sorted((rename.get(name, name), coefficient) for name, coefficient in index))
                             for index in key)
            analysis["writes"] = {name: renamed(key) for name, key in analysis["writes"].items()}
            analysis["accesses"] = {name: [renamed(key) for key in keys] for name, keys in analysis["accesses"].items()}
    if (left["scalar_writes"] & right["incoming"] or right["scalar_writes"] & left["incoming"]):
        return False, "scalar dependence across loop nests"
    for name in left["writes"].keys() | right["writes"].keys():
        keys = left["accesses"].get(name, []) + right["accesses"].get(name, [])
        if any(key is None or key != keys[0] for key in keys):
            return False, "cross-iteration dependence across loop nests: " + name
    return True, None


def contains_unsupported(node):
    return isinstance(node, Unsupported) or (isinstance(node, Loop) and any(contains_unsupported(child) for child in node.body))


def may_move_before(assignment, loop, scope):
    """Move already-executed invariant setup across an independent loop.

    This does not speculate computations out of a possibly zero-trip loop.
    """
    if not isinstance(assignment, Assignment) or assignment.target.indices is not None:
        return False
    name = assignment.target.name
    if name not in scope.locals or name in scope.unsafe:
        return False
    if assignment.calls or any(ref.indices is not None for ref in assignment.reads):
        return False
    analysis, reason = analyze_nest(loop, scope)
    if reason:
        return False
    reads = {ref.name for ref in assignment.reads}
    touched = analysis["scalar_writes"] | analysis["incoming"] | set(analysis["variables"])
    return name not in touched and not (reads & (analysis["scalar_writes"] | set(analysis["variables"])))
