"""Source helpers still specific to ELMM's unpreprocessed files.

Anchors and the loop rewrites are the compiler's (``compiler.resident``). What
stays here: ELMM's raw sources carry ``#ifdef PAR`` blocks for its MPI build,
which is not part of the resident configuration, and the pressure preparation
expands one PURE function by hand.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

from fparser.two.utils import walk

from compiler.resident import anchors
from compiler.resident.anchors import (  # noqa: F401
    _comment_mask,
    _line_char_offsets,
    _split_top_level_args,
    find_all_calls,
    find_call_span,
    find_once,
    find_unique_span,
    replace_once,
)

_UNDEFINED = ("PAR",)


def parse_fortran(text: str):
    return anchors.parse_fortran(text, undefined=_UNDEFINED)


def find_routine_span(text: str, name: str, *, kind: str = "subroutine") -> tuple[int, int]:
    return anchors.find_routine_span(text, name, kind=kind, undefined=_UNDEFINED)


def inline_pure_scalar_function(source_text: str, func_name: str | None = None) -> dict:
    """Extract a PURE single-expression scalar function for general call-site inlining."""
    tree = parse_fortran(source_text)
    matches = []
    for node in walk(tree):
        if type(node).__name__ != "Function_Subprogram":
            continue
        header = next((x for x in node.content if type(x).__name__ == "Function_Stmt"), None)
        if header is None:
            continue
        name = str(header.items[1]).lower()
        if func_name is not None and name != func_name.lower():
            continue
        if "pure" not in str(header.items[0]).lower():
            continue
        matches.append((node, header))
    if len(matches) != 1:
        label = func_name or "PURE scalar function"
        raise ValueError(f"Expected one original PURE scalar {label}")

    function, header = matches[0]
    name = str(header.items[1]).lower()
    args_node = header.items[2]
    if args_node is None:
        dummy_args: list[str] = []
    elif hasattr(args_node, "items"):
        dummy_args = [str(x).lower() for x in args_node.items]
    else:
        dummy_args = [str(args_node).lower()]

    result_name = name
    if len(header.items) > 3 and header.items[3] is not None:
        suffix = header.items[3]
        if getattr(suffix, "items", None) and suffix.items[0] is not None:
            result_name = str(suffix.items[0]).lower()

    execution = next((x for x in function.content if type(x).__name__ == "Execution_Part"), None)
    if execution is None:
        raise ValueError(f"Missing execution body in {name}")
    statements = [x for x in execution.content if type(x).__name__ != "Comment"]
    if (
        len(statements) != 1
        or type(statements[0]).__name__ != "Assignment_Stmt"
        or str(statements[0].items[0]).lower() != result_name
    ):
        raise ValueError(f"{name} must remain one scalar result expression")

    constants = [
        x
        for x in walk(function)
        if type(x).__name__ == "Type_Declaration_Stmt"
        and x.items[1] is not None
        and "parameter" in str(x.items[1]).lower()
    ]
    expression = str(statements[0].items[2])

    def expand_call(call_args: Sequence[str]) -> str:
        if len(call_args) != len(dummy_args):
            raise ValueError(
                f"Call to {name} expected {len(dummy_args)} arguments, got {len(call_args)}"
            )
        if not dummy_args:
            return expression
        mapping = {}
        for formal, actual in zip(dummy_args, call_args):
            cleaned = actual.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", cleaned):
                cleaned = f"({cleaned})"
            mapping[formal] = cleaned
        pattern = re.compile(
            r"(?<![.%_])\b(" + "|".join(re.escape(a) for a in dummy_args) + r")\b",
            re.IGNORECASE,
        )
        return pattern.sub(lambda m: mapping[m.group(1).lower()], expression)

    return {
        "name": name,
        "result_name": result_name,
        "dummy_args": dummy_args,
        "constants": constants,
        "constants_source": "".join(f"    {x}\n" for x in constants),
        "expression": expression,
        "expand_call": expand_call,
    }


def expand_pure_calls_in_text(text: str, info: dict) -> tuple[str, int]:
    """Replace all calls to ``info['name']`` in ``text`` with expanded scalar expressions."""
    name = info["name"]
    pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(name)}\s*\(", re.IGNORECASE)
    out: list[str] = []
    pos = 0
    count = 0
    mask = _comment_mask(text)
    n = len(text)
    while pos < n:
        m = pattern.search(text, pos)
        if m is None:
            out.append(text[pos:])
            break
        if m.start() < len(mask) and mask[m.start()]:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        open_paren = m.end() - 1
        depth = 0
        in_single = False
        in_double = False
        i = open_paren
        close_paren = None
        while i < n:
            ch = text[i]
            if ch == "'" and not in_double:
                in_single = not in_single
            elif ch == '"' and not in_single:
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        close_paren = i
                        break
            i += 1
        if close_paren is None:
            out.append(text[pos : m.end()])
            pos = m.end()
            continue
        args = _split_top_level_args(text[open_paren + 1 : close_paren])
        expanded = info["expand_call"](args)
        out.append(text[pos : m.start()])
        out.append(f"({expanded})")
        count += 1
        pos = close_paren + 1
    return "".join(out), count
