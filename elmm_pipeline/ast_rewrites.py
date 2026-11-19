"""Anchors for ELMM's sources, including the unpreprocessed ones.

The anchor matcher is the compiler's (``compiler.resident.anchors``). ELMM's
raw sources carry ``#ifdef PAR`` blocks for its MPI build, which is not part
of the resident configuration; they are dropped before such a source is parsed.
"""
from __future__ import annotations



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
