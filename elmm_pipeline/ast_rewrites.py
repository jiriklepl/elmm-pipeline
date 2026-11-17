"""Structural Fortran AST and token-normalized source rewrite helpers."""
from __future__ import annotations

import re
from typing import Sequence

from fparser.common.readfortran import FortranStringReader
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

_PARSER = None


def _get_parser():
    global _PARSER
    if _PARSER is None:
        _PARSER = ParserFactory().create(std="f2008")
    return _PARSER


def _sanitize_for_parse(text: str) -> str:
    """Replace C preprocessor macros/directives while preserving exact line numbers."""
    lines = []
    skip_inactive = False
    depth = 0
    inactive_depth = None
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            directive = stripped[1:].strip().lower()
            if directive.startswith("if"):
                depth += 1
                if not skip_inactive and re.match(r"ifdef\s+par\b", directive):
                    skip_inactive = True
                    inactive_depth = depth
            elif directive.startswith("else") and skip_inactive and depth == inactive_depth:
                skip_inactive = False
            elif directive.startswith("endif"):
                if skip_inactive and depth == inactive_depth:
                    skip_inactive = False
                    inactive_depth = None
                depth = max(0, depth - 1)
            lines.append("\n" if line.endswith("\n") else "")
            continue
        if skip_inactive:
            lines.append("\n" if line.endswith("\n") else "")
            continue
        cleaned = line.replace("__FILE__", "'__FILE__'").replace("__LINE__", "0")
        lines.append(cleaned)
    return "".join(lines)


def parse_fortran(text: str):
    """Parse free-form Fortran 2008 text, tolerating preprocessor macros."""
    parser = _get_parser()
    try:
        return parser(FortranStringReader(text, ignore_comments=False))
    except Exception:
        sanitized = _sanitize_for_parse(text)
        return parser(FortranStringReader(sanitized, ignore_comments=False))


def _strip_inline_comment(line: str) -> tuple[str, str]:
    """Split a Fortran line into (code, comment) respecting quotes and !$omp."""
    in_single = False
    in_double = False
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'" and not in_double:
            if in_single and i + 1 < len(line) and line[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < len(line) and line[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif ch == "!" and not in_single and not in_double:
            if line[i:].lower().startswith("!$omp"):
                return line, ""
            return line[:i], line[i:]
        i += 1
    return line, ""


def _logical_lines(text: str) -> list[str]:
    """Join Fortran '&' line continuations into logical lines."""
    raw_lines = text.splitlines()
    logical: list[str] = []
    current = ""
    continuing = False
    for raw in raw_lines:
        code, comment = _strip_inline_comment(raw)
        stripped = code.rstrip()
        if continuing:
            lead = stripped.lstrip()
            if lead.startswith("&"):
                lead = lead[1:]
            piece = lead
        else:
            piece = stripped
        has_cont = piece.endswith("&")
        if has_cont:
            piece = piece[:-1].rstrip()
        if continuing:
            current = f"{current} {piece}".strip() if current and piece else (current or piece)
        else:
            current = piece + (comment if not has_cont and comment else "")
        if has_cont:
            continuing = True
        else:
            logical.append(current)
            current = ""
            continuing = False
    if continuing or current:
        logical.append(current)
    return logical


_TOKEN_SPEC = re.compile(
    r"""
    (?P<string>'(?:''|[^'])*'|"(?:""|[^"])*")
    | (?P<omp>!\$omp\b[^\n]*)
    | (?P<comment>![^\n]*)
    | (?P<dotop>\.[a-z_]+\.)
    | (?P<two>::|/=|==|<=|>=|=>|\*\*|//)
    | (?P<word>[a-z_][a-z0-9_]*)
    | (?P<number>(?:\d+\.\d*|\.\d+|\d+)(?:[ed][+-]?\d+)?(?:_[a-z0-9_]+)?)
    | (?P<punct>[()\[\],=+\-*/%:])
    | (?P<ws>\s+)
    | (?P<other>.)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_COMPOUND_NEXT = {
    "end": {"do", "if", "subroutine", "function", "module", "block", "select", "associate", "type"},
    "else": {"if"},
}


def _tokenize_line(line: str, *, preserve_comments: bool) -> list[tuple[str, str]]:
    """Tokenize one logical Fortran line into (kind, raw_text) pairs."""
    tokens: list[tuple[str, str]] = []
    raw_tokens: list[tuple[str, str]] = []
    for match in _TOKEN_SPEC.finditer(line):
        kind = match.lastgroup
        value = match.group()
        if kind == "ws":
            continue
        if kind == "comment" and not preserve_comments:
            continue
        raw_tokens.append((kind, value))

    i = 0
    while i < len(raw_tokens):
        kind, value = raw_tokens[i]
        if kind == "word" and value.lower() in _COMPOUND_NEXT and i + 1 < len(raw_tokens):
            nkind, nval = raw_tokens[i + 1]
            if nkind == "word" and nval.lower() in _COMPOUND_NEXT[value.lower()]:
                tokens.append(("compound", f"{value.lower()}:{nval.lower()}"))
                i += 2
                continue
        if kind == "word" and value.lower() in {
            "enddo", "endif", "endsubroutine", "endfunction", "endmodule",
            "endblock", "endselect", "endassociate", "endtype", "elseif",
        }:
            prefix = "else" if value.lower().startswith("else") else "end"
            suffix = value.lower()[len(prefix):]
            tokens.append(("compound", f"{prefix}:{suffix}"))
            i += 1
            continue
        tokens.append((kind, value))
        i += 1
    return tokens


def _token_to_regex(kind: str, value: str) -> str:
    if kind == "compound":
        prefix, suffix = value.split(":", 1)
        return rf"\b{re.escape(prefix)}\s*{re.escape(suffix)}\b"
    if kind == "omp":
        sub = [_token_to_regex(k, v) for k, v in _tokenize_line(value[5:], preserve_comments=False)]
        return r"!\$omp" + (r"\s+" + r"\s*".join(sub) if sub else r"")
    if kind == "comment":
        body = " ".join(value[1:].split())
        if not body:
            return r"![^\n]*"
        words = [re.escape(part) for part in body.split()]
        return r"!\s*" + r"\s+".join(words)
    if kind in {"word", "number"}:
        return rf"\b{re.escape(value)}\b"
    return re.escape(value)


def _build_line_regex(tokens: list[tuple[str, str]]) -> str:
    if not tokens:
        return ""
    parts = [_token_to_regex(tokens[0][0], tokens[0][1])]
    wordlike = {"word", "number", "compound", "dotop"}
    sep_opt = r"(?:\s|&\s*(?:![^\n]*)?\r?\n\s*&?)*"
    sep_req = r"(?:\s|&\s*(?:![^\n]*)?\r?\n\s*&?)+"
    for (pk, _), (ck, cv) in zip(tokens, tokens[1:]):
        sep = sep_req if (pk in wordlike and ck in wordlike) else sep_opt
        parts.append(sep + _token_to_regex(ck, cv))
    return "".join(parts)


def _comment_mask(text: str) -> list[bool]:
    """Compute per-character boolean mask indicating characters inside non-OMP comments."""
    mask = [False] * len(text)
    in_single = False
    in_double = False
    in_comment = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "\n":
            in_single = False
            in_double = False
            in_comment = False
            i += 1
            continue
        if in_comment:
            mask[i] = True
            i += 1
            continue
        if ch == "'" and not in_double:
            if in_single and i + 1 < n and text[i + 1] == "'":
                i += 2
                continue
            in_single = not in_single
        elif ch == '"' and not in_single:
            if in_double and i + 1 < n and text[i + 1] == '"':
                i += 2
                continue
            in_double = not in_double
        elif ch == "!" and not in_single and not in_double:
            if not text[i:].lower().startswith("!$omp"):
                in_comment = True
                mask[i] = True
        i += 1
    return mask


def _find_token_matches(text: str, anchor: str, *, preserve_comments: bool) -> list[tuple[int, int]]:
    logical = _logical_lines(anchor)
    stmt_regexes: list[str] = []
    for line in logical:
        tokens = _tokenize_line(line, preserve_comments=preserve_comments)
        if tokens:
            stmt_regexes.append(_build_line_regex(tokens))
    if not stmt_regexes:
        return []

    has_nl = "\n" in anchor
    whole_line = has_nl or anchor[:1] in {" ", "\t"}
    if len(stmt_regexes) == 1:
        body = stmt_regexes[0]
    else:
        between = r"[ \t]*(?:![^\n]*)?\r?\n(?:\s*(?:![^\n]*)?\r?\n)*[ \t]*"
        body = between.join(stmt_regexes)

    if whole_line:
        prefix = r"(?im)^[ \t]*"
        suffix = r"[ \t]*(?:![^\n]*)?\r?\n" if anchor.endswith("\n") else r""
    else:
        prefix = r"(?im)"
        suffix = r""

    pattern = re.compile(prefix + body + suffix)
    anchor_has_comment = anchor.lstrip().startswith("!") and not anchor.lstrip().lower().startswith("!$omp")
    mask = None if anchor_has_comment else _comment_mask(text)

    matches: list[tuple[int, int]] = []
    for m in pattern.finditer(text):
        start, end = m.span()
        if mask is not None and start < len(mask) and mask[start]:
            continue
        # Ensure inline expression anchors do not match starting inside a comment.
        if mask is not None and not whole_line and any(mask[pos] for pos in range(start, end)):
            continue
        matches.append((start, end))
    return matches


def find_unique_span(text: str, anchor: str, *, message: str | None = None) -> tuple[int, int]:
    """Locate ``anchor`` uniquely in ``text``, tolerating formatting and comment edits."""
    exact_count = text.count(anchor)
    if exact_count == 1:
        idx = text.index(anchor)
        return idx, idx + len(anchor)
    if exact_count > 1:
        raise ValueError(message or f"Pressure outlining requires one original anchor: {anchor!r}")

    has_plain_comment = any(
        _strip_inline_comment(line)[1].strip() for line in anchor.splitlines()
    )
    if has_plain_comment:
        matches = _find_token_matches(text, anchor, preserve_comments=True)
        if len(matches) == 1:
            return matches[0]
    matches = _find_token_matches(text, anchor, preserve_comments=False)
    if len(matches) == 1:
        return matches[0]

    # Allow optional trailing procedure/module name on END SUBROUTINE / END MODULE anchors.
    end_match = re.fullmatch(
        r"\s*end\s*(subroutine|function|module)\s+[a-z_][a-z0-9_]*\s*\n?",
        anchor,
        re.IGNORECASE,
    )
    if end_match:
        stripped_anchor = re.sub(
            r"(end\s*(?:subroutine|function|module))\s+[a-z_][a-z0-9_]*",
            r"\1",
            anchor,
            flags=re.IGNORECASE,
        )
        matches = _find_token_matches(text, stripped_anchor, preserve_comments=False)
        if len(matches) == 1:
            return matches[0]

    raise ValueError(message or f"Pressure outlining requires one original anchor: {anchor!r}")


def find_once(text: str, anchor: str, *, message: str | None = None) -> int:
    """Return the starting character offset of a unique anchor in ``text``."""
    return find_unique_span(text, anchor, message=message)[0]


def replace_once(
    text: str,
    anchor: str,
    replacement: str,
    *,
    after: bool = False,
    insert: bool = False,
    message: str | None = None,
) -> str:
    """Replace or insert relative to a unique anchor in ``text``."""
    start, end = find_unique_span(text, anchor, message=message)
    if insert:
        pos = end if after else start
        return text[:pos] + replacement + text[pos:]
    return text[:start] + replacement + text[end:]


def _line_char_offsets(text: str) -> list[tuple[int, int]]:
    """Return 1-indexed line offsets ``[(line_start, line_end_no_nl), ...]``."""
    offsets: list[tuple[int, int]] = [(0, 0)]  # dummy index 0
    pos = 0
    for line in text.splitlines(keepends=True):
        no_nl = line.rstrip("\r\n")
        offsets.append((pos, pos + len(no_nl)))
        pos += len(line)
    return offsets


def find_routine_span(text: str, name: str, *, kind: str = "subroutine") -> tuple[int, int]:
    """Return ``(start_char, end_char)`` for the named Fortran subroutine or function."""
    target = name.lower()
    sub_kind = "Subroutine_Subprogram" if kind == "subroutine" else "Function_Subprogram"
    stmt_kind = "Subroutine_Stmt" if kind == "subroutine" else "Function_Stmt"
    end_kind = "End_Subroutine_Stmt" if kind == "subroutine" else "End_Function_Stmt"

    try:
        tree = parse_fortran(text)
    except Exception:
        tree = None

    if tree is not None:
        top_matches: list[tuple[int, int]] = []
        all_matches: list[tuple[int, int]] = []
        offsets = _line_char_offsets(text)
        for node in walk(tree):
            if type(node).__name__ != sub_kind:
                continue
            stmt = next((c for c in node.content if type(c).__name__ == stmt_kind), None)
            end_stmt = next((c for c in reversed(node.content) if type(c).__name__ == end_kind), None)
            if stmt is None or end_stmt is None or len(stmt.items) < 2:
                continue
            if str(stmt.items[1]).lower() != target:
                continue
            s_line = stmt.item.span[0]
            e_line = end_stmt.item.span[1]
            char_span = (offsets[s_line][0], offsets[e_line][1])
            all_matches.append(char_span)
            parent_type = type(getattr(node, "parent", None)).__name__
            if parent_type not in {"Internal_Subprogram_Part"}:
                top_matches.append(char_span)
        chosen = top_matches if top_matches else all_matches
        if len(chosen) == 1:
            return chosen[0]
        if len(chosen) > 1:
            raise ValueError(f"Expected one complete original {name} procedure")

    # Fallback line scanner if raw syntax contains unsupported preprocessor blocks.
    start_pat = re.compile(
        rf"(?im)^[ \t]*(?:(?:pure|elemental|recursive|impure|module)\s+)*{kind}\s+{re.escape(name)}\b(?:\s*\(|[ \t]*(?:!.*)?$)"
    )
    starts = list(start_pat.finditer(text))
    if len(starts) == 1:
        named_end = re.search(
            rf"(?im)^[ \t]*end\s*{kind}\s+{re.escape(name)}\b[^\n]*",
            text[starts[0].end():],
        )
        end = named_end or re.search(
            rf"(?im)^[ \t]*end\s*{kind}[ \t]*(?:!.*)?$",
            text[starts[0].end():],
        )
        if end is not None:
            return starts[0].start(), starts[0].end() + end.end()
    raise ValueError(f"Expected one complete original {name} procedure")


def _split_top_level_args(arg_str: str) -> list[str]:
    args: list[str] = []
    cur: list[str] = []
    depth = 0
    in_single = False
    in_double = False
    i = 0
    while i < len(arg_str):
        ch = arg_str[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            cur.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            cur.append(ch)
        elif not in_single and not in_double:
            if ch in "([":
                depth += 1
                cur.append(ch)
            elif ch in ")]":
                depth = max(0, depth - 1)
                cur.append(ch)
            elif ch == "&":
                # Skip continuation marker and following newline/comment/leading &
                rest = arg_str[i + 1 :]
                nl = rest.find("\n")
                if nl != -1:
                    i += 1 + nl
                    after = arg_str[i + 1 :].lstrip(" \t")
                    skip = len(arg_str[i + 1 :]) - len(after)
                    if after.startswith("&"):
                        skip += 1
                    i += skip
            elif ch == "," and depth == 0:
                args.append(" ".join("".join(cur).split()))
                cur = []
                i += 1
                continue
            else:
                cur.append(ch)
        else:
            cur.append(ch)
        i += 1
    tail = " ".join("".join(cur).split())
    if tail or args:
        args.append(tail)
    return args


def find_all_calls(text: str, callee_name: str) -> list[tuple[int, int, str, list[str]]]:
    """Find all non-comment ``call <callee_name>(...)`` occurrences in ``text``."""
    mask = _comment_mask(text)
    pattern = re.compile(rf"(?im)(?:^[ \t]*|(?<=\))[ \t]*)call\s+{re.escape(callee_name)}\b")
    results: list[tuple[int, int, str, list[str]]] = []
    n = len(text)
    for m in pattern.finditer(text):
        call_start = m.start()
        if call_start < len(mask) and mask[call_start]:
            continue
        pos = m.end()
        # Skip whitespace and line continuations between callee name and '('
        while pos < n:
            if text[pos] in " \t":
                pos += 1
            elif text[pos] == "&":
                nl = text.find("\n", pos)
                if nl == -1:
                    pos = n
                else:
                    pos = nl + 1
                    while pos < n and text[pos] in " \t&":
                        pos += 1
            else:
                break
        if pos >= n or text[pos] != "(":
            # Argumentless CALL
            end_line = text.find("\n", m.end())
            stmt_end = n if end_line == -1 else end_line
            results.append((call_start, stmt_end, text[call_start:stmt_end], []))
            continue
        depth = 0
        in_single = False
        in_double = False
        in_comment = False
        arg_start = pos + 1
        end_pos = None
        i = pos
        while i < n:
            ch = text[i]
            if ch == "\n":
                in_comment = False
                i += 1
                continue
            if in_comment:
                i += 1
                continue
            if ch == "'" and not in_double:
                if in_single and i + 1 < n and text[i + 1] == "'":
                    i += 2
                    continue
                in_single = not in_single
            elif ch == '"' and not in_single:
                if in_double and i + 1 < n and text[i + 1] == '"':
                    i += 2
                    continue
                in_double = not in_double
            elif not in_single and not in_double:
                if ch == "!":
                    in_comment = True
                elif ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        end_pos = i + 1
                        break
            i += 1
        if end_pos is None:
            continue
        args = _split_top_level_args(text[arg_start : end_pos - 1])
        results.append((call_start, end_pos, text[call_start:end_pos], args))
    return results


def find_call_span(
    text: str,
    callee_name: str,
    *,
    within_routine: str | None = None,
) -> tuple[int, int, str, list[str]]:
    """Locate a unique ``call <callee_name>(...)`` statement."""
    if within_routine is not None:
        r_lo, r_hi = find_routine_span(text, within_routine)
        sub = find_all_calls(text[r_lo:r_hi], callee_name)
        if len(sub) != 1:
            raise ValueError(f"Expected one call to {callee_name} in {within_routine}")
        s, e, raw, args = sub[0]
        return r_lo + s, r_lo + e, raw, args
    matches = find_all_calls(text, callee_name)
    if len(matches) != 1:
        raise ValueError(f"Expected one call to {callee_name}")
    return matches[0]


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


def _do_loop_info(loop_node) -> tuple[str, list]:
    header = next(
        (item for item in loop_node.content if type(item).__name__ == "Nonlabel_Do_Stmt"),
        None,
    )
    if header is None or len(header.items) < 2 or header.items[1] is None:
        raise ValueError("Expected counted DO construct")
    counted = header.items[1].items[1]
    if not counted:
        raise ValueError("Expected counted DO loop control")
    var = str(counted[0]).lower()
    bounds = list(counted[1])
    return var, bounds


def _loop_statements(node) -> list:
    return [
        item
        for item in node.content
        if type(item).__name__ not in {"Comment", "Nonlabel_Do_Stmt", "End_Do_Stmt"}
    ]


def _extract_consumer_upper_bound(bound_expr: str, outer_var: str, outer_hi: str) -> str:
    text = re.sub(r"\s+", "", bound_expr)
    m = re.fullmatch(r"min\((.+)\)", text, re.IGNORECASE)
    if m:
        args = _split_top_level_args(m.group(1))
        for arg in args:
            if not re.search(rf"\b{re.escape(outer_var)}\b", arg, re.IGNORECASE):
                return arg
    if not re.search(rf"\b{re.escape(outer_var)}\b", text, re.IGNORECASE):
        return bound_expr.strip()
    return outer_hi.strip()


def _parse_local_index_assignment(lhs_str: str, rhs_str: str, inner_vars: set[str], outer_vars: set[str]):
    """Match ``lvar = inner_var - outer_var (+ int)?``."""
    if "(" in lhs_str:
        return None
    clean_rhs = re.sub(r"\s+", "", rhs_str).lower()
    for ivar in inner_vars:
        for ovar in outer_vars:
            m = re.fullmatch(
                rf"{re.escape(ivar)}-{re.escape(ovar)}([+-]\d+)?",
                clean_rhs,
            )
            if m:
                offset = int(m.group(1)) if m.group(1) else 0
                return lhs_str.strip().lower(), ivar, ovar, offset
    return None


def _shift_vars(expression: str, var_map: dict[str, str], offsets: Sequence[int], ordered_axes: Sequence[str]) -> str:
    """Shift loop variables in ``expression`` in a single pass."""
    mapping = {}
    for axis_idx, pvar in enumerate(ordered_axes):
        cvar = var_map[pvar]
        off = offsets[axis_idx]
        mapping[pvar] = f"({cvar}+({off}))" if off != 0 else cvar
    if not mapping:
        return expression
    pattern = re.compile(
        r"(?<![.%_])\b(" + "|".join(re.escape(v) for v in mapping) + r")\b",
        re.IGNORECASE,
    )
    return pattern.sub(lambda m: mapping[m.group(1).lower()], expression)


def _unpack_3d_nest(nest_node):
    """Unpack a 3-level loop nest, allowing hoisted assignments between loop headers."""
    v2, b2 = _do_loop_info(nest_node)
    stmts2 = _loop_statements(nest_node)
    assigns2 = [s for s in stmts2 if type(s).__name__ == "Assignment_Stmt"]
    loops2 = [s for s in stmts2 if type(s).__name__ == "Block_Nonlabel_Do_Construct"]
    if len(loops2) != 1 or len(assigns2) + 1 != len(stmts2):
        raise ValueError("Expected single nested loop at level 2")
    v1, b1 = _do_loop_info(loops2[0])
    stmts1 = _loop_statements(loops2[0])
    assigns1 = [s for s in stmts1 if type(s).__name__ == "Assignment_Stmt"]
    loops1 = [s for s in stmts1 if type(s).__name__ == "Block_Nonlabel_Do_Construct"]
    if len(loops1) != 1 or len(assigns1) + 1 != len(stmts1):
        raise ValueError("Expected single nested loop at level 1")
    v0, b0 = _do_loop_info(loops1[0])
    return v2, b2, v1, b1, v0, b0, assigns2 + assigns1 + _loop_statements(loops1[0])


def inline_tile_scratch_nests(
    source_text: str,
    *,
    target_routines: set[str] | None = None,
) -> tuple[str, list[dict]]:
    """Inline 3D producer-consumer private tile scratch arrays into pointwise loops."""
    lines = source_text.splitlines(keepends=True)
    tree = parse_fortran(source_text)
    changes: list[tuple[int, int, str, dict]] = []

    for routine in (n for n in walk(tree) if type(n).__name__ == "Subroutine_Subprogram"):
        header = next(n for n in routine.content if type(n).__name__ == "Subroutine_Stmt")
        rname = str(header.items[1]).lower()
        if target_routines is not None and rname not in target_routines:
            continue
        execution = next((n for n in routine.content if type(n).__name__ == "Execution_Part"), None)
        if execution is None:
            continue
        tiles = [n for n in execution.content if type(n).__name__ == "Block_Nonlabel_Do_Construct"]
        for tile in tiles:
            # Verify 3-level outer tile nest containing [producer, consumer]
            try:
                o_var2, o_b2 = _do_loop_info(tile)
                c2 = _loop_statements(tile)
                if len(c2) != 1 or type(c2[0]).__name__ != "Block_Nonlabel_Do_Construct":
                    continue
                mid = c2[0]
                o_var1, o_b1 = _do_loop_info(mid)
                c1 = _loop_statements(mid)
                if len(c1) != 1 or type(c1[0]).__name__ != "Block_Nonlabel_Do_Construct":
                    continue
                inner = c1[0]
                o_var0, o_b0 = _do_loop_info(inner)
                pair = _loop_statements(inner)
                if len(pair) != 2 or any(type(n).__name__ != "Block_Nonlabel_Do_Construct" for n in pair):
                    continue
                producer, consumer = pair
                # Unpack 3-level producer and consumer nests
                p_var2, _, p_var1, _, p_var0, _, prod_stmts = _unpack_3d_nest(producer)
                c_var2, c_b2, c_var1, c_b1, c_var0, c_b0, cons_stmts = _unpack_3d_nest(consumer)
            except ValueError:
                continue

            outer_vars = {o_var0, o_var1, o_var2}
            prod_vars = {p_var0, p_var1, p_var2}
            cons_vars = {c_var0, c_var1, c_var2}

            prod_local_idx: dict[str, tuple[str, int]] = {}  # lvar -> (p_loop_var, const_offset)
            scalar_defs: dict[str, str] = {}
            scratch_defs: dict[tuple[str, tuple[str, str, str]], str] = {}
            straight_line = True
            for stmt in prod_stmts:
                if type(stmt).__name__ != "Assignment_Stmt":
                    straight_line = False
                    break
                lhs, _, rhs = stmt.items
                lhs_str = str(lhs).strip()
                parsed_idx = _parse_local_index_assignment(lhs_str, str(rhs), prod_vars, outer_vars)
                if parsed_idx is not None:
                    lvar, ivar, _ovar, off = parsed_idx
                    prod_local_idx[lvar] = (ivar, off)
                    continue
                expr = str(rhs)
                if scalar_defs:
                    pat = re.compile(
                        r"(?<![.%_])\b(" + "|".join(re.escape(k) for k in scalar_defs) + r")\b",
                        re.IGNORECASE,
                    )
                    expr = pat.sub(lambda m: f"({scalar_defs[m.group(1).lower()]})", expr)
                if type(lhs).__name__ == "Part_Ref":
                    arr_name = str(lhs.items[0]).lower()
                    idx_items = tuple(re.sub(r"\s+", "", str(x)).lower() for x in lhs.items[1].items)
                    if len(idx_items) != 3 or not all(idx in prod_local_idx for idx in idx_items):
                        straight_line = False
                        break
                    scratch_defs[(arr_name, idx_items)] = expr
                else:
                    scalar_defs[lhs_str.lower()] = expr

            if not straight_line or not scratch_defs:
                continue

            scratch_names = {arr for arr, _ in scratch_defs}
            # Determine consumer local index variables and map producer->consumer axes
            cons_local_idx: dict[str, tuple[str, int]] = {}
            cons_body_stmts = []
            for stmt in cons_stmts:
                if type(stmt).__name__ == "Assignment_Stmt":
                    lhs_str = str(stmt.items[0]).strip()
                    parsed_idx = _parse_local_index_assignment(lhs_str, str(stmt.items[2]), cons_vars, outer_vars)
                    if parsed_idx is not None:
                        lvar, ivar, _ovar, off = parsed_idx
                        cons_local_idx[lvar] = (ivar, off)
                        continue
                cons_body_stmts.append(stmt)

            # Map producer loop var -> consumer loop var via matching outer loop variable
            outer_to_cvar = {}
            for cvar, cb in ((c_var0, c_b0), (c_var1, c_b1), (c_var2, c_b2)):
                for ovar in outer_vars:
                    if re.search(rf"\b{re.escape(ovar)}\b", str(cb[0]), re.IGNORECASE):
                        outer_to_cvar[ovar] = cvar
            if len(outer_to_cvar) != 3:
                outer_to_cvar = {o_var0: c_var0, o_var1: c_var1, o_var2: c_var2}

            prod_to_cons_var = {p_var0: c_var0, p_var1: c_var1, p_var2: c_var2}
            ordered_p_axes = [p_var0, p_var1, p_var2]

            rendered_body: list[str] = []
            valid_consumer = True
            for stmt in cons_body_stmts:
                rendered = str(stmt)
                for ref in sorted(
                    (
                        n
                        for n in walk(stmt)
                        if type(n).__name__ == "Part_Ref" and str(n.items[0]).lower() in scratch_names
                    ),
                    key=lambda n: len(str(n)),
                    reverse=True,
                ):
                    arr_name = str(ref.items[0]).lower()
                    ref_indices = getattr(ref.items[1], "items", ())
                    matching_keys = [k for k in scratch_defs if k[0] == arr_name]
                    if len(matching_keys) != 1 or len(ref_indices) != 3:
                        valid_consumer = False
                        break
                    _, def_idx_tuple = matching_keys[0]
                    offsets_by_pvar: dict[str, int] = {}
                    for def_lvar, idx_expr in zip(def_idx_tuple, ref_indices):
                        p_var, p_off = prod_local_idx[def_lvar]
                        clean_idx = re.sub(r"\s+", "", str(idx_expr)).lower()
                        # Find matching consumer local index variable
                        matched_off = None
                        for clvar, (_cvar, c_off) in cons_local_idx.items():
                            m = re.fullmatch(rf"{re.escape(clvar)}(?:([+-])(\d+))?", clean_idx)
                            if m:
                                ref_off = 0 if m.group(1) is None else int(m.group(1) + m.group(2))
                                matched_off = ref_off + (c_off - p_off)
                                break
                        if matched_off is None:
                            valid_consumer = False
                            break
                        offsets_by_pvar[p_var] = matched_off
                    if not valid_consumer:
                        break
                    axis_offsets = [offsets_by_pvar[pvar] for pvar in ordered_p_axes]
                    expanded = _shift_vars(
                        scratch_defs[(arr_name, def_idx_tuple)],
                        prod_to_cons_var,
                        axis_offsets,
                        ordered_p_axes,
                    )
                    rendered = rendered.replace(str(ref), f"({expanded})")
                if not valid_consumer:
                    break
                rendered_body.append(rendered)

            if not valid_consumer or not rendered_body:
                continue

            lo2 = str(o_b2[0]).strip()
            hi2 = _extract_consumer_upper_bound(str(c_b2[1]), o_var2, str(o_b2[1]))
            lo1 = str(o_b1[0]).strip()
            hi1 = _extract_consumer_upper_bound(str(c_b1[1]), o_var1, str(o_b1[1]))
            lo0 = str(o_b0[0]).strip()
            hi0 = _extract_consumer_upper_bound(str(c_b0[1]), o_var0, str(o_b0[1]))
            replacement = (
                f"    do {c_var2}={lo2},{hi2}\n"
                f"      do {c_var1}={lo1},{hi1}\n"
                f"        do {c_var0}={lo0},{hi0}\n"
                + "\n".join(rendered_body)
                + "\n        enddo\n      enddo\n    enddo\n"
            )
            first = next(n for n in tile.content if type(n).__name__ == "Nonlabel_Do_Stmt").item.span[0]
            last = tile.content[-1].item.span[1]
            changes.append(
                (
                    first,
                    last,
                    replacement,
                    {"routine": rname, "scratch_arrays": sorted(scratch_names)},
                )
            )

    for first, last, replacement, _meta in sorted(changes, key=lambda item: item[0], reverse=True):
        lines[first - 1 : last] = [replacement]
    metadata = [meta for _, _, _, meta in sorted(changes, key=lambda item: item[0])]
    return "".join(lines), metadata


def promote_line_snapshots(
    source_text: str,
    *,
    routine_name: str | None = None,
) -> tuple[str, list[dict]]:
    """Lift 1D line-copy buffers (``tmp = arr(:,j,k)``) into 3D snapshots."""
    tree = parse_fortran(source_text)
    edits: list[dict] = []
    text = source_text

    candidate_routines = []
    for routine in (n for n in walk(tree) if type(n).__name__ == "Subroutine_Subprogram"):
        header = next(n for n in routine.content if type(n).__name__ == "Subroutine_Stmt")
        rname = str(header.items[1]).lower()
        if routine_name is not None and rname != routine_name.lower():
            continue
        # Look for 1D declaration using ubound(<arr>, <dim>)
        found = False
        for decl in walk(routine):
            if type(decl).__name__ != "Type_Declaration_Stmt":
                continue
            dtype, attrs, entities = decl.items
            if attrs is not None and "allocatable" in str(attrs).lower():
                continue
            for entity in entities.items:
                shape = entity.items[1]
                if shape is None or len(getattr(shape, "items", ())) != 1:
                    continue
                dim_spec = shape.items[0]
                hi = str(getattr(dim_spec, "items", (None, None))[1] or "")
                m = re.fullmatch(
                    r"ubound\(\s*([a-z_][a-z0-9_]*)\s*,\s*[a-z_][a-z0-9_]*\s*\)",
                    re.sub(r"\s+", "", hi),
                    re.I,
                )
                if m:
                    lo_expr = re.sub(r"\s+", "", str(dim_spec.items[0]))
                    tmp_name = str(entity.items[0])
                    arr_name = m.group(1)
                    candidate_routines.append((rname, str(dtype), tmp_name, arr_name, lo_expr))
                    found = True
                    break
            if found:
                break

    if routine_name is not None and not candidate_routines:
        raise ValueError(f"Expected 1D line-snapshot buffer in {routine_name}")

    for rname, dtype_str, tmp_name, arr_name, lo_expr in reversed(candidate_routines):
        r_lo, r_hi = find_routine_span(text, rname)
        body = text[r_lo:r_hi]

        # 1. Replace 1D declaration with 3D allocatable declaration via AST span.
        sub_tree = parse_fortran(body)
        decl_span = None
        for decl in walk(sub_tree):
            if type(decl).__name__ != "Type_Declaration_Stmt":
                continue
            for entity in decl.items[2].items:
                if str(entity.items[0]).lower() == tmp_name.lower() and entity.items[1] is not None:
                    decl_span = decl.item.span
                    break
            if decl_span is not None:
                break
        if decl_span is None:
            raise ValueError(f"Missing 1D snapshot declaration for {tmp_name} in {rname}")
        body_offsets = _line_char_offsets(body)
        d_lo = body_offsets[decl_span[0]][0]
        d_hi = body_offsets[decl_span[1]][1]
        body = body[:d_lo] + f"    {dtype_str}, allocatable :: {tmp_name}(:,:,:)" + body[d_hi:]

        # 2. Insert 3D allocation & copy before the first executable assignment / control statement.
        sub_tree = parse_fortran(body)
        sub_exec = next(n for n in walk(sub_tree) if type(n).__name__ == "Execution_Part")
        first_exec = next((n for n in sub_exec.content if type(n).__name__ != "Comment"), None)
        if first_exec is None:
            raise ValueError(f"Empty execution body in {rname}")
        body_offsets = _line_char_offsets(body)
        insert_line = first_exec.item.span[0]
        insert_pos = body_offsets[insert_line][0]
        alloc_block = (
            f"    allocate({tmp_name}({lo_expr}:ubound({arr_name},1),{lo_expr}:ubound({arr_name},2),{lo_expr}:ubound({arr_name},3)))\n"
            f"    do k={lo_expr},ubound({arr_name},3)\n"
            f"      do j={lo_expr},ubound({arr_name},2)\n"
            f"        do i={lo_expr},ubound({arr_name},1)\n"
            f"          {tmp_name}(i,j,k)={arr_name}(i,j,k)\n"
            f"        enddo\n"
            f"      enddo\n"
            f"    enddo\n\n"
        )
        body = body[:insert_pos] + alloc_block + body[insert_pos:]

        # 3. Replace line copies tmp = U(:,j,k) / U(i,:,k) / U(i,j,:) and 1D tmp(idx) reads.
        line_patterns = [
            (rf"(?im)^[ \t]*{re.escape(tmp_name)}\s*=\s*{re.escape(arr_name)}\s*\(\s*:\s*,\s*([a-z_]\w*)\s*,\s*([a-z_]\w*)\s*\)[ \t]*$", 0),
            (rf"(?im)^[ \t]*{re.escape(tmp_name)}\s*=\s*{re.escape(arr_name)}\s*\(\s*([a-z_]\w*)\s*,\s*:\s*,\s*([a-z_]\w*)\s*\)[ \t]*$", 1),
            (rf"(?im)^[ \t]*{re.escape(tmp_name)}\s*=\s*{re.escape(arr_name)}\s*\(\s*([a-z_]\w*)\s*,\s*([a-z_]\w*)\s*,\s*:\s*\)[ \t]*$", 2),
        ]
        # Replace single-axis reads inside each branch (or by matching index variables ii/jj/kk)
        for pat_str, axis in line_patterns:
            m_line = re.search(pat_str, body)
            if m_line is None:
                raise ValueError(f"Missing line snapshot assignment for axis {axis} in {rname}")
            v1, v2 = m_line.group(1), m_line.group(2)
            body = body[: m_line.start()] + "! Input line is in the complete snapshot." + body[m_line.end() :]
            idx_name = ("ii", "jj", "kk")[axis]
            if not re.search(rf"\b{re.escape(tmp_name)}\s*\(\s*{idx_name}\s*\)", body, re.IGNORECASE):
                # Fallback: infer 1D index name from inner do-loop in the surrounding construct
                inner_do = re.search(
                    r"(?im)do\s+([a-z_]\w*)\s*=\s*[a-z_]\w*\s*-\s*(\d+)\s*,\s*[a-z_]\w*\s*\+\s*\2\b",
                    body[m_line.start() :],
                )
                if inner_do:
                    idx_name = inner_do.group(1)
            coords = [v1, v2]
            coords.insert(axis, idx_name)
            coord_str = ",".join(coords)
            read_pat = re.compile(rf"\b{re.escape(tmp_name)}\s*\(\s*{re.escape(idx_name)}\s*\)", re.IGNORECASE)
            if not read_pat.search(body):
                raise ValueError(f"Missing 1D snapshot read {tmp_name}({idx_name}) in {rname}")
            body = read_pat.sub(f"{tmp_name}({coord_str})", body)

        # 4. Express fixed symmetric stencil offsets directly: do ii = i-R, i+R -> do ii = -R, R
        stencil_pat = re.compile(
            r"(?ims)^([ \t]*)do\s+([a-z_]\w*)\s*=\s*([a-z_]\w*)\s*-\s*(\d+)\s*,\s*\3\s*\+\s*\4\b(.*?)^\s*end\s*do\b"
        )
        found_stencils = list(stencil_pat.finditer(body))
        if not found_stencils:
            raise ValueError(f"Missing original symmetric filter accumulation in {rname}")
        for m_loop in reversed(found_stencils):
            idx_v, axis_v, radius = m_loop.group(2), m_loop.group(3), m_loop.group(4)
            original_loop = m_loop.group(0)
            shifted = re.sub(
                rf"\b{re.escape(idx_v)}\b",
                f"({axis_v}+{idx_v})",
                original_loop,
                flags=re.IGNORECASE,
            )
            shifted = re.sub(
                rf"\(\s*{re.escape(axis_v)}\s*\+\s*{re.escape(idx_v)}\s*\)\s*-\s*{re.escape(axis_v)}\b",
                idx_v,
                shifted,
                flags=re.IGNORECASE,
            )
            shifted = re.sub(
                rf"(?i)do\s+\(\s*{re.escape(axis_v)}\s*\+\s*{re.escape(idx_v)}\s*\)\s*=\s*{re.escape(axis_v)}\s*-\s*{radius}\s*,\s*{re.escape(axis_v)}\s*\+\s*{radius}\b",
                f"do {idx_v} = -{radius}, {radius}",
                shifted,
                count=1,
            )
            body = body[: m_loop.start()] + shifted + body[m_loop.end() :]

        # 5. Release snapshot before end subroutine
        ending = re.search(rf"(?im)^[ \t]*end\s*subroutine(?:\s+{re.escape(rname)})?\b", body)
        if ending is None:
            raise ValueError(f"Missing procedure end in {rname}")
        release_block = (
            "\n    block\n      use elmm_resident_pressure_state, only: resident_flow\n"
            f"      call resident_flow%release_real({tmp_name})\n    end block\n"
        )
        body = body[: ending.start()] + release_block + body[ending.start() :]
        text = text[:r_lo] + body + text[r_hi:]
        edits.append({"routine": rname, "snapshot": tmp_name, "source_array": arr_name})

    return text, edits
