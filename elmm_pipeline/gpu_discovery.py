"""Conservative, name-independent lifting of complete loop domains for GPUs.

The outliner proves storage, liveness, domains and OpenMP dispatch. With an
explicit local compiler, numerical legality is checked through its public CLI;
otherwise the conservative CPU pointwise proof remains in force.
A failed parallel/tiled domain is never retried as a GPU call per CPU tile.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import copy
import fnmatch
import hashlib
from pathlib import Path
import re
from typing import Any

from . import structural as s
from .local_adapter import check as compiler_check, LocalAdapterError


@dataclass
class Directive:
    start: int
    end: int
    text: str


def directives(parsed: s.Parsed) -> list[Directive]:
    """Read only active preprocessed directives, keeping original line spans."""
    found = {}
    for node in s.walk(parsed.tree):
        if s.kind(node) != "Comment" or not re.match(r"\s*!\$omp\b", str(node), re.I):
            continue
        try:
            first, last = parsed.origin_span(node)
        except s.Unsupported:
            continue
        found[first] = Directive(first, last, re.sub(r"^\s*!\$omp\s*&?\s*", "", str(node), flags=re.I).strip().lower())
    result = []
    pending = None
    for directive in sorted(found.values(), key=lambda value: value.start):
        if pending is not None:
            pending.text = pending.text.rstrip("& ") + " " + directive.text
            pending.end = directive.end
        else:
            pending = directive
        if not pending.text.endswith("&"):
            result.append(pending)
            pending = None
    if pending is not None:
        raise s.Unsupported("unterminated OpenMP continuation")
    return result


def clause_list(text: str, prefix: str) -> dict[str, str]:
    rest = text[len(prefix):].strip()
    clauses = {}
    while rest:
        match = re.match(r"([a-z]+)\s*\(", rest)
        if not match:
            raise s.Unsupported("unsupported OpenMP clause: " + rest)
        name = match[1]
        depth, index = 1, match.end()
        first = index
        while index < len(rest) and depth:
            depth += (rest[index] == "(") - (rest[index] == ")")
            index += 1
        if depth or name in clauses:
            raise s.Unsupported("ambiguous OpenMP clauses")
        value = rest[first:index - 1].strip()
        if name not in {"private", "firstprivate", "shared", "default", "schedule", "collapse"}:
            raise s.Unsupported("unsupported OpenMP clause: " + name)
        if name in {"private", "firstprivate", "shared"} and not re.fullmatch(r"[a-z]\w*(?:\s*,\s*[a-z]\w*)*", value):
            raise s.Unsupported("unsupported OpenMP data-sharing list")
        if name == "default" and value not in {"shared", "none"}:
            raise s.Unsupported("unsupported OpenMP default data-sharing")
        if name == "schedule" and value not in {"runtime", "static", "dynamic", "guided", "auto"}:
            # A chunk expression may have side effects or depend on a private
            # value. Keep the complete native construct in that case.
            raise s.Unsupported("unsupported OpenMP schedule/chunk expression")
        if name == "collapse" and not re.fullmatch(r"[1-9][0-9]*", value):
            raise s.Unsupported("nonconstant OpenMP collapse")
        clauses[name] = value
        rest = rest[index:].strip().lstrip(",").strip()
    return clauses


class OpenMP:
    def __init__(self, parsed: s.Parsed, routine: Any):
        self.parsed = parsed
        if any(s.kind(node) == "Comment" and str(node).lstrip().startswith("!$")
               and not re.match(r"\s*!\$omp\b", str(node), re.I) for node in s.walk(routine)):
            raise s.Unsupported("unsupported non-OpenMP directive/context")
        first, last = parsed.origin_span(routine)
        self.items = [d for d in directives(parsed) if first <= d.start <= last]
        self.lines = parsed.text.splitlines()
        self.parallel = []
        stack = []
        self.unsupported = []
        for directive in self.items:
            text = directive.text
            if re.match(r"parallel\b(?!\s+do\b)", text):
                stack.append(directive)
            elif text == "end parallel":
                if not stack:
                    raise s.Unsupported("unmatched OpenMP parallel end")
                self.parallel.append((stack.pop(), directive))
            elif text != "barrier" and not re.match(r"(?:parallel do|do|end do|end parallel do|threadprivate)\b", text):
                # No inner extraction around opaque OpenMP synchronization,
                # SIMD, tasks, sections, single/master, or nested offload.
                self.unsupported.append(directive)
        if stack:
            raise s.Unsupported("unmatched OpenMP parallel scope")

    def blank_between(self, first: int, last: int) -> bool:
        return all(not line.strip() or (line.lstrip().startswith("!") and not line.lstrip().startswith("!$"))
                   for line in self.lines[first:last - 1])

    def describe(self, node: Any) -> tuple[dict, list[dict], set[str]]:
        first, last = self.parsed.origin_span(node)
        governing = next((d for d in reversed(self.items) if d.end < first
                          and self.blank_between(d.end, first)), None)
        enclosing = [(begin, end) for begin, end in self.parallel if begin.start < first < end.start]
        if self.unsupported:
            # This is intentionally routine-conservative: parsing arbitrary
            # nested directive grammars is outside this lifting contract.
            raise s.Unsupported("unsupported OpenMP directive/context: " + self.unsupported[0].text)
        if len(enclosing) > 1:
            raise s.Unsupported("nested OpenMP teams require a dispatch proof")
        clauses = []
        if enclosing:
            clauses.append(clause_list(enclosing[0][0].text, "parallel"))
        dispatch = "serial"
        consumed = []
        if governing and re.match(r"(?:parallel do|do)\b", governing.text):
            combined = governing.text.startswith("parallel do")
            if combined and enclosing:
                raise s.Unsupported("nested combined OpenMP parallel do")
            if not combined and not enclosing:
                raise s.Unsupported("orphan OpenMP do has no proved enclosing parallel team")
            prefix = "parallel do" if combined else "do"
            clauses.append(clause_list(governing.text, prefix))
            consumed.append(governing)
            if not combined:
                dispatch = "collective_single"
            expected = "end " + prefix
            following = next((d for d in self.items if d.start > last and self.blank_between(last, d.start)), None)
            if following and following.text.startswith(expected):
                if following.text != expected and not (not combined and following.text == expected + " nowait"):
                    raise s.Unsupported("unsupported OpenMP end clause")
                consumed.append(following)
            start = governing.start
            end = consumed[-1].end if len(consumed) > 1 else last
        else:
            if enclosing:
                raise s.Unsupported("loop inside parallel team lacks a proved worksharing dispatch")
            start, end = first, last
        private = set()
        for clause in clauses:
            for key in ("private", "firstprivate"):
                private.update(value.strip() for value in clause.get(key, "").split(",") if value.strip())
        return {"dispatch": dispatch, "start_line": start, "end_line": end,
                "team_start_line": enclosing[0][0].start if enclosing else None,
                "barriers": "before_and_after" if dispatch == "collective_single" else None,
                "consumed_directives": [{"start_line": d.start, "end_line": d.end, "text": d.text} for d in consumed]}, clauses, private


def equal(left: Any, right: Any, scope: s.Scope) -> bool:
    if re.sub(r"\s+", "", str(left)).lower() == re.sub(r"\s+", "", str(right)).lower():
        return True
    try:
        return s.affine(left, scope) == s.affine(right, scope)
    except s.Unsupported:
        return False


def live_after(routine, end, written, parsed, region_node):
    """CPU liveness with a narrowly proved later-loop definition refinement.

    A later conditional loop need not execute, so its iterator cannot kill a
    pending value globally. Inside that loop, however, the iterator has been
    defined on every path reaching its body. Its own limit expressions still
    read the old value. Reads after a possibly empty loop remain live as well.
    """
    pending = set(written)
    active = s.ancestors(region_node)
    for statement in s.walk(routine):
        if not pending:
            break
        if not getattr(getattr(statement, "item", None), "span", None):
            continue
        start, _ = parsed.origin_span(statement)
        if start <= end:
            continue
        kind = s.kind(statement)
        if kind == "Comment":
            if re.match(r"\s*!\$omp\b", str(statement), re.I):
                references = set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", str(statement).lower()))
                if references & pending:
                    return references & pending
            continue
        if kind == "Assignment_Stmt":
            lhs, _, rhs = statement.items
            used = s.names(rhs) | (s.names(lhs) if s.kind(lhs) != "Name" else set())
        elif kind == "Nonlabel_Do_Stmt":
            control = statement.items[1]
            used = (set().union(*(s.names(bound) for bound in control.items[1][1]))
                    if control is not None and control.items[1] is not None else s.names(statement))
        else:
            used = s.names(statement)
        parent = getattr(statement, "parent", None)
        while parent is not None and parent is not routine:
            if s.kind(parent) == "Block_Nonlabel_Do_Construct":
                first, _ = parsed.origin_span(parent)
                if first > end and not (kind == "Nonlabel_Do_Stmt" and parent is getattr(statement, "parent", None)):
                    try:
                        iterator, _ = s.loop_info(parent)
                        used.discard(iterator.lower())
                    except s.Unsupported:
                        pass
            parent = getattr(parent, "parent", None)
        if used & pending:
            return used & pending
        if kind == "Assignment_Stmt" and s.kind(statement.items[0]) == "Name" and s.ancestors(statement) <= active:
            pending.discard(str(statement.items[0]).lower())
        elif kind == "Nonlabel_Do_Stmt":
            control = statement.items[1]
            if control is not None and control.items[1] is not None:
                parents = s.ancestors(statement)
                parents.discard(id(getattr(statement, "parent", None)))
                if parents <= active:
                    pending.discard(str(control.items[1][0]).lower())
    return set()


def lifted_nest(node: Any, scope: s.Scope, *, structured: bool = False) -> tuple[s.Nest, list[dict], set[str], int]:
    # Reuse rectangular's ancestor checks even when tile strides prevent its
    # ordinary domain proof. Checking a leaf preserves the same ancestors.
    current = node
    chain = []
    while True:
        iterator, bounds = s.loop_info(current)
        chain.append((iterator, bounds))
        children = s.code_children(current)[1:-1]
        if len(children) == 1 and s.kind(children[0]) == "Block_Nonlabel_Do_Construct":
            current = children[0]
        else:
            if not children or (not structured and any(s.kind(child) != "Assignment_Stmt" for child in children)):
                raise s.Unsupported("complete GPU loop nest is not a perfect straight-line assignment nest")
            assignments = children
            break
    # fparser attaches a preceding directive to the following DO construct.
    # Only directives strictly within its header/end boundaries are internal.
    leaf = copy(current)
    leaf_start = s.span(s.code_children(current)[0])[0]
    leaf.content = [child for child in s.content(current)
                    if s.kind(child) != "Comment" or s.span(child)[0] >= leaf_start]
    if structured:
        s.scope_checks(node)
    else:
        s.rectangular(leaf, scope, set())
    loop_start = s.span(s.code_children(node)[0])[0]
    loop_end = s.span(s.code_children(node)[-1])[1]
    if any(s.kind(n) == "Comment" and str(n).lstrip().startswith("!$")
           and loop_start <= s.span(n)[0] <= loop_end for n in s.walk(node)):
        raise s.Unsupported("directive inside complete GPU loop nest")
    all_names = {name.lower() for name, _ in chain}
    if len(all_names) != len(chain):
        raise s.Unsupported("reused induction variable in GPU loop nest")
    pairs = {}
    point_to_tile = {}
    for position, (point, bounds) in enumerate(chain):
        if len(bounds) > 2:
            try:
                if s.integer_constant(bounds[2], scope) != 1:
                    continue
            except s.Unsupported:
                continue
        if s.kind(bounds[0]) != "Name" or s.kind(bounds[1]) != "Intrinsic_Function_Reference":
            continue
        function, arguments = bounds[1].items
        if str(function).lower() != "min" or len(arguments.items) != 2 or not scope.intrinsic_unshadowed("min"):
            continue
        tile = str(bounds[0]).lower()
        outer = next(((name, values) for name, values in chain[:position] if name.lower() == tile), None)
        if outer is None or len(outer[1]) != 3:
            continue
        name, (lower, upper, step) = outer
        endpoint = s.F.Level_2_Expr(f"{name} + ({step}) - 1")
        a, b = arguments.items
        if not ((equal(a, endpoint, scope) and equal(b, upper, scope)) or
                (equal(b, endpoint, scope) and equal(a, upper, scope))):
            continue
        if tile in pairs:
            raise s.Unsupported("tile iterator controls multiple point loops")
        pairs[tile] = {"tile": name, "point": point, "lower": str(lower), "upper": str(upper), "step": str(step)}
        point_to_tile[point.lower()] = outer
    loops = []
    for name, bounds in chain:
        if name.lower() in pairs:
            continue
        if name.lower() in point_to_tile:
            bounds = point_to_tile[name.lower()][1][:2]
        elif len(bounds) > 2 and s.integer_constant(bounds[2], scope) != 1:
            raise s.Unsupported("GPU nonunit stride is not a complete strip-mined domain")
        if any(s.names(bound) & all_names for bound in bounds[:2]):
            raise s.Unsupported("GPU point domain depends on an induction variable")
        loops.append((name, bounds[:2]))
    tile_names = set(pairs)
    if not loops:
        raise s.Unsupported("GPU domain has no point loops")
    if any(s.names(assignment) & tile_names for assignment in assignments):
        raise s.Unsupported("tile iterator is used by point arithmetic")
    # Outer tile bounds and strides must be invariant across every lifted loop.
    for name, bounds in chain:
        if name.lower() in tile_names:
            if any(s.names(value) & all_names for value in bounds):
                raise s.Unsupported("tile limits/stride depend on induction variables")
    return s.Nest(node, loops, assignments), list(pairs.values()), tile_names, len(chain)


def make_gpu_region(nodes, scope, module, dummies, routine, parsed, insertion, omp, ordinal, *, structured=False):
    """Prove one contiguous group without weakening the per-nest contract."""
    descriptions = [omp.describe(node) for node in nodes]
    contexts = [description[0] for description in descriptions]
    context = dict(contexts[0])
    parent = getattr(nodes[0], "parent", None)
    for index in range(1, len(nodes)):
        previous, current = contexts[index - 1], contexts[index]
        if getattr(nodes[index], "parent", None) is not parent:
            raise s.Unsupported("GPU groups cannot cross a control-flow boundary")
        if any(current[key] != context[key] for key in ("dispatch", "team_start_line")):
            raise s.Unsupported("GPU groups require the same dispatch and parallel team")
        if not omp.blank_between(previous["end_line"], current["start_line"]):
            raise s.Unsupported("GPU groups cannot cross statements or synchronization directives")
    nests, tiles, tile_names, original_ranks = [], [], set(), []
    private = set()
    for node, (_, clauses, own_private) in zip(nodes, descriptions):
        try:
            nest, own_tiles, own_tile_names, original_rank = lifted_nest(node, scope, structured=structured)
        except s.Unsupported:
            if not structured:
                raise
            # Retain imperfect/nonrectangular loops for the selected compiler.
            # No native directive or enclosing scope contract may disappear.
            s.scope_checks(node)
            first = s.span(s.code_children(node)[0])[0]
            last = s.span(s.code_children(node)[-1])[1]
            if any(s.kind(n) == "Comment" and str(n).lstrip().startswith("!$")
                   and first <= s.span(n)[0] <= last for n in s.walk(node)):
                raise s.Unsupported("directive inside complete GPU loop nest")
            current, original_rank = node, 0
            while s.kind(current) == "Block_Nonlabel_Do_Construct":
                s.loop_info(current)
                original_rank += 1
                children = s.code_children(current)[1:-1]
                current = children[0] if len(children) == 1 else None
            nest, own_tiles, own_tile_names = s.Nest(node, [], [node]), [], set()
        for clause in clauses:
            if int(clause.get("collapse", "1")) > original_rank:
                raise s.Unsupported("OpenMP collapse exceeds complete loop nest")
        nests.append(nest)
        tiles.extend(own_tiles)
        tile_names.update(own_tile_names)
        original_ranks.append(original_rank)
        private.update(own_private)
    if structured:
        region = s.make_block_region(nests, scope, module, dummies, routine, parsed, insertion, ordinal,
                                     liveness=live_after)
    else:
        region = s.make_region(nests, scope, module, dummies, routine, parsed, insertion, set(), ordinal,
                               liveness=live_after)
    locals_ = {name.lower() for nest in nests for name, _ in nest.loops} | tile_names
    for nest in nests:
        for statement in s.walk(nest.assignments):
            if s.kind(statement) == "Assignment_Stmt" and s.kind(statement.items[0]) == "Name":
                locals_.add(str(statement.items[0]).lower())
            elif s.kind(statement) == "Block_Nonlabel_Do_Construct":
                locals_.add(s.loop_info(statement)[0].lower())
    captured = {symbol["name"].lower() for symbol in region["symbols"]}
    for nest in nests:
        for _, bounds in nest.loops:
            bound_names = set().union(*(s.names(value) for value in bounds))
            # Retained bounds execute in place; removed tile variables have no
            # corresponding native final value in the outlined block.
            if structured and bound_names & tile_names:
                raise s.Unsupported("lifted tile iterator is read by another grouped loop bound")
            if not structured:
                captured.update(bound_names)
    if structured and any(s.names(statement) & tile_names for nest in nests for statement in nest.assignments):
        raise s.Unsupported("lifted tile iterator is read by another grouped loop body")
    written = {symbol["name"].lower() for symbol in region["symbols"] if symbol["intent"] == "inout"} | locals_
    for tile in tiles:
        symbol = scope.resolve(tile["tile"])
        if (symbol is None or symbol.scope is not scope or symbol.rank or s.type_spec(symbol) != "integer"
                or symbol.name.lower() in dummies or "save" in symbol.attributes or symbol.initializer is not None):
            raise s.Unsupported("tile iterator is not an ephemeral local default integer")
        s.symbol_checks(symbol, scope, dummies)
        for value in (tile["lower"], tile["upper"], tile["step"]):
            expression = s.F.Level_2_Expr(value)
            s.safe_bound(expression, scope, written, dummies)
            captured.update(s.names(expression))
    # Bounds are evaluated once at the grouped call site. In particular, a
    # lifted-away tile iterator still changes in the native first nest and
    # cannot supply a later nest's eagerly captured limit or tile guard.
    if captured & locals_:
        raise s.Unsupported("GPU local value is captured by another grouped nest")
    _, last = parsed.origin_span(nodes[-1])
    if tile_names and (live := live_after(routine, last, tile_names, parsed, nodes[-1])):
        raise s.Unsupported("tile iterator remains live after GPU region: " + ", ".join(sorted(live)))
    if captured & private - locals_:
        raise s.Unsupported("GPU capture is OpenMP private/firstprivate: " + ", ".join(sorted(captured & private - locals_)))
    # Imported symbols may have been lazily parsed while make_region resolved
    # captures. Check THREADPRIVATE on their original declaration names too.
    threadprivate = set()
    for source in list(scope.project.files.values()):
        if isinstance(source, s.Parsed):
            for directive in directives(source):
                if directive.text.startswith("threadprivate"):
                    threadprivate.update(re.findall(r"\b[a-z]\w*\b", directive.text)[1:])
    for name in captured - locals_:
        symbol = scope.resolve(name)
        if name in threadprivate or (symbol and symbol.name.lower() in threadprivate):
            raise s.Unsupported("GPU capture is OpenMP threadprivate: " + name)
    first, last = contexts[0]["start_line"], contexts[-1]["end_line"]
    original = "".join(parsed.text.splitlines(keepends=True)[first - 1:last])
    if any(line.lstrip().startswith("#") for line in original.splitlines()):
        raise s.Unsupported("preprocessor directive crosses complete GPU replacement span")
    context.pop("start_line")
    context.pop("end_line")
    point_ranks = [len(nest.loops) for nest in nests]
    guards = list(dict.fromkeys(f"({tile['step']}) > 0" for tile in tiles))
    context.update({"kind": "strip_mined" if tiles else ("structured" if structured else "rectangular"), "tiles": tiles,
                    "allocated_arrays": region.pop("allocated_arrays", []),
                    "guard": " .and. ".join(guards) or None,
                    "native_fallback": True, "expected_omp_level": 1 if context["dispatch"] == "collective_single" else 0,
                    "original_loop_rank": original_ranks[0] if len(nodes) == 1 else None,
                    "point_loop_rank": point_ranks[0] if len(nodes) == 1 else None,
                    "original_loop_ranks": original_ranks, "point_loop_ranks": point_ranks,
                    "grouped_nests": len(nodes),
                    "member_spans": [{"start_line": item["start_line"], "end_line": item["end_line"]} for item in contexts],
                    "consumed_directives": [directive for item in contexts for directive in item["consumed_directives"]]})
    region.update(start_line=first, end_line=last, original_text=original,
                  original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                  source_span_sha256=hashlib.sha256(original.encode()).hexdigest(), gpu_scope=context)
    region["context"] = {"nested_parallel": False, "gpu_dispatch": context["dispatch"]}
    return region


def discover_sources(sources: list[str], project_root: str, output_dir: str,
                     defines: list[str] | None = None, select: list[str] | None = None,
                     compiler_root: str | None = None, compiler_python: str | None = None) -> dict:
    if bool(compiler_root) != bool(compiler_python):
        raise ValueError("GPU compiler checking requires both compiler_root and compiler_python")
    structured = compiler_root is not None
    project = s.Project(Path(project_root), list(defines or ()))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    regions, rejected = [], []
    for source in sources:
        path = Path(source).resolve()
        try:
            parsed = project.parse(path)
        except Exception as error:
            rejected.append({"file": str(path), "routine": None, "reason": f"source parse/preprocessing: {error}"})
            continue
        for routine in (node for node in s.walk(parsed.tree) if s.kind(node) == "Subroutine_Subprogram"):
            statement = next(node for node in s.content(routine) if s.kind(node) == "Subroutine_Stmt")
            name = str(statement.items[1])
            if select and not any(fnmatch.fnmatchcase(name.lower(), pattern.lower()) for pattern in select):
                continue
            try:
                scope, module, dummies, insertion = s.original_scope(parsed, routine, project)
                omp = OpenMP(parsed, routine)
            except Exception as error:
                rejected.append({"file": str(path), "routine": name, "reason": str(error)})
                continue
            diagnostics = set()

            def scan(parent):
                children = s.content(parent)
                index = 0
                while index < len(children):
                    child = children[index]
                    call_group = structured and s.kind(child) == "Call_Stmt"
                    if s.kind(child) != "Block_Nonlabel_Do_Construct" and not call_group:
                        if s.kind(child) not in {"Subroutine_Subprogram", "Internal_Subprogram_Part"}:
                            scan(child)
                        index += 1
                        continue
                    # Only siblings of the same kind, separated by comments,
                    # are proposed. The region builder checks exact directive,
                    # source and control-flow boundaries before acceptance.
                    candidates, ends = [child], [index + 1]
                    after = index + 1
                    while after < len(children):
                        next_node = children[after]
                        if s.kind(next_node) == "Comment":
                            after += 1
                            continue
                        if s.kind(next_node) != s.kind(child):
                            break
                        candidates.append(next_node)
                        ends.append(after + 1)
                        after += 1
                    # Try the longest prefix first. A rejected union must not
                    # discard independently eligible original regions.
                    for count in range(len(candidates), 0, -1):
                        try:
                            if call_group:
                                from .gpu_calls import make_call_region
                                region = make_call_region(candidates[:count], scope, module, dummies, routine,
                                                          parsed, insertion, omp, len(regions))
                            else:
                                region = make_gpu_region(candidates[:count], scope, module, dummies, routine,
                                                         parsed, insertion, omp, len(regions), structured=structured)
                            input_path = destination / ("region_" + region["id"] + ".f90")
                            input_path.write_text(region["module_source"])
                            if structured:
                                report = compiler_check(input_path, Path(compiler_root), Path(compiler_python),
                                                        region["entry"], destination / ("check_" + region["id"]))
                                if not report["supported"]:
                                    raise s.Unsupported("selected compiler: " + str(report.get("reason") or "unsupported region"))
                                region["compiler_check"] = report
                            region["input_path"] = str(input_path)
                            regions.append(region)
                            index = ends[count - 1]
                            break
                        except LocalAdapterError:
                            raise
                        except Exception as error:
                            if call_group and count == 2:
                                diagnostics.add(str(error))
                            if count == 1:
                                if not call_group:
                                    diagnostics.add(str(error))
                                # A complete failed loop stays native. Never
                                # launch once per unproved tile or team lane.
                                index += 1

            for execution in (node for node in s.content(routine) if s.kind(node) == "Execution_Part"):
                scan(execution)
            rejected.extend({"file": str(path), "routine": name, "reason": reason} for reason in sorted(diagnostics))
    return {"schema": 1, "backend": "gpu", "project_root": str(project.root), "defines": project.defines,
            "regions": regions, "rejected": rejected,
            "selection": {"sources": sources, "routine_patterns": list(select or ()), "markers_required": False},
            "parser": "fparser Fortran 2008; gfortran preprocessing with original source mapping",
            "policy": ("adjacent complete loops or direct leaf calls; capture/liveness/OpenMP proofs and selected compiler numerical legality"
                       if structured else "adjacent complete rectangular or proved strip-mined domains; original CPU pointwise and capture proofs")}
