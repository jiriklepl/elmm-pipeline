"""Name-independent, conservative outlining of rectangular Fortran loops.

This module runs only inside the explicitly configured fparser environment.
It proves a deliberately small pointwise contract and leaves all surrounding
native statements, tiling, and OpenMP scopes in place. Unsupported constructs
are reported, never guessed from names or silently substituted.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import fnmatch
import hashlib
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from fparser.common.readfortran import FortranStringReader
from fparser.two import Fortran2003 as F
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk


class Unsupported(ValueError):
    pass


def kind(node: Any) -> str:
    return type(node).__name__


def names(node: Any) -> set[str]:
    return {str(n).lower() for n in walk(node) if kind(n) == "Name"}


def content(node: Any) -> list:
    return list(getattr(node, "content", ()))


def span(node: Any) -> tuple[int, int]:
    spans = [n.item.span for n in walk(node) if getattr(getattr(n, "item", None), "span", None)]
    if not spans:
        raise Unsupported("missing source location")
    return min(s[0] for s in spans), max(s[1] for s in spans)


def code_children(node: Any) -> list:
    return [n for n in content(node) if kind(n) != "Comment"]


@dataclass
class Symbol:
    name: str
    type_name: str
    kind_expr: Any
    shape: Any
    attributes: set[str]
    intent: str | None
    initializer: Any
    scope: "Scope"

    @property
    def rank(self) -> int:
        return len(self.shape.items) if self.shape is not None else 0


@dataclass
class Scope:
    name: str
    project: "Project"
    parent: "Scope | None" = None
    symbols: dict[str, Symbol] = field(default_factory=dict)
    imports: list[tuple[str, dict[str, str] | None]] = field(default_factory=list)
    storage_aliases: bool = False
    default_public: bool = True
    accessibility: dict[str, bool] = field(default_factory=dict)

    def exports(self, name: str) -> bool:
        name = name.lower()
        if name in self.accessibility:
            return self.accessibility[name]
        symbol = self.symbols.get(name)
        if symbol is not None:
            if "private" in symbol.attributes:
                return False
            if "public" in symbol.attributes:
                return True
        return self.default_public

    def intrinsic_unshadowed(self, name: str, seen: frozenset = frozenset()) -> bool:
        """Unknown wildcard imports cannot establish intrinsic binding."""
        marker = (id(self), name)
        if marker in seen:
            return True
        seen = seen | {marker}
        if name in self.symbols:
            return "intrinsic" in self.symbols[name].attributes
        for module_name, aliases in self.imports:
            if aliases is not None and name not in aliases:
                continue
            imported = aliases.get(name, name) if aliases is not None else name
            module = self.project.module(module_name)
            if module is not None and not module.exports(imported):
                continue
            if module is None or not module.intrinsic_unshadowed(imported, seen):
                return False
        return self.parent.intrinsic_unshadowed(name, seen) if self.parent is not None else True

    def resolve(self, name: str, seen: frozenset = frozenset()) -> Symbol | None:
        name = name.lower()
        marker = (id(self), name)
        if marker in seen:
            return None
        seen = seen | {marker}
        if name in self.symbols:
            return self.symbols[name]
        matches = []
        for module_name, aliases in self.imports:
            if aliases is not None and name not in aliases:
                continue
            imported = aliases.get(name, name) if aliases is not None else name
            module = self.project.module(module_name)
            if module is not None and module.exports(imported):
                symbol = module.resolve(imported, seen)
                if symbol is not None and all(symbol is not old for old in matches):
                    matches.append(symbol)
        if len(matches) > 1:
            raise Unsupported(f"ambiguous imported symbol {name}")
        if matches:
            return matches[0]
        return self.parent.resolve(name, seen) if self.parent is not None else None

    def populate(self, specification: Any) -> None:
        for node in content(specification):
            if kind(node) == "Access_Stmt":
                access, selected = node.items
                public = str(access).lower() == "public"
                if selected is None:
                    self.default_public = public
                else:
                    self.accessibility.update((name, public) for name in names(selected))
            if kind(node) == "Use_Stmt":
                _, _, module, only, values = node.items
                aliases = None
                if only is not None and "ONLY" in str(only).upper():
                    aliases = {}
                    for value in values.items if values is not None else ():
                        if kind(value) == "Rename":
                            aliases[str(value.items[1]).lower()] = str(value.items[2]).lower()
                        elif kind(value) == "Name":
                            aliases[str(value).lower()] = str(value).lower()
                elif values is not None:
                    # Wildcard imports with renaming require excluding the
                    # original name; fail resolution rather than guessing.
                    aliases = {str(v.items[1]).lower(): str(v.items[2]).lower()
                               for v in values.items if kind(v) == "Rename"}
                self.imports.append((str(module).lower(), aliases))
            elif kind(node) == "Type_Declaration_Stmt":
                type_spec, attributes, entities = node.items
                if kind(type_spec) != "Intrinsic_Type_Spec":
                    type_name, selector = "unsupported", None
                else:
                    type_name, selector = type_spec.items
                    type_name = str(type_name).lower()
                kind_expr = selector.items[1] if selector is not None and kind(selector) == "Kind_Selector" else None
                attrs = set()
                intent = None
                shape = None
                for attr in attributes.items if attributes is not None else ():
                    if kind(attr) == "Dimension_Attr_Spec":
                        shape = attr.items[1]
                    elif kind(attr) == "Intent_Attr_Spec":
                        intent = str(attr.items[1]).lower()
                    else:
                        attrs.add(str(attr).lower())
                for entity in entities.items:
                    name, own_shape, _, initializer = entity.items
                    key = str(name).lower()
                    value = initializer.items[1] if initializer is not None else None
                    self.symbols[key] = Symbol(str(name), type_name, kind_expr, own_shape or shape,
                                               attrs, intent, value, self)
        # Separate attribute statements carry the same semantic restrictions as
        # attributes attached to a type declaration. EQUIVALENCE/COMMON require
        # a storage-association analysis beyond the pointwise name proof.
        for node in content(specification):
            if kind(node) in {"Equivalence_Stmt", "Common_Stmt", "Bind_Stmt", "Namelist_Stmt"}:
                self.storage_aliases = True
            attributes = {"Pointer_Stmt": "pointer", "Volatile_Stmt": "volatile",
                          "Asynchronous_Stmt": "asynchronous", "Allocatable_Stmt": "allocatable",
                          "Save_Stmt": "save", "Optional_Stmt": "optional", "Contiguous_Stmt": "contiguous",
                          "Target_Stmt": "target"}
            if kind(node) in attributes:
                selected = names(node) or set(self.symbols)
                for name in selected:
                    if name in self.symbols:
                        self.symbols[name].attributes = self.symbols[name].attributes | {attributes[kind(node)]}
            if kind(node) == "Data_Stmt":
                # DATA initialization implies SAVE even when the declaration
                # itself has no initializer or explicit SAVE attribute.
                for name in names(node):
                    if name in self.symbols:
                        self.symbols[name].attributes = self.symbols[name].attributes | {"save"}
            if kind(node) == "External_Stmt":
                for name in names(node):
                    if name in self.symbols:
                        self.symbols[name].attributes = self.symbols[name].attributes | {"external"}
                    else:
                        self.symbols[name] = Symbol(name, "procedure", None, None, {"external"}, None, None, self)
            if kind(node) == "Interface_Block":
                header = next((n for n in content(node) if kind(n) == "Interface_Stmt"), None)
                if header is not None:
                    for name in names(header):
                        self.symbols.setdefault(name, Symbol(name, "procedure", None, None, set(), None, None, self))


@dataclass
class Parsed:
    path: Path
    tree: Any
    mapping: list[tuple[Path | None, int]]
    text: str
    modules: dict[int, Scope]
    preprocessed_text: str = ""
    preprocess_command: list[str] = field(default_factory=list)

    def origin_span(self, node: Any) -> tuple[int, int]:
        if kind(node) == "Block_Nonlabel_Do_Construct":
            # fparser may attach the trailing inline comment of an enclosing
            # DO header to its first child loop. Descendant-comment extrema
            # would then include (and replace) the enclosing DO itself.
            headers = [child for child in content(node) if kind(child) == "Nonlabel_Do_Stmt"]
            endings = [child for child in content(node) if kind(child) == "End_Do_Stmt"]
            if len(headers) != 1 or len(endings) != 1:
                raise Unsupported("loop lacks an unambiguous header/end source boundary")
            start = span(headers[0])[0]
            end = span(endings[0])[1]
        else:
            start, end = span(node)
        lines = self.mapping[start - 1:end]
        if not lines or any(path != self.path for path, _ in lines):
            raise Unsupported("region crosses an included file or lacks an original source span")
        return lines[0][1], lines[-1][1]


class Project:
    def __init__(self, root: Path, defines: list[str]):
        self.root = root.resolve()
        self.defines = defines
        self.parser = ParserFactory().create(std="f2008")
        self.files: dict[Path, Parsed | Exception] = {}
        self.modules: dict[str, Scope] = {}
        self.procedures: dict[int, tuple[Parsed, Any]] = {}
        self.module_files: dict[str, list[Path]] = {}
        for path in sorted(self.root.rglob("*")):
            if path.suffix.lower() not in {".f90", ".f95", ".f03", ".f08"}:
                continue
            for match in re.finditer(r"(?im)^\s*module\s+(?!procedure\b|function\b|subroutine\b)(\w+)", path.read_text(errors="replace")):
                self.module_files.setdefault(match[1].lower(), []).append(path.resolve())
        self._intrinsics()

    def _intrinsics(self) -> None:
        for module_name, constants in {
            "iso_fortran_env": {"real32": 4, "real64": 8, "int32": 4, "int64": 8},
            "iso_c_binding": {"c_float": 4, "c_double": 8, "c_int": 4, "c_int32_t": 4, "c_int64_t": 8},
            "omp_lib": {}, "omp_lib_kinds": {},
        }.items():
            scope = Scope(module_name, self)
            for name, value in constants.items():
                scope.symbols[name] = Symbol(name, "integer", None, None, {"parameter"}, None,
                                              F.Int_Literal_Constant(str(value)), scope)
            self.modules[module_name] = scope

    def module(self, name: str) -> Scope | None:
        paths = list(dict.fromkeys(self.module_files.get(name, ())))
        # Alternate source implementations may declare the same module (e.g.
        # serial and distributed builds). Without an explicit build-source
        # selection they remain ambiguous even if one happened to be parsed
        # first during a whole-tree scan.
        if len(paths) > 1:
            return None
        if name in self.modules:
            return self.modules[name]
        if len(paths) != 1:
            return None
        try:
            self.parse(paths[0])
        except Exception:
            return None
        return self.modules.get(name)

    def parse(self, path: Path) -> Parsed:
        path = path.resolve()
        if path in self.files:
            previous = self.files[path]
            if isinstance(previous, Exception):
                raise previous
            return previous
        try:
            command = [*shlex.split(os.environ.get("FC", "gfortran")), "-cpp", "-E", "-O3", "-ffree-line-length-none", "-fopenmp",
                       *["-D" + value for value in self.defines], "-I" + str(path.parent), str(path)]
            result = subprocess.run(command, text=True, capture_output=True, check=False)
            if result.returncode:
                raise Unsupported("preprocessing failed: " + result.stderr.strip()[-700:])
            lines = []
            mapping = []
            current_path: Path | None = path
            current_line = 1
            for line in result.stdout.splitlines():
                directive = re.match(r'^#\s+(\d+)\s+"([^"]+)"', line)
                if directive:
                    current_line = int(directive[1])
                    current_path = None if directive[2].startswith("<") else Path(directive[2]).resolve()
                    continue
                if line.startswith("#"):
                    continue
                lines.append(line)
                mapping.append((current_path, current_line))
                current_line += 1
            reader = FortranStringReader("\n".join(lines), ignore_comments=False)
            tree = self.parser(reader)
            parsed = Parsed(path, tree, mapping, path.read_text(), {}, "\n".join(lines)+"\n",command)
            self.files[path] = parsed
            # Register every module before parsing specifications to permit
            # mutual imports; symbol resolution itself is lazy and cycle-safe.
            for module in (n for n in walk(tree) if kind(n) == "Module"):
                statement = next(n for n in content(module) if kind(n) == "Module_Stmt")
                scope = Scope(str(statement.items[1]), self)
                parsed.modules[id(module)] = scope
                self.modules[scope.name.lower()] = scope
            for module in (n for n in walk(tree) if kind(n) == "Module"):
                scope = parsed.modules[id(module)]
                for specification in (n for n in content(module) if kind(n) == "Specification_Part"):
                    scope.populate(specification)
                for part in (n for n in content(module) if kind(n) == "Module_Subprogram_Part"):
                    for procedure in content(part):
                        for header in (n for n in content(procedure) if kind(n) in {"Subroutine_Stmt", "Function_Stmt"}):
                            name = str(header.items[1]).lower()
                            symbol = scope.symbols.setdefault(name, Symbol(name, "procedure", None, None, set(), None, None, scope))
                            if kind(header) == "Subroutine_Stmt":
                                self.procedures[id(symbol)] = (parsed, procedure)
            return parsed
        except Exception as error:
            self.files[path] = error
            raise


def integer_constant(node: Any, scope: Scope, active: frozenset[str] = frozenset()) -> int:
    if node is None:
        raise Unsupported("unresolved integer constant")
    if kind(node) == "Int_Literal_Constant":
        return int(node.items[0])
    if kind(node) == "Name":
        name = str(node).lower()
        symbol = scope.resolve(name)
        if name in active or symbol is None or "parameter" not in symbol.attributes:
            raise Unsupported(f"unresolved integer/kind constant {name}")
        return integer_constant(symbol.initializer, symbol.scope, active | {name})
    items = getattr(node, "items", ())
    if kind(node) == "Intrinsic_Function_Reference":
        name = str(items[0]).lower()
        args = items[1].items
        if name == "kind" and len(args) == 1:
            value = args[0]
            if kind(value) == "Real_Literal_Constant":
                return literal_kind(value, scope)
            if kind(value) == "Int_Literal_Constant":
                selector = value.items[1]
                if selector is None:
                    return 4
                try:
                    return int(str(selector))
                except ValueError:
                    return integer_constant(F.Name(str(selector)), scope, active)
        if name == "selected_real_kind":
            values = [integer_constant(arg.items[1] if kind(arg) == "Actual_Arg_Spec" else arg, scope, active) for arg in args]
            if values and values[0] <= 6 and (len(values) < 2 or values[1] <= 37):
                return 4
            if values and values[0] <= 15 and (len(values) < 2 or values[1] <= 307):
                return 8
    if len(items) == 3 and str(items[0]) == "(" and str(items[-1]) == ")":
        return integer_constant(items[1], scope, active)
    if len(items) == 2 and str(items[0]) in {"+", "-"}:
        value = integer_constant(items[1], scope, active)
        return value if str(items[0]) == "+" else -value
    if len(items) == 3 and str(items[1]) in {"+", "-", "*", "/"}:
        left = integer_constant(items[0], scope, active)
        right = integer_constant(items[2], scope, active)
        operator = str(items[1])
        if operator == "+": return left + right
        if operator == "-": return left - right
        if operator == "*": return left * right
        if operator == "/" and right: return int(left / right)
    raise Unsupported(f"unsupported integer/kind constant {node}")


def literal_kind(node: Any, scope: Scope) -> int:
    value, selector = node.items
    if selector is not None:
        try:
            result = int(str(selector))
        except ValueError:
            result = integer_constant(F.Name(str(selector)), scope)
    else:
        result = 8 if "d" in str(value).lower() else 4
    if result not in {4, 8}:
        raise Unsupported(f"unsupported real kind {result}")
    return result


def type_spec(symbol: Symbol, *, logical: bool = False) -> str:
    if symbol.type_name in {"real", "double precision"}:
        value = integer_constant(symbol.kind_expr, symbol.scope) if symbol.kind_expr is not None else (8 if symbol.type_name == "double precision" else 4)
        if value not in {4, 8}:
            raise Unsupported(f"unsupported real kind for {symbol.name}: {value}")
        return f"real(kind={value})"
    if symbol.type_name == "integer":
        if symbol.kind_expr is not None and integer_constant(symbol.kind_expr, symbol.scope) != 4:
            raise Unsupported(f"non-default integer kind for {symbol.name}")
        return "integer"
    if logical and symbol.type_name == "logical" and not symbol.rank:
        if symbol.kind_expr is not None and integer_constant(symbol.kind_expr, symbol.scope) != 4:
            raise Unsupported(f"non-default logical kind for {symbol.name}")
        return "logical"
    raise Unsupported(f"unsupported type for {symbol.name}: {symbol.type_name}")


def array_lowers(symbol: Symbol) -> list[str]:
    result = []
    for dimension in symbol.shape.items:
        if kind(dimension) in {"Assumed_Shape_Spec", "Explicit_Shape_Spec", "Deferred_Shape_Spec"}:
            # Declaration expressions were evaluated on procedure entry (or
            # allocation); variables in them may since have changed. The
            # descriptor is the authority at the replacement call site.
            result.append(f"lbound({symbol.name}, {len(result) + 1})")
        else:
            raise Unsupported(f"unsupported array shape for {symbol.name}")
    return result


def loop_info(node: Any) -> tuple[str, list[Any]]:
    children = code_children(node)
    if not children or kind(children[0]) != "Nonlabel_Do_Stmt":
        raise Unsupported("only unlabelled counted DO loops are supported")
    control = children[0].items[1]
    if control is None or control.items[0] is not None or control.items[1] is None:
        raise Unsupported("only counted DO loops are supported")
    name, limits = control.items[1]
    return str(name), list(limits)


def affine(node: Any, scope: Scope) -> tuple[dict[str, int], int]:
    """A deliberately narrow affine recognizer, without symbolic guesses."""
    if kind(node) == "Name":
        symbol = scope.resolve(str(node))
        if symbol is None or symbol.rank or symbol.type_name != "integer":
            raise Unsupported(f"non-scalar/integer array index {node}")
        if "parameter" in symbol.attributes:
            return {}, integer_constant(node, scope)
        return {str(node).lower(): 1}, 0
    try:
        return {}, integer_constant(node, scope)
    except Unsupported:
        pass
    items = getattr(node, "items", ())
    if len(items) == 3 and str(items[0]) == "(" and str(items[-1]) == ")":
        return affine(items[1], scope)
    if len(items) == 2 and str(items[0]) in {"+", "-"}:
        coefficients, constant = affine(items[1], scope)
        sign = 1 if str(items[0]) == "+" else -1
        return {key: sign * value for key, value in coefficients.items()}, sign * constant
    if len(items) == 3 and str(items[1]) in {"+", "-", "*"}:
        left, lc = affine(items[0], scope)
        right, rc = affine(items[2], scope)
        if str(items[1]) == "*":
            if left and right:
                raise Unsupported("non-affine array subscript")
            return ({key: value * rc for key, value in left.items()}, lc * rc) if left else ({key: value * lc for key, value in right.items()}, lc * rc)
        sign = 1 if str(items[1]) == "+" else -1
        combined = dict(left)
        for key, value in right.items():
            combined[key] = combined.get(key, 0) + sign * value
        return {key: value for key, value in combined.items() if value}, lc + sign * rc
    raise Unsupported(f"non-affine array subscript {node}")


PURE_INTRINSICS = {"abs", "sqrt", "min", "max", "sin", "cos", "tan", "exp", "log", "tanh", "mod", "modulo", "int", "real", "sign"}


@dataclass
class Nest:
    node: Any
    loops: list[tuple[str, list[Any]]]
    assignments: list[Any]


def scope_checks(node: Any) -> None:
    """Reject enclosing constructs with storage or concurrency contracts."""
    parent = getattr(node, "parent", None)
    while parent is not None and kind(parent) != "Subroutine_Subprogram":
        if kind(parent) in {"Block_Construct", "Associate_Construct", "Select_Type_Construct", "Select_Rank_Construct",
                            "Forall_Construct", "Critical_Construct", "Change_Team_Construct"}:
            raise Unsupported(f"enclosing {kind(parent)} requires scope/association/concurrency analysis")
        if kind(parent) == "Block_Nonlabel_Do_Construct":
            header = next((n for n in content(parent) if kind(n) == "Nonlabel_Do_Stmt"), None)
            control = header.items[1] if header is not None else None
            if control is not None and control.items[1] is None and str(control).lower().startswith("concurrent"):
                raise Unsupported("enclosing DO CONCURRENT requires a pure generated-call contract")
        parent = getattr(parent, "parent", None)
def rectangular(node: Any, scope: Scope, blocked: set[int]) -> Nest:
    scope_checks(node)
    if any(kind(n) == "Comment" and str(n).lstrip().startswith("!$") for n in walk(node)):
        raise Unsupported("directive inside outlined region requires preserved native scope")
    loops = []
    current = node
    while True:
        if id(current) in blocked:
            raise Unsupported("loop is required by an existing OpenMP worksharing/collapse directive")
        iterator, bounds = loop_info(current)
        if len(bounds) > 2 and integer_constant(bounds[2], scope) != 1:
            raise Unsupported("only unit-stride loop regions are currently outlined")
        loops.append((iterator, bounds[:2]))
        children = code_children(current)[1:-1]
        if len(children) == 1 and kind(children[0]) == "Block_Nonlabel_Do_Construct":
            current = children[0]
            continue
        if not children or any(kind(child) != "Assignment_Stmt" for child in children):
            raise Unsupported("loop body is not a straight-line assignment region")
        body = children
        break
    iterators = {name.lower() for name, _ in loops}
    if len(iterators) != len(loops):
        raise Unsupported("reused induction variable in nested loops")
    for _, bounds in loops:
        if any(names(bound) & iterators for bound in bounds):
            raise Unsupported("loop bounds depend on another outlined induction variable")
    return Nest(node, loops, body)


def original_scope(parsed: Parsed, routine: Any, project: Project, *, declarations_only=False) -> tuple[Scope, str, set[str], int]:
    parent = getattr(routine, "parent", None)
    module_scope = None
    while parent is not None:
        if id(parent) in parsed.modules:
            module_scope = parsed.modules[id(parent)]
            break
        if kind(parent) in {"Subroutine_Subprogram", "Function_Subprogram", "Main_Program"}:
            raise Unsupported("internal procedures require host-association analysis")
        parent = getattr(parent, "parent", None)
    statement = next(n for n in content(routine) if kind(n) == "Subroutine_Stmt")
    _, name, args, suffix = statement.items
    prefix = statement.items[0]
    if prefix is not None and any(str(value).lower() in {"pure", "elemental"} for value in prefix.items):
        raise Unsupported("PURE/ELEMENTAL procedure requires a pure generated-call contract")
    if suffix is not None:
        raise Unsupported("procedure with BIND/suffix is not outlined")
    if any(kind(node) == "Entry_Stmt" for node in walk(routine)):
        raise Unsupported("ENTRY procedure requires argument-availability analysis")
    if not declarations_only and any(kind(node) == "Internal_Subprogram_Part" for node in content(routine)):
        raise Unsupported("contained procedures require analysis of hidden host-associated reads")
    scope = Scope(str(name), project, module_scope)
    for specification in (n for n in content(routine) if kind(n) == "Specification_Part"):
        scope.populate(specification)
    for part in (n for n in content(routine) if kind(n) == "Internal_Subprogram_Part"):
        for procedure in content(part):
            for header in (n for n in content(procedure) if kind(n) in {"Subroutine_Stmt", "Function_Stmt"}):
                internal_name = str(header.items[1]).lower()
                scope.symbols.setdefault(internal_name, Symbol(internal_name, "procedure", None, None, set(), None, None, scope))
    dummies = {str(arg).lower() for arg in args.items} if args is not None else set()
    # Use association belongs immediately after the complete subroutine header,
    # before IMPLICIT and all declarations. The line number is an insertion
    # boundary: insert before this one-based original-source line.
    _, header_end = parsed.origin_span(statement)
    return scope, module_scope.name if module_scope else "", dummies, header_end + 1


def omp_constraints(parsed: Parsed, routine: Any) -> tuple[set[int], set[int]]:
    blocked = set()
    nested = set()
    source_lines = parsed.text.splitlines()
    for loop in (n for n in walk(routine) if kind(n) == "Block_Nonlabel_Do_Construct"):
        header = next(n for n in content(loop) if kind(n) == "Nonlabel_Do_Stmt")
        start, _ = parsed.origin_span(header)
        index = start - 2
        directives = []
        while index >= 0:
            text = source_lines[index].strip().lower()
            if text.startswith("!$omp"):
                directives.append(text)
            elif text and not text.startswith("!"):
                break
            index -= 1
        if any(re.search(r"\b(?:do|simd)\b", text) for text in directives):
            levels = max([int(match[1]) for text in directives if (match := re.search(r"collapse\s*\(\s*(\d+)\s*\)", text))] or [1])
            current = loop
            for _ in range(levels):
                blocked.add(id(current))
                children = code_children(current)[1:-1]
                if len(children) != 1 or kind(children[0]) != "Block_Nonlabel_Do_Construct":
                    break
                current = children[0]
        # An orphaned !$omp do is also executed by an existing team; later
        # worksharing loops in a parallel region need the same serial outlined
        # helper as the first combined parallel-do construct.
        if any(re.search(r"\b(?:parallel|do|simd)\b", text) for text in directives):
            nested.update(id(child) for child in walk(loop) if kind(child) == "Block_Nonlabel_Do_Construct")
    return blocked, nested


def symbol_checks(symbol: Symbol, scope: Scope, dummies: set[str], *, stable_allocatable: bool = False) -> None:
    if symbol.scope.storage_aliases:
        raise Unsupported(f"EQUIVALENCE/COMMON/BIND/NAMELIST storage association in the scope of {symbol.name}")
    allowed_allocatable = stable_allocatable and symbol.rank > 0
    unsafe = {"pointer", "target", "volatile", "asynchronous", "optional"}
    if not allowed_allocatable:
        unsafe.add("allocatable")
    if symbol.attributes & unsafe:
        raise Unsupported(f"unsafe storage attributes for {symbol.name}")
    unknown = symbol.attributes - {"parameter", "save", "contiguous", "value", "protected", "public", "private"} - ({"allocatable"} if allowed_allocatable else set())
    if unknown:
        raise Unsupported(f"unsupported declaration attributes for {symbol.name}: {', '.join(sorted(unknown))}")
    if symbol.rank and symbol.scope is scope and symbol.name.lower() not in dummies:
        raise Unsupported(f"local scratch array {symbol.name}")
    type_spec(symbol, logical=stable_allocatable)


def ancestors(node: Any) -> set[int]:
    result = set()
    parent = getattr(node, "parent", None)
    while parent is not None:
        result.add(id(parent))
        parent = getattr(parent, "parent", None)
    return result


def directive_value_reads(text: str) -> set[str]:
    """Names whose incoming scalar value may be used by an OpenMP directive.

    PRIVATE declares a new value; it does not read the old scalar induction
    variable. Keep FIRSTPRIVATE and all unrecognized clauses conservative.
    Callers use this for local scalar liveness, not descriptor/lifetime effects.
    """
    text = re.sub(r"\bprivate\s*\(\s*[a-z]\w*(?:\s*,\s*[a-z]\w*)*\s*\)", "", text, flags=re.I)
    return set(re.findall(r"\b[a-zA-Z][a-zA-Z0-9_]*\b", text.lower()))


def live_after(routine: Any, end: int, written: set[str], parsed: Parsed, region_node: Any) -> set[str]:
    """Conservatively inspect later source statements, killing scalar defs."""
    pending = set(written)
    active = ancestors(region_node)
    for statement in walk(routine):
        if not pending:
            break
        if not getattr(getattr(statement, "item", None), "span", None):
            continue
        start, _ = parsed.origin_span(statement)
        if start <= end:
            continue
        if kind(statement) == "Comment":
            # Directive clauses are opaque to fparser's expression tree, but
            # e.g. FIRSTPRIVATE evaluates the old value of its listed names.
            if re.match(r"\s*!\$omp\b", str(statement), re.I):
                referenced = directive_value_reads(str(statement))
                if referenced & pending:
                    return referenced & pending
            continue
        if kind(statement) == "Assignment_Stmt":
            lhs, _, rhs = statement.items
            used = names(rhs) | (names(lhs) if kind(lhs) != "Name" else set())
            if used & pending:
                return used & pending
            # A definition in a conditional branch or a later loop may not
            # execute; it cannot kill a previously produced live value.
            if kind(lhs) == "Name" and ancestors(statement) <= active:
                pending.discard(str(lhs).lower())
        elif kind(statement) == "Nonlabel_Do_Stmt":
            control = statement.items[1]
            if control is not None and control.items[1] is not None:
                iterator, bounds = control.items[1]
                used = set().union(*(names(bound) for bound in bounds))
                if used & pending:
                    return used & pending
                parents = ancestors(statement)
                own_loop = getattr(statement, "parent", None)
                if own_loop is not None:
                    parents.discard(id(own_loop))
                if parents <= active:
                    pending.discard(str(iterator).lower())
        elif names(statement) & pending:
            return names(statement) & pending
    return set()


def safe_bound(expression: Any, scope: Scope, written: set[str], dummies: set[str]) -> None:
    if names(expression) & written:
        raise Unsupported("loop bound reads storage written inside the outlined region")
    allowed = {"Name", "Int_Literal_Constant", "Level_2_Expr", "Level_2_Unary_Expr", "Add_Operand",
               "Mult_Operand", "Parenthesis", "Intrinsic_Function_Reference", "Intrinsic_Name", "Actual_Arg_Spec_List"}
    for node in walk(expression):
        if node is None or isinstance(node, (str, int, float, bool)):
            continue
        if kind(node) not in allowed:
            raise Unsupported("array elements or opaque calls in loop bounds cannot be evaluated eagerly")
        items = getattr(node, "items", ())
        if len(items) == 3 and str(items[1]) in {"/", "**"}:
            # Inner bounds are normally not evaluated for an empty outer
            # domain. Eager argument evaluation must not introduce a divide
            # by zero or another trapping operation on that path.
            try:
                integer_constant(node, scope)
            except Unsupported:
                raise Unsupported("potentially trapping loop-bound arithmetic cannot be evaluated eagerly")
        if kind(node) == "Intrinsic_Function_Reference":
            name = str(node.items[0]).lower()
            if name not in {"min", "max", "size", "lbound", "ubound"} or not scope.intrinsic_unshadowed(name):
                raise Unsupported("loop-bound function needs an effect/evaluation-order proof")
        if kind(node) == "Name":
            symbol = scope.resolve(str(node))
            if symbol is None:
                raise Unsupported(f"unresolved loop-bound symbol {node}")
            symbol_checks(symbol, scope, dummies)


def visible_symbol(scope: Scope, name: str) -> Symbol | None:
    symbol = scope.resolve(name)
    return replace(symbol, name=name) if symbol is not None and symbol.name.lower() != name.lower() else symbol


def normalize_statement(statement: Any, scope: Scope, lower_names: dict[str, list[str]]) -> str:
    node = F.Assignment_Stmt(str(statement))
    # Normalize real literal kinds explicitly; source kind aliases need not be
    # imported into the independently compilable generated module.
    for literal in (n for n in walk(node) if kind(n) == "Real_Literal_Constant"):
        value = re.sub("[dD]", "e", literal.items[0])
        literal.items = (value, str(literal_kind(literal, scope)))
    for literal in (n for n in walk(node) if kind(n) == "Int_Literal_Constant" and n.items[1] is not None):
        selector = literal.items[1]
        try:
            precision = int(str(selector))
        except ValueError:
            precision = integer_constant(F.Name(str(selector)), scope)
        if precision != 4:
            raise Unsupported("non-default integer literal kind in region arithmetic")
        literal.items = (literal.items[0], "4")
    for call in (n for n in walk(node) if kind(n) == "Intrinsic_Function_Reference"):
        name, arguments = call.items
        if str(name).lower() in {"real", "int"} and len(arguments.items) == 2:
            precision = integer_constant(arguments.items[1], scope)
            call.items = (name, F.Actual_Arg_Spec_List(f"{arguments.items[0]}, {precision}"))
    for reference in (n for n in walk(node) if kind(n) == "Part_Ref"):
        name, subscripts = reference.items
        key = str(name).lower()
        if key in lower_names:
            values = [f"({index}) - ({lower}) + 1" for index, lower in zip(subscripts.items, lower_names[key])]
            reference.items = (name, F.Section_Subscript_List(", ".join(values)))
    return str(node)


def make_region(nests: list[Nest], scope: Scope, module: str, dummies: set[str],
                routine: Any, parsed: Parsed, insertion: int, nested: set[int], ordinal: int,
                *, liveness=live_after) -> dict:
    iterator_names = {name.lower() for nest in nests for name, _ in nest.loops}
    assigned_temporaries = {str(assignment.items[0]).lower() for nest in nests for assignment in nest.assignments
                            if kind(assignment.items[0]) == "Name"}
    local_written = set(iterator_names)
    arrays: dict[str, Symbol] = {}
    captured: dict[str, Symbol] = {}
    local_symbols: dict[str, Symbol] = {}
    write_maps: dict[str, list[tuple[dict[str, int], int]]] = {}
    all_reads: dict[str, list[list[tuple[dict[str, int], int]]]] = {}
    for nest in nests:
        defined = {name.lower() for name, _ in nest.loops}
        active = set(defined)
        for iterator in active:
            symbol = scope.resolve(iterator)
            if symbol is None or symbol.rank or type_spec(symbol) != "integer" or symbol.scope is not scope or iterator in dummies:
                raise Unsupported(f"induction variable {iterator} is not a local default integer")
            symbol_checks(symbol, scope, dummies)
            if "save" in symbol.attributes or symbol.initializer is not None:
                raise Unsupported(f"persistent induction variable {iterator}")
            local_symbols[iterator] = symbol
        for assignment in nest.assignments:
            lhs, _, rhs = assignment.items
            if kind(lhs) not in {"Name", "Part_Ref"}:
                raise Unsupported("unsupported assignment target")
            for call in (n for n in walk(rhs) if kind(n) == "Intrinsic_Function_Reference"):
                if str(call.items[0]).lower() not in PURE_INTRINSICS or not scope.intrinsic_unshadowed(str(call.items[0]).lower()):
                    raise Unsupported(f"unsupported/non-elemental intrinsic {call.items[0]}")
                if str(call.items[0]).lower() in {"real", "int"} and len(call.items[1].items) == 2:
                    precision = integer_constant(call.items[1].items[1], scope)
                    allowed_kinds = {4, 8} if str(call.items[0]).lower() == "real" else {4}
                    if precision not in allowed_kinds:
                        raise Unsupported("unsupported conversion kind")
            if any(kind(n) in {"Data_Ref", "Structure_Constructor", "Array_Constructor", "Subscript_Triplet", "Actual_Arg_Spec"} for n in walk(assignment)):
                raise Unsupported("array sections, derived data, constructors, or keyword expressions")
            for reference in (n for n in walk(assignment) if kind(n) == "Part_Ref"):
                name, indices = reference.items
                key = str(name).lower()
                symbol = visible_symbol(scope, str(name))
                if symbol is None or not symbol.rank:
                    raise Unsupported(f"unresolved array or opaque function {name}")
                symbol_checks(symbol, scope, dummies)
                if len(indices.items) != symbol.rank:
                    raise Unsupported(f"array rank mismatch for {name}")
                if any(names(index) & assigned_temporaries for index in indices.items):
                    raise Unsupported("array index uses a region-written scalar; injectivity needs value propagation")
                signature = [affine(index, scope) for index in indices.items]
                arrays[key] = symbol
                if reference is lhs:
                    # Each active iterator must own one distinct subscript;
                    # translations and permutations are permitted. Mixing
                    # iterators in one subscript needs a stronger injection
                    # proof and is deliberately excluded.
                    owned = set()
                    for coefficients, _ in signature:
                        varying = {key: value for key, value in coefficients.items() if key in active}
                        if len(varying) > 1 or any(abs(value) != 1 for value in varying.values()):
                            raise Unsupported("output subscript is not an injective pointwise map")
                        owned.update(varying)
                    if owned != active:
                        raise Unsupported("output does not distinguish every induction variable")
                    if key in write_maps and write_maps[key] != signature:
                        raise Unsupported(f"multiple different output index maps for {name}")
                    write_maps[key] = signature
                else:
                    all_reads.setdefault(key, []).append(signature)
            array_names = set(arrays)
            references = names(rhs) | (names(lhs) if kind(lhs) == "Part_Ref" else set())
            for name in references - array_names:
                symbol = visible_symbol(scope, name)
                if symbol is None:
                    raise Unsupported(f"unresolved captured symbol {name}")
                symbol_checks(symbol, scope, dummies)
                if name in defined:
                    continue
                if name in iterator_names:
                    raise Unsupported(f"induction variable used before definition: {name}")
                # A local scalar assigned inside this nest is a temporary;
                # reading it before that assignment implies carried state.
                assigned = {str(a.items[0]).lower() for a in nest.assignments if kind(a.items[0]) == "Name"}
                if name in assigned:
                    raise Unsupported(f"temporary {name} is read before definition")
                captured[name] = symbol
            if kind(lhs) == "Name":
                key = str(lhs).lower()
                symbol = scope.resolve(key)
                if symbol is None or symbol.rank or symbol.scope is not scope or key in dummies or "parameter" in symbol.attributes:
                    raise Unsupported(f"write to captured or nonlocal scalar {lhs}")
                symbol_checks(symbol, scope, dummies)
                if "save" in symbol.attributes or symbol.initializer is not None:
                    raise Unsupported(f"write to persistent local scalar {lhs}")
                if key in active:
                    raise Unsupported("assignment changes an induction variable")
                local_symbols[key] = symbol
                local_written.add(key)
                defined.add(key)
    if not write_maps:
        raise Unsupported("region has no array output")
    if not scope.intrinsic_unshadowed("lbound"):
        raise Unsupported("LBOUND intrinsic binding cannot be proved for generated call arguments")
    storage_names: dict[int, set[str]] = {}
    for name in arrays:
        storage_names.setdefault(id(scope.resolve(name)), set()).add(name)
    if any(len(aliases) > 1 and aliases & write_maps.keys() for aliases in storage_names.values()):
        raise Unsupported("multiple visible aliases of the same written array storage")
    for name, signatures in all_reads.items():
        if name in write_maps and any(signature != write_maps[name] for signature in signatures):
            raise Unsupported(f"neighbor/dependent read of written array {name}")
    for nest in nests:
        for _, bounds in nest.loops:
            for expression in bounds:
                safe_bound(expression, scope, set(write_maps) | local_written, dummies)
    check_local_liveness(nests, routine, parsed, local_written, liveness)
    return render_region(nests, scope, module, dummies, routine, parsed, insertion, nested, ordinal,
                         arrays, captured, local_symbols, set(write_maps))


def check_local_liveness(nests, routine, parsed, local_written, liveness=live_after):
    first, _ = parsed.origin_span(nests[0].node)
    _, last = parsed.origin_span(nests[-1].node)
    if live := liveness(routine, last, local_written, parsed, nests[-1].node):
        raise Unsupported("written local value remains live after region: " + ", ".join(sorted(live)))
    # A scalar read on an enclosing loop's next iteration is also live-out.
    ancestor = getattr(nests[0].node, "parent", None)
    while ancestor is not None and ancestor is not routine:
        if kind(ancestor) == "Block_Nonlabel_Do_Construct":
            start, _ = parsed.origin_span(ancestor)
            for statement in walk(ancestor):
                if not getattr(getattr(statement, "item", None), "span", None) or kind(statement) == "Comment":
                    continue
                line, _ = parsed.origin_span(statement)
                if kind(statement) == "Assignment_Stmt":
                    lhs, _, rhs = statement.items
                    reads = names(rhs) | (names(lhs) if kind(lhs) != "Name" else set())
                elif kind(statement) == "Nonlabel_Do_Stmt":
                    control = statement.items[1]
                    reads = set().union(*(names(value) for value in control.items[1][1])) if control is not None and control.items[1] is not None else names(statement)
                else:
                    reads = names(statement)
                if start < line < first and reads & local_written:
                    raise Unsupported("local scalar is read across an enclosing-loop backedge")
        ancestor = getattr(ancestor, "parent", None)


def render_region(nests, scope, module, dummies, routine, parsed, insertion, nested, ordinal,
                  arrays, captured, local_symbols, write_maps, *, structured=False, statement_span=False,
                  body_builder=None):
    first, _ = parsed.origin_span(nests[0].node)
    _, last = parsed.origin_span(nests[-1].node)
    source = str(parsed.path.relative_to(parsed.path.parents[0]))
    try:
        source = str(parsed.path.relative_to(scope.project.root))
    except ValueError:
        source = str(parsed.path)
    original = "".join(parsed.text.splitlines(keepends=True)[first - 1:last])
    if any(re.match(r"\s*#", line) for line in original.splitlines()):
        raise Unsupported("preprocessor directive crosses the original replacement span")
    if ";" in original:
        raise Unsupported("multiple statements on a source line need finer replacement spans")
    original_lines = original.splitlines()
    if not statement_span and (not re.match(r"\s*do\b", original_lines[0], re.I) or not re.match(r"\s*end\s*do\b", original_lines[-1], re.I)):
        raise Unsupported("source macro/comment span does not exactly delimit the outlined DO region")
    digest = hashlib.sha256((source + "\0" + scope.name + "\0" + str(first) + "\0" + original).encode()).hexdigest()
    slug = digest[:12]
    prefix = "outline_" + slug
    generated_module = prefix + "_module"
    entry = prefix
    occupied = set(scope.symbols) | set(arrays) | set(captured)
    if entry in occupied:
        raise Unsupported("generated entry name conflicts with a source symbol")
    args = []
    actuals = []
    declarations = []
    metadata_symbols = []
    for name, symbol in sorted({**arrays, **captured}.items()):
        args.append(symbol.name)
        actuals.append(symbol.name)
        intent = "inout" if name in write_maps else "in"
        shape = "(" + ",".join(":" for _ in range(symbol.rank)) + ")" if symbol.rank else ""
        declarations.append(f"    {type_spec(symbol, logical=structured)}, intent({intent}) :: {symbol.name}{shape}")
        metadata_symbols.append({"name": symbol.name, "type": type_spec(symbol, logical=structured), "rank": symbol.rank,
                                 "intent": intent, "origin": "dummy" if name in dummies else "host/local",
                                 "contiguous_proven": bool(symbol.rank and ("contiguous" in symbol.attributes
                                     or kind(symbol.shape) == "Explicit_Shape_Spec_List"
                                     or (symbol.scope is not scope and "pointer" not in symbol.attributes)))})
    lower_names = {}
    for name, symbol in sorted(arrays.items()):
        lower_names[name] = []
        for dimension, expression in enumerate(array_lowers(symbol)):
            argument = f"{prefix}_lb_{len(lower_names)}_{dimension}"
            args.append(argument)
            actuals.append(expression)
            declarations.append(f"    integer, intent(in) :: {argument}")
            lower_names[name].append(argument)
    body = []
    if body_builder is not None:
        body = body_builder(lower_names)
    elif structured:
        body = normalize_blocks(nests, scope, lower_names)
    else:
        bound_index = 0
        bound_pairs = {}
        for nest in nests:
            for depth, (iterator, bounds) in enumerate(nest.loops):
                bound_key = tuple(re.sub(r"\s+", "", str(expression)).lower() for expression in bounds)
                bound_arguments = bound_pairs.get(bound_key)
                if bound_arguments is None:
                    bound_arguments = []
                    for side, expression in zip(("lo", "hi"), bounds):
                        argument = f"{prefix}_{side}_{bound_index}"
                        args.append(argument)
                        actuals.append(str(expression))
                        declarations.append(f"    integer, intent(in) :: {argument}")
                        bound_arguments.append(argument)
                    bound_pairs[bound_key] = bound_arguments
                body.append("    " + "  " * depth + f"do {iterator} = {', '.join(bound_arguments)}")
                bound_index += 1
            for assignment in nest.assignments:
                body.append("    " + "  " * len(nest.loops) + normalize_statement(assignment, scope, lower_names))
            for depth in reversed(range(len(nest.loops))):
                body.append("    " + "  " * depth + "end do")
    for name, symbol in sorted(local_symbols.items()):
        declarations.append(f"    {type_spec(symbol, logical=structured)} :: {symbol.name}")
    header = f"  subroutine {entry}( &\n" + ", &\n".join("    " + name for name in args) + ")"
    module_source = "\n".join([f"module {generated_module}", "  implicit none", "contains", header,
                                  *declarations, *body, f"  end subroutine {entry}", f"end module {generated_module}", ""])
    call = f"call {entry}( &\n" + ", &\n".join("  " + name for name in actuals) + ")"
    statement = next(n for n in content(routine) if kind(n) == "Subroutine_Stmt")
    header_start, _ = parsed.origin_span(statement)
    previous_lines = parsed.text.splitlines()[:header_start - 1]
    annotated = False
    for line in reversed(previous_lines):
        if line.strip().lower() == "! kernel":
            annotated = True
            break
        if line.strip() and not line.lstrip().startswith("!"):
            break
    return {"id": slug, "source": source, "module": module, "routine": scope.name,
            "start_line": first, "end_line": last, "insertion_line": insertion,
            "original_text": original, "source_span_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "original_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "source_sha256": hashlib.sha256(parsed.text.encode()).hexdigest(),
            "generated_module": generated_module, "entry": entry,
            "args": args, "actual_args": actuals, "call": call,
            "use": f"use {generated_module}, only: {entry}",
            "module_source": module_source, "symbols": metadata_symbols,
            "loop_nests": len(nests), "loop_ranks": [len(nest.loops) for nest in nests],
            "context": {"nested_parallel": any(id(nest.node) in nested for nest in nests)},
            "annotated": annotated,
            "legality": ("capture storage and native liveness proved; numerical and parallel legality delegated to selected compiler"
                         if structured else "rectangular unit-stride loops; injective pointwise writes; written arrays read only at the same point; no calls with unknown effects; scalar temporaries defined before read and dead after region"),
            "ordinal": ordinal}


def block_tree(nests: list[Nest], scope: Scope):
    """Clone the retained DO/IF/assignment structure, without native directives."""
    body = []
    for nest in nests:
        body.extend(f"do {iterator} = {', '.join(map(str, bounds))}" for iterator, bounds in nest.loops)
        body.extend(str(statement) for statement in nest.assignments)
        body.extend("end do" for _ in nest.loops)
    source = "\n".join(line for line in "\n".join(body).splitlines() if not line.lstrip().startswith("!"))
    tree = scope.project.parser(FortranStringReader("subroutine outline_body\n" + source + "\nend subroutine"))
    return next(node for node in walk(tree) if kind(node) == "Execution_Part")


def expression_names(node):
    # Keyword labels are parsed as Names, but do not refer to source storage.
    labels = {id(value.items[0]) for value in walk(node) if kind(value) == "Actual_Arg_Spec"}
    return {str(value).lower() for value in walk(node) if kind(value) == "Name" and id(value) not in labels}


def make_block_region(nests, scope, module, dummies, routine, parsed, insertion, ordinal, *, liveness=live_after,
                      statement_span=False):
    """Prove capture/storage contracts; the public compiler proves computation.

    Bounds and scalar/conditional statements remain at their original positions.
    This path is usable only when the caller checks the complete emitted entry
    with its selected compiler before accepting the replacement.
    """
    body = block_tree(nests, scope)
    statements = {"Execution_Part", "Block_Nonlabel_Do_Construct", "Nonlabel_Do_Stmt", "End_Do_Stmt",
                  "Assignment_Stmt", "If_Construct", "If_Then_Stmt", "Else_If_Stmt", "Else_Stmt",
                  "End_If_Stmt", "If_Stmt", "Comment"}

    def check_statements(node):
        if kind(node) not in statements:
            raise Unsupported(f"unsupported statement in complete GPU block: {kind(node)}")
        for child in content(node):
            check_statements(child)
        if kind(node) == "If_Stmt":
            check_statements(node.items[1])

    check_statements(body)
    loops = [node for node in walk(body) if kind(node) == "Block_Nonlabel_Do_Construct"]
    iterators = {loop_info(node)[0].lower() for node in loops}
    assignments = [node for node in walk(body) if kind(node) == "Assignment_Stmt"]
    scalar_writes = {str(node.items[0]).lower() for node in assignments if kind(node.items[0]) == "Name"}
    local_written = iterators | scalar_writes
    arrays, captured, local_symbols = {}, {}, {}
    written_arrays = set()
    for name in local_written:
        symbol = scope.resolve(name)
        if (symbol is None or symbol.rank or symbol.scope is not scope or name in dummies
                or "parameter" in symbol.attributes):
            raise Unsupported(f"write to captured or nonlocal scalar {name}")
        symbol_checks(symbol, scope, dummies, stable_allocatable=True)
        if "save" in symbol.attributes or symbol.initializer is not None:
            raise Unsupported(f"write to persistent local scalar {name}")
        if name in iterators and type_spec(symbol) != "integer":
            raise Unsupported(f"induction variable {name} is not a local default integer")
        local_symbols[name] = symbol
    for assignment in assignments:
        lhs = assignment.items[0]
        if kind(lhs) not in {"Name", "Part_Ref"}:
            raise Unsupported("unsupported assignment target")
        if kind(lhs) == "Name" and str(lhs).lower() in iterators:
            raise Unsupported("assignment changes an induction variable")
        if kind(lhs) == "Part_Ref":
            written_arrays.add(str(lhs.items[0]).lower())
    if not written_arrays:
        raise Unsupported("region has no array output")
    # Array sections/whole-array assignments may allocate or change descriptors;
    # derived references introduce hidden storage aliases. Neither is a capture.
    if any(kind(node) in {"Data_Ref", "Structure_Constructor", "Array_Constructor", "Subscript_Triplet"}
           for node in walk(body)):
        raise Unsupported("array sections, derived data, or constructors in GPU capture")
    for call in (node for node in walk(body) if kind(node) == "Intrinsic_Function_Reference"):
        name = str(call.items[0]).lower()
        if not scope.intrinsic_unshadowed(name):
            raise Unsupported(f"intrinsic binding cannot be proved: {name}")
    for reference in (node for node in walk(body) if kind(node) == "Part_Ref"):
        name, indices = reference.items
        symbol = visible_symbol(scope, str(name))
        if symbol is None or not symbol.rank:
            raise Unsupported(f"unresolved array or opaque function {name}")
        if len(indices.items) != symbol.rank:
            raise Unsupported(f"array rank mismatch for {name}")
    # Fortran model inquiries inspect type even when their argument is
    # undefined. Preserve scalar declarations used only there; passing them
    # through a value ABI would introduce a native value read.
    model_names = {id(value) for call in walk(body) if kind(call) == "Intrinsic_Function_Reference"
                   and str(call.items[0]).lower() in {"kind", "epsilon", "tiny", "huge"}
                   for value in walk(call.items[1]) if kind(value) == "Name"}
    keyword_names = {id(value.items[0]) for value in walk(body) if kind(value) == "Actual_Arg_Spec"}
    runtime_names = {str(value).lower() for value in walk(body) if kind(value) == "Name"
                     and id(value) not in model_names | keyword_names}
    for name in expression_names(body) - local_written:
        symbol = visible_symbol(scope, name)
        if symbol is None:
            raise Unsupported(f"unresolved captured symbol {name}")
        symbol_checks(symbol, scope, dummies, stable_allocatable=True)
        if not symbol.rank and name not in runtime_names:
            local_symbols[name] = symbol
        else:
            (arrays if symbol.rank else captured)[name] = symbol
    if not scope.intrinsic_unshadowed("lbound"):
        raise Unsupported("LBOUND intrinsic binding cannot be proved for generated call arguments")
    allocated = sorted(symbol.name for symbol in arrays.values() if "allocatable" in symbol.attributes)
    if allocated and not scope.intrinsic_unshadowed("allocated"):
        raise Unsupported("ALLOCATED intrinsic binding cannot be proved for generated guard")
    storage_names = {}
    for name in arrays:
        storage_names.setdefault(id(scope.resolve(name)), set()).add(name)
    if any(len(aliases) > 1 and aliases & written_arrays for aliases in storage_names.values()):
        raise Unsupported("multiple visible aliases of the same written array storage")
    check_local_liveness(nests, routine, parsed, local_written, liveness)
    region = render_region(nests, scope, module, dummies, routine, parsed, insertion, set(), ordinal,
                           arrays, captured, local_symbols, written_arrays, structured=True, statement_span=statement_span)
    region["allocated_arrays"] = allocated
    return region


def normalize_blocks(nests, scope, lower_names):
    """Normalize descriptors and literal kinds without moving expressions."""
    body = block_tree(nests, scope)

    def normalize(node):
        if isinstance(node, (list, tuple)):
            return type(node)(normalize(value) for value in node)
        if node is None or isinstance(node, (str, int, float, bool)):
            return node
        if hasattr(node, "items"):
            node.items = tuple(normalize(value) for value in node.items)
        if hasattr(node, "content"):
            node.content = [normalize(value) for value in node.content]
        if kind(node) == "Real_Literal_Constant":
            precision = literal_kind(node, scope)
            node.items = (re.sub("[dD]", "e", node.items[0]), str(precision))
        elif kind(node) in {"Int_Literal_Constant", "Logical_Literal_Constant"} and node.items[1] is not None:
            selector = str(node.items[1])
            precision = int(selector) if selector.isdigit() else integer_constant(F.Name(selector), scope)
            if precision != 4:
                raise Unsupported("non-default integer/logical literal kind in GPU block")
            node.items = (node.items[0], None)
        elif kind(node) == "Part_Ref":
            name, subscripts = node.items
            lowers = lower_names[str(name).lower()]
            values = [f"({index}) - ({lower}) + 1" for index, lower in zip(subscripts.items, lowers)]
            node.items = (name, F.Section_Subscript_List(", ".join(values)))
        elif kind(node) == "Intrinsic_Function_Reference":
            function, arguments = node.items
            name = str(function).lower()
            values = list(arguments.items)
            if name in {"real", "int", "nint", "floor", "ceiling"}:
                for index, value in enumerate(values):
                    keyword = str(value.items[0]).lower() if kind(value) == "Actual_Arg_Spec" else None
                    if keyword == "kind" or (keyword is None and index == 1):
                        selector = value.items[1] if keyword else value
                        precision = integer_constant(selector, scope)
                        values[index] = (F.Actual_Arg_Spec(f"kind={precision}") if keyword else
                                         F.Int_Literal_Constant(str(precision)))
                node.items = (function, F.Actual_Arg_Spec_List(", ".join(map(str, values))))
            if name in {"allocated", "lbound", "ubound"}:
                bound_args = {}
                formal = ["array"] if name == "allocated" else ["array", "dim", "kind"]
                for index, value in enumerate(values):
                    keyword = str(value.items[0]).lower() if kind(value) == "Actual_Arg_Spec" else None
                    key = keyword or (formal[index] if index < len(formal) else "invalid")
                    if key not in formal or key in bound_args:
                        raise Unsupported("unsupported descriptor inquiry arguments")
                    bound_args[key] = value.items[1] if keyword else value
                array_node = bound_args.get("array")
                array = str(array_node).lower()
                symbol = scope.resolve(array) if kind(array_node) == "Name" else None
                if name == "allocated":
                    if symbol is None or "allocatable" not in symbol.attributes:
                        raise Unsupported("ALLOCATED inquiry requires an allocatable capture")
                    return F.Logical_Literal_Constant(".true.")
                if "dim" not in bound_args or array not in lower_names:
                    raise Unsupported("array bound inquiry requires a captured array and constant DIM")
                if "kind" in bound_args and integer_constant(bound_args["kind"], scope) != 4:
                    raise Unsupported("non-default integer array bound inquiry")
                dimension = integer_constant(bound_args["dim"], scope)
                if not 1 <= dimension <= len(lower_names[array]):
                    raise Unsupported("invalid array bound DIM")
                lower = lower_names[array][dimension - 1]
                return (F.Name(lower) if name == "lbound" else
                        F.Level_2_Expr(f"({lower}) + size({array}, {dimension}) - 1"))
        return node

    return ["    " + line for line in str(normalize(body)).splitlines()]


def discover_sources(sources: list[str], project_root: str, output_dir: str,
                     defines: list[str] | None = None, select: list[str] | None = None) -> dict:
    project = Project(Path(project_root), list(defines or ()))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    regions = []
    rejected = []
    for source in sources:
        path = Path(source).resolve()
        try:
            parsed = project.parse(path)
        except Exception as error:
            rejected.append({"file": str(path), "routine": None, "reason": f"source parse/preprocessing: {error}"})
            continue
        for routine in (n for n in walk(parsed.tree) if kind(n) == "Subroutine_Subprogram"):
            statement = next(n for n in content(routine) if kind(n) == "Subroutine_Stmt")
            routine_name = str(statement.items[1])
            if select and not any(fnmatch.fnmatchcase(routine_name.lower(), pattern.lower()) for pattern in select):
                continue
            try:
                scope, module, dummies, insertion = original_scope(parsed, routine, project)
                blocked, nested = omp_constraints(parsed, routine)
            except Exception as error:
                rejected.append({"file": str(path), "routine": routine_name, "reason": str(error)})
                continue
            diagnostics = set()

            def scan(parent: Any) -> None:
                children = content(parent)
                index = 0
                while index < len(children):
                    child = children[index]
                    if kind(child) != "Block_Nonlabel_Do_Construct":
                        if kind(child) not in {"Subroutine_Subprogram", "Internal_Subprogram_Part"}:
                            scan(child)
                        index += 1
                        continue
                    try:
                        nest = rectangular(child, scope, blocked)
                        candidates = [nest]
                        after = index + 1
                        # Adjacency is structural: comments are harmless except
                        # directives, which form transformation barriers.
                        while after < len(children):
                            next_node = children[after]
                            if kind(next_node) == "Comment" and not str(next_node).lstrip().startswith("!$"):
                                after += 1
                                continue
                            if kind(next_node) != "Block_Nonlabel_Do_Construct":
                                break
                            try:
                                candidates.append(rectangular(next_node, scope, blocked))
                            except Unsupported:
                                break
                            after += 1
                        try:
                            region = make_region(candidates, scope, module, dummies, routine, parsed, insertion, nested, len(regions))
                        except Unsupported:
                            if len(candidates) == 1:
                                raise
                            region = make_region([nest], scope, module, dummies, routine, parsed, insertion, nested, len(regions))
                            after = index + 1
                        input_path = destination / ("region_" + region["id"] + ".f90")
                        input_path.write_text(region["module_source"])
                        region["input_path"] = str(input_path)
                        regions.append(region)
                        index = after
                    except Exception as error:
                        diagnostics.add(str(error))
                        scan(child)
                        index += 1

            for execution in (n for n in content(routine) if kind(n) == "Execution_Part"):
                scan(execution)
            for reason in sorted(diagnostics):
                rejected.append({"file": str(path), "routine": routine_name, "reason": reason})
    return {"schema": 1, "project_root": str(project.root), "defines": project.defines,
            "regions": regions, "rejected": rejected,
            "selection": {"sources": sources, "routine_patterns": list(select or ()), "markers_required": False},
            "parser": "fparser Fortran 2008; gfortran preprocessing with original source mapping"}
