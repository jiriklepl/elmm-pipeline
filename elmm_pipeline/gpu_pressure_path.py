"""Opt-in, source-bound pressure path preparation in the original SAVE frame.

The added flag is a *restriction*, never a readable/allocated/definedness fact.
The compiler must independently admit every retained edge and consume the
flag before it can create or reuse a domain. This module neither emits a GPU
call nor supplies a compiler effect or lifetime proof.
"""
from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from .build_receipts import file_identity

_ENTRY = "pressure::pressurecorrection"
_ROOTS = ("pressure::prepoisson", "poissonsolvers::poiss_poisfft", "pressure::postpoisson")
_EXCLUDED = {"poissonsolvers::poisssor", "poissonsolvers::poiss_poisfft_variable_z",
             "multigrid2d::poissmg2d", "multigrid::poissmg"}
_FLAG = "elmm_pressure_epoch_eligible"


def _span(node):
    item = getattr(node, "item", None)
    value = getattr(item, "fort_original_span", getattr(item, "span", None))
    if value is None or len(value) != 2 or value[0] != value[1]:
        raise ValueError("Pressure path preparation requires exact single-line original edit anchors")
    return tuple(value)


def _reference(value):
    if (not isinstance(value, dict) or set(value) != {"path", "sha256"}
            or not isinstance(value["path"], str) or not Path(value["path"]).is_absolute()
            or file_identity(value["path"]) != value):
        raise ValueError("Pressure path preparation requires the exact original source reference")
    return dict(value)


def materialize_supported_pressure_path(source_paths, analysis_sources, directory, *, original_source):
    """Add a sticky private gate without replacing the original dispatch.

    Original statements keep their relative order and occur once. The only
    executable additions set a new private SAVE LOGICAL. It is armed inside
    the original first-allocation guard after Phi's original initialization,
    and is permanently cleared before an excluded solver implementation.
    There is no second Phi/RHS/Solver frame and no copied original interval.

    The returned source must be preprocessed/exported and qualified afresh.
    In particular the old configured source map must never be retagged with
    the replacement's hash. This is an implementation preparation, not a
    ready-to-execute pipeline request.
    """
    from fparser.two.utils import walk

    from compiler.frontend.call_bindings import resolve_original_call
    from compiler.frontend.source_effects import SourceEffects
    from compiler.frontend.supported_paths import prove_supported_path
    from compiler.ir import CompilationError

    original_source = _reference(original_source)
    if not isinstance(analysis_sources, dict):
        raise TypeError("Pressure path preparation requires an authentic configured-source export")
    analysis = SourceEffects(source_paths, analysis_sources=analysis_sources)
    # The factory owns this analysis. Seal its captured source exactly as the
    # normal ScopeBuilder does; publication still rechecks every live input.
    analysis.seal_session()
    routine = analysis.routines.get(_ENTRY)
    if (routine is None or str(routine.scope.path) != original_source["path"]
            or analysis.sources[str(routine.scope.path)] != original_source["sha256"]):
        raise ValueError("Pressure path preparation differs from its pinned original SAVE frame")
    if _FLAG in routine.scope.bindings:
        raise ValueError("Pressure path preparation cannot reuse an existing eligibility binding")
    nodes = tuple(routine.execution.content)
    calls = []
    for node in walk(routine.execution):
        if type(node).__name__ != "Call_Stmt":
            continue
        try:
            resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node)
        except CompilationError:
            continue  # The unchanged clock/excluded edge still belongs to the path proof.
        calls.append((node, resolved.procedure))
    endpoints = []
    for name in (_ROOTS[0], _ROOTS[-1]):
        found = [node for node, callee in calls if callee == name and any(node is item for item in nodes)]
        if len(found) != 1:
            raise ValueError("Pressure path preparation requires one original top-level Pre/Post pair")
        endpoints.append(next(index for index, node in enumerate(nodes) if node is found[0]))
    first, last = endpoints
    if first >= last:
        raise ValueError("Pressure path preparation requires original Pre before Post")
    selection = prove_supported_path(analysis, routine, nodes[first:last+1], _ROOTS)
    excluded = [(node, name) for node, name in calls if name in _EXCLUDED]
    if {name for _node, name in excluded} != _EXCLUDED or len(excluded) != len(_EXCLUDED):
        raise ValueError("Pressure path preparation requires every exact original excluded solver edge")

    phi, rhs, called = (routine.scope.bindings.get(name) for name in ("phi", "rhs", "called"))
    if (any(binding is None or binding.declaring_scope is not routine.scope for binding in (phi, rhs, called))
            or any(binding.dtype != "real" or binding.kind != 8 or binding.rank != 3
                   or not {"save", "allocatable"} <= binding.attributes
                   or binding.attributes & {"pointer", "target", "optional", "volatile", "asynchronous"}
                   for binding in (phi, rhs))):
        raise ValueError("Pressure path preparation requires the original independent REAL64 SAVE arrays")
    initializers = [node for node in walk(routine.execution) if type(node).__name__ == "If_Construct"
                    and str(node.content[0]).replace(" ", "").lower() == "if(called==0)then"]
    if len(initializers) != 1:
        raise ValueError("Pressure path preparation requires the original first-call allocation guard")
    guard, = initializers
    significant = [node for node in guard.content[1:-1] if type(node).__name__ != "Comment"]
    if (len(significant) < 3 or [type(node).__name__ for node in significant[:3]] !=
            ["Allocate_Stmt", "Allocate_Stmt", "Assignment_Stmt"]
            or str(significant[2]).replace(" ", "").lower() not in {"phi=0", "phi=0.0"}
            or [str(node.items[1].items[0].items[0]).lower() for node in significant[:2]] != ["phi", "rhs"]):
        raise ValueError("Pressure path preparation requires exact original Phi/RHS allocation and Phi initialization")
    # No branch/loop/function substitution is inferred from observed settings.
    # The fresh compiler proof must check the private flag and all other edges.
    source = Path(original_source["path"])
    original_text = source.read_bytes().decode("utf-8")
    lines = original_text.splitlines(keepends=True)
    first_exec = next(node for node in nodes if type(node).__name__ != "Comment")
    declaration_anchor = first_exec.content[0] if type(first_exec).__name__ == "If_Construct" else first_exec
    insertions = [(_span(declaration_anchor)[0], "    logical, save :: " + _FLAG + " = .false.\n\n", "private declaration"),
                  (_span(guard.content[-1])[0], "      " + _FLAG + " = .true.\n", "first allocation epoch only")]
    for node, name in excluded:
        insertions.append((_span(node)[0], "        " + _FLAG + " = .false.\n", name))
    edited = list(lines)
    for line, text, _reason in sorted(insertions, key=lambda item: item[0], reverse=True):
        edited[line-1:line-1] = [text]
    replacement = "".join(edited)
    # Removing exactly the new insertions reconstructs every original byte.
    # Reconstruction uses the edit recipe below rather than text replacement,
    # so an equal-text original line can never be accidentally removed.
    offset = 0
    spans = []
    for line, text, reason in sorted(insertions, key=lambda item: item[0]):
        count = len(text.splitlines(keepends=True))
        spans.append({"original_before_line": line, "replacement_span": [line+offset, line+offset+count-1],
                      "text": text, "reason": reason})
        offset += count
    reverse = replacement.splitlines(keepends=True)
    for edit in reversed(spans):
        lower, upper = edit["replacement_span"]
        del reverse[lower-1:upper]
    if "".join(reverse) != original_text:
        raise ValueError("Pressure path preparation failed its exact original byte reconstruction")
    analysis.verify_publication()
    if file_identity(source) != original_source:
        raise ValueError("Original pressure source changed during path preparation")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / source.name
    destination.write_bytes(replacement.encode("utf-8"))
    record = {"version": 1, "kind": "explicit_supported_pressure_path_source_implementation", "entry": _ENTRY,
              "original": original_source, "replacement": file_identity(destination),
              "original_path_selection": selection.public(), "edits": spans,
              "eligibility": {"binding": _FLAG, "storage": "private local SAVE LOGICAL",
                  "initial": False, "first_fresh_allocation": True,
                  "excluded_calls": sorted(_EXCLUDED), "rearm_after_excluded_edge": False,
                  "scope": "restricts all managed allocations of this region; grants no alias or allocation facts"},
              "state_frame": "original PressureCorrection Phi/RHS/called/timers retained exactly once",
              "dispatch": "all original predicates and selected calls remain at original relative positions",
              "required_consumer": "exact fresh source path, typed sticky gate, independent canonical allocation/capture/effects guards; publish/close before excluded edge; never reopen while barred",
              "requires_fresh_configured_source_export": True,
              "ready_for_execution": False, "source_authority": False, "lifetime_authority": False,
              "payload_definition_authority": False, "execution_authority": False, "performance_claim": False}
    (directory / "source-implementation.json").write_text(json.dumps(record, indent=2) + "\n")
    analysis.verify_publication()
    _reference(original_source)
    return record


def prepare_supported_pressure_selection(source_paths, analysis_sources, directory, *, implementation):
    """Prepare an explicit selector from the replacement's *fresh* export.

    No capture facts are synthesized. The caller must supply its independent
    storage/alias/definition authority and merge this root selector with the
    separately prepared inner solver provider contract. The compiler then
    reissues the source path, gate, descriptor and execution plan itself.
    """
    from fparser.two.utils import walk

    from compiler.frontend.call_bindings import resolve_original_call
    from compiler.frontend.source_effects import SourceEffects
    from compiler.frontend.supported_paths import requested_supported_path
    from compiler.ir import CompilationError

    implementation = _reference(implementation)
    selected = json.loads(Path(implementation["path"]).read_text())
    if (selected.get("version") != 1 or selected.get("kind") != "explicit_supported_pressure_path_source_implementation"
            or selected.get("entry") != _ENTRY or selected.get("ready_for_execution") is not False
            or selected.get("payload_definition_authority") is not False
            or selected.get("source_authority") is not False or selected.get("lifetime_authority") is not False):
        raise ValueError("Supported pressure selection requires its exact explicit source implementation")
    _reference(selected["original"])
    replacement = _reference(selected["replacement"])
    if not isinstance(analysis_sources, dict):
        raise TypeError("Supported pressure selection requires a fresh configured-source export")
    analysis = SourceEffects(source_paths, analysis_sources=analysis_sources)
    analysis.seal_session()
    routine = analysis.routines.get(_ENTRY)
    if routine is None or str(routine.scope.path) != replacement["path"]:
        raise ValueError("Supported pressure selection must analyze the exact changed original frame")
    # Authenticate the implementation recipe without granting it authority.
    restored = Path(replacement["path"]).read_bytes().decode("utf-8").splitlines(keepends=True)
    texts = {"private declaration": "    logical, save :: " + _FLAG + " = .false.\n\n",
             "first allocation epoch only": "      " + _FLAG + " = .true.\n",
             **{name: "        " + _FLAG + " = .false.\n" for name in _EXCLUDED}}
    edits = selected.get("edits")
    if (not isinstance(edits, list) or len(edits) != len(texts)
            or any(not isinstance(edit, dict) or set(edit) !=
                   {"original_before_line", "replacement_span", "text", "reason"}
                   or not isinstance(edit["reason"], str) for edit in edits)
            or {edit["reason"] for edit in edits} != set(texts)):
        raise ValueError("Supported pressure source requires the exact six restriction-only insertions")
    offset, previous = 0, 0
    for edit in edits:
        line = edit["original_before_line"]
        if (type(line) is not int or line <= previous or edit["text"] != texts[edit["reason"]]
                or edit["replacement_span"] != [line+offset, line+offset+len(edit["text"].splitlines())-1]):
            raise ValueError("Supported pressure source insertion differs from the exact restriction recipe")
        offset += len(edit["text"].splitlines())
        previous = line
    for edit in reversed(selected["edits"]):
        first, last = edit["replacement_span"]
        if "".join(restored[first-1:last]) != edit["text"]:
            raise ValueError("Supported pressure source differs from its exact insertion recipe")
        del restored[first-1:last]
    if "".join(restored).encode("utf-8") != Path(selected["original"]["path"]).read_bytes():
        raise ValueError("Supported pressure source no longer reconstructs every original byte")

    def anchor(node, callee):
        item = node.item
        return {"span": list(getattr(item, "fort_original_span", item.span)),
                "sha256": sha256(str(node).encode()).hexdigest(), "callee": callee}

    calls = []
    for node in walk(routine.execution):
        if type(node).__name__ == "Call_Stmt":
            try:
                resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node)
            except CompilationError:
                continue
            calls.append((node, resolved.procedure))
    endpoints = []
    for name in (_ROOTS[0], _ROOTS[-1]):
        found = [(node, callee) for node, callee in calls if callee == name]
        if len(found) != 1:
            raise ValueError("Supported pressure selection requires the exact original Pre/Post endpoints")
        endpoints.append(anchor(*found[0]))
    barred = [anchor(node, callee) for node, callee in calls if callee in _EXCLUDED]
    if len(barred) != len(_EXCLUDED) or {value["callee"] for value in barred} != _EXCLUDED:
        raise ValueError("Supported pressure selection requires every exact excluded solver edge")
    request = {"version": 1, "source": replacement, "first": endpoints[0], "last": endpoints[1],
               "eligibility": {"binding": _FLAG, "resources": [routine.scope.bindings[name].root for name in ("phi", "rhs")],
                               "barred_calls": barred}}
    path, gate = requested_supported_path(analysis, routine, request, _ROOTS)
    selector = {"version": 4, "lifetime": "supported_original_path", "device_roots": list(_ROOTS), "path": request}
    contracts = {"schema_version": 1, "procedures": {_ENTRY: {"native_ancestor": selector}}}
    analysis.verify_publication()
    _reference(implementation)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    payloads = {"effect-contracts.json": contracts, "source-path-proof.json": path.public(),
                "source-gate-proof.json": gate.public()}
    for name, value in payloads.items():
        (directory / name).write_text(json.dumps(value, indent=2) + "\n")
    result = {"version": 1, "kind": "prepared_supported_pressure_path_selection", "entry": _ENTRY,
              "source_implementation": implementation, "replacement": replacement,
              "effect_contracts": file_identity(directory / "effect-contracts.json"),
              "source_path": file_identity(directory / "source-path-proof.json"),
              "source_gate": file_identity(directory / "source-gate-proof.json"),
              "capture_facts": "independent caller facts required; no default initialization or noescape inferred",
              "provider": "separately prepared exact inner solver contract required",
              "ready_for_execution": False, "source_authority": False, "payload_definition_authority": False,
              "execution_authority": False, "performance_claim": False}
    (directory / "prepared-selection.json").write_text(json.dumps(result, indent=2) + "\n")
    analysis.verify_publication()
    _reference(implementation)
    return result
