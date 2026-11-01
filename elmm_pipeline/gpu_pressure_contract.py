"""Issue fresh source inputs for the explicitly qualified PoisFFT adapter.

This is an opt-in adapter for the pinned ELMM/PoisFFT sources. It packages a
contract, not a transformed source or an execution plan. The public compiler
must issue its own checked plan again when these inputs are consumed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from .gpu_pressure_build import prepare_poisfft_provider_manifest

_CALLER = "poissonsolvers::poiss_poisfft"
_CALLEE = "poisfft_dp::poisfft_solver3d__execute"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _reference(reference):
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256"}
            or not isinstance(reference["path"], str) or not Path(reference["path"]).is_absolute()
            or not isinstance(reference["sha256"], str) or not _SHA.fullmatch(reference["sha256"])
            or _digest(reference["path"]) != reference["sha256"]):
        raise ValueError("Pressure source preparation requires an exact qualification reference")
    return json.loads(Path(reference["path"]).read_text())


def _same_reference(record, reference, *, key="path"):
    return (isinstance(record, dict) and record.get(key) == reference[key]
            and record.get("sha256") == reference["sha256"])


def _qualified_sources(analysis, routine, opaque, numerical, backend, opaque_reference):
    """Match bytes at fresh paths, never copy old source identities to them."""
    source = opaque["source_contract"]
    caller = source["caller"]
    if (caller["procedure"] != _CALLER or routine.qualified != _CALLER
            or analysis.sources[str(routine.scope.path)] != caller["sha256"]):
        raise ValueError("Fresh pressure wrapper differs from its explicit source qualification")
    backend_reference = {"path": backend["manifest"], "sha256": backend["sha256"]}
    if (not _same_reference(numerical.get("source_and_receiver_observability_qualification"), opaque_reference)
            or not _same_reference(numerical.get("prepared_backend"), backend_reference)):
        raise ValueError("Pressure numerical qualification differs from the receiver or prepared backend")
    identities = {**analysis.sources, **analysis.inputs.dependencies}
    for expected in source["pinned_reference"]["sources"]:
        expected_path = Path(expected["path"])
        suffix = expected_path.parts[-3:] if expected_path.parent.name == "src" else (expected_path.name,)
        matches = [digest for path, digest in identities.items() if Path(path).parts[-len(suffix):] == suffix]
        if not matches or any(digest != expected["sha256"] for digest in matches):
            raise ValueError("Fresh PoisFFT source/include differs from the explicitly qualified library")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _entry_read_prerequisite(reference, analysis, opaque, task, provider):
    """Check an explicit reviewed library premise, then bind it to fresh inputs.

    This is a trusted source/library precondition. Numerical agreement and
    provider may-read permissions cannot create it. The compiler independently
    authenticates the resulting exact original CALL and provider binding.
    """
    from compiler.frontend.external_entry_reads import (
        _CLAIM,
        _FLAGS,
        provider_read_binding,
        requirement_scope,
    )

    record = _reference(reference)
    keys = {"version", "kind", "claim", "authority", "pinned_sources", "implementation_sha256",
            "callee", "metadata_query", "required_reads", "layout_equalities", "configuration", "evidence"}
    required = [{"formal": "rhs", "lower": [1] * 3, "upper": [
        {"descriptor": "size", "formal": "rhs", "axis": axis, "offset": -1} for axis in range(1, 4)]}]
    guards = [{"component": "solver%sizes", "index": axis, "descriptor": {
        "descriptor": "size", "formal": formal, "axis": axis, "offset": offset}}
        for formal, offset in (("rhs", -2), ("phi", -4)) for axis in range(1, 4)]
    configuration = {"required_defines": ["DPREC"], "forbidden_defines": ["PAR", "MPI"]}
    if (type(record) is not dict or set(record) != keys
            or type(record["version"]) is not int or record["version"] != 1
            or record["kind"] != "reviewed_pinned_poisfft_entry_read_prerequisite"
            or record["claim"] != _CLAIM
            or type(record["authority"]) is not dict or set(record["authority"]) != set(_FLAGS)
            or any(record["authority"][key] is not value for key, value in _FLAGS.items())
            or type(record["pinned_sources"]) is not list or not record["pinned_sources"]
            or _encoded(record["pinned_sources"]) != _encoded(opaque["source_contract"]["pinned_reference"]["sources"])
            or record["implementation_sha256"] != opaque["implementation"]["sha256"]
            or _encoded(record["callee"]) != _encoded({"procedure": task["source"]["callee"],
                "body_sha256": task["source"]["callee_body_sha256"]})
            or _encoded(record["metadata_query"]) != _encoded(provider["metadata_query"])
            or _encoded(record["required_reads"]) != _encoded(required)
            or _encoded(record["layout_equalities"]) != _encoded(guards)
            or _encoded(record["configuration"]) != _encoded(configuration)):
        raise ValueError("PoisFFT entry reads require the exact reviewed unconditional library prerequisite")
    defines = analysis.inputs.public()["configuration"].get("defines", [])
    if (not isinstance(defines, list) or any(not isinstance(item, str) for item in defines)
            or "DPREC" not in {item.split("=", 1)[0] for item in defines}
            or {"PAR", "MPI"} & {item.split("=", 1)[0] for item in defines}):
        raise ValueError("PoisFFT entry reads require configured DPREC without PAR or MPI")
    if type(record["evidence"]) is not list or not record["evidence"]:
        raise ValueError("PoisFFT entry reads require explicit reviewed source/library evidence")
    for evidence in record["evidence"]:
        _reference(evidence)
    # The selected body must come from the authenticated lazy implementation,
    # not merely another module/procedure with the same public spelling.
    callee = analysis.routines[task["source"]["callee"]]
    implementation = opaque["implementation"]
    if (str(callee.scope.path) != implementation["path"]
            or analysis.sources[str(callee.scope.path)] != implementation["sha256"]):
        raise ValueError("PoisFFT entry reads require the selected qualified lazy implementation")
    return {"version": 1, "kind": "qualified_original_external_entry_reads", "claim": _CLAIM,
            "authority": dict(_FLAGS), "source_inputs": analysis.inputs.identity(),
            "task_scope": requirement_scope(task), "metadata_query": provider["metadata_query"],
            "provider_binding": provider_read_binding(provider), "required_reads": required,
            "layout_equalities": guards, "evidence": [dict(reference), *record["evidence"]]}


def prepare_poisfft_contract(source_paths, analysis_sources, directory, *, backend, fortran,
                             numerical_qualification, opaque_qualification, workspace_cap_bytes):
    """Prepare one fresh forced-sections pressure root without compiling.

    ``analysis_sources`` is the authentic configured-source export for these
    exact original paths, including the real PoisFFT sources/includes. Only
    the inner Execute is contracted. Dimensions, lengths, boundaries and FD
    order come from the reached typed query of the original SAVE receiver;
    the caller supplies a workspace cap, not guessed numerical parameters.
    """
    return _prepare_poisfft_contract(source_paths, analysis_sources, directory, backend=backend, fortran=fortran,
        numerical_qualification=numerical_qualification, opaque_qualification=opaque_qualification,
        workspace_cap_bytes=workspace_cap_bytes, callee=_CALLEE)


def _prepare_poisfft_contract(source_paths, analysis_sources, directory, *, backend, fortran,
                              numerical_qualification, opaque_qualification, workspace_cap_bytes, callee,
                              required_entry_read_prerequisite=None, provider_only_lifetime=None,
                              provider_retention_prerequisite=None):
    """Shared mechanics; public producers fix and authenticate the implementation."""
    from fparser.two.utils import walk

    from compiler.frontend.call_bindings import resolve_original_call
    from compiler.frontend.device_tasks import external_device_task, task_scope_identity
    from compiler.frontend.source_effects import SourceEffects

    if type(workspace_cap_bytes) is not int or not 1 <= workspace_cap_bytes < (1 << 63):
        raise ValueError("Pressure workspace cap requires a positive signed 64-bit byte count")
    if not isinstance(analysis_sources, dict):
        raise TypeError("Pressure source preparation requires an authentic configured-source export")
    prerequisite = (None if required_entry_read_prerequisite is None
                    else dict(required_entry_read_prerequisite))
    lifetime_request = None if provider_only_lifetime is None else json.loads(_encoded(provider_only_lifetime))
    if lifetime_request is not None and (
            prerequisite is None or type(lifetime_request) is not dict
            or set(lifetime_request) != {"version", "call_path"}
            or type(lifetime_request["version"]) is not int or lifetime_request["version"] != 1
            or type(lifetime_request["call_path"]) is not list or len(lifetime_request["call_path"]) < 2
            or any(type(name) is not str or name != name.lower() for name in lifetime_request["call_path"])
            or len(set(lifetime_request["call_path"])) != len(lifetime_request["call_path"])
            or lifetime_request["call_path"][-1] != _CALLER):
        raise ValueError("PoisFFT provider-only lifetime requires an explicit parent call-path request and reviewed entry reads")
    if (lifetime_request is None) != (provider_retention_prerequisite is None):
        raise ValueError("provider-only lifetime requires its explicit retained-state prerequisite")
    opaque = _reference(opaque_qualification)
    numerical = _reference(numerical_qualification)
    from compiler.frontend.provider_lifetime import program_roots
    metadata = program_roots(lifetime_request)
    analysis = SourceEffects(source_paths, analysis_sources=analysis_sources,
                             **({"program_roots": metadata} if metadata else {}))
    routine = analysis.routines.get(_CALLER)
    if routine is None:
        raise ValueError("Fresh configured sources do not contain the qualified pressure wrapper")
    _qualified_sources(analysis, routine, opaque, numerical, backend, opaque_qualification)
    if lifetime_request is not None and (
            len(lifetime_request["call_path"]) > analysis.procedure_limit
            or len(lifetime_request["call_path"])-1 > analysis.depth_limit
            or any(name not in analysis.routines
                   or analysis.routines[name].source_kind != ("program" if index == 0 and name.startswith("$program::") else "module")
                   for index, name in enumerate(lifetime_request["call_path"]))):
        raise ValueError("PoisFFT provider-only lifetime request has unavailable original members or exceeds source budgets")
    calls = [node for node in walk(routine.execution)
             if type(node).__name__ == "Call_Stmt" and str(node.items[0]).lower() == "execute"]
    if len(calls) != 1:
        raise ValueError("Pressure adapter requires one exact original inner Execute call")
    node, = calls
    resolved = resolve_original_call(analysis, analysis.source_scope_for(node, routine.scope), node)
    if resolved.callee.qualified != callee or tuple(item.formal for item in resolved.associations) != ("d", "phi", "rhs"):
        raise ValueError("Pressure adapter requires the original double precision PoisFFT Execute interface")
    receiver, phi, rhs = resolved.associations
    constructors = [item for item in walk(routine.execution) if type(item).__name__ == "Assignment_Stmt"
                    and type(item.items[0]).__name__ == "Name"
                    and analysis._binding(analysis.source_scope_for(item, routine.scope), item.items[0]) is receiver.binding]
    if len(constructors) != 1:
        raise ValueError("Pressure adapter requires one authenticated configured receiver constructor")
    constructor, = constructors
    if (any(item.binding is None or item.binding.rank != 3 or item.binding.dtype != "real"
            or item.binding.kind != 8 or item.binding.attributes & {"allocatable", "pointer", "target", "optional"}
            for item in (phi, rhs)) or phi.binding.root == rhs.binding.root):
        raise ValueError("Pressure adapter requires independent ordinary rank-three REAL64 entry arrays")

    def span(item):
        result = getattr(item.item, "fort_original_span", item.item.span)
        if result is None:
            raise ValueError("Pressure source operation lacks an exact original span")
        return list(result)

    def descriptor(name, formal, axis, offset):
        return {"descriptor": name, "formal": formal, "axis": axis, "offset": offset}

    task = {"version": 2, "source": {
        "sha256": analysis.sources[str(routine.scope.path)], "span": span(node),
        "call_sha256": hashlib.sha256(str(node).encode()).hexdigest(), "callee": resolved.callee.qualified,
        "callee_body_sha256": hashlib.sha256(str(resolved.callee.scope.node).encode()).hexdigest()},
        "provider_manifest": {}, "resources": [
            {"formal": "phi", "effects": [{"kind": "overwrite", "lower": [2] * 3,
                "upper": [descriptor("size", "phi", axis, -2) for axis in range(1, 4)]}]},
            {"formal": "rhs", "effects": [{"kind": "read", "lower": [1] * 3,
                "upper": [descriptor("size", "rhs", axis, -1) for axis in range(1, 4)]}]}],
        "configuration": [
            {"component": "solver%device_budget_bytes", "kind": "size", "values": [workspace_cap_bytes]},
            *({"component": formal + "_lower_bounds", "kind": "int64",
               "values": [descriptor("lbound", formal, axis, 0) for axis in range(1, 4)]}
              for formal in ("phi", "rhs"))],
        "opaque_arguments": [{"formal": "d", "constructor": {"span": span(constructor),
            "statement_sha256": hashlib.sha256(str(constructor).encode()).hexdigest()},
            "qualification": dict(opaque_qualification)}], "completion": "wait_without_export"}
    analysis.inputs.verify_live()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    task["provider_manifest"] = prepare_poisfft_provider_manifest(
        backend, directory / "provider.json", fortran=fortran, qualification=numerical_qualification,
        original_body_sha256=task["source"]["callee_body_sha256"], task_contract_sha256=task_scope_identity(task))
    required_record = None
    if prerequisite is not None:
        initial_provider = _reference(task["provider_manifest"])
        required_record = _entry_read_prerequisite(prerequisite, analysis, opaque, task, initial_provider)
        required_path = directory / "required-entry-reads.json"
        required_path.write_text(json.dumps(required_record, indent=2) + "\n")
        task["entry_read_requirements"] = {"path": str(required_path), "sha256": _digest(required_path)}
        # The task hash includes the new prerequisite. Only this final provider
        # and the subsequently issued source proof may be returned as authority.
        task["provider_manifest"] = prepare_poisfft_provider_manifest(
            backend, directory / "provider-entry-reads.json", fortran=fortran, qualification=numerical_qualification,
            original_body_sha256=task["source"]["callee_body_sha256"], task_contract_sha256=task_scope_identity(task))
        final_provider = _reference(task["provider_manifest"])
        from compiler.frontend.external_entry_reads import provider_read_binding
        if _encoded(provider_read_binding(final_provider)) != _encoded(required_record["provider_binding"]):
            raise ValueError("PoisFFT provider changed while binding the entry-read prerequisite")
    retained_reference = None
    if lifetime_request is not None:
        from compiler.frontend.provider_lifetime import validate_retained_state_record
        retained_provider = _reference(task["provider_manifest"])
        original_state = _reference(provider_retention_prerequisite)
        validate_retained_state_record(original_state, retained_provider, prerequisite=True)
        state = {**original_state, "kind": "external_provider_retained_plan_state",
                 "prerequisite": dict(provider_retention_prerequisite)}
        state_path = directory / "retained-provider-state.json"
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        retained_reference = {"path": str(state_path), "sha256": _digest(state_path)}
        retained_provider["retained_state_qualification"] = retained_reference
        provider_path = Path(task["provider_manifest"]["path"])
        provider_path.write_text(json.dumps(retained_provider, indent=2) + "\n")
        task["provider_manifest"] = {"path": str(provider_path), "sha256": _digest(provider_path)}
    final_provider_reference = dict(task["provider_manifest"])
    required_reference = None if prerequisite is None else dict(task["entry_read_requirements"])
    contracts = {_CALLER: {"device_tasks": [task]}}
    analysis.contracts = contracts
    proof = external_device_task(analysis, routine, node)
    # Neither INTENT(IN) nor an interface establishes caller definitions.
    # Independent caller/source-position evidence must supply any readable
    # coverage, including the RHS interior; unused ghost bytes remain unknown.
    # The original inner Phi INTENT(OUT) kill remains mandatory.
    fact = {"storage": "stable", "escapes": False, "allocation_changes": False}
    facts = {"schema_version": 1, "participation": "serial", "sources": dict(analysis.sources),
             "captures": {phi.binding.root: {**fact, "initialized": "none"},
                          rhs.binding.root: {**fact, "initialized": "none"}}}
    if lifetime_request is not None:
        # This is an explicit selection request, not a source/lifetime proof.
        # ScopeBuilder must issue the original-call selection on its own sealed
        # session before any native companion or provider-only owner is emitted.
        facts["provider_only_lifetime"] = lifetime_request
    payloads = {"effect_contracts.json": {"schema_version": 1, "procedures": contracts},
                "capture_facts.json": facts, "source-proof.json": proof.public(),
                "analysis-sources.json": analysis.inputs.public()}
    for name, value in payloads.items():
        with (directory / name).open("x") as stream:
            stream.write(json.dumps(value, indent=2) + "\n")
    analysis.inputs.verify_live()
    _reference(opaque_qualification)
    _reference(numerical_qualification)
    if prerequisite is not None:
        if (task["provider_manifest"] != final_provider_reference
                or task["entry_read_requirements"] != required_reference):
            raise ValueError("PoisFFT final provider or prerequisite reference changed during source preparation")
        _reference(prerequisite)
        for evidence in required_record["evidence"]:
            _reference(evidence)
        if _encoded(_reference(task["entry_read_requirements"])) != _encoded(required_record):
            raise ValueError("PoisFFT entry-read prerequisite changed during source preparation")
        _reference(task["provider_manifest"])
        analysis.inputs.verify_live()
    if retained_reference is not None:
        _reference(provider_retention_prerequisite)
        if _encoded(_reference(retained_reference)) != _encoded(state):
            raise ValueError("provider-only retained-state qualification changed during preparation")
        validate_retained_state_record(state, _reference(task["provider_manifest"]))
        analysis.inputs.verify_live()
    def reference(name):
        return {"path": str(directory / name), "sha256": _digest(directory / name)}
    result = {"schema_version": 1, "kind": "prepared_explicit_poisfft_source_contract",
              "entry": _CALLER, "source": str(routine.scope.path), "source_files": [str(path) for path in analysis.inputs.paths
                  if str(path) != str(routine.scope.path)], "gpu_policy": "sections", "scope_execution": "graph",
              "effect_contracts": reference("effect_contracts.json"), "capture_facts": reference("capture_facts.json"),
              "analysis_sources": reference("analysis-sources.json"),
              "source_proof": reference("source-proof.json"), "provider_manifest": task["provider_manifest"],
              "pressure_backend": dict(backend), "performance_claim": False,
              "execution_qualification": "requires fresh compiler plan, actual correctness gate and observed device execution"}
    if prerequisite is not None:
        result["required_entry_read_prerequisite"] = prerequisite
        result["entry_read_requirements"] = dict(task["entry_read_requirements"])
    if lifetime_request is not None:
        result["provider_retention_prerequisite"] = dict(provider_retention_prerequisite)
        result["retained_state_qualification"] = retained_reference
        result["provider_only_lifetime_request"] = lifetime_request
        result["provider_only_lifetime_qualification"] = "requires compiler-issued exact call spine and complete per-call field publication"
    with (directory / "prepared-contract.json").open("x") as stream:
        stream.write(json.dumps(result, indent=2) + "\n")
    return result
