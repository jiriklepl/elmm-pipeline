"""Validate compiler-issued diagnostic plans without choosing execution.

The compiler retains the executable plan and its proof objects. This consumer
checks the versioned public record's integrity and source links only; operation,
definition, policy and cost facts do not authorize additional transformations.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


_PLAN_FIELDS = {
    "schema_version", "identity", "execution_identity", "entry", "sources", "source_inputs_identity",
    "configuration_identity", "capture_facts_identity", "procedures", "resources",
    "numerical_invocations", "gpu_leaves", "policy", "cost_requirements",
}
_PROCEDURE_FIELDS = {
    "procedure", "source", "source_sha256", "structured_summary_identity", "emission_identity",
    "units", "boundaries", "calls",
}
_UNIT_FIELDS = {
    "ordinal", "selected_node_ids", "first_line", "last_line", "resources", "ordered_definitions", "operations",
}
_APPLICABILITY_FIELDS = {
    "schema_version", "kind", "identity", "plan_identity", "execution_identity",
    "entry", "sources", "source_inputs_identity", "configuration_identity", "capture_facts_identity",
    "canonical_resources", "key_candidates", "unclassified_dependencies", "original_control_trees",
    "native_operations", "numerical_operations", "requirements", "existing_exact_window",
    "placement_authorized", "execution_authorized", "existing_placement_unchanged", "identity_policy", "coverage",
}


def _require(condition, reason):
    if not condition:
        raise ValueError("Compiler checked execution plan " + reason)


def _text(value):
    return isinstance(value, str) and bool(value)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Compiler checked execution plan requires finite JSON facts") from error
    return hashlib.sha256(encoded).hexdigest()


def _source_inputs_identity(manifest):
    analysis = manifest.get("analysis_sources")
    _require(analysis is None or isinstance(analysis, dict), "requires configured source metadata")
    if analysis is None:
        return _digest({"schema_version": 1, "sources": manifest["source_inputs"],
                        "dependencies": {}, "configuration": None, "entries": []})
    entries = analysis.get("entries")
    _require(isinstance(entries, list) and all(isinstance(item, dict) and _text(item.get("source"))
                                              for item in entries), "requires configured source entries")
    _require(analysis.get("source_inputs") == manifest["source_inputs"], "configured source inputs differ")
    return _digest({"schema_version": 1, "sources": manifest["source_inputs"],
                    "dependencies": analysis.get("dependencies"), "configuration": analysis.get("configuration"),
                    "entries": sorted(entries, key=lambda item: item["source"])})


def validate_checked_execution_plans(manifest, *, capture_facts=None):
    """Check optional v1 records in existing scope-manifest schema 1.

    Missing records preserve compatibility with earlier compiler packages.
    Capture facts are checked against the request when available. Their compact
    plan identity differs from the outer manifest's legacy JSON hash recipe.
    Saved packages retain both identities unchanged and remain subject to the
    existing saved-manifest, artifact and source verification before application.
    """
    scopes = manifest.get("scopes", [])
    _require(isinstance(scopes, list), "requires a list of source scopes")
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        _require("interval_applicability" not in scope or "checked_execution_plan" in scope,
                 "interval applicability requires its checked plan")
        if "checked_execution_plan" not in scope:
            continue
        _validate_plan(scope["checked_execution_plan"], manifest, scope, capture_facts)
        if "interval_applicability" in scope:
            _validate_applicability(scope["interval_applicability"], scope["checked_execution_plan"], scope)


def _validate_plan(plan, manifest, scope, capture_facts):
    _require(isinstance(plan, dict) and set(plan) == _PLAN_FIELDS, "requires the complete v1 schema")
    _require(type(plan["schema_version"]) is int and plan["schema_version"] == 1,
             "has an unsupported schema version")
    _require(all(_sha256(plan[key]) for key in ("identity", "execution_identity", "source_inputs_identity",
                                               "configuration_identity", "capture_facts_identity")),
             "requires SHA256 identities")
    _require(plan["identity"] == _digest({key: value for key, value in plan.items() if key != "identity"}),
             "identity differs from its contents")
    sources = plan["sources"]
    _require(isinstance(sources, dict) and bool(sources)
             and all(_text(path) and Path(path).is_absolute() and _sha256(digest)
                     for path, digest in sources.items())
             and sources == manifest.get("source_inputs"), "original source identities differ")
    _require(plan["source_inputs_identity"] == _source_inputs_identity(manifest),
             "configured source identity differs")
    if capture_facts is not None:
        _require(plan["capture_facts_identity"] == _digest(capture_facts), "capture facts identity differs")
    _require(_text(plan["entry"]) and plan["entry"] == manifest.get("entry"), "entry differs from its manifest")

    resources = plan["resources"]
    _require(isinstance(resources, list), "requires canonical resource records")
    roots, registrations = set(), set()
    for resource in resources:
        _require(isinstance(resource, dict) and set(resource) == {"resource", "registration_identity", "descriptor"}
                 and _text(resource["resource"]) and isinstance(resource["descriptor"], dict),
                 "requires canonical resource descriptors")
        number = resource["registration_identity"]
        _require(type(number) is int and number > 0 and number not in registrations
                 and resource["resource"] not in roots, "repeats or invalidates canonical registrations")
        roots.add(resource["resource"])
        registrations.add(number)

    procedures = plan["procedures"]
    _require(isinstance(procedures, list) and bool(procedures), "requires original procedure records")
    names = set()
    for procedure in procedures:
        _require(isinstance(procedure, dict) and set(procedure) == _PROCEDURE_FIELDS,
                 "requires complete procedure records")
        name, source = procedure["procedure"], procedure["source"]
        _require(_text(name) and name not in names, "repeats or invalidates an original procedure")
        names.add(name)
        _require(_text(source) and source in sources and procedure["source_sha256"] == sources[source]
                 and _sha256(procedure["structured_summary_identity"]), "procedure source linkage differs")
        _require(_sha256(procedure["emission_identity"]), "requires a procedure emission SHA256 identity")
        _require(all(isinstance(procedure[key], list) and all(isinstance(item, dict) for item in procedure[key])
                     for key in ("boundaries", "calls")), "requires native boundary and call records")
        units = procedure["units"]
        _require(isinstance(units, list), "requires ordered procedure units")
        for ordinal, unit in enumerate(units):
            _require(isinstance(unit, dict) and set(unit) == _UNIT_FIELDS, "requires complete unit records")
            _require(type(unit["ordinal"]) is int and unit["ordinal"] == ordinal, "unit ordinals are not ordered")
            _require(isinstance(unit["selected_node_ids"], list)
                     and all(_text(node) for node in unit["selected_node_ids"]), "requires original node identities")
            _require(type(unit["first_line"]) is int and type(unit["last_line"]) is int
                     and 1 <= unit["first_line"] <= unit["last_line"], "requires original ordered line spans")
            _require(isinstance(unit["resources"], dict)
                     and all(_text(local) and _text(root) and root in roots
                             for local, root in unit["resources"].items()), "unit canonical resource linkage differs")
            _require(isinstance(unit["ordered_definitions"], dict) and isinstance(unit["operations"], dict),
                     "requires definition and operation records")
    root = procedures[0]
    _require(root["procedure"] == plan["entry"] and root["source"] == scope.get("path"),
             "root procedure differs from its source scope")
    authority = scope.get("source_authority")
    _require(isinstance(authority, dict)
             and authority.get("structured_summary_identity") == root["structured_summary_identity"],
             "root summary differs from its source scope")
    _require(type(plan["numerical_invocations"]) is int and plan["numerical_invocations"] >= 0,
             "requires a nonnegative numerical invocation count")
    _require(isinstance(plan["gpu_leaves"], list) and all(_text(leaf) for leaf in plan["gpu_leaves"])
             and len(set(plan["gpu_leaves"])) == len(plan["gpu_leaves"]), "requires distinct numerical leaf identities")
    _require(isinstance(plan["policy"], dict), "requires compiler policy facts")
    _require(isinstance(plan["cost_requirements"], list)
             and all(isinstance(item, dict) and _text(item.get("kind")) and _text(item.get("reason"))
                     for item in plan["cost_requirements"]), "requires explanatory cost requirements")


def _validate_applicability(record, plan, scope):
    """Transport explanatory key candidates, never accept them as calibration."""
    _require(isinstance(record, dict) and set(record) == _APPLICABILITY_FIELDS,
             "interval applicability requires the complete v1 schema")
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1
             and record["kind"] == "checked-interval-cost-applicability",
             "interval applicability has an unsupported schema version or kind")
    _require(_sha256(record["identity"])
             and record["identity"] == _digest({key: value for key, value in record.items() if key != "identity"}),
             "interval applicability identity differs from its contents")
    _require(record["plan_identity"] == plan["identity"], "interval applicability plan identity differs")
    for key in ("execution_identity", "entry", "sources", "source_inputs_identity", "configuration_identity",
                "capture_facts_identity"):
        _require(record[key] == plan[key], "interval applicability " + key + " differs")
    _require(record["canonical_resources"] == plan["resources"],
             "interval applicability canonical resources differ")
    _require(record["placement_authorized"] is False and record["execution_authorized"] is False
             and record["existing_placement_unchanged"] is True,
             "interval applicability cannot authorize execution or placement")
    exact = scope.get("exact_window")
    _require(exact is None or isinstance(exact, dict), "requires a structured exact-window attachment")
    projection = None if exact is None else {key: exact[key] for key in (
        "available", "selection", "reason", "contract_sha256", "evidence_sha256", "capture") if key in exact}
    _require(record["existing_exact_window"] == projection,
             "interval applicability cannot add or replace an exact-window attachment")
    _require(all(_text(record[key]) for key in ("identity_policy", "coverage")),
             "interval applicability requires its evidence limitations")
    for key in ("key_candidates", "unclassified_dependencies", "original_control_trees", "native_operations",
                "numerical_operations", "requirements"):
        _require(isinstance(record[key], list) and all(isinstance(item, dict) for item in record[key]),
                 "interval applicability requires structured " + key)
    for candidate in record["key_candidates"]:
        _require(_text(candidate.get("resource")) and _text(candidate.get("kind"))
                 and candidate.get("safe_evaluation_authorized") is False
                 and isinstance(candidate.get("properties"), list)
                 and isinstance(candidate.get("missing_proofs"), list),
                 "interval applicability candidates cannot authorize runtime evaluation")
    for requirement in record["requirements"]:
        _require(_text(requirement.get("kind")) and _text(requirement.get("reason"))
                 and requirement.get("status") == "not_established",
                 "interval applicability requires unresolved explanatory requirements")
    procedures = {item["procedure"]: item for item in plan["procedures"]}
    for key in ("original_control_trees", "native_operations", "numerical_operations"):
        for item in record[key]:
            procedure, ordinal = item.get("procedure"), item.get("unit")
            _require(_text(procedure) and procedure in procedures and type(ordinal) is int
                     and 0 <= ordinal < len(procedures[procedure]["units"]),
                     "interval applicability operation is outside its checked interval")
