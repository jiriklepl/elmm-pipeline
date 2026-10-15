"""Authenticate compiler-issued call observations without granting placement authority."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_FIELDS = {
    "schema_version", "protocol", "variant", "identity", "execution_identity",
    "source_inputs_identity", "configuration_identity", "capture_facts_identity",
    "runtime_identity", "compiler_identity", "call_sites", "placement_authority",
    "automatic_estimate_available", "calibration_evidence", "interval", "clock",
    "observation_prefix", "limitations",
}
_CALL_FIELDS = {"identity", "caller", "source", "source_sha256", "first_line", "last_line", "call_sha256"}
_LIMITATIONS = [
    "No runtime workload key; observations are not automatic placement evidence.",
    "Resident variant may execute guarded native fallbacks; verify runtime traces separately.",
    "Only listed direct call sites in registered subroutines of the authenticated supplied source graph are observed.",
]


def _require(condition, reason):
    if not condition:
        raise ValueError("Compiler owner measurement " + reason)


def _sha256(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _digest(value):
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError("Compiler owner measurement requires finite JSON facts") from error
    return hashlib.sha256(encoded).hexdigest()


def validate_owner_measurements(manifest):
    """Validate the optional v1 diagnostic contract and return its variant.

    The checked plan remains compiler authority. This consumer verifies source,
    original call spans and public identities; it never creates a schedule or
    treats observed durations as calibration or automatic placement evidence.
    """
    variant = manifest.get("scope_measurement")
    scopes = manifest.get("scopes", [])
    _require(isinstance(scopes, list) and all(isinstance(scope, dict) for scope in scopes),
             "requires source scope records")
    records = [scope["owner_measurement"] for scope in scopes if "owner_measurement" in scope]
    if variant is None:
        _require(not records, "requires an explicit top-level variant")
        return None
    _require(variant in ("native", "resident"), "has an unsupported variant")
    _require(manifest.get("scope_execution") == "reached" and type(manifest.get("scope_count")) is int
             and manifest.get("scope_count") == len(scopes) == len(records) == 1,
             "requires one admitted reached owner")
    _require(manifest.get("automatic_estimate_available") is False,
             "cannot supply automatic placement estimates")
    scope, record = scopes[0], records[0]
    _require(isinstance(record, dict) and set(record) == _FIELDS, "requires the complete v1 schema")
    _require(type(record["schema_version"]) is int and record["schema_version"] == 1
             and record["protocol"] == "complete-lexical-call-clock-v1", "has an unsupported protocol")
    _require(record["variant"] == variant, "variant differs from its manifest")
    _require(all(record[key] is False for key in (
        "placement_authority", "automatic_estimate_available", "calibration_evidence")),
        "cannot supply placement or calibration authority")
    _require(record["interval"] == "before original call argument evaluation through return and finalization"
             and record["clock"] == "system_clock:int64"
             and record["observation_prefix"] == "FORT_SCOPE_MEASUREMENT "
             and record["limitations"] == _LIMITATIONS, "requires the declared diagnostic clock boundaries")
    identity_fields = ("identity", "execution_identity", "source_inputs_identity", "configuration_identity",
                       "capture_facts_identity", "runtime_identity", "compiler_identity")
    _require(all(_sha256(record[key]) for key in identity_fields), "requires SHA256 identities")
    _require(record["identity"] == _digest({key: value for key, value in record.items()
                                          if key not in {"identity", "variant"}}),
             "identity differs from its contents")
    plan = scope.get("checked_execution_plan")
    _require(isinstance(plan, dict) and all(record[key] == plan.get(key) for key in (
        "execution_identity", "source_inputs_identity", "configuration_identity", "capture_facts_identity")),
        "identity differs from the checked execution plan")
    if variant == "resident":
        _require(isinstance(manifest.get("runtime"), dict)
                 and record["runtime_identity"] == _digest(manifest["runtime"]),
                 "runtime identity differs from resident artifacts")

    sources = manifest.get("source_inputs", {})
    calls = record["call_sites"]
    _require(isinstance(calls, list) and bool(calls), "requires authenticated original call sites")
    identities, call_sources, text = set(), set(), {}
    for call in calls:
        _require(isinstance(call, dict) and set(call) == _CALL_FIELDS, "requires complete call-site records")
        _require(all(_sha256(call[key]) for key in ("identity", "source_sha256", "call_sha256"))
                 and call["identity"] not in identities, "requires distinct call-site identities")
        _require(call["identity"] == _digest({key: value for key, value in call.items() if key != "identity"}),
                 "call-site identity differs from its contents")
        source = call["source"]
        _require(isinstance(source, str) and source in sources and Path(source).is_absolute()
                 and call["source_sha256"] == sources[source]
                 and isinstance(call["caller"], str) and bool(call["caller"]),
                 "call-site source linkage differs")
        first, last = call["first_line"], call["last_line"]
        _require(type(first) is int and type(last) is int and 1 <= first <= last,
                 "requires original call line spans")
        if source not in text:
            try:
                raw = Path(source).read_bytes()
                _require(hashlib.sha256(raw).hexdigest() == sources[source], "original call source changed")
                text[source] = raw.decode().splitlines(keepends=True)
            except (OSError, UnicodeError) as error:
                raise ValueError("Compiler owner measurement original call source unavailable") from error
        _require(last <= len(text[source])
                 and hashlib.sha256("".join(text[source][first - 1:last]).encode()).hexdigest() == call["call_sha256"],
                 "original call text differs")
        identities.add(call["identity"])
        call_sources.add(source)
    replacements = manifest.get("sources", {})
    _require(isinstance(replacements, dict) and call_sources <= replacements.keys(),
             "call sites lack compiler-issued source replacements")
    if variant == "native":
        _require(manifest.get("runtime") is None, "native variant must not declare a resident runtime")
        _require(set(replacements) == call_sources, "native variant may replace only original callers")
        units = manifest.get("build_sources")
        _require(isinstance(units, list) and bool(units) and all(
            isinstance(unit, dict) and unit.get("language") == "fortran"
            and unit.get("role") == "original_source" for unit in units),
            "native variant may build only compiler-issued original caller sources")
        _require({unit.get("path") for unit in units} == {item.get("replacement") for item in replacements.values()},
                 "native build inputs differ from caller replacements")
    return variant
