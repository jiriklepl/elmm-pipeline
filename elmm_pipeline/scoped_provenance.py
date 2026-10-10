"""Attribute observed scope operations through compiler-published identities.

These counters describe API submissions and coherence commits. They do not
measure device duration, prove overlap, or reconstruct an execution policy.
"""
from __future__ import annotations

import re

FIELDS = ("owner", "procedure", "segment", "operation", "implementation", "boundary")
OBSERVED = frozenset({"create", "initialize", "allocate", "upload", "download", "launch", "wait",
    "release", "close", "device_commit", "host_commit", "reconcile_device", "reconcile_host",
    "forget_definition", "forget_sections", "scratch_allocate", "scratch_free", "scratch_acquire",
    "scratch_release"})
HASH = re.compile(r"[0-9a-f]{64}\Z")
OPERATION_KINDS = frozenset({"source_call", "borrowed_call", "numerical_segment", "native_operation"})


def _registry(manifests):
    records, supported = {}, False
    for manifest in manifests:
        mapping = manifest.get("runtime_provenance")
        if mapping is None:
            continue
        if not isinstance(mapping, dict) or type(mapping.get("schema_version")) is not int or mapping["schema_version"] != 1:
            raise ValueError("unsupported public runtime provenance version")
        if not isinstance(mapping.get("records"), list):
            raise TypeError("public runtime provenance requires identity records")
        supported = True
        local = set()
        for record in mapping["records"]:
            if not isinstance(record, dict) or not HASH.fullmatch(str(record.get("id", ""))) or not isinstance(record.get("kind"), str):
                raise ValueError("invalid public runtime provenance identity")
            identity = record["id"]
            if identity in local:
                raise ValueError("duplicate public runtime provenance identity")
            local.add(identity)
            if identity in records and records[identity] != record:
                raise ValueError("conflicting public runtime provenance identities across packages")
            records[identity] = record
    return records, supported


def _integer(fields, name, *, positive=False):
    value = fields.get(name, "")
    if not re.fullmatch(r"[0-9]+", value) or (positive and int(value) == 0):
        raise ValueError("invalid observed scoped provenance " + name)
    return int(value)


def _counters():
    return {"events": 0, "upload_bytes": 0, "download_bytes": 0, "kernel_launches": 0}


def _count(target, event, amount):
    target["events"] += 1
    if event in {"upload", "download"}:
        target[event + "_bytes"] += amount
    elif event == "launch":
        target["kernel_launches"] += 1


def scoped_provenance(log, manifests):
    """Keep unknown/legacy events visible; never guess their source identity."""
    records, supported = _registry(manifests)
    groups, used, total = {}, set(), _counters()
    unknown = {field: _counters() for field in FIELDS}
    versioned, legacy = 0, 0
    for line in log.splitlines():
        match = re.match(r"^FORT_SCOPED ([a-z_]+) (.*)$", line)
        if not match or match[1] not in OBSERVED:
            continue
        event, encoded = match.groups()
        fields = {}
        for token in encoded.split():
            key, separator, value = token.partition("=")
            if not separator or key in fields:
                raise ValueError("malformed observed scoped provenance fields")
            fields[key] = value
        amount = _integer(fields, "bytes")
        tagged = "provenance_version" in fields
        context = None
        if tagged:
            if fields["provenance_version"] != "1":
                raise ValueError("unsupported observed runtime provenance version")
            context = _integer(fields, "context", positive=True)
            if "buffer_handle" in fields:
                _integer(fields, "buffer_handle", positive=True)
            for field in FIELDS:
                identity = fields.get(field)
                if identity == "unknown":
                    continue
                if not isinstance(identity, str) or not HASH.fullmatch(identity) or identity not in records:
                    raise ValueError("observed scoped provenance identity absent from public manifest: " + field)
                kind = records[identity]["kind"]
                if (field == "operation" and kind not in OPERATION_KINDS) or (field != "operation" and kind != field):
                    raise ValueError("observed scoped provenance identity has incorrect kind: " + field)
                used.add(identity)
            if event in {"upload", "download"}:
                _integer(fields, "buffer_handle", positive=True)
                _integer(fields, "buffer", positive=True)
                _integer(fields, "generation")
            versioned += 1
        else:
            if any(field in fields for field in FIELDS) or "context" in fields:
                raise ValueError("observed source attribution lacks runtime provenance version")
            legacy += 1
        identities = tuple(fields[field] if tagged else "unknown" for field in FIELDS)
        key = (tagged, *identities)
        if key not in groups:
            groups[key] = {"versioned": tagged, **dict(zip(FIELDS, identities)),
                           **_counters(), "event_counts": {}, "contexts": set()}
        group = groups[key]
        if context is not None:
            group["contexts"].add(context)
        group["event_counts"][event] = group["event_counts"].get(event, 0) + 1
        _count(group, event, amount)
        _count(total, event, amount)
        for field, identity in zip(FIELDS, identities):
            if identity == "unknown":
                _count(unknown[field], event, amount)
    for group in groups.values():
        group["contexts"] = sorted(group["contexts"])
    return {"schema_version": 1, "available": supported and versioned > 0,
            "public_mapping_available": supported, "versioned_events": versioned, "legacy_events": legacy,
            "totals": total, "unattributed_fields": unknown,
            "groups": [groups[key] for key in sorted(groups)],
            "records": [records[identity] for identity in sorted(used)],
            "timing_sample": False,
            "semantics": "actual API submissions and coherence commits; no device durations or policy inference"}
