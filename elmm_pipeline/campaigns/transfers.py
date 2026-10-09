"""Read completed public transfer statistics; no implementation decisions."""
from __future__ import annotations

import math

COUNTERS = ("fallbacks", "pinned_uploads", "pinned_downloads", "pinned_upload_bytes", "pinned_download_bytes",
            "packed_bytes", "unpacked_bytes", "tiles", "events", "event_waits", "staging_allocations", "staging_reuses")
MEMORY = ("slot_capacity", "staging_device_bytes", "process_reserved_bytes", "process_peak_bytes")
DURATIONS = ("packing_seconds", "unpacking_seconds", "event_wait_seconds")


def aggregate_final_statistics(records, *, require_pinned=False):
    rows = {}
    for row in records:
        if row.get("event") != "transfer_statistics":
            continue
        if type(row.get("stats_version")) is not int or row["stats_version"] != 1 or row.get("complete") is not True:
            raise ValueError("complete public transfer statistics version1 required")
        for name in ("context", "requested_mode", "effective_mode", "fallback_reason", *COUNTERS, *MEMORY):
            if type(row.get(name)) is not int or not 0 <= row[name] < 2**64:
                raise ValueError("invalid public transfer counter: " + name)
        for name in DURATIONS:
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("invalid public transfer duration: " + name)
        if row["process_reserved_bytes"] > row["process_peak_bytes"] or row["process_peak_bytes"] > 64 * 2**20:
            raise ValueError("public transfer pool exceeds the shared64MiB budget")
        if row["staging_device_bytes"] != 0:
            raise ValueError("scoped staging must retain existing full-layout device buffers")
        old = rows.get(row["context"])
        if old is not None and old != row:
            raise ValueError("conflicting completed public statistics for one context")
        rows[row["context"]] = row
    if require_pinned and (not rows or any(row["requested_mode"] != 1 or row["effective_mode"] != 1
                                         or row["fallbacks"] or row["fallback_reason"] for row in rows.values())):
        raise ValueError("active pinned control must prove effective pinned copies without fallback")
    result = {name: sum(row[name] for row in rows.values()) for name in (*COUNTERS, *DURATIONS)}
    # Pool usage is process-wide and repeated by context; these fields must
    # never be summed. Successful-close rows include publication and drain.
    result.update({name: max((row[name] for row in rows.values()), default=0) for name in MEMORY})
    result.update(contexts=len(rows), source="complete public successful-close transfer_statistics rows",
                  before_close_abi_stats_are_partial=True, performance_samples=False,
                  requested_modes=sorted({row["requested_mode"] for row in rows.values()}),
                  effective_modes=sorted({row["effective_mode"] for row in rows.values()}))
    if require_pinned and (not result["pinned_uploads"] or not result["pinned_downloads"]):
        raise ValueError("active pinned control must observe both upload and publication")
    return result
