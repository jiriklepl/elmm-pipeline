"""Preserve public batch records independently of realized CUDA activity."""
from __future__ import annotations

from collections import Counter
import math


def batch_summary(records):
    rows = [dict(row) for row in records if row.get("event") == "batch_statistics"]
    for row in rows:
        if row.get("stats_version") != 1 or row.get("schema_version") != 1:
            raise ValueError("versioned batch statistics required")
        if any(type(row.get(name)) is not int or row[name] not in (0, 1)
               for name in ("available", "applied", "owner_cost_available")):
            raise ValueError("invalid public batch availability")
        if row.get("selected_transfers") not in (0, 1, 2):
            raise ValueError("invalid selected transfer implementation")
        for name in ("preparation_operations", "batches", "chunk_iterations", "slot_bytes", "upload_bytes",
                     "download_bytes", "prefix_upload_bytes", "prefix_uploads", "launches", "completed_batches",
                     "actual_upload_bytes", "actual_download_bytes", "actual_launches"):
            if type(row.get(name)) is not int or not 0 <= row[name] < 2**64:
                raise ValueError("invalid public batch counter: " + name)
        for name in ("baseline_seconds", "pinned_seconds", "pipelined_seconds", "execution_seconds",
                     "estimated_seconds", "terminal_delta_seconds"):
            value = row.get(name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError("nonfinite public batch cost: " + name)
            if name not in ("estimated_seconds", "terminal_delta_seconds") and value < 0:
                raise ValueError("negative public execution cost")
        if row["completed_batches"] > row["batches"] or (row["applied"] and (
                not row["available"] or not row["batches"] or row["completed_batches"] != row["batches"])):
            raise ValueError("applied batch lacks complete drain")
    return {"records": rows, "record_count": len(rows),
            "applied_records": sum(row["applied"] for row in rows),
            "pipelined_applied_records": sum(row["applied"] and row["selected_transfers"] == 2 for row in rows),
            "selection_records": dict(Counter(str(row["selected_transfers"]) for row in rows)),
            "reasons": dict(Counter(row["reason"] for row in rows)),
            "source": "public batch_statistics; estimates and actual-prefix counters preserved without additive inference",
            "owner_reprice_note": "Applied owner_cost_available=0 excludes legacy whole-owner report_v2 as a batch prediction",
            "overlap_proven": False, "timing_samples": False}
