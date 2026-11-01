"""Exclusive per-thread source-function budgets, separate from performance samples."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT_PARENT = 2**64 - 1


def category(symbol):
    """Only explicit routine ownership; unknown descendants inherit their owner."""
    name = symbol.lower()
    if name.startswith("__wrap_fort_scope_"):
        return "runtime"
    if name.startswith("__wrap_cpp_fort_"):
        return "generated_planning"
    if "_mod_subgridstresses" in name:
        return "sgs"
    if "_mod_convection" in name:
        return "convection"
    if "momentumdiffusion" in name or "momentum_diffusion" in name:
        return "diffusion"
    if "_mod_prepoisson" in name:
        return "pre_fft"
    if "_mod_postpoisson" in name:
        return "post_fft"
    if "__execute" in name and "poisfft" in name:
        return "fft_execute"
    if "_mod_poiss_poisfft" in name:
        return "fft_setup_and_wrapper"
    if ("_mod_bound" in name or "_mod_ibmomentum" in name
            or "_mod_ibmodif" in name):
        return "boundaries"
    if (name.startswith("__outputs_") or name.startswith("__simplevtk_")
            or name.startswith("__vtkframes_")):
        return "output"
    if name.startswith("__initial_"):
        return "initialize"
    return None


def read_symbols(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]+)\s+[tTwW]\s+(\S+)", line)
        if match:
            address, symbol = int(match[1], 16), match[2]
            result.setdefault(address, []).append(symbol)
    return result


def analyze(csv_path, symbols_path):
    """Partition main-thread wall once; workers report CPU seconds independently."""
    lines = Path(csv_path).read_text().splitlines()
    header = re.fullmatch(r"# schema=1 main_tid=(\d+) anchor=([0-9a-f]+) clock_errors=(\d+)", lines[0])
    if header is None or int(header[3]):
        raise ValueError("missing timer authority or failed clocks")
    main_tid, anchor = int(header[1]), int(header[2], 16)
    symbols = read_symbols(symbols_path)
    anchors = [pc for pc, names in symbols.items() if "fort_budget_anchor" in names]
    if len(anchors) != 1:
        raise ValueError("exact diagnostic ELF anchor required")
    bias = anchor - anchors[0]
    threads = {}
    for line in lines:
        if line.startswith("# thread="):
            match = re.fullmatch(r"# thread=(\d+) open_frames=(\d+) nesting_errors=(\d+)", line)
            if match is None or int(match[2]) or int(match[3]):
                raise ValueError("incomplete or malformed callback stack")
            threads[int(match[1])] = []
    records = csv.DictReader(line for line in lines[1:] if not line.startswith("#"))
    for record in records:
        row = {key: int(value, 16 if key in {"function", "caller"} else 10)
               for key, value in record.items()}
        if row["tid"] not in threads or row["event"] != len(threads[row["tid"]]):
            raise ValueError("callback event order/thread differs")
        names = symbols.get(row["function"] - bias)
        if not names:
            raise ValueError("recorded function lacks exact ELF symbol")
        row["symbol"] = names[0]
        row["category"] = category(row["symbol"])
        row["wall_ns"] = row["stop_ns"] - row["start_ns"]
        row["cpu_ns"] = row["cpu_stop_ns"] - row["cpu_start_ns"]
        if row["wall_ns"] < 0 or row["cpu_ns"] < 0:
            raise ValueError("negative timer interval")
        row["exclusive_wall_ns"], row["exclusive_cpu_ns"] = row["wall_ns"], row["cpu_ns"]
        row["children"] = []
        rows = threads[row["tid"]]
        if row["parent"] != ROOT_PARENT:
            if row["parent"] >= len(rows):
                raise ValueError("parent is not an earlier event")
            parent = rows[row["parent"]]
            if not parent["start_ns"] <= row["start_ns"] <= row["stop_ns"] <= parent["stop_ns"]:
                raise ValueError("child interval is outside parent")
            if parent["children"] and parent["children"][-1][1] > row["start_ns"]:
                raise ValueError("sibling callback intervals overlap")
            parent["children"].append((row["start_ns"], row["stop_ns"]))
            parent["exclusive_wall_ns"] -= row["wall_ns"]
            parent["exclusive_cpu_ns"] -= row["cpu_ns"]
            row["category"] = row["category"] or parent["category"]
        row["category"] = row["category"] or "other"
        rows.append(row)
    if main_tid not in threads:
        raise ValueError("application main thread was not observed")
    budgets, timeline = {}, []
    epoch = min(row["start_ns"] for rows in threads.values() for row in rows)
    first_step = min((row["start_ns"] for row in threads[main_tid]
                      if "_mod_tmarchrk3" in row["symbol"].lower()), default=epoch)
    for tid, rows in threads.items():
        totals = defaultdict(lambda: {"exclusive_wall_ns": 0, "exclusive_cpu_ns": 0, "events": 0})
        for row in rows:
            if row["exclusive_wall_ns"] < 0 or row["exclusive_cpu_ns"] < 0:
                raise ValueError("overlapping children or inconsistent CPU clocks")
            total = totals[row["category"]]
            total["exclusive_cpu_ns"] += row["exclusive_cpu_ns"]
            if tid == main_tid and row["category"] == "other":
                previous = row["start_ns"]
                for begin, end in [*row["children"], (row["stop_ns"], row["stop_ns"])]:
                    before = max(0, min(begin, first_step) - previous)
                    totals["initialize"]["exclusive_wall_ns"] += before
                    total["exclusive_wall_ns"] += begin - previous - before
                    previous = end
            else:
                total["exclusive_wall_ns"] += row["exclusive_wall_ns"]
            total["events"] += 1
            timeline.append({"name": row["symbol"], "cat": row["category"], "ph": "X",
                             "ts": (row["start_ns"] - epoch) / 1000, "dur": row["wall_ns"] / 1000,
                             "pid": 1, "tid": tid, "args": {"event": row["event"], "parent": row["parent"]}})
        roots = [row for row in rows if row["parent"] == ROOT_PARENT]
        if sum(item["exclusive_wall_ns"] for item in totals.values()) != sum(row["wall_ns"] for row in roots):
            raise ValueError("exclusive wall partition differs from root intervals")
        budgets[str(tid)] = {"main_thread": tid == main_tid, "categories": dict(totals),
                             "root_wall_ns": sum(row["wall_ns"] for row in roots), "events": len(rows)}
    return {"schema_version": 1, "diagnostic_only": True, "performance_sample": False,
            "main_tid": main_tid, "load_bias": bias, "threads": budgets,
            "wall_partition": "main-thread exclusive intervals only; never add worker wall",
            "cpu_work": "per-thread exclusive CPU nanoseconds; worker CPU is not elapsed wall",
            "limitations": ["Instrumentation perturbs execution; no production speedup claim.",
                            "Runtime intervals can include blocked GPU work; they are not pure CPU instructions.",
                            "Source ancestors retain work not independently instrumented, including native generated bodies."]}, {"traceEvents": timeline}


def save_analysis(csv_path, symbols_path, output):
    output = Path(output)
    output.mkdir(exist_ok=False)
    budget, timeline = analyze(csv_path, symbols_path)
    for name, value in (("budget.json", budget), ("timeline.json", timeline)):
        (output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return budget
