"""Versioned per-case complete-wall targets and paired qualification evidence."""
from __future__ import annotations

import math
import statistics


def validate_target(target, modes, candidate):
    required = {"schema_version", "metric", "limit", "comparison", "control_mode"}
    if (not isinstance(target, dict) or type(target.get("schema_version")) is not int
            or target["schema_version"] not in {1, 2}):
        raise ValueError("performance_target requires a supported versioned target contract")
    version = target["schema_version"]
    if version == 2:
        required.add("control_comparison")
    optional = {"require_control_improvement"} if version == 1 else set()
    if not required <= target.keys() or set(target) - required - optional:
        raise ValueError("performance_target requires its complete versioned target contract")
    metrics = {"seconds", "ratio_to_native"} | ({"ratio_to_control"} if version == 2 else set())
    if (not isinstance(target["metric"], str) or target["metric"] not in metrics
            or not isinstance(target["comparison"], str) or target["comparison"] not in {"lt", "le"}):
        raise ValueError("performance_target requires a supported metric and lt/le")
    if version == 2 and (not isinstance(target["control_comparison"], str)
                          or target["control_comparison"] not in {"lt", "le"}):
        raise ValueError("performance_target control_comparison requires lt/le")
    limit = target["limit"]
    if type(limit) not in (int, float) or not math.isfinite(limit) or limit <= 0:
        raise ValueError("performance_target limit must be finite and positive")
    if (not candidate or candidate == "native" or not isinstance(target["control_mode"], str)
            or target["control_mode"] not in modes - {candidate, "native"}):
        raise ValueError("performance_target requires distinct native, candidate and configured control modes")
    if type(target.get("require_control_improvement", True)) is not bool:
        raise ValueError("performance_target require_control_improvement must be boolean")


def target_seconds(target, summary):
    reference = {"ratio_to_native": "native", "ratio_to_control": target["control_mode"]}.get(target["metric"])
    return target["limit"] * (summary[reference]["median"] if reference else 1)


def target_label(target):
    operator = "<" if target["comparison"] == "lt" else "≤"
    value = (f"{target['limit']:g} s" if target["metric"] == "seconds"
             else f"{100 * target['limit']:g}% of " + (
                 target["control_mode"] if target["metric"] == "ratio_to_control" else "native"))
    return operator + " " + value


def evaluate_target(state, case, summary, runs):
    target = case["performance_target"]
    candidate, control = state["candidate_mode"], target["control_mode"]
    for mode in ("native", candidate, control):
        median = summary[mode].get("median")
        if type(median) not in (int, float) or not math.isfinite(median) or median <= 0:
            raise ValueError("qualification summary requires finite positive medians")
    threshold = target_seconds(target, summary)

    def meets(value):
        return value < threshold if target["comparison"] == "lt" else value <= threshold

    paired, warmups = {}, {}
    for mode in ("native", candidate, control):
        observations = [run for run in runs if run["mode"] == mode]
        if any(type(run.get("warmup")) is not bool or run.get("traced") is not False
               or type(run.get("round")) is not int or run["round"] < 0
               or run["warmup"] != (run["round"] == 0)
               or type(run.get("wall_seconds")) not in (int, float)
               or not math.isfinite(run["wall_seconds"]) or run["wall_seconds"] <= 0
               or run.get("validation", {}).get("passed") is not True for run in observations):
            raise ValueError("qualification requires valid untraced validated rounds and excluded warmups")
        selected = [run for run in observations if not run["warmup"]]
        warmups[mode] = sum(run["warmup"] for run in observations)
        if warmups[mode] > 1:
            raise ValueError("qualification requires one unique excluded warmup per mode")
        paired[mode] = {run["round"]: run["wall_seconds"] for run in selected}
        if len(paired[mode]) != len(selected):
            raise ValueError("qualification requires unique measured rounds per mode")
        # A partial diagnostic may retain a separately supplied median for a
        # mode with no observations. It cannot establish the complete protocol.
        # Every available sample set must authenticate its reported statistics.
        if selected:
            values = [row["wall_seconds"] for row in selected]
            row = summary[mode]
            expected = {"median": statistics.median(values), "min": min(values), "max": max(values),
                        "ratio_to_native": statistics.median(values) / summary["native"]["median"]}
            if (type(row.get("repetitions")) is not int or row["repetitions"] != len(values)
                    or not isinstance(row.get("samples"), list)
                    or any(type(value) not in (int, float) for value in row["samples"])
                    or sorted(row["samples"]) != sorted(values)
                    or any(type(row.get(key)) not in (int, float) or row[key] != value
                           for key, value in expected.items())):
                raise ValueError("qualification summary differs from observed measured rounds")
    complete = all(set(rounds) == set(range(1, 8)) for rounds in paired.values())
    warmups_complete = all(count == 1 for count in warmups.values())
    all_target = complete and all(meets(value) for value in paired[candidate].values())
    all_control = complete and all(paired[candidate][round_] < paired[control][round_]
                                   for round_ in range(1, 8))
    median_target = meets(summary[candidate]["median"])
    median_control = summary[candidate]["median"] < summary[control]["median"]
    control_operator = target.get("control_comparison", "lt")

    def meets_control(value, reference):
        return value < reference if control_operator == "lt" else value <= reference

    control_met = meets_control(summary[candidate]["median"], summary[control]["median"])
    all_control_met = complete and all(meets_control(paired[candidate][round_], paired[control][round_])
                                      for round_ in range(1, 8))
    result = {"schema_version": 4 if target["schema_version"] == 2 else 3,
            "baseline_mode": "native", "baseline_seconds": summary["native"]["median"],
            "candidate_mode": candidate, "candidate_ratio": summary[candidate]["median"] / summary["native"]["median"],
            "performance_target": target, "target_seconds": threshold, "target_label": target_label(target),
            "target_met": median_target, "faster_than_control": median_control,
            "accepted": median_target and (control_met or not target.get("require_control_improvement", True)),
            "comfortable": {"required_measured_rounds": 7, "paired_rounds_complete": complete,
                            "required_excluded_warmups_per_mode": 1, "warmups_complete": warmups_complete,
                            "measured_rounds_by_mode": {mode: sorted(rounds) for mode, rounds in paired.items()},
                            "excluded_warmups_by_mode": warmups,
                            "all_samples_meet_target": all_target, "beats_control_each_round": all_control,
                            "met": complete and warmups_complete and all_target and all_control_met}}
    if target["schema_version"] == 2:
        result.update(control_comparison=control_operator, control_requirement_met=control_met)
        result["comfortable"]["meets_control_each_round"] = all_control_met
    return result
