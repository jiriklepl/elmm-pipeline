"""Immutable plots and receipts for complete application comparisons."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from .identity import digest, guard_passed, read, tree_hashes, write
from .prepare import check


def plot(root, output=None):
    root = Path(root).resolve(strict=True)
    state = check(root)
    report = read(root / "timings/report.json")
    verification = read(root / "verification/report.json")
    if report["status"] != "passed" or verification["status"] != "passed":
        raise ValueError("complete correctness and timing reports required")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/elmm-campaign-plots")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = root / "figures" if output is None else Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    fig, axes = plt.subplots(1, len(report["cases"]), figsize=(17, 5), squeeze=False, sharey=True)
    for ax, case in zip(axes[0], report["cases"], strict=True):
        rows = case["summary"]
        baseline = case["gates"]["baseline_seconds"]
        values = [rows[mode["name"]]["median"] / baseline for mode in state["modes"]]
        lower = [values[i] - rows[mode["name"]]["min"] / baseline for i, mode in enumerate(state["modes"])]
        upper = [rows[mode["name"]]["max"] / baseline - values[i] for i, mode in enumerate(state["modes"])]
        bars = ax.bar(np.arange(len(values)), values, yerr=[lower, upper], capsize=2)
        observed = next(row for row in verification["cases"] if row["case"] == case["case"])
        launches = {run["mode"]: run["activity"]["kernel_launches"] for run in observed["runs"]}
        for bar, mode in zip(bars, state["modes"], strict=True):
            if mode["name"] != "native" and not launches[mode["name"]]:
                bar.set_hatch("//")
        ax.axhline(1, color="black", linewidth=1)
        ax.axhline(1.05, color="#b91c1c", linestyle="--", linewidth=1)
        ax.axhline(.70, color="#15803d", linestyle="--", linewidth=1)
        ax.set_xticks(np.arange(len(values)), [mode["label"] for mode in state["modes"]], rotation=45, ha="right")
        ax.set_title(f"{case['case']}\nBest frozen baseline: {baseline:.3f} s")
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    # Shared limits must be fixed only after every panel contributes its
    # whiskers; setting a bound inside the loop disables later autoscaling.
    axes[0, 0].set_ylim(bottom=0)
    axes[0, 0].set_ylabel("Complete wall time / better native or frozen implementation")
    fig.suptitle(state["title"] + " · four host threads")
    footer = fig.text(.5, .025, "Fresh medians; whiskers: sample min–max. Green: 30% reduction; red: 5% regression.\n"
             "Hatching: no CUDA kernels in separate validation. Startup, output and teardown included.", ha="center")
    # Reserve a real plotting height before accommodating rotated labels.
    # tight_layout can otherwise spend almost the entire fixed-height figure
    # on long labels and collapse all three axes into a few pixels.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    label_height = max(label.get_window_extent(renderer).height for ax in axes[0]
                       for label in ax.get_xticklabels()) / fig.dpi
    footer_height = footer.get_window_extent(renderer).height / fig.dpi
    bottom = label_height + footer_height + .65
    top = .8
    height = 4.5 + bottom + top
    fig.set_size_inches(17, height)
    fig.subplots_adjust(left=.075, right=.99, bottom=bottom / height,
                        top=1 - top / height, wspace=.14)
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / ("complete-wall-relative." + suffix), dpi=180, bbox_inches="tight")
    plt.close(fig)
    write(output / "source.json", {"report_sha256": digest(root / "timings/report.json"),
                                   "verification_sha256": digest(root / "verification/report.json"),
                                   "campaign_sha256": digest(root / "campaign.json"),
                                   "rendering_only": True,
                                   "plotter_sha256": digest(__file__)})


def seal(root, output):
    root, output = Path(root).resolve(strict=True), Path(output).resolve()
    state = check(root)
    phases = ["verification", "halos", "timings", *state.get("commands", {})]
    reports = {}
    for phase in phases:
        guard_passed(root / "guards" / (phase + ".json"))
        paths = state.get("commands", {}).get(phase, {}).get("reports", [phase + "/report.json"])
        for path in paths:
            report = read(root / path)
            if report.get("status") != "passed":
                raise ValueError("successful report required: " + path)
            reports[path] = digest(root / path)
    if not (root / "figures/source.json").is_file():
        raise ValueError("current comparison plots required")
    plotted = read(root / "figures/source.json")
    if plotted["report_sha256"] != reports["timings/report.json"] or plotted["verification_sha256"] != reports["verification/report.json"]:
        raise ValueError("comparison plots are stale")
    timing = read(root / "timings/report.json")
    output.mkdir(parents=True, exist_ok=False)
    for name in ("figures", *state.get("additional_figure_directories", [])):
        shutil.copytree(root / name, output / name)
    shutil.copy2(root / "campaign.json", output / "campaign.json")
    shutil.copy2(root / "timings/report.json", output / "timings.json")
    shutil.copy2(root / "verification/report.json", output / "verification.json")
    previous = state["previous_snapshot"]
    pair = {name: value["commit"] for name, value in state["repositories"].items()}
    lines = ["# " + state["title"], "", "Fresh complete-application runs; no timings reused.", "",
             f"Compiler `{pair['compiler']}` · pipeline `{pair['pipeline']}`.", "",
             ("[Current application comparison](figures/complete-wall-relative.png) · "
              f"[Previous snapshot]({previous})."), "",
             "| Case | Better baseline | Median (s) | 30% target (s) |", "|---|---|---:|---:|"]
    for case in timing["cases"]:
        gate = case["gates"]
        lines.append(f"| {case['case']} | {gate['baseline_mode']} | {gate['baseline_seconds']:.3f} | {gate['target_seconds']:.3f} |")
    lines += ["", "All complete final fields and configured full-array/halo drivers passed.",
              "Diagnostic planning, transfer and allocation counters are separate from untraced timing samples.",
              "This receipt does not promote a policy or claim unmeasured overlap.", ""]
    (output / "report.md").write_text("\n".join(lines))
    receipt = {"schema_version": 1, "status": "sealed", "created_utc": datetime.now(timezone.utc).isoformat(),
               "campaign": str(root), "campaign_sha256": digest(root / "campaign.json"),
               "compatible_commits": pair, "previous_snapshot": previous, "reports_sha256": reports,
               "guard_receipts_sha256": {phase: digest(root / "guards" / (phase + ".json")) for phase in phases},
               "timings_reused": False, "policy_promoted": False, "files_sha256": tree_hashes(output)}
    write(output / "snapshot.json", receipt)
    return output
