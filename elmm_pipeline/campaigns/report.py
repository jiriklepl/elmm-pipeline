"""Immutable plots and receipts for complete application comparisons."""
from __future__ import annotations

import os
import shutil
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path

from .identity import check_files, digest, guard_passed, read, tree_hashes, write
from .prepare import check


def plot(root, output=None, *, historical=False):
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
    fig.suptitle(("Historical measurements · " if historical else "") + state["title"] + " · four host threads")
    freshness = "Retained historical medians; rendering only" if historical else "Fresh medians"
    footer = fig.text(.5, .025, freshness + "; whiskers: sample min–max. Green: 30% reduction; red: 5% regression.\n"
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
                                   "historical": historical,
                                   "rendering_only": True,
                                   "plotter_sha256": digest(__file__)})


def plot_snapshot(snapshot, output, *, gpu_label="Scoped GPU", native_label="Native"):
    """Render the paired complete-wall summaries of a verified SGS snapshot."""
    snapshot = Path(snapshot).resolve(strict=True)
    root, output = snapshot.parent, Path(output).resolve()
    state, snapshot_sha256 = read(snapshot), digest(snapshot)
    if (state.get("schema_version") != 1
            or state.get("kind") != "verified_current_compiler_complete_application"
            or state["resources"]["runtime_threads"] != 4
            or state["selection"]["gpu_policy"] != "sections"
            or state["selection"]["scope_transfers"] != "direct"):
        raise ValueError("verified four-thread SGS sections/direct snapshot required")
    if not gpu_label.strip() or not native_label.strip():
        raise ValueError("nonempty plot labels required")
    hashes = state["receipt_sha256"]
    required = {"timings/timings.json", "regressions/runs.json", "full-fields/validation.json"}
    if not required.issubset(hashes) or not hashes:
        raise ValueError("snapshot timing and validation receipt hashes required")
    check_files(root, hashes)
    for name in state["guards"]:
        if name not in hashes:
            raise ValueError("unhashed guard: " + name)
        guard_passed(root / name)
    primary, supplementary = read(root / "timings/timings.json"), read(root / "regressions/runs.json")
    if any(row["status"] != "passed" for row in (primary, supplementary, read(root / "full-fields/validation.json"))):
        raise ValueError("successful timing and field validation required")
    cases = [state["primary"], *state["supplementary_cases"]]
    sources = [primary, *supplementary["cases"]]
    if [row["case"] for row in cases] != [primary["case"]["name"], *(row["case"] for row in sources[1:])]:
        raise ValueError("snapshot cases differ from timing receipts")
    for index, (case, source) in enumerate(zip(cases, sources, strict=True)):
        repetitions = 7 if index == 0 else 3
        if case["summary"] != source["summary"] or set(case["summary"]) != {"native", "current"}:
            raise ValueError("snapshot summary differs from timing receipt")
        for mode, summary in case["summary"].items():
            runs = [row for row in source["runs"] if row["mode"] == mode]
            measured = [row for row in runs if not row["warmup"]]
            samples = [row["wall_seconds"] for row in measured]
            if (len(runs) != repetitions + 1 or len(measured) != repetitions
                    or summary["repetitions"] != repetitions or summary["samples"] != samples
                    or sorted(row["round"] for row in runs) != list(range(repetitions + 1))
                    or any(row["warmup"] != (row["round"] == 0) or row["traced"]
                           or not row["validation"]["passed"] for row in runs)
                    or any(not math.isfinite(value) or value <= 0 for value in samples)):
                raise ValueError("untraced measured samples and one excluded warmup required")
            expected = {"median": statistics.median(samples), "min": min(samples), "max": max(samples),
                        "ratio_to_native": statistics.median(samples) / case["summary"]["native"]["median"]}
            if any(not math.isclose(summary[key], value, rel_tol=1e-12, abs_tol=1e-12)
                   for key, value in expected.items()):
                raise ValueError("summary statistics differ from measured samples")
        reduction = 100 * (1 - case["summary"]["current"]["ratio_to_native"])
        if not math.isclose(case["wall_reduction_percent"], reduction, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("snapshot wall reduction differs from samples")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/elmm-campaign-plots")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output.mkdir(parents=True, exist_ok=False)
    labels, modes, colors = [native_label, gpu_label], ["native", "current"], ["#475569", "#0284c7"]
    with plt.rc_context({"font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
                         "xtick.labelsize": 13, "ytick.labelsize": 12, "pdf.fonttype": 42}):
        for relative in (False, True):
            fig, axes = plt.subplots(1, len(cases), figsize=(5.4 * len(cases), 8), squeeze=False, sharey=True)
            maximum = max(case["summary"][mode]["max"] / (case["summary"]["native"]["median"] if relative else 1)
                          for case in cases for mode in modes)
            for index, (ax, case) in enumerate(zip(axes[0], cases, strict=True)):
                baseline = case["summary"]["native"]["median"] if relative else 1
                rows = [case["summary"][mode] for mode in modes]
                values = [row["median"] / baseline for row in rows]
                errors = [[(row["median"] - row["min"]) / baseline for row in rows],
                          [(row["max"] - row["median"]) / baseline for row in rows]]
                ax.bar([0, 1], values, width=.58, color=colors, alpha=.85,
                       yerr=errors, capsize=5, error_kw={"linewidth": 1.5}, zorder=2)
                for position, row in enumerate(rows):
                    ax.scatter(position + np.linspace(-.14, .14, len(row["samples"])),
                               np.array(row["samples"]) / baseline, s=30,
                               color="white", edgecolor="#172033", linewidth=.9, zorder=3)
                    ax.text(position, row["max"] / baseline + .035 * maximum,
                            f"{values[position]:.3f}×" if relative else f"{values[position]:.3f} s",
                            ha="center", va="bottom", fontsize=12)
                if relative:
                    ax.axhline(1, color="#64748b", linewidth=1, zorder=1)
                    ax.axhline(.85, color="#15803d", linestyle="--", linewidth=1.5,
                               label="Approx. 15% lower wall time (0.85 × native)", zorder=1)
                ending = "natural end" if index == 0 else "configured cap"
                ax.set_title(f"{case['case'].replace('-', ' ').capitalize()}\n"
                             f"{case['completed_steps']} steps · {ending} · n={rows[0]['repetitions']}", pad=14)
                ax.set_xticks([0, 1], labels)
                ax.set_ylim(0, 1.16 * maximum)
                ax.grid(axis="y", alpha=.2)
                ax.set_axisbelow(True)
            axes[0, 0].set_ylabel("Complete wall time / native median" if relative else "Complete wall time (seconds)")
            fig.suptitle("Complete application wall time" + (" relative to native" if relative else ""), fontsize=17, y=.97)
            if relative:
                fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .925), frameon=False)
            fig.text(.5, .055, "Dots: measured runs; bars: medians; whiskers: sample min–max. One excluded warmup per case.\n"
                     "Current compiler SGS only · forced sections/direct · four host threads · automatic policy not promoted.\n"
                     "Complete process wall time includes startup, solver, transfers, output and teardown.", ha="center", fontsize=12, linespacing=1.5)
            fig.subplots_adjust(left=.075, right=.985, bottom=.235, top=.80, wspace=.16)
            name = "complete-wall-relative" if relative else "complete-wall-absolute"
            for suffix in ("png", "svg", "pdf"):
                fig.savefig(output / (name + "." + suffix), dpi=180)
            plt.close(fig)
    check_files(root, hashes)
    if digest(snapshot) != snapshot_sha256:
        raise ValueError("snapshot changed during rendering")
    write(output / "source.json", {"schema_version": 1, "rendering_only": True,
          "created_utc": datetime.now(timezone.utc).isoformat(),
          "snapshot": {"path": str(snapshot), "sha256": snapshot_sha256},
          "receipt_sha256": hashes, "plotter_sha256": digest(__file__),
          "cli_sha256": digest(Path(__file__).with_name("__main__.py")),
          "labels": dict(zip(modes, labels, strict=True)), "files_sha256": tree_hashes(output)})
    return output


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
