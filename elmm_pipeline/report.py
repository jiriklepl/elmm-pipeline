import statistics


def summary(records):
    measured = [r for r in records if not r["warmup"]]
    result = {}
    for key in ("wall_seconds", "time_steps_seconds", "poisson_seconds"):
        values = [r[key] for r in measured if r.get(key) is not None]
        result[key] = {"median": statistics.median(values), "min": min(values), "max": max(values)} if values else None
    return result


def render(report):
    configuration = report["configuration"]
    settings = report["settings"]
    lines = ["# ELMM application benchmark", "",
             f"Configuration: `{configuration['source']}`; effective advection method **{configuration['advection_method']}**.", "",
             f"Overrides: `{configuration['overrides']}`. Effective input files are in `config/`.", "",
             f"CPU/OpenMP; double precision; {settings['threads']} threads; {settings['repeats']} measured repeats and {settings['warmups']} warmups per variant.",
             f"ELMM revision: `{report['dependencies']['elmm']['revision']}`.", "",
             "| Variant | Status | Steps/run | Step time, median (s) | Wall time, median (s) | Baseline/step time | Validation |",
             "|---|---|---:|---:|---:|---:|---|"]
    if report.get("error"):
        lines[2:2] = [f"**Pipeline failed:** {report['error']}", ""]
    baseline = report["variants"].get("baseline", {}).get("summary", {}).get("time_steps_seconds")
    for name, variant in report["variants"].items():
        if variant["status"] != "ok":
            lines.append(f"| {name} | {variant['status']} | — | — | — | — | {variant.get('reason', '')} |")
            continue
        summary = variant["summary"]
        step = summary["time_steps_seconds"]["median"]
        wall = summary["wall_seconds"]["median"]
        speedup = f"{baseline['median'] / step:.3f}×" if baseline and step else "—"
        counts = ', '.join(str(n) for n in sorted({r['completed_steps'] for r in variant['runs']}))
        lines.append(f"| {name} | ok | {counts} | {step:.2f} | {wall:.3f} | {speedup} | pass |")
    transformed = [v["summary"]["time_steps_seconds"] for name, v in report["variants"].items()
                   if name != "baseline" and v["status"] == "ok"]
    if baseline and transformed and settings["repeats"] >= 3 and all(
            v["min"] <= baseline["max"] and baseline["min"] <= v["max"] for v in transformed):
        lines.extend(["", "All transformed variants' timing ranges overlap the baseline's range. These samples do not establish a clear application-level speedup."])
    lines.extend(["", "The timings measure complete ELMM runs. Step time is ELMM's internal timer (including periodic output); wall time also includes startup and final output. Internal timers have 0.01 s resolution.", "",
                  "Numerical validation compares every saved final pressure and velocity value and every other field in `output/out.vtk`, including finite-value checks. ELMM writes these fields as float32 even in double-precision builds. The JSON report records tolerances, per-field errors, every repetition, build commands, coverage, hashes, and tool versions.", "",
                  "Regions are discovered from program structure and symbol information. Source markers and selection filters are optional. Unsupported regions and surrounding native code are preserved. Normalized region modules pass through each backend's generic transformations. These are CPU comparisons; no GPU/offload performance is claimed.", "",
                  "| Variant | Discovered | Outlined | Backend changes | Executed regions |",
                  "|---|---:|---:|---:|---:|"])
    for name, variant in report["variants"].items():
        adapter = variant.get("build", {}).get("adapter", {})
        if name == "baseline" or "outlined_regions" not in adapter:
            continue
        executed = {rid for run in variant.get("runs", [])
                    for rid in run.get("execution_coverage", {}).get("executed_regions", [])}
        count = str(len(executed)) if adapter["coverage_enabled"] else "unverified"
        lines.append(f"| {name} | {adapter['discovered_regions']} | {adapter['outlined_regions']} | {adapter['optimized_regions']} | {count} |")
    lines.extend(["", "Outlining replaces a proved eligible loop region with a generated helper call. Backend changes count additional tool transformations or C++ translation; they do not imply a speedup. A tool can accept an outlined region while finding no further legal optimization. The JSON manifests record rejected regions and reasons, pass diagnostics, and which regions this configuration exercised.", ""])
    if settings.get("coverage_enabled", True):
        lines.extend(["Coverage is enabled: transformed-run timings include marker guards and first-call output; the baseline has no coverage instrumentation.", ""])
    for name, variant in report["variants"].items():
        if variant["status"] == "ok":
            data = variant["summary"]
            steps = data["time_steps_seconds"]
            poisson = data["poisson_seconds"]
            share = f"; Poisson timer / step timer: {100 * poisson['median'] / steps['median']:.1f}%" if poisson and steps["median"] else ""
            lines.append(f"- {name}: step-time range {steps['min']:.2f}–{steps['max']:.2f} s{share}.")
    return "\n".join(lines) + "\n"
