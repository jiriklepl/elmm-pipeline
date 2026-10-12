import statistics

from .gpu_validation import native_policy_exercised


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
    gpu = settings.get("target", "cpu") == "gpu"
    execution = ("CPU/OpenMP baseline; local CUDA offload"
                 if gpu else "CPU/OpenMP")
    lines = ["# ELMM application benchmark", "",
             f"Configuration: `{configuration['source']}`; effective advection method **{configuration['advection_method']}**.", "",
             f"Overrides: `{configuration['overrides']}`. Effective input files are in `config/`.", "",
             f"{execution}; double precision; {settings['threads']} host threads; {settings['repeats']} measured repeats and {settings['warmups']} warmups per variant.",
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
                  "Regions are discovered from program structure and symbol information. Source markers and selection filters are optional. Unsupported regions and surrounding native code are preserved. Normalized region modules pass through each backend's generic transformations." +
                  (" The selected GPU policy can retain eligible work on the CPU; unsupported regions and the native Poisson solver remain native." if gpu else
                   " These are CPU comparisons; no GPU/offload performance is claimed."), "",
                  "| Variant | Discovered regions | Outlined regions | Loop nests outlined | Backend changes | Executed regions |",
                  "|---|---:|---:|---:|---:|---:|"])
    for name, variant in report["variants"].items():
        adapter = variant.get("build", {}).get("adapter", {})
        if name == "baseline" or "outlined_regions" not in adapter:
            continue
        executed = {rid for run in variant.get("runs", [])
                    for rid in run.get("execution_coverage", {}).get("executed_regions", [])}
        count = str(len(executed)) if adapter["coverage_enabled"] else "unverified"
        nests = sum(region.get("loop_nests", 1) for region in adapter["regions"])
        lines.append(f"| {name} | {adapter['discovered_regions']} | {adapter['outlined_regions']} | {nests} | {adapter['optimized_regions']} | {count} |")
    lines.extend(["", "Outlining replaces a proved eligible loop region with a generated helper call. Backend changes count additional tool transformations or C++ translation; they do not imply a speedup. A tool can accept an outlined region while finding no further legal optimization. The JSON manifests record rejected regions and reasons, pass diagnostics, and which regions this configuration exercised.", ""])
    scoped = any(variant.get("build", {}).get("adapter", {}).get("memory_model") == "scoped"
                 for variant in report["variants"].values())
    if scoped:
        lines.extend([("Compiler source scopes use explicit capture facts and original-source provenance. "
                      "The compiler forms groups, chooses GPU leaves, and implements coherence. "
                      "Adapter call markers are unavailable for source scopes; transformed-call coverage remains unverified."), ""])
    if settings.get("coverage_enabled", True) and not scoped:
        lines.extend(["Coverage is enabled: transformed-run timings include marker guards and first-call output; the baseline has no coverage instrumentation.", ""])
    if gpu:
        devices = (report.get("gpu_toolchain") or {}).get("device", {}).get("devices", [])
        if devices:
            lines.extend(["GPU: " + "; ".join(str(device.get("name", device)) for device in devices) + ".", ""])
        lines.extend([
            "GPU evidence comes from separate, numerically validated runs with runtime tracing. These runs and their profiler overhead are excluded from the timing samples above. Nsight totals sum device activity durations; they are not elapsed critical-path time and should not be subtracted from the unprofiled application timer.", "",
            "| Variant | GPU computation verified | Compute launches | Device kernel time (s) | Device transfer time (s) | Host→device (GiB) | Device→host (GiB) |",
            "|---|---|---:|---:|---:|---:|---:|"])
        for name, variant in report["variants"].items():
            if name == "baseline":
                continue
            validation = variant.get("gpu_validation", {})
            activity = validation.get("gpu_activity", {})
            profile = validation.get("gpu_profile", {})
            def number(value, scale=1):
                return "—" if value is None else f"{value / scale:.4f}"
            # Failed/incomplete profiles cannot establish durations or byte totals.
            available = profile.get("available", False)
            kernel = number(profile.get("kernel_seconds") if available else None)
            transfer = number(profile.get("transfer_seconds") if available else None)
            upload = number(profile.get("h2d_bytes") if available else None, 1024 ** 3)
            download = number(profile.get("d2h_bytes") if available else None, 1024 ** 3)
            native = native_policy_exercised(activity, variant.get("build", {}).get("adapter", {}).get("regions", []))
            proof = "yes" if activity.get("verified") else ("no (native policy)" if native else ("no" if activity else "pending"))
            lines.append(f"| {name} | {proof} | {activity.get('kernel_launches', '—')} | {kernel} | {transfer} | {upload} | {download} |")
        lines.extend(["", "| Variant | Runtime policy | Native decisions | GPU decisions | Mixed decisions | Hybrid decisions | Chunked decisions |",
                      "|---|---|---:|---:|---:|---:|---:|"])
        for name, variant in report["variants"].items():
            if name == "baseline":
                continue
            decisions = variant.get("gpu_validation", {}).get("gpu_activity", {}).get("offload_decisions", {})
            modes = decisions.get("modes", {})
            policy = variant.get("build", {}).get("identity", {}).get("gpu_policy", "always")
            lines.append(f"| {name} | {policy} | " + " | ".join(str(modes.get(mode, 0))
                         for mode in ("native", "gpu", "mixed", "hybrid", "chunked")) + " |")
        lines.extend(["", ("Policy decisions describe CPU/GPU placement and are separate from successful device launches. "
                      "An intentional all-native policy run is valid after numerical validation and makes no GPU execution claim. "
                      "Cost decisions use reusable offline hardware calibration and runtime dimensions; application profiling is not an input."), ""])
        if scoped:
            lines.extend(["", ("Scoped storage is shared across compiler-approved source calls within one scope. "
                          "Required native consumers reconcile their data before running, and scope exit restores host-visible outputs. "
                          "Automatic mode currently selects the original native span when coherent estimates are unavailable; "
                          "this selection is established by the versioned compiler manifest, not a claimed runtime decision count. "
                          "Complete application timings include every native boundary and final export. No speedup is implied by offload."), ""])
        else:
            lines.extend(["", "Arrays remain on the GPU across kernels within one normalized region invocation, including adjacent compatible loop nests and supported complete tile domains. Grouped nests share device allocations and transfer common inputs once per invocation. Data is synchronized with the host at each invocation boundary; this pipeline does not yet retain arrays across separate regions or timesteps. Transfers, allocation, launch overhead and native CPU work are included in application timings. No speedup is implied by successful offload.", ""])
        for name, variant in report["variants"].items():
            profile = variant.get("gpu_validation", {}).get("gpu_profile", {})
            if profile and not profile.get("available"):
                lines.extend([f"- {name}: device timing unavailable: {profile.get('reason', 'incomplete profile')}."])
        lines.append("")
    for name, variant in report["variants"].items():
        if variant["status"] == "ok":
            data = variant["summary"]
            steps = data["time_steps_seconds"]
            poisson = data["poisson_seconds"]
            share = f"; Poisson timer / step timer: {100 * poisson['median'] / steps['median']:.1f}%" if poisson and steps["median"] else ""
            lines.append(f"- {name}: step-time range {steps['min']:.2f}–{steps['max']:.2f} s{share}.")
    return "\n".join(lines) + "\n"
