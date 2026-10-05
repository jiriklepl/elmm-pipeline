import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys

from .bootstrap import setup, ensure_repo
from .build import build, preprocessing_defines
from .common import ROOT, clean_environment, command, digest_tree, python_environment, write_json
from .configuration import prepare_config
from .instrumentation import execution_coverage
from .report import render, summary
from .validation import compare_vtk, read_vtk, validate_log

TOOLS = ("baseline", "loki", "psyclone", "local")


def tool_list(value):
    values = list(dict.fromkeys(value.split(",")))
    if not values or any(v not in TOOLS for v in values):
        raise argparse.ArgumentTypeError("Choose comma-separated baseline,loki,psyclone,local")
    # Validation always requires a native reference.
    return ["baseline", *[v for v in values if v != "baseline"]]


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be positive")
    return result


def compiler_identity():
    versions = {}
    for env, default in (("FC", "gfortran"), ("CC", "gcc"), ("CXX", "g++")):
        executable = shutil.which(os.environ.get(env, default))
        if not executable:
            raise RuntimeError(f"Compiler {os.environ.get(env, default)} not found")
        versions[env] = {"executable": executable,
                         "version": command([executable, "--version"])[0]}
    return versions


def prepare_build(tool, args, setup_state, compilers):
    defines = preprocessing_defines(args.build_arg)
    repositories = ROOT / ".cache" / "repos"
    key_data = {"tool": tool, "dependencies": setup_state["pins"], "compilers": compilers,
                "packages": setup_state["packages"].get(tool), "build_args": args.build_arg,
                "preprocessor_defines": defines,
                "native_libraries": setup_state["native"],
                "source_hashes": {name: digest_tree(repositories / name / "src") for name in ("elmm", "PoisFFT")}}
    paths = [ROOT / "elmm_pipeline" / "build.py"]
    if tool == "local":
        key_data["local_compiler"] = digest_tree(repositories / "local-compiler" / "compiler")
    if tool != "baseline":
        paths += [ROOT / "elmm_pipeline" / name for name in (
            "adaptation.py", "discovery.py", "structural.py", "local_adapter.py", "recipes.py",
            "instrumentation.py", "common.py")]
        key_data["workers"] = digest_tree(ROOT / "elmm_pipeline" / "tool_workers")
        key_data["selection"] = {"sources": args.source_pattern, "include": args.select,
                                 "exclude": args.exclude, "marked_only": args.marked_only,
                                 "coverage": not args.no_coverage}
    key_data["implementation"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    # Custom compiled hooks affect build identity even when their argument path is unchanged.
    key_data["custom_inputs"] = {}
    for arg in args.build_arg:
        value = arg.split("=", 1)[-1]
        path = Path(value)
        if path.is_file():
            key_data["custom_inputs"][str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:16]
    directory = ROOT / "work" / "builds" / f"{tool}-{key}"
    stamp = directory / "complete.json"
    if stamp.is_file():
        data = json.loads(stamp.read_text())
        binary = directory / "bin" / "ELMM"
        if binary.is_file() and hashlib.sha256(binary.read_bytes()).hexdigest() == data["binary_sha256"]:
            return binary, data
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    ignore = shutil.ignore_patterns(".git", "__pycache__", "*.o", "*.mod", ".sconsign*")
    for name in ("elmm", "PoisFFT"):
        shutil.copytree(repositories / name, directory / name, ignore=ignore)
    source_root = directory / "elmm" / "src"
    adapter = {"tool": "baseline", "coverage": "Unmodified ELMM source"}
    if tool != "baseline":
        from .adaptation import adapt_project
        adapter = adapt_project(tool, source_root, ROOT / ".cache/envs" / tool / "bin/python",
                                directory / "generated", compiler_root=repositories / "local-compiler",
                                source_patterns=args.source_pattern, include=args.select,
                                exclude=args.exclude, marked_only=args.marked_only,
                                defines=defines, coverage=not args.no_coverage)
        if not adapter["regions"]:
            return None, {"identity": key_data, "adapter": adapter,
                          "reason": "No eligible regions accepted by this backend; native code preserved."}
    binary = build(directory / "elmm", directory / "PoisFFT", directory / "bin",
                   ROOT / ".cache/envs/build/bin/python", ROOT / ".cache/native/lib",
                   args.jobs, args.threads, args.build_arg,
                   extra_sources=[Path(p) for p in adapter.get("extra_sources", [])],
                   patched_sources=[source_root / p for p in adapter.get("changed_files", [])])
    data = {"identity": key_data, "adapter": adapter,
            "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "changed_source_sha256": {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                                      for name in adapter.get("changed_files", [])},
            "build_command": json.loads((binary.parent / "build-command.json").read_text())}
    # Adapter metadata can contain Paths; serialize these as explicit artifact paths.
    data = json.loads(json.dumps(data, default=str))
    write_json(stamp, data)
    return binary, data


def run(args):
    if args.warmups < 0:
        raise ValueError("--warmups cannot be negative")
    defines = preprocessing_defines(args.build_arg)
    if not (ROOT / ".cache/setup.json").exists():
        raise ValueError("Run pipeline.py setup first")
    setup_state = json.loads((ROOT / ".cache/setup.json").read_text())
    pins = json.loads((ROOT / "dependencies.json").read_text())
    if setup_state["pins"] != pins:
        raise ValueError("Dependency pins changed; run setup again")
    for name, spec in pins.items():
        ensure_repo(name, spec)
    for tool in args.tools:
        if tool != "baseline":
            installed = command([ROOT / ".cache/envs" / tool / "bin/python", "-m", "pip", "freeze"],
                                env=python_environment())[0].splitlines()
            if installed != setup_state["packages"].get(tool):
                raise ValueError(f"{tool} environment changed; run setup to record updated packages")
    # SCons executes from src/, so resolve explicitly supplied custom files now.
    normalized_args = []
    for arg in args.build_arg:
        key, separator, value = arg.partition("=")
        if separator and Path(value).is_file():
            value = str(Path(value).resolve())
        normalized_args.append(key + separator + value)
    args.build_arg = normalized_args
    missing = set(args.tools) - set(setup_state["tools"])
    if missing:
        raise ValueError(f"Run setup for missing tools: {', '.join(sorted(missing))}")
    source = Path(args.config).resolve() if args.config else ROOT / ".cache/repos/elmm/examples" / args.example
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    destination = Path(args.output).resolve() if args.output else ROOT / "results" / tag
    if destination.exists():
        raise ValueError(f"Output directory must be new: {destination}")
    destination.mkdir(parents=True)
    configuration = prepare_config(source, destination / "config", steps=args.steps, advection=args.advection)
    configuration["sha256"] = digest_tree(destination / "config")
    configuration["original_sha256"] = digest_tree(source if source.is_dir() else source.parent)
    compilers = compiler_identity()
    report = {"schema_version": 2, "created_utc": tag, "configuration": configuration,
              "settings": {"threads": args.threads, "repeats": args.repeats, "warmups": args.warmups,
                           "timeout_seconds": args.timeout, "atol": args.atol, "rtol": args.rtol,
                           "build_args": args.build_arg,
                           "preprocessor_defines": defines,
                           "source_patterns": args.source_pattern, "selection": args.select,
                           "exclusions": args.exclude, "marked_only": args.marked_only,
                           "coverage_enabled": not args.no_coverage,
                           "environment": {k: v for k, v in clean_environment(args.threads).items()
                                           if k.startswith(("OMP_", "OPENBLAS_", "MKL_"))}},
              "dependencies": setup_state["pins"], "packages": setup_state["packages"],
              "host": {"platform": platform.platform(), "machine": platform.machine(),
                       "processor": platform.processor(), "cpu_count": os.cpu_count()},
              "compilers": compilers, "native_libraries": setup_state["native"], "variants": {}}
    compiler_info = ROOT / ".cache/repos/local-compiler/source.json"
    if "local" in args.tools and compiler_info.exists():
        report["local_compiler"] = json.loads(compiler_info.read_text())
        actual = digest_tree(compiler_info.parent / "compiler")
        if actual != report["local_compiler"]["compiler_sha256"]:
            raise ValueError("Cached local compiler changed; repeat setup with its explicit source checkout")
    binaries = {}

    def save():
        write_json(destination / "report.json", report)
        (destination / "report.md").write_text(render(report))

    try:
        # Finish all compilation before starting timed runs.
        for tool in args.tools:
            print(f"Preparing {tool} application build", flush=True)
            report["variants"][tool] = {"status": "pending", "runs": []}
            binary, build_metadata = prepare_build(tool, args, setup_state, compilers)
            if binary is None:
                report["variants"][tool] = {"status": "unsupported", "build": build_metadata,
                                            "reason": build_metadata["reason"]}
                continue
            binaries[tool] = binary
            report["variants"][tool] = {"status": "pending", "build": build_metadata, "runs": []}
            artifact_dir = destination / "sources" / tool
            artifact_dir.mkdir(parents=True)
            for relative in build_metadata["adapter"].get("changed_files", []):
                target = artifact_dir / "native" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(binary.parent.parent / "elmm/src" / relative, target)
            shutil.copy2(binary.parent / "build.log", artifact_dir / "build.log")
            generated = binary.parent.parent / "generated"
            if generated.exists():
                shutil.copytree(generated, artifact_dir / "generated")
        reference = None
        reference_steps = None
        sequence = list(binaries)
        for index in range(args.warmups + args.repeats):
            # Rotate order to reduce systematic thermal/order bias; baseline runs first
            # in round zero to establish the deterministic reference state.
            order = sequence[index % len(sequence):] + sequence[:index % len(sequence)]
            for tool in order:
                variant = report["variants"][tool]
                warmup = index < args.warmups
                label = (f"warmup-{index + 1:03d}" if warmup else f"repeat-{index - args.warmups + 1:03d}")
                directory = destination / "runs" / tool / label
                shutil.copytree(destination / "config", directory)
                output = directory / "output"
                if output.exists():
                    shutil.rmtree(output)
                print(f"Running {tool} {label}", flush=True)
                log, wall = command([binaries[tool], *configuration["runtime_arguments"]], cwd=directory, env=clean_environment(args.threads),
                                    log=directory / "run.log", timeout=args.timeout)
                timings = validate_log(log, configuration["expected_max_steps"],
                                       expected_completed_steps=reference_steps)
                if tool != "baseline":
                    adapter = variant["build"]["adapter"]
                    timings["execution_coverage"] = execution_coverage(
                        log, adapter["regions"], enabled=adapter["coverage_enabled"])
                vtk = read_vtk(output / "out.vtk")
                timings["grid_shape"] = list(vtk[0])
                if reference is None:
                    if tool != "baseline":
                        raise RuntimeError("First run must be baseline")
                    reference = vtk
                    reference_steps = timings["completed_steps"]
                validation = compare_vtk(reference, vtk, atol=args.atol, rtol=args.rtol)
                record = {"warmup": warmup, "label": label, "wall_seconds": wall, **timings,
                          "validation": validation, "log": str(directory.relative_to(destination) / "run.log")}
                variant["runs"].append(record)
                if not validation["passed"]:
                    variant["status"] = "failed"
                    variant["reason"] = "Final application state differs from baseline beyond tolerance"
                    raise RuntimeError(f"{tool}: numerical validation failed; inspect {destination / 'report.json'}")
        for variant in report["variants"].values():
            if variant["status"] == "pending":
                variant["status"] = "ok"
                variant["summary"] = summary(variant["runs"])
                adapter = variant["build"]["adapter"]
                if adapter["tool"] != "baseline" and adapter["coverage_enabled"] and not any(
                        r["execution_coverage"]["executed_regions"] for r in variant["runs"]):
                    variant["status"] = "not_exercised"
                    variant["reason"] = "No outlined region executed for this configuration."
        save()
    except Exception as error:
        report["error"] = str(error)
        if "tool" in locals() and tool in report["variants"]:
            report["variants"][tool]["status"] = "failed"
            report["variants"][tool]["reason"] = str(error)
        save()
        raise
    print(f"Report: {destination / 'report.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Portable ELMM application benchmark pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    install = sub.add_parser("setup", help="Fetch pinned sources and provision private Python environments")
    install.add_argument("--tools", type=tool_list, default=list(TOOLS))
    install.add_argument("--local-compiler", type=Path, help="Explicit checkout containing compiler/; copied into cache")
    install.add_argument("--fc", help="GNU Fortran compiler executable (e.g. gfortran-15)")
    analyze = sub.add_parser("analyze", help="Discover eligible regions in any Fortran source tree")
    analyze.add_argument("--source-tree", type=Path, required=True)
    analyze.add_argument("--python", type=Path, required=True, help="Explicit Python environment with fparser")
    analyze.add_argument("--output", type=Path, required=True, help="New analysis artifact directory")
    analyze.add_argument("--source-pattern", action="append", help="Relative glob; default **/*.f90")
    analyze.add_argument("--define", action="append", default=[], help="Active preprocessor symbol")
    analyze.add_argument("--fc", help="GNU Fortran compiler used for preprocessing")
    measure = sub.add_parser("run", help="Build, execute, validate and report")
    select = measure.add_mutually_exclusive_group()
    select.add_argument("--example", default="channel", help="ELMM examples/ subdirectory (default: channel)")
    select.add_argument("--config", help="Any configuration directory, or a .conf file in it")
    measure.add_argument("--tools", type=tool_list, default=list(TOOLS))
    measure.add_argument("--steps", type=positive, help="Explicit cap; omitted preserves the original duration")
    measure.add_argument("--advection", type=int, choices=(2, 4, 5, 6), help="Explicitly override main.conf method")
    measure.add_argument("--threads", type=positive, default=4)
    measure.add_argument("--jobs", type=positive, default=min(8, os.cpu_count() or 1))
    measure.add_argument("--repeats", type=positive, default=3)
    measure.add_argument("--warmups", type=int, default=1)
    measure.add_argument("--timeout", type=positive, default=3600, help="Maximum seconds per application process")
    measure.add_argument("--atol", type=float, default=1e-6)
    measure.add_argument("--rtol", type=float, default=1e-5)
    measure.add_argument("--output", help="New result directory (default: results/<UTC timestamp>)")
    measure.add_argument("--build-arg", action="append", default=[], help="Repeat SCons name=value options, including custom hooks")
    measure.add_argument("--fc", help="GNU Fortran compiler executable (e.g. gfortran-15)")
    measure.add_argument("--source-pattern", action="append", help="Optional relative source glob; default **/*.f90")
    measure.add_argument("--select", action="append", default=[], help="Optional procedure or module::procedure glob")
    measure.add_argument("--exclude", action="append", default=[], help="Procedure glob to leave native")
    measure.add_argument("--marked-only", action="store_true", help="Restrict to explicitly marked source procedures")
    measure.add_argument("--no-coverage", action="store_true", help="Disable inserted execution markers; coverage is unverified")
    args = parser.parse_args()
    if getattr(args, "fc", None):
        os.environ["FC"] = args.fc
    try:
        if args.command == "setup":
            setup(args.tools, args.local_compiler)
        elif args.command == "analyze":
            from .discovery import discover_project
            if args.output.exists():
                raise ValueError("Analysis output directory must be new")
            data = discover_project(args.source_tree, args.python, args.output,
                                    patterns=args.source_pattern or ["**/*.f90"], defines=args.define)
            print(f"Discovered {len(data['regions'])} eligible regions; diagnostics in {args.output}")
        else:
            args.source_pattern = args.source_pattern or ["**/*.f90"]
            if any(not math.isfinite(t) or t < 0 for t in (args.atol, args.rtol)):
                raise ValueError("Validation tolerances must be finite and nonnegative")
            run(args)
    except (ValueError, RuntimeError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
