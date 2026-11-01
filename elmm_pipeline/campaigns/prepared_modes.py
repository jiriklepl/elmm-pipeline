"""Import four independently authenticated modes without rebuilding the controls."""
from __future__ import annotations

import copy
from pathlib import Path

from .identity import digest, read, write

BASELINE_REVISION = "7a612e40b5790e2d112ea137c6a38a9ff0ef7d41"


def load_adapter(path):
    """Resolve exact public manifest sidecars; embedded and referenced data must agree."""
    def load(package, active):
        result = dict(package)
        reference = package.get("scope_manifest_reference")
        if reference is not None:
            if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
                raise ValueError("exact scope manifest reference required")
            manifest = Path(reference["path"]).resolve(strict=True)
            if manifest in active:
                raise ValueError("cyclic scope manifest reference")
            if (manifest != (Path(package["artifact_root"]) / "scope-manifest.json").resolve()
                    or digest(manifest) != reference["sha256"]):
                raise ValueError("scope manifest reference path or hash differs")
            contents = read(manifest)
            if not isinstance(contents, dict):
                raise ValueError("scope manifest must be an object")
            if "scope_manifest" in package and package["scope_manifest"] != contents:
                raise ValueError("scope manifest reference contents differ")
            result["scope_manifest"] = contents
            active = {*active, manifest}
        if "scope_packages" in package:
            if not isinstance(package["scope_packages"], list):
                raise ValueError("scope packages must be a list")
            result["scope_packages"] = [load(child, active) for child in package["scope_packages"]]
        return result

    return load(read(path), {Path(path).resolve(strict=True)})


def import_modes(baseline_path, candidate_path, identity_path, output, *, validate_prepared):
    """Use the public prepared validator for each build, retaining its own proof/profile."""
    baseline_path, candidate_path, identity_path = map(Path, (baseline_path, candidate_path, identity_path))
    baseline, candidate, identity = map(read, (baseline_path, candidate_path, identity_path))
    if (identity.get("revisions", {}).get("main") != BASELINE_REVISION
            or identity.get("prepared_sha256") != digest(baseline_path)):
        raise ValueError("exact frozen 7a612 baseline identity required")
    for name in ("original_application", "contract_sha256", "field_policy", "threads", "build_args"):
        if baseline.get(name) != candidate.get(name):
            raise ValueError("four-mode common basis differs: " + name)
    if baseline.get("threads") != 4:
        raise ValueError("four-thread comparison required")
    selected, guards = {}, {}
    for key, path, prepared in (("baseline", baseline_path, baseline), ("candidate", candidate_path, candidate)):
        authenticated, bound = validate_prepared(prepared, prepared, threads=4, build_args=prepared["build_args"])
        selected[key] = {name: authenticated[name] for name in ("builds", "diagnostic_builds")}
        del authenticated  # Expanded public proof metadata is not duplicated into the campaign catalog.
        bound = dict(bound)
        for built in prepared["builds"].values():
            bound[built["build_receipt"]] = built["build_receipt_sha256"]
        profile = prepared.get("calibration_profile")
        if profile:
            bound[profile["path"]] = profile["sha256"]
        for filename, expected in bound.items():
            if filename in guards and guards[filename] != expected:
                raise ValueError("conflicting prepared artifact authority")
            guards[filename] = expected
        guards[str(path.resolve())] = digest(path)
    guards[str(identity_path.resolve())] = digest(identity_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    modes = []
    for name, origin, source_mode in (("native", "baseline", "native"), ("be58", "baseline", "committed"),
                                      ("baseline-7a612", "baseline", "reworked"), ("candidate", "candidate", "reworked")):
        prepared = baseline if origin == "baseline" else candidate
        built = selected[origin]["builds"][source_mode]
        mode = {"name": name, "binary": built["binary"], "sha256": built["sha256"],
                "build_receipt": built["build_receipt"], "prepared_source_mode": source_mode,
                "prepared_manifest": str((baseline_path if origin == "baseline" else candidate_path).resolve()),
                "calibration_profile": copy.deepcopy(prepared.get("calibration_profile")) if source_mode == "reworked" else None,
                "compiler": copy.deepcopy(prepared.get("compilers", {}).get(source_mode)),
                "diagnostic_build": copy.deepcopy(selected[origin]["diagnostic_builds"][source_mode])}
        if source_mode != "native":
            # References remain compact; validation above already authenticated the complete public proof.
            adapter = output / (name + "-adapter.json")
            write(adapter, prepared["generation"][source_mode])
            mode["adapter"] = str(adapter)
            guards[str(adapter)] = digest(adapter)
        modes.append(mode)
    for path, expected in guards.items():
        if digest(path) != expected:
            raise ValueError("prepared input changed while importing modes: " + path)
    result = {"schema_version": 1, "status": "authenticated_modes_no_runs", "modes": modes,
              "baseline_modes": ["native", "be58", "baseline-7a612"], "candidate_mode": "candidate",
              "threads": 4, "repetitions": 7, "warmups": 1, "timings_reused": False,
              "original_application": baseline["original_application"], "contract_sha256": baseline["contract_sha256"],
              "field_policy": baseline["field_policy"], "external_files": guards,
              "qualification": "Mode import only: same-config complete-field/halo, natural correctness, fixed FFTW and fresh rotated timing gates remain mandatory."}
    write(output / "modes.json", result)
    return result
