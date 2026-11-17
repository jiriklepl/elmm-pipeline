"""Resolve prepared campaign-mode adapters from their public manifests."""
from __future__ import annotations

from pathlib import Path

from .identity import digest, read



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
