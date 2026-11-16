"""Declarative application project profiles for generic Fortran offloading."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

_CONFIG_MODES = {"elmm-conf", "generic"}
_BUILD_MODES = {"scons-elmm", "cmake", "command"}
_VALIDATION_MODES = {"elmm-vtk", "json-stdout", "files"}


@dataclass(frozen=True)
class ProjectProfile:
    """Configuration describing how an application stages, builds, and validates."""

    name: str = "elmm"
    source_subdir: str = "src"
    executable_name: str = "ELMM"
    default_defines: tuple[str, ...] = ("DPREC",)
    config_mode: str = "elmm-conf"
    build_mode: str = "scons-elmm"
    validation_mode: str = "elmm-vtk"
    build_command: tuple[str, ...] | None = None
    runtime_arguments: tuple[str, ...] = ()
    output_files: tuple[str, ...] = ()
    env_strip_prefixes: tuple[str, ...] = ("ELMM_",)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("ProjectProfile name must be a non-empty string")
        if not isinstance(self.executable_name, str) or not self.executable_name.strip():
            raise ValueError("ProjectProfile executable_name must be a non-empty string")
        if self.config_mode not in _CONFIG_MODES:
            raise ValueError(f"Unsupported config_mode {self.config_mode!r}; expected one of {sorted(_CONFIG_MODES)}")
        if self.build_mode not in _BUILD_MODES:
            raise ValueError(f"Unsupported build_mode {self.build_mode!r}; expected one of {sorted(_BUILD_MODES)}")
        if self.validation_mode not in _VALIDATION_MODES:
            raise ValueError(
                f"Unsupported validation_mode {self.validation_mode!r}; expected one of {sorted(_VALIDATION_MODES)}"
            )
        if self.build_mode == "command" and not self.build_command:
            raise ValueError("ProjectProfile with build_mode='command' requires a non-empty build_command")

    def to_dict(self) -> dict:
        data = asdict(self)
        data["default_defines"] = list(self.default_defines)
        data["runtime_arguments"] = list(self.runtime_arguments)
        data["output_files"] = list(self.output_files)
        data["env_strip_prefixes"] = list(self.env_strip_prefixes)
        if data["build_command"] is not None:
            data["build_command"] = list(self.build_command)
        return {"schema_version": 1, **data}


def ELMMProfile() -> ProjectProfile:
    """Return the default profile matching the ELMM application conventions."""
    return ProjectProfile()


def load_project_profile(path: str | Path | None = None) -> ProjectProfile:
    """Load and validate a schema-v1 JSON project profile, or return ELMM defaults."""
    if path is None:
        return ELMMProfile()
    raw = json.loads(Path(path).resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ValueError("Project profile requires a JSON object with schema_version=1")
    allowed = {
        "schema_version",
        "name",
        "source_subdir",
        "executable_name",
        "default_defines",
        "config_mode",
        "build_mode",
        "validation_mode",
        "build_command",
        "runtime_arguments",
        "output_files",
        "env_strip_prefixes",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown project profile fields: {sorted(unknown)}")
    build_cmd = raw.get("build_command")
    if build_cmd is not None:
        if not isinstance(build_cmd, list) or not all(isinstance(x, str) and x for x in build_cmd):
            raise ValueError("Project profile build_command must be a list of non-empty strings")
        build_cmd = tuple(build_cmd)
    return ProjectProfile(
        name=raw.get("name", "elmm"),
        source_subdir=raw.get("source_subdir", "src"),
        executable_name=raw.get("executable_name", "ELMM"),
        default_defines=tuple(raw.get("default_defines", ("DPREC",))),
        config_mode=raw.get("config_mode", "elmm-conf"),
        build_mode=raw.get("build_mode", "scons-elmm"),
        validation_mode=raw.get("validation_mode", "elmm-vtk"),
        build_command=build_cmd,
        runtime_arguments=tuple(raw.get("runtime_arguments", ())),
        output_files=tuple(raw.get("output_files", ())),
        env_strip_prefixes=tuple(raw.get("env_strip_prefixes", ("ELMM_",))),
    )
