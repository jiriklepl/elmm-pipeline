"""Freeze public OpenMP launch settings without interpreting placement costs."""
from __future__ import annotations

import os
import re

OPENMP_LAUNCH_VARIABLES = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT")
_IDENTITY_FIELDS = {"OMP_WAIT_POLICY": "omp_wait_policy", "GOMP_SPINCOUNT": "gomp_spincount"}


def _values(values, *, complete):
    if (not isinstance(values, dict) or not set(values) <= set(OPENMP_LAUNCH_VARIABLES)
            or complete and set(values) != set(OPENMP_LAUNCH_VARIABLES)):
        raise ValueError("OpenMP launch environment requires wait/spin settings only")
    for value in values.values():
        if value is not None and (not isinstance(value, str) or len(value) > 128
                                 or any(ord(char) < 32 or ord(char) > 126 for char in value)):
            raise ValueError("OpenMP launch settings must be bounded printable strings or null (unset)")
    return dict(values)


def freeze_launch_environment(profile, configured=None, *, environment=None):
    """Transport compatible profile values, explicit values or preparation state.

    Rejected cost families still identify the environment in which their raw
    observations were collected. This does not admit those costs or choose an
    implementation. Never let a conflicting override disguise another launch.
    """
    configured = _values({} if configured is None else configured, complete=False)
    observed = os.environ if environment is None else environment
    recorded, sections = None, []
    numerical = profile.get("numerical", {})
    numerical_identity = numerical.get("identity") if isinstance(numerical, dict) else None
    affinity = numerical_identity.get("cpu_affinity") if isinstance(numerical_identity, dict) else None
    for name in ("cpu_dependency", "cpu_execution_protocol"):
        if name not in profile:
            continue
        section = profile[name]
        identity = section.get("execution_identity") if isinstance(section, dict) else None
        if (not isinstance(section, dict) or type(section.get("schema_version")) is not int
                or section["schema_version"] != 1 or not isinstance(identity, dict)
                or any(type(identity.get(field)) is not int or identity[field] != profile.get(field)
                       for field in ("cpu_threads", "precision_bits"))
                or not isinstance(affinity, list) or not affinity
                or any(type(cpu) is not int or cpu < 0 for cpu in affinity)
                or affinity != sorted(set(affinity)) or identity.get("cpu_affinity") != affinity
                or any(type(cpu) is not int for cpu in identity.get("cpu_affinity", []))):
            raise ValueError("incompatible OpenMP launch environment identity: " + name)
        if any(field not in identity for field in _IDENTITY_FIELDS.values()):
            raise ValueError("missing OpenMP launch environment identity: " + name)
        values = _values({variable: identity[field] for variable, field in _IDENTITY_FIELDS.items()}, complete=True)
        if recorded is not None and values != recorded:
            raise ValueError("profile OpenMP launch environment identities disagree")
        recorded = values
        sections.append(name)
    if recorded is not None:
        if any(recorded[name] != value for name, value in configured.items()):
            raise ValueError("configured OpenMP launch environment conflicts with calibration")
        values, provenance = recorded, {name: "profile" for name in OPENMP_LAUNCH_VARIABLES}
    else:
        values = _values({name: configured[name] if name in configured else observed.get(name)
                          for name in OPENMP_LAUNCH_VARIABLES}, complete=True)
        provenance = {name: "configuration" if name in configured else "preparation_environment"
                      for name in OPENMP_LAUNCH_VARIABLES}
    return {"schema_version": 1, "values": values, "provenance": provenance, "profile_sections": sections}


def apply_launch_environment(environment, contract, *, overrides=None):
    """Apply to a fresh subprocess environment; None preserves legacy campaigns."""
    if contract is None:
        return
    if (not isinstance(contract, dict) or type(contract.get("schema_version")) is not int
            or contract["schema_version"] != 1):
        raise ValueError("invalid frozen OpenMP launch environment")
    values = _values(contract.get("values"), complete=True)
    if overrides is not None and any(name in overrides and overrides[name] != value
                                     for name, value in values.items()):
        raise ValueError("external phase conflicts with frozen OpenMP launch environment")
    if "OMP_THREAD_LIMIT" in environment:
        raw_limit = environment["OMP_THREAD_LIMIT"]
        if (not isinstance(raw_limit, str) or len(raw_limit) > 128
                or re.fullmatch(r"[ \t]*\+?[0-9]+[ \t]*", raw_limit) is None
                or not 0 < int(raw_limit) <= (1 << 31) - 1):
            raise ValueError("OMP_THREAD_LIMIT must be a bounded positive decimal integer")
        if int(raw_limit) < 4:
            raise ValueError("OMP_THREAD_LIMIT is below the four-thread campaign budget")
    for name, value in values.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value


def launch_environment_values(environment):
    return {name: environment.get(name) for name in (*OPENMP_LAUNCH_VARIABLES, "OMP_THREAD_LIMIT")}
