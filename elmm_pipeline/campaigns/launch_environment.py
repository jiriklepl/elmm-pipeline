"""Freeze public OpenMP launch settings for every process of a campaign."""
from __future__ import annotations

import os
import re

OPENMP_LAUNCH_VARIABLES = ("OMP_WAIT_POLICY", "GOMP_SPINCOUNT")


def _values(values, *, complete):
    if (not isinstance(values, dict) or not set(values) <= set(OPENMP_LAUNCH_VARIABLES)
            or complete and set(values) != set(OPENMP_LAUNCH_VARIABLES)):
        raise ValueError("OpenMP launch environment requires wait/spin settings only")
    for value in values.values():
        if value is not None and (not isinstance(value, str) or len(value) > 128
                                 or any(ord(char) < 32 or ord(char) > 126 for char in value)):
            raise ValueError("OpenMP launch settings must be bounded printable strings or null (unset)")
    return dict(values)


def freeze_launch_environment(configured=None, *, environment=None):
    """Record explicit values, and the preparation environment for the others."""
    configured = _values({} if configured is None else configured, complete=False)
    observed = os.environ if environment is None else environment
    values = _values({name: configured[name] if name in configured else observed.get(name)
                      for name in OPENMP_LAUNCH_VARIABLES}, complete=True)
    provenance = {name: "configuration" if name in configured else "preparation_environment"
                  for name in OPENMP_LAUNCH_VARIABLES}
    return {"schema_version": 1, "values": values, "provenance": provenance}


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
