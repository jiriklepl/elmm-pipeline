"""Execution evidence for arbitrary discovered regions, without source annotations."""
import re


def execution_coverage(log, regions, *, enabled=True):
    if not enabled:
        return {"enabled": False, "executed_regions": [], "unobserved_regions": [],
                "note": "Execution coverage disabled; transformed-call execution is not verified."}
    seen = set(re.findall(r"(?<![A-Za-z0-9_])ELMM_PIPELINE_REGION[ \t]+([A-Za-z0-9_]+)(?![A-Za-z0-9_])", log))
    expected = {region["id"] for region in regions}
    if seen - expected:
        raise ValueError("Application emitted coverage absent from the build manifest")
    return {"enabled": True, "executed_regions": sorted(seen),
            "unobserved_regions": sorted(expected - seen)}
