"""Execution evidence for arbitrary discovered regions, without source annotations."""
import re


def region_call(region, *, coverage=True):
    """Wrap a call with one marker per OpenMP thread when requested.

    Thread-local saved state avoids races inside existing parallel loops. Only
    first invocations enter the I/O lock. These are benchmark instrumentation,
    independent of optional !kernel source hints.
    """
    call = region["call"].strip()
    if not re.match(r"(?i)^call\s+[A-Za-z]\w*\s*\(", call):
        raise ValueError("Discovery must supply a direct generated CALL")
    if not coverage:
        return call + "\n"
    identifier = region["id"]
    if not re.fullmatch(r"[A-Za-z0-9_]+", identifier):
        raise ValueError("Invalid region identifier")
    flag = "pipeline_seen_" + identifier[:32]
    marker = "ELMM_PIPELINE_REGION " + identifier
    return "\n".join([
        "block", f"  logical, save :: {flag} = .false.",
        f"  !$omp threadprivate({flag})", f"  if (.not. {flag}) then",
        "    !$omp critical(pipeline_coverage_output)", f"    write(*,'(a)') '{marker}'",
        "    !$omp end critical(pipeline_coverage_output)", f"    {flag} = .true.",
        "  end if", "  " + call, "end block", "",
    ])


def execution_coverage(log, regions, *, enabled=True):
    if not enabled:
        return {"enabled": False, "executed_regions": [], "unobserved_regions": [],
                "note": "Execution coverage disabled; transformed-call execution is not verified."}
    seen = set(re.findall(r"(?m)^\s*ELMM_PIPELINE_REGION\s+([A-Za-z0-9_]+)\s*$", log))
    expected = {region["id"] for region in regions}
    if seen - expected:
        raise ValueError("Application emitted coverage absent from the build manifest")
    return {"enabled": True, "executed_regions": sorted(seen),
            "unobserved_regions": sorted(expected - seen)}
