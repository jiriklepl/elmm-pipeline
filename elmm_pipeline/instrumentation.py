"""Execution evidence for arbitrary discovered regions, without source annotations."""
import re


def gpu_level_name(region):
    return "pipeline_level_" + region["id"]


def _instrumented_call(region, *, coverage=True):
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


def offload_contract(region):
    contract = region.get("transformation", {}).get("compiler_report", {}).get("offload") or {}
    if not isinstance(contract, dict):
        raise ValueError("Compiler offload dispatch contract must be an object")
    query = contract.get("native_fallback_query")
    if query is not None and not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", str(query)):
        raise ValueError("Compiler fallback query must be a simple Fortran function name")
    return contract


def _policy_call(region, call, scope, contract, guard):
    """Query a compiler-owned policy only after native preconditions are safe."""
    query = contract["native_fallback_query"]
    if not isinstance(region.get("actual_args"), list):
        raise ValueError("Compiler policy query requires discovered actual_args")
    expression = query + "( &\n" + ", &\n".join("  " + value for value in region["actual_args"]) + ")"
    allocated = scope.get("allocated_arrays", [])
    if any(not re.fullmatch(r"[A-Za-z]\w*", name) for name in allocated):
        raise ValueError("Allocated captures must be simple source identifiers")
    ready = "pipeline_ready_" + region["id"]
    while re.search(r"\b" + ready + r"\b", region["original_text"] + call + guard + expression, re.I):
        ready += "_"
    # Every team member must observe preceding native writes before querying
    # shared shape/scalar inputs and choosing GPU or native worksharing.
    before = "!$omp barrier\n" if scope["dispatch"] == "collective_single" else ""
    if contract.get("collective_entry"):
        if scope["dispatch"] != "collective_single":
            raise ValueError("Collective compiler entry requires a proved native OpenMP team")
        call += "!$omp barrier\n"
    elif scope["dispatch"] == "collective_single":
        call = "!$omp single\n" + call + "!$omp end single\n"
    conditions = [" .and. ".join(f"allocated({name})" for name in allocated)] if allocated else []
    conditions.append(guard)
    checks = "".join(f"if ({condition}) then\n" for condition in conditions)
    checks += f"{ready} = {expression}\n" + "end if\n" * len(conditions)
    return (before + f"block\nlogical :: {ready}\n{ready} = .false.\n" + checks
            + f"if ({ready}) then\n" + call + "else\n"
            + region["original_text"].rstrip() + "\nend if\nend block\n")


def region_call(region, *, coverage=True):
    call = _instrumented_call(region, coverage=coverage)
    scope = region.get("gpu_scope")
    if scope is None:
        return call
    dispatch = scope["dispatch"]
    expected = scope["expected_omp_level"]
    if (dispatch, expected) not in {("serial", 0), ("collective_single", 1)}:
        raise ValueError("Unsupported GPU/OpenMP execution scope")
    guard = f"{gpu_level_name(region)}() == {expected}"
    if scope.get("guard"):
        guard += " .and. (" + scope["guard"] + ")"
    contract = offload_contract(region)
    if contract.get("native_fallback_query"):
        return _policy_call(region, call, scope, contract, guard)
    allocated = scope.get("allocated_arrays", [])
    if allocated:
        if any(not re.fullmatch(r"[A-Za-z]\w*", name) for name in allocated):
            raise ValueError("Allocated captures must be simple source identifiers")
        # Fortran does not require short-circuit evaluation. Never evaluate
        # bounds, strides or descriptors until all captured arrays exist.
        ready = "pipeline_ready_" + region["id"]
        used = region["original_text"] + call + guard
        while re.search(r"\b" + ready + r"\b", used, re.I):
            ready += "_"
        checks = " .and. ".join(f"allocated({name})" for name in allocated)
        before = ""
        if dispatch == "collective_single":
            # Complete preceding native work before every team member reads
            # shared allocation status and chooses the same execution path.
            before = "!$omp barrier\n"
            call = "!$omp single\n" + call + "!$omp end single\n"
        return (before + f"block\nlogical :: {ready}\n{ready} = .false.\n"
                + f"if ({checks}) then\n{ready} = {guard}\nend if\n"
                + f"if ({ready}) then\n" + call + "else\n"
                + region["original_text"].rstrip() + "\nend if\nend block\n")
    if dispatch == "collective_single":
        # Every team member reaches this guard. Finish preceding native work
        # before one thread transfers complete arrays; publish results before
        # any team member continues. The fallback remains collective too.
        call = "!$omp barrier\n!$omp single\n" + call + "!$omp end single\n"
    return (f"if ({guard}) then\n" + call + "else\n" +
            region["original_text"].rstrip() + "\nend if\n")


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
