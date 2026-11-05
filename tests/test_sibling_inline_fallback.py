"""A lazy domain's first helper tag is not the later helper's CALL ancestor."""

import json
from copy import deepcopy

import pytest
from elmm_pipeline.gpu_validation import offload_decisions
from test_conditional_return_record import seal

from compiler.tests.test_conditional_return import compile_fixture


@pytest.fixture(scope="module")
def original(tmp_path_factory):
    _outputs, report, scope, _contract = compile_fixture(tmp_path_factory.mktemp("sibling-fallback"))
    records = report["runtime_provenance"]["records"]
    owner = next(r for r in records if r["kind"] == "owner" and r["procedure"] == "ancestor_fixture::producer")
    caller = next(r for r in records if r["kind"] == "procedure" and r["procedure"] == "ancestor_fixture::consumer")
    catalog = next(r for r in report["child_numerical_regions"] if r["procedure"] == caller["procedure"])
    region = next(r for r in catalog["regions"] if r.get("used"))
    worker = next(r for r in records if r["kind"] == "procedure" and r["procedure"] == region["entry"])
    implementation = next(
        r
        for r in records
        if r["kind"] == "implementation" and r.get("procedure_id") == worker["id"] and r["backend"] == "cuda"
    )
    segment = next(
        r
        for r in records
        if r["kind"] == "segment"
        and r["procedure"] == caller["procedure"]
        and r["first_line"] <= region["first_line"] <= region["last_line"] <= r["last_line"]
    )
    unknown = dict.fromkeys(("owner", "procedure", "segment", "operation", "implementation", "boundary"), "unknown")
    create = {
        "schema_version": 1,
        "event": "domain_create",
        "domain": 1,
        "context": 0,
        "physical_context": 1,
        "last_context": 0,
        "device": 0,
        "failed": False,
        **unknown,
    }
    attach = {**create, "event": "domain_attach", "context": 2}
    position = (
        "FORT_SCOPED position bytes=0 provenance_version=1 context=2 domain_version=1 domain=1 domain_role=execution "
        f"owner={owner['id']} procedure={caller['id']} segment={segment['id']} operation=unknown implementation={implementation['id']} boundary=unknown"
    )
    decision = "FORT_OFFLOAD entry=region mode=native-scoped-resource gpu_units=0 cpu_units=1 unit_kind=regions"
    return report, ["FORT_SCOPED domain " + json.dumps(row) for row in (create, attach)] + [position, decision]


def test_siblings_use_the_validated_return_contract_and_two_root_paths(original):
    manifest, lines = original
    before = deepcopy(manifest)
    result = offload_decisions("\n".join(lines), [], scope_manifest=manifest)
    (row,) = result["authenticated_inline_fallbacks"]
    route = row["borrowed_owner_route"]
    assert route["kind"] == "checked sibling helper storage epoch"
    assert route["root_to_owner"][-1]["callee"] == "ancestor_fixture::producer"
    assert route["root_to_caller"][-1]["callee"] == "ancestor_fixture::consumer"
    assert route["dynamic_call_ancestry_inferred"] is False
    assert row["gpu_execution_confirmed"] is False
    assert result["gpu_units"] == 0
    assert manifest == before


@pytest.mark.parametrize(
    "fault",
    [
        "contract_missing",
        "contract_identity",
        "escape",
        "payload",
        "missing_owner_call",
        "missing_caller_call",
        "missing_coordinator",
        "foreign_owner_source",
        "foreign_caller_source",
        "missing_member",
    ],
)
def test_sibling_membership_alone_cannot_authorize_fallback(original, fault):
    manifest, lines = deepcopy(original)
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    members = {p["procedure"]: p for p in plan["procedures"]}
    if fault == "contract_missing":
        epoch.pop("return_contract")
    elif fault == "contract_identity":
        epoch["return_contract"]["identity"] = "f" * 64
    elif fault in {"escape", "payload"}:
        epoch["return_contract"]["escape_completion_assumed" if fault == "escape" else "payload_access_authorized"] = (
            True
        )
        seal(manifest)
    elif fault in {"missing_owner_call", "missing_caller_call"}:
        members["ancestor_fixture::producer" if fault == "missing_owner_call" else "ancestor_fixture::consumer"][
            "calls"
        ] = []
        seal(manifest)
    elif fault == "missing_coordinator":
        scope["module_coordinators"] = [
            c for c in scope["module_coordinators"] if c["procedure"] != "ancestor_fixture::consumer"
        ]
    elif fault in {"foreign_owner_source", "foreign_caller_source"}:
        members["ancestor_fixture::producer" if fault == "foreign_owner_source" else "ancestor_fixture::consumer"][
            "source_sha256"
        ] = "f" * 64
        seal(manifest)
    else:
        epoch["return_contract"]["members"] = epoch["return_contract"]["members"][:-1]
        seal(manifest)
    with pytest.raises(ValueError, match="Inline fallback|checked execution plan"):
        offload_decisions("\n".join(lines), [], scope_manifest=manifest)
