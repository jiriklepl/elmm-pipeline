"""Direct source-task fields seed only their checked original CALL ancestors."""
import json
from copy import deepcopy
from hashlib import sha256

import pytest
from elmm_pipeline.checked_execution_plan import validate_checked_execution_plans
from elmm_pipeline.local_adapter import _scope_package_build
from test_conditional_return_record import seal
from test_ordinary_lexical_use_closure import parts, route_to, slot

from compiler.tests.test_hierarchical_child_regions import SOURCE, fixture
from compiler.tests.test_tile_private import SOURCE as TILE


@pytest.fixture(scope="module")
def issued(tmp_path_factory):
    directory = tmp_path_factory.mktemp("direct-source-closure")
    consumer = TILE.split("contains\n", 1)[1].rsplit("end module", 1)[0]
    consumer = consumer.replace("advance(a,b,n,t)", "consumer(b,a,n)")
    consumer = consumer.replace("integer,intent(in)::n,t", "integer,intent(in)::n\ninteger,parameter::t=4")
    consumer = consumer.replace("tmp(li)=3.0_8*a(i)", "tmp(li)=3.0_8*a(i)+real(mask(i),8)")
    old = SOURCE.split("subroutine consumer", 1)[1].split("end subroutine", 1)[0]
    source = SOURCE.replace("subroutine consumer" + old + "end subroutine", consumer.strip())
    source = source.replace("real(8)::observed", "real(8)::observed\ninteger,allocatable::mask(:)")
    path, builder = fixture(directory, source)
    builder.facts["captures"]["hierarchical::mask"] = dict(builder.facts["captures"]["hierarchical::left"])
    outputs, _report = builder.run()
    manifest = json.loads(outputs["scope-manifest.json"])
    package = directory / "package"
    for name, content in outputs.items():
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content)
    assert sha256(path.read_bytes()).hexdigest() == builder.facts["sources"][str(path)]
    return manifest, package, builder.facts


def test_direct_module_field_survives_conditional_child_and_package_validation(issued):
    manifest, package, facts = issued
    report = deepcopy(manifest)
    scope, plan, epoch, contract = parts(report)
    task, = next(m for m in epoch["members"] if m["procedure"] == "hierarchical::consumer")["source_device_tasks"]
    assert "call_projection" not in task["execution_binding"]
    root = "hierarchical::mask"
    assert task["execution_binding"]["canonical_resources"][root] == root
    assert not any(root in unit["resources"].values() for p in plan["procedures"] for unit in p["units"])
    assert route_to(contract, "hierarchical::middle")["canonical_resources"][root] == root
    assert route_to(contract, "hierarchical::consumer")["canonical_resources"][root] == root
    assert root not in route_to(contract, "hierarchical::producer")["canonical_resources"]
    validate_checked_execution_plans(report, capture_facts=facts)
    _scope_package_build({"artifact_root": str(package), "scope_manifest": report})
    assert report == manifest


def test_direct_source_sibling_cannot_justify_foreign_implicit_capture(issued):
    report = deepcopy(issued[0])
    scope, plan, _epoch, contract = parts(report)
    root, name = "hierarchical::mask", "hierarchical::producer"
    route = route_to(contract, name)
    number = next(row["registration_identity"] for row in plan["resources"] if row["resource"] == root)
    local_slot = slot(name, root)
    route["canonical_resources"][root] = root
    route["canonical_handles"][local_slot] = number
    route["handle_forwarding"][local_slot] = number
    for rows in (plan["procedures"], scope["module_coordinators"]):
        next(row for row in rows if row["procedure"] == name)["calls"][0]["handle_bindings"][local_slot] = number
    seal(report)
    with pytest.raises(ValueError, match="conditional return child formal-to-canonical association differs"):
        validate_checked_execution_plans(report, capture_facts=issued[2])
