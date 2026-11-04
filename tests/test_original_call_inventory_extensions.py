"""Original host associations and graph sequence order grant no device storage."""
import json
from copy import deepcopy

import pytest
from compiler.tests.test_graph_source_owner import emit
from compiler.tests.test_hierarchical_child_regions import SOURCE as HIERARCHICAL
from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_native_host_fixed_arrays import SOURCE

from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.selected_epoch_record import _original_sequence
from elmm_pipeline.source_task_call_record import (
    _original_call,
    validate_call_inventory,
)


@pytest.fixture(scope="module", params=["unused", "unused_rank4", "scalar_element", "renamed_parent"])
def authentic(request, tmp_path_factory):
    source = SOURCE
    if request.param.startswith("unused"):
        rank4 = request.param == "unused_rank4"
        source = source.replace("smooth(field,kinds,n)", "smooth(field,kinds,n,unused)")
        source = source.replace("real(8),intent(inout)::field(:)",
                                "real(8),intent(inout)::field(:)\nreal(8),intent(in)::unused(" +
                                (":,:,:,:" if rank4 else ":") + ")")
        if rank4:
            source = source.replace("implicit none", "implicit none\nreal(8)::unused4(2,2,2,2)")
        source = source.replace("smooth(a,types,n)", "smooth(a,types,n," + ("unused4" if rank4 else "b") + ")")
        source = source.replace("smooth(b,other_types,n)", "smooth(b,other_types,n," + ("unused4" if rank4 else "a") + ")")
    elif request.param == "scalar_element":
        from compiler.tests.test_hierarchical_scalar_elements import SOURCE_ELEMENTS
        from compiler.tests.test_hierarchical_scalar_elements import (
            emit as emit_elements,
        )

        source = SOURCE_ELEMENTS.replace("real(8)::observed", "real(8)::observed\ninteger::weights(8)")
        source = source.replace("integer::i\ndo i=1,n\na(i)=scale*b(i)\nenddo", """integer::i,j
real(8)::pencil(n)
!$omp parallel do private(i,j,pencil) schedule(runtime)
do j=1,1
pencil=b
do i=2,n-1
if(weights(i)==0) a(i)=scale*0.5_8*(pencil(i-1)+pencil(i+1))
enddo
enddo
!$omp end parallel do""")
        _, _, _, _, outputs, report = emit_elements(tmp_path_factory.mktemp(request.param), source)
        return request.param, json.loads(outputs["scope-manifest.json"])
    else:
        source = source.replace("contains\n", """contains
subroutine bridge(renamed,kinds,n)
real(8),intent(inout)::renamed(:)
integer,intent(in)::kinds(:),n
call smooth(renamed,kinds,n)
end subroutine
""", 1).replace("call smooth(a,types,n)", "call bridge(a,types,n)").replace(
            "call smooth(b,other_types,n)", "call bridge(b,other_types,n)")
    _, outputs, report = emit(tmp_path_factory.mktemp(request.param), source)
    assert report["scope_count"] == 1
    return request.param, json.loads(outputs["scope-manifest.json"])


def test_original_host_call_inventory_round_trips(authentic):
    kind, manifest = authentic
    before = deepcopy(manifest)
    validate_checked_execution_plans(manifest)
    assert manifest == before
    projections = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["native_opaque_accesses"]
    assert projections
    if kind.startswith("unused"):
        nodes = [n for p in projections for n in p["call_projection"]["members"]
                 if n["procedure"] == "graph_fixture::smooth"]
        assert nodes and all("argument::unused" not in n["resource_slots"] for n in nodes)


@pytest.mark.parametrize("authentic", ["unused", "unused_rank4"], indirect=True)
@pytest.mark.parametrize("fault", ["actual", "type", "rank", "storage", "section", "caller_slot"])
def test_unregistered_formal_does_not_weaken_registered_associations(authentic, fault):
    kind, manifest = authentic
    assert kind.startswith("unused")
    value = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["native_opaque_accesses"][0]["call_projection"]
    nodes = {n["procedure"]: deepcopy(n) for n in value["members"]}
    edge = next(e for e in value["calls"] if e["callee"] == "graph_fixture::smooth")
    call = deepcopy(edge["source_call"])
    caller, callee = nodes[edge["caller"]], nodes[edge["callee"]]
    _original_call(call, callee, caller)
    row = next(r for r in call["resource_mappings"] if r["formal"] == "field")
    if fault == "actual": row["resource"] = "argument::foreign"
    elif fault == "type": row["actual_descriptor"]["type"] = "integer"
    elif fault == "rank": row["formal_descriptor"]["rank"] = 4
    elif fault == "storage": row["storage"] = "section"
    elif fault == "section": row["section"] = {"lower": 2}
    elif fault == "caller_slot": caller["resource_slots"].pop(row["resource"])
    with pytest.raises(ValueError):
        _original_call(call, callee, caller)


@pytest.fixture(scope="module")
def commented_graph(tmp_path_factory):
    source = HIERARCHICAL.replace("call producer(", "! before selected producer\ncall producer(").replace(
        "call middle(", "! comment between siblings\n! another comment\ncall middle(")
    _, builder = fixture(tmp_path_factory.mktemp("commented-graph"), source)
    _, manifest = builder.run()
    assert manifest["scope_count"] == 1
    validate_checked_execution_plans(manifest)
    selection = manifest["scopes"][0]["checked_execution_plan"]["storage_epoch"]["return_contract"]["root_selection"]
    return manifest, selection


def test_comments_do_not_break_original_sequence_adjacency(commented_graph):
    _, selection = commented_graph
    indices = [int(n.rsplit("/", 1)[1]) for n in [*selection["selected_node_ids"], selection["cut_before"]]]
    assert indices != list(range(indices[0], indices[0] + len(indices)))


@pytest.mark.parametrize("fault", ["digest", "reverse", "omit", "cut_inside", "repeat", "parent_kind"])
def test_sparse_indices_still_require_the_original_order(commented_graph, fault):
    manifest, selection = deepcopy(commented_graph)
    structure = manifest["native_effects"]["structured_effects"]
    selected = selection["selected_node_ids"]
    cut = selection["cut_before"]
    parent = next(n for n in structure["nodes"] if n["id"] == selection["parent_sequence"])
    if fault == "digest":
        structure["structured_identity"] = "f" * 64
    elif fault == "reverse":
        selected.reverse()
    elif fault == "omit":
        assert len(selected) >= 2
        selected.pop()
    elif fault == "cut_inside":
        cut = selected[-1]
    else:
        if fault == "repeat":
            parent["children"].append(parent["children"][0])
        else:
            parent["kind"] = "branch"
        payload = {k: v for k, v in structure.items()
                   if k not in {"schema_version", "structured_identity", "node_limit"}}
        payload["version"] = structure["schema_version"]
        structure["structured_identity"] = _digest(payload)
    with pytest.raises(ValueError):
        _original_sequence(selected, cut, structure["procedure"], source_graph=structure)


@pytest.mark.parametrize("authentic", ["scalar_element"], indirect=True)
@pytest.mark.parametrize("fault", ["missing_contract", "changed_element", "unchecked_call"])
def test_scalar_projection_requires_the_original_coordinate_contract(authentic, fault):
    _, manifest = authentic
    manifest = deepcopy(manifest)
    validate_checked_execution_plans(manifest)
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    epoch = plan["storage_epoch"]
    access = next(p for p in epoch["native_opaque_accesses"]
                  if any(r["storage"] == "scalar_element" for e in p["call_projection"]["calls"]
                         for r in e["source_call"]["resource_mappings"]))
    projection = access["call_projection"]
    edge = next(e for e in projection["calls"]
                if any(r["storage"] == "scalar_element" for r in e["source_call"]["resource_mappings"]))
    procedure = next(p for p in plan["procedures"] if p["procedure"] == access["procedure"])
    resources = {r["resource"]: r for r in epoch["resources"]}

    def check():
        validate_call_inventory(projection, list(projection["field_slots"]), procedure,
                                resources, plan["entry"], manifest)

    check()
    if fault == "unchecked_call":
        nodes = {n["procedure"]: n for n in projection["members"]}
        with pytest.raises(ValueError, match="lacks its checked coordinate contract"):
            _original_call(edge["source_call"], nodes[edge["callee"]], nodes[edge["caller"]])
        return
    if fault == "changed_element":
        original = deepcopy(edge["source_call"])
        row = next(r for r in edge["source_call"]["resource_mappings"] if r["storage"] == "scalar_element")
        row["scalar_element"] = {}
        manifest["resolved_calls"] = [deepcopy(edge["source_call"]) if c == original else c
                                      for c in manifest["resolved_calls"]]
    else:
        callee = next(p for p in plan["procedures"] if p["procedure"] == edge["callee"])
        site = next(c for c in callee["calls"] if c["caller"] == edge["caller"]
                    and c["first_line"] == edge["first_line"] and c["last_line"] == edge["last_line"])
        site.pop("original_scalar_element_coordinates")
        coordinator = next(c for c in scope["module_coordinators"] if c["procedure"] == edge["callee"])
        counterpart = next(c for c in coordinator["calls"] if c["caller"] == edge["caller"]
                           and c["first_line"] == edge["first_line"] and c["last_line"] == edge["last_line"])
        counterpart.pop("original_scalar_element_coordinates")
    with pytest.raises(ValueError, match="scalar elements differ from their original coordinate contract"):
        check()
