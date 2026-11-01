"""A local bound needs exact reached source authority, not a fabricated entry fact."""
from copy import deepcopy

import pytest

from compiler.tests.test_line_source_tasks import source_builder
from compiler.tests.test_reached_scalar_definitions import local_source
from elmm_pipeline.checked_execution_plan import _digest, validate_checked_execution_plans
from elmm_pipeline.source_task_record import validate_source_task
from test_line_source_task_record import values, check, reseal


@pytest.fixture(scope="module", params=[False, True], ids=["descriptor", "parameter"])
def authentic(request, tmp_path_factory):
    text = local_source("mx=ubound(a,1)-3\n\n! original preparation comment\nmx=mx-1", guard="if(nx>0)then")
    if request.param:
        text = text.replace("integer::i,j,k,ii,jj,kk,mx,undefined",
                            "integer::i,j,k,ii,jj,kk,mx,undefined\ninteger,parameter::trim=3")
        text = text.replace("mx=ubound(a,1)-3", "mx=nx-trim")
        text = text.replace("if(nx>0)then", "if(nx<0)then\n! untouched empty sibling\nelseif(nx>0)then")
    builder, _nodes = source_builder(tmp_path_factory.mktemp("scalar-reader"), text)
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    return report, deepcopy(builder.facts)


def prefix(args):
    capture = next(c for c in args[0]["scalar_captures"]
                   if c["authority"]["kind"] == "original reached scalar prefix definition")
    return capture, capture["authority"]["proof"]


def test_real_original_scalar_preparation_round_trips_without_local_entry_facts(authentic):
    report, facts = authentic
    before = deepcopy(report)
    validate_checked_execution_plans(report, capture_facts=facts)
    args = values(report)
    check(args)
    capture, proof = prefix(args)
    assert capture["resource"] not in facts["captures"]
    assert "" in proof["selection"]["prefix"]
    assert any("original preparation comment" in line for line in proof["selection"]["prefix"])
    assert [a["resource"] for a in proof["assignments"]] == [capture["resource"]] * 2
    assert report == before


@pytest.mark.parametrize("fault", [
    "kind", "version", "hash", "procedure", "graph", "inputs", "nodes", "effects", "effects_count", "effects_type", "legacy_effect", "legacy_version", "semantics", "extra",
    "entry_authority", "array_authority", "finite_authority", "motion", "execution", "lifetime", "capture_finite",
    "selection_version", "selection_procedure", "selection_nodes", "selection_hash", "selection_bool_span",
    "selection_order", "selection_activation", "selection_fallback", "prefix_empty", "prefix_text", "prefix_nonstring", "controls",
    "resources_empty", "resources_duplicate", "resource_descriptor", "resource_foreign", "resource_save",
    "assignments_empty", "assignment_future", "assignment_reorder", "assignment_text", "assignment_resource",
    "dependency_duplicate", "dependency_descriptor", "dependency_kind", "dependency_local_fact",
    "dependency_undefined", "dependency_lifetime", "dependency_extra", "parameter_without_declaration",
])
def test_resealed_prefix_cannot_add_definition_range_or_foreign_source_authority(authentic, fault):
    report, _facts = authentic
    args = values(deepcopy(report))
    check(args)
    capture, proof = prefix(args)
    selection = proof["selection"]
    dependency = next(d for d in proof["dependencies"]
                      if d["authority"]["kind"] == "explicit checked whole-defined scalar entry fact")
    if fault == "kind": proof["kind"] = "caller_scalar_definitions"
    elif fault == "version": proof["version"] = True
    elif fault == "hash": proof["identity"] = "f" * 64
    elif fault == "procedure": proof["procedure"] = "foreign::advance"
    elif fault == "graph": proof["source_graph"] = "f" * 64
    elif fault == "inputs": proof["source_inputs"] = {}
    elif fault == "nodes": proof["selected_node_ids"] = [proof["selected_node_ids"][0]+"0"]
    elif fault == "effects": proof["effects_identities"][0] = "bad"
    elif fault == "effects_count": proof["effects_identities"].pop()
    elif fault == "effects_type": proof["effects_identities"] = "f"*64
    elif fault == "legacy_effect": proof["effects_identity"] = proof.pop("effects_identities")[0]
    elif fault == "legacy_version": proof["version"] = 1
    elif fault == "semantics": proof["semantics"] = "defined at entry"
    elif fault == "extra": proof["placement_authorized"] = True
    elif fault in {"entry_authority", "array_authority", "finite_authority", "motion", "execution", "lifetime"}:
        key = {"entry_authority":"entry_definition_authorized", "array_authority":"array_payload_definition_authorized",
               "finite_authority":"finite_or_constant_authorized", "motion":"motion_authorized",
               "execution":"execution_authorized", "lifetime":"lifetime_authorized"}[fault]
        proof[key] = True
    elif fault == "capture_finite": capture["authority"]["finite_or_constant_authorized"] = True
    elif fault == "selection_version": selection["schema_version"] = True
    elif fault == "selection_procedure": selection["procedure"] = "foreign::advance"
    elif fault == "selection_nodes": selection["selected_node_ids"] = []
    elif fault == "selection_hash": selection["original_source_sha256"] = "bad"
    elif fault == "selection_bool_span": selection["first_line"] = True
    elif fault == "selection_order": selection["last_line"] = selection["first_line"]-1
    elif fault == "selection_activation": selection["activation"] = "entry"
    elif fault == "selection_fallback": selection["fallback"] = "replay after execution"
    elif fault == "prefix_empty": selection["prefix"] = []
    elif fault == "prefix_text": selection["prefix"][0] = "other=0"
    elif fault == "prefix_nonstring": selection["prefix"][0] = False
    elif fault == "controls": selection["enclosing_controls"] = [False]
    elif fault == "resources_empty": proof["resources"] = []
    elif fault == "resources_duplicate": proof["resources"].append(deepcopy(proof["resources"][0]))
    elif fault == "resource_descriptor": proof["resources"][0]["kind"] = 8
    elif fault == "resource_foreign": proof["resources"][0]["resource"] = "foreign::mx"
    elif fault == "resource_save": proof["resources"][0]["attributes"].append("save")
    elif fault == "assignments_empty": proof["assignments"] = []
    elif fault == "assignment_future": proof["assignments"][0]["last_line"] = selection["first_line"]
    elif fault == "assignment_reorder": proof["assignments"].reverse()
    elif fault == "assignment_text": proof["assignments"][0]["source"] = "mx=0"
    elif fault == "assignment_resource":
        for assignment in proof["assignments"]: assignment["resource"] = proof["procedure"]+"::other"
    elif fault == "dependency_duplicate": proof["dependencies"].append(deepcopy(dependency))
    elif fault == "dependency_descriptor": dependency["descriptor"]["resource"] = "argument::foreign"
    elif fault == "dependency_kind": dependency["authority"]["kind"] = "original positive-domain integer read"
    elif fault == "dependency_local_fact":
        dependency["resource"] = dependency["descriptor"]["resource"] = proof["procedure"]+"::uninitialized"
    elif fault == "dependency_undefined": dependency["authority"]["fact"]["initialized"] = "none"
    elif fault == "dependency_lifetime": dependency["authority"]["fact"]["lifetime"] = "selected_window"
    elif fault == "dependency_extra": dependency["authority"]["finite_or_constant_authorized"] = True
    else: dependency["authority"] = {"kind":"original PARAMETER declaration"}
    if fault != "hash": proof["identity"] = _digest({k:v for k,v in proof.items() if k != "identity"})
    reseal(args)
    with pytest.raises(ValueError): check(args)


def test_prefix_dependency_is_bound_to_supplied_original_entry_fact(authentic):
    report, facts = authentic
    args = values(deepcopy(report))
    record, member, procedure, inputs, resources, owner, manifest = args
    validate_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest, capture_facts=facts)
    _capture, proof = prefix(args)
    dep = next(d for d in proof["dependencies"] if "fact" in d["authority"])
    facts = deepcopy(facts)
    facts["captures"][dep["resource"]]["initialized"] = "none"
    with pytest.raises(ValueError, match="supplied request"):
        validate_source_task(record, member, procedure, inputs, resources, owner=owner, manifest=manifest, capture_facts=facts)


@pytest.mark.parametrize("kind", [1, 4])
def test_original_host_logical_predicate_does_not_expand_device_scalar_abi(tmp_path, kind):
    text = local_source(guard="if(flag)then")
    text = text.replace("subroutine advance(a,mask,nx,ny,nz)", "subroutine advance(a,mask,nx,ny,nz,flag)")
    text = text.replace("integer,intent(in)::nx,ny,nz", "integer,intent(in)::nx,ny,nz\nlogical(" + str(kind) + "),intent(in)::flag")
    builder, _nodes = source_builder(tmp_path, text)
    builder.facts["captures"]["argument::flag"] = dict(builder.facts["captures"]["argument::nx"])
    if kind == 4:
        from compiler.ir import CompilationError
        from compiler.tests.test_reached_scalar_definitions import issue
        with pytest.raises(CompilationError, match="unsupported scoped capture type"):
            issue(builder, _nodes)
        return
    _outputs, report = builder.run()
    assert report["scope_count"] == 1
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    args = values(report)
    check(args)
    _capture, proof = prefix(args)
    assert any(d["descriptor"]["type"] == "logical" for d in proof["dependencies"])
    assert all(s["type"] != "logical" for s in args[0]["scalars"])

    dependency = next(d for d in proof["dependencies"] if d["descriptor"]["type"] == "logical")
    dependency["descriptor"]["kind"] = 4
    proof["identity"] = _digest({k:v for k,v in proof.items() if k != "identity"})
    reseal(args)
    with pytest.raises(ValueError, match="unsupported capture kinds"):
        check(args)


def test_three_original_if_elseif_else_line_tasks_keep_separate_reached_effect_lists(tmp_path):
    from compiler.tests.test_reached_scalar_definitions import branching_builder

    builder = branching_builder(tmp_path)
    _outputs, report = builder.run()
    assert report['scope_count'] == 1
    validate_checked_execution_plans(report, capture_facts=builder.facts)
    member = report['scopes'][0]['checked_execution_plan']['storage_epoch']['members'][0]
    assert len(member['source_device_tasks']) == 3
    counts = []
    for index in range(3):
        args = values(report, index)
        check(args)
        capture, proof = prefix(args)
        assert capture['descriptor']['name'] == ['maxi', 'maxj', 'maxk'][index]
        assert proof['version'] == 2 and 'effects_identity' not in proof
        counts.append(len(proof['effects_identities']))
    assert counts == [7, 8, 8]
