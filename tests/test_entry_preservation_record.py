"""Transport actual issued preservation proofs without issuing source authority."""
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.entry_preservation_record import validate_entry_preservation
from elmm_pipeline.source_placement_record import _tasks
from test_owner_local_call_group_record import parts, seal

from compiler.frontend.entry_preservation import prove_preserved_entry_definitions
from compiler.scopes.call_groups import _candidates
from compiler.tests.test_owner_local_call_groups import SOURCE, fixture


def issue_transport(path, *, nested=False, comments=False, anchor="endif"):
    source = SOURCE.replace("call unknown_prefix()", "a(:)=a(:)+1.0_8")
    if comments:
        source = source.replace("a(:)=a(:)+1.0_8", "a(:)=a(:)+1.0_8\n! native prefix comment\nb(:)=b(:)\n! another comment")
    if nested:
        source = source.replace("if(flag)then\ncall first", "if(n>=0)then\nif(flag)then\ncall first")
        source = source.replace("endif\ncall unknown_suffix()", "endif\nendif\ncall unknown_suffix()")
    if anchor in {"else", "elseif"}:
        header = "else" if anchor == "else" else "else if(.not.flag)then"
        source = source.replace("endif\ncall unknown_suffix()", header+"\ncall unknown_other()\nendif\ncall unknown_suffix()")
    elif anchor in {"last_else", "middle_elseif"}:
        header = "else" if anchor == "last_else" else "else if(flag)then"
        source = source.replace("if(flag)then\ncall first", "if(.not.flag)then\ncall unknown_other()\n"+header+"\ncall first")
        if anchor == "middle_elseif":
            source = source.replace("endif\ncall unknown_suffix()", "else\ncall unknown_last()\nendif\ncall unknown_suffix()")
    _path, builder = fixture(path, source, lifetime="invocation", unused=True)
    for fact in builder.facts["captures"].values():
        fact["initialized"] = "whole"
    originals = next(_candidates(builder))
    bindings = tuple(builder.entry.scope.bindings[name] for name in ("a", "b", "unused"))
    proof = prove_preserved_entry_definitions(builder.analysis, builder.entry.qualified,
                                              originals, bindings, builder.facts)
    assert proof is not None
    _outputs, report = builder.run()
    assert report["scope_count"] == 1, report["boundaries"]
    _scope, _plan, _epoch, group = parts(report)
    # This fixture tests portable cross-links with an actual issued source
    # proof. It does not claim that transformed metadata emitted GPU control.
    group.update(version=3, entry_preservation=proof.public(),
        repeated_activation="fresh local context; only independently preserved invocation-entry definitions; NONE after retirement; close and clear handles at original sequence exit")
    seal(report)
    return report, deepcopy(builder.facts)


@pytest.fixture(scope="module", params=[(False, anchor) for anchor in
                                      ("endif", "else", "elseif", "last_else", "middle_elseif")]+[(True, "endif")])
def original(request, tmp_path_factory):
    nested, anchor = request.param
    return issue_transport(tmp_path_factory.mktemp("entry-preservation-reader"), nested=nested, anchor=anchor)


def reseal(report):
    _scope, _plan, _epoch, group = parts(report)
    record = group.get("entry_preservation")
    if isinstance(record, dict):
        record["identity"] = _digest({key: value for key, value in record.items() if key != "identity"})
    seal(report)


def test_real_issued_definition_proof_has_exact_plan_source_exit_and_fact_links(original):
    report, facts = original
    validate_checked_execution_plans(report, capture_facts=facts)
    scope, plan, epoch, group = parts(report)
    record = group["entry_preservation"]
    tasks = _tasks(scope["graph_execution"]["procedures"][0]["task_graph"])
    assert validate_entry_preservation(record, group, epoch, plan,
                                       original_tasks=tasks, capture_facts=facts) is record
    assert record["version"] == 2
    assert record["new_definition_authority"] is False
    assert len(record["segments"]) >= 2
    assert "argument::unused" in record["preserved_resources"]
    assert "argument::unused" in group["descriptor_guard_resources"]


@pytest.mark.parametrize("fault", [
    "version_bool", "version_old", "version_unknown", "extra", "missing", "kind", "procedure", "source",
    "graph", "exit", "facts", "definitions", "execution", "payload", "lifetime", "aliases", "retention",
    "policy_definitions", "policy_prefix", "policy_values", "no_segments", "segments_type", "segment_extra",
    "segment_proof", "segment_empty", "segment_duplicate", "segment_foreign", "segment_leading_zero",
    "segment_cross_parent", "segment_reverse", "segment_repeat", "segment_selected_call", "segments_reverse",
    "segment_omitted", "predicate_omitted", "roots_empty",
    "roots_unknown", "roots_duplicate", "roots_type", "group_old", "activation", "descriptor_out",
    "descriptor_pointer", "descriptor_attributes_type", "entry_selected_window", "entry_unknown", "entry_source_sections",
])
def test_resealed_preservation_cannot_promote_or_detach_independent_entry_definitions(original, fault):
    report, facts = deepcopy(original)
    _scope, plan, epoch, group = parts(report)
    value = group["entry_preservation"]
    replacements = {
        "version_bool": ("version", True), "version_old": ("version", 1), "version_unknown": ("version", 3),
        "kind": ("kind", "new_definition"), "procedure": ("procedure", "foreign::owner"),
        "source": ("source_inputs", {}), "graph": ("source_graph", "a"*64),
        "exit": ("sequence_exit_identity", "a"*64), "facts": ("entry_capture_facts_identity", "a"*64),
        "definitions": ("new_definition_authority", True), "execution": ("execution_authorized", True),
        "payload": ("payload_access_authorized", True), "lifetime": ("descriptor_lifetime_authority", True),
        "aliases": ("alias_authority", True), "retention": ("cross_exit_retention_authorized", 0),
        "policy_definitions": ("definition_source", "infer from read envelope"),
        "policy_prefix": ("prefix", "skip original guards"), "policy_values": ("semantics", "unchanged bytes"),
        "no_segments": ("segments", []), "segments_type": ("segments", None),
        "roots_empty": ("preserved_resources", []), "roots_unknown": ("preserved_resources", ["unknown"]),
        "roots_type": ("preserved_resources", [True]),
    }
    if fault in replacements:
        key, replacement = replacements[fault]
        value[key] = replacement
    elif fault == "extra":
        value["preservation_identity"] = "a"*64
    elif fault == "missing":
        value.pop("segments")
    elif fault == "segment_extra":
        value["segments"][0]["initialized"] = "whole"
    elif fault == "segment_proof":
        value["segments"][0]["preservation_identity"] = "invalid"
    elif fault == "segment_empty":
        value["segments"][0]["selected_node_ids"] = []
    elif fault == "segment_duplicate":
        nodes = value["segments"][0]["selected_node_ids"]
        nodes.append(nodes[0])
    elif fault == "segment_foreign":
        value["segments"][0]["selected_node_ids"] = ["foreign::owner/body/0"]
    elif fault == "segment_leading_zero":
        value["segments"][0]["selected_node_ids"] = [plan["entry"]+"/body/00"]
    elif fault == "segment_cross_parent":
        value["segments"][0]["selected_node_ids"] = [plan["entry"]+"/body/0", plan["entry"]+"/body/1/body0/0"]
    elif fault == "segment_reverse":
        value["segments"][0]["selected_node_ids"] = [plan["entry"]+"/body/1", plan["entry"]+"/body/0"]
    elif fault == "segment_repeat":
        value["segments"].append(deepcopy(value["segments"][0]))
    elif fault == "segment_selected_call":
        value["segments"][0]["selected_node_ids"] = [group["selected_node_ids"][0]]
    elif fault == "segments_reverse":
        value["segments"].reverse()
    elif fault == "segment_omitted":
        value["segments"].pop(0)
    elif fault == "predicate_omitted":
        value["segments"] = [segment for segment in value["segments"]
                             if "condition" not in segment["selected_node_ids"][0]]
    elif fault == "roots_duplicate":
        value["preserved_resources"].append(value["preserved_resources"][0])
    elif fault == "group_old":
        group["version"] = 2
    elif fault == "activation":
        group["repeated_activation"] = "all independent input is unchanged and device-current"
    elif fault.startswith("descriptor_"):
        descriptor = epoch["resources"][0]["descriptor"]
        descriptor["intent"] = "out" if fault == "descriptor_out" else descriptor["intent"]
        if fault == "descriptor_pointer":
            descriptor["attributes"].append("pointer")
        elif fault == "descriptor_attributes_type":
            descriptor["attributes"] = None
        root = epoch["resources"][0]["resource"]
        next(row for row in plan["resources"] if row["resource"] == root)["descriptor"] = deepcopy(descriptor)
    else:
        fact = facts["captures"]["argument::a"]
        if fault == "entry_selected_window":
            fact["lifetime"] = "selected_window"
        elif fault == "entry_unknown":
            fact["initialized"] = "none"
        else:
            fact["source_sections"] = []
        plan["capture_facts_identity"] = value["entry_capture_facts_identity"] = _digest(facts)
    reseal(report)
    with pytest.raises(ValueError):
        validate_checked_execution_plans(report, capture_facts=facts)


def test_v2_group_cannot_smuggle_entry_preservation(original):
    report, facts = deepcopy(original)
    _scope, _plan, _epoch, group = parts(report)
    group["version"] = 2
    group["repeated_activation"] = "fresh local context and NONE at every reached group; close and clear handles at original sequence exit"
    reseal(report)
    with pytest.raises(ValueError, match="exact versioned"):
        validate_checked_execution_plans(report, capture_facts=facts)


def test_original_comments_may_leave_numeric_gaps_without_omitting_a_graph_child(tmp_path):
    report, facts = issue_transport(tmp_path, comments=True)
    validate_checked_execution_plans(report, capture_facts=facts)
    _scope, _plan, _epoch, group = parts(report)
    nodes = group["entry_preservation"]["segments"][0]["selected_node_ids"]
    assert [node.rsplit("/", 1)[1] for node in nodes] == ["0", "2"]
