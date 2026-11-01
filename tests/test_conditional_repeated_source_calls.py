"""Repeated exact CALL sites do not borrow another site's local handles."""

import json
from copy import deepcopy

import pytest
from elmm_pipeline.checked_execution_plan import (
    _digest,
    validate_checked_execution_plans,
)
from elmm_pipeline.conditional_return_record import _projected_call_site

from compiler.tests.test_hierarchical_child_regions import fixture
from compiler.tests.test_same_family_section_composition_cuda import HELPERS
from compiler.tests.test_same_family_section_composition_cuda import (
    source as combined_source,
)


@pytest.fixture(
    scope="module", params=[False, True], ids=["repeated-site", "two-projections"]
)
def authentic_repeated(tmp_path_factory, request):
    source = combined_source()
    if request.param:
        team = HELPERS[
            HELPERS.index("!$omp parallel") : HELPERS.index("!$omp end parallel")
            + len("!$omp end parallel")
        ]
        source = source.replace("!$omp end parallel", "!$omp end parallel\n" + team, 1)
    _path, builder = fixture(tmp_path_factory.mktemp("conditional-repeat"), source)
    outputs, report = builder.run()
    assert report["scope_count"] == 1
    manifest = json.loads(outputs["scope-manifest.json"])
    validate_checked_execution_plans(manifest)
    return manifest, request.param


def target(manifest):
    scope = manifest["scopes"][0]
    plan = scope["checked_execution_plan"]
    returned = plan["storage_epoch"]["return_contract"]
    middle = next(
        member
        for member in returned["members"]
        if member["procedure"] == "hierarchical::middle"
    )
    routes = [
        route
        for route in middle["routes"]
        if route.get("procedure") == "hierarchical::relay"
    ]
    tasks = [
        task
        for member in plan["storage_epoch"]["members"]
        for task in member.get("source_device_tasks", [])
    ]
    return scope, returned, routes, tasks


def test_repeated_sites_choose_their_exact_authenticated_edges(authentic_repeated):
    manifest, twice = authentic_repeated
    before = deepcopy(manifest)
    scope, _returned, routes, tasks = target(manifest)
    assert len(routes) == 3 and len(tasks) == (2 if twice else 1)
    assert routes[0]["handle_forwarding"] == routes[2]["handle_forwarding"]
    assert routes[0]["selected_node_ids"] != routes[2]["selected_node_ids"]
    chosen = [
        _projected_call_site(
            route, "hierarchical::middle", "hierarchical::relay", scope
        )
        for route in routes
    ]
    assert len({(edge["first_line"], edge["last_line"]) for edge in chosen}) == 3
    assert [edge["selected_node_ids"] for edge in chosen] == [
        route["selected_node_ids"] for route in routes
    ]
    validate_checked_execution_plans(manifest)
    assert manifest == before


@pytest.mark.parametrize(
    "fault", ["missing_site", "other_pair", "wrong_forward", "conflicting_projection"]
)
def test_projected_site_never_falls_back_to_another_equal_forwarding(
    authentic_repeated, fault
):
    original, _twice = authentic_repeated
    bad = deepcopy(original)
    scope, returned, routes, tasks = target(bad)
    route = routes[2]
    if fault == "missing_site":
        route["selected_node_ids"] = ["hierarchical::middle/body/999"]
    elif fault == "other_pair":
        route["procedure"] = "hierarchical::helper"
    elif fault == "wrong_forward":
        key = next(iter(route["handle_forwarding"]))
        route["handle_forwarding"][key] += 123
    else:
        # Parent validation authenticates whole DAGs. Isolate the return reader's
        # duplicate-edge consistency gate after the authentic baseline passed.
        duplicate = deepcopy(tasks[0])
        projection = duplicate["execution_binding"]["call_projection"]
        edge = next(
            edge
            for edge in projection["calls"]
            if edge["selected_node_ids"] == route["selected_node_ids"]
        )
        edge["source_call"]["resource_mapping"]["argument::a"] = "argument::foreign"
        scope["checked_execution_plan"]["storage_epoch"]["members"][0].setdefault(
            "source_device_tasks", []
        ).append(duplicate)
    with pytest.raises(ValueError):
        _projected_call_site(route, "hierarchical::middle", route["procedure"], scope)
    if fault != "conflicting_projection":
        returned["identity"] = _digest(
            {k: v for k, v in returned.items() if k != "identity"}
        )
        with pytest.raises(ValueError):
            validate_checked_execution_plans(bad)


def test_another_valid_repeated_site_cannot_replace_return_route(authentic_repeated):
    original, _twice = authentic_repeated
    bad = deepcopy(original)
    _scope, returned, routes, _tasks = target(bad)
    routes[2]["selected_node_ids"] = routes[0]["selected_node_ids"][:]
    returned["identity"] = _digest(
        {k: v for k, v in returned.items() if k != "identity"}
    )
    with pytest.raises(ValueError):
        validate_checked_execution_plans(bad)


def test_exact_selected_edge_cannot_match_another_retained_call_span(
    authentic_repeated,
):
    original, _twice = authentic_repeated
    bad = deepcopy(original)
    scope, _returned, routes, _tasks = target(bad)
    edge = _projected_call_site(
        routes[2], "hierarchical::middle", "hierarchical::relay", scope
    )
    first = _projected_call_site(
        routes[0], "hierarchical::middle", "hierarchical::relay", scope
    )
    procedure = next(
        p
        for p in scope["checked_execution_plan"]["procedures"]
        if p["procedure"] == "hierarchical::relay"
    )
    call = next(
        c
        for c in procedure["calls"]
        if c["caller"] == edge["caller"] and c["first_line"] == edge["first_line"]
    )
    call["first_line"], call["last_line"] = first["first_line"], first["last_line"]
    with pytest.raises(ValueError):
        validate_checked_execution_plans(bad)
