"""Pressure adapter requirements never substitute for compiler source proofs."""
import hashlib
import itertools
import math
from dataclasses import replace

import pytest
from elmm_pipeline.gpu_pressure import PressureProblem, prepare_adapter, source_manifest


def problem(**changes):
    return PressureProblem(**({"sizes": (7, 5, 9), "lengths": (2.75, 1.25, 3.5),
                              "boundary_conditions": (0, 0, 0, 0, 4, 4), "approximation": 4,
                              "workspace_cap_bytes": 256 << 20} | changes))


def test_channel_requirements_preserve_original_geometry_and_no_authority():
    p = problem(sizes=[7, 5, 9])
    assert p.sizes == (7, 5, 9)
    record = p.record()
    assert record["phi"] == {"extents": [11, 9, 13], "lower_bounds": [-1, -1, -1],
                             "ghosts": [2, 2, 2], "dense_first_index_fastest": True}
    assert record["rhs"] == {"extents": [9, 7, 11], "lower_bounds": [0, 0, 0],
                             "ghosts": [1, 1, 1], "dense_first_index_fastest": True}
    assert record["boundary_conditions"] == [0, 0, 0, 0, 4, 4]
    assert not record["source_authority"]
    assert not record["placement_authority"]
    assert not record["estimates_available"]
    assert len(record["compiler_requirements"]) == 4


@pytest.mark.parametrize("change", [
    {"sizes": (1, 5, 9)}, {"sizes": (True, 5, 9)}, {"sizes": (7, 5)},
    {"lengths": (math.nan, 1, 1)}, {"lengths": (0, 1, 1)}, {"lengths": (1 << 4096, 1, 1)},
    {"approximation": True}, {"approximation": 6}, {"workspace_cap_bytes": 0},
    {"boundary_conditions": (0, 4, 0, 0, 4, 4)}, {"boundary_conditions": (2, 2, 0, 0, 4, 4)},
    {"phi_lower_bounds": ((1 << 63) - 2, -1, -1)}, {"rhs_lower_bounds": (0, 0, -(1 << 63) - 1)},
])
def test_unqualified_operator_or_layout_is_rejected(change):
    with pytest.raises(ValueError):
        problem(**change)


def test_physical_effects_distinguish_rhs_input_phi_interior_and_halos():
    effects = problem().effects()
    assert effects["solve"] == {
        "phi": {"reads": [], "writes": [{"lower": [2, 2, 2], "upper": [9, 7, 11]}],
                "overwrites": [{"lower": [2, 2, 2], "upper": [9, 7, 11]}]},
        "rhs": {"reads": [{"lower": [1, 1, 1], "upper": [8, 6, 10]}], "writes": [], "overwrites": []},
    }


def points(box):
    return set(itertools.product(*(range(a, b) for a, b in zip(box["lower"], box["upper"]))))


def test_six_face_effects_leave_edges_and_corners_untouched():
    p = problem()
    halo = p.effects()["halo"]["phi"]
    faces = [points(box) for box in halo["writes"]]
    assert all(not a & b for a, b in itertools.combinations(faces, 2))
    all_points = set(itertools.product(*(range(n + 4) for n in p.sizes)))
    expected = {q for q in all_points if sum(not 2 <= q[i] < p.sizes[i] + 2 for i in range(3)) == 1}
    assert set.union(*faces) == expected
    interior = points({"lower": [2, 2, 2], "upper": [n + 2 for n in p.sizes]})
    assert all(points(box) <= interior for box in halo["reads"])
    assert halo["writes"] == halo["overwrites"]
    # Periodic west face reads the last two interior planes; reflecting bottom
    # face reads its first two planes, matching original Bound_Phi order.
    assert halo["reads"][0] == {"lower": [7, 2, 2], "upper": [9, 7, 11]}
    assert halo["reads"][4] == {"lower": [2, 2, 2], "upper": [9, 7, 4]}


def test_preparation_copies_exact_sources_and_is_not_a_build(tmp_path):
    before = source_manifest()
    result = prepare_adapter(problem(), tmp_path / "fresh")
    assert source_manifest() == before
    assert len(result["sources"]) == 4
    for entry in result["sources"]:
        from pathlib import Path
        assert hashlib.sha256(Path(entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    assert {entry["path"].rsplit("/", 1)[-1] for entry in result["sources"]} == {
        "gpu_pressure.h", "gpu_pressure.cpp", "gpu_pressure_halo.cu", "gpu_pressure.f90"}
    assert not list((tmp_path / "fresh").glob("*.o"))
    assert (tmp_path / "fresh" / "pressure-adapter.json").is_file()
    with pytest.raises(FileExistsError):
        prepare_adapter(problem(), tmp_path / "fresh")


def test_public_records_do_not_mutate_the_immutable_problem():
    p = problem()
    changed = p.record()
    changed["effects"]["solve"]["phi"]["writes"][0]["lower"][0] = 99
    assert p.effects()["solve"]["phi"]["writes"][0]["lower"] == [2, 2, 2]
    assert replace(p, approximation=2).record()["approximation"] == 2
    with pytest.raises(TypeError):
        prepare_adapter(p.record(), "must-not-be-created")
