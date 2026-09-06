from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy", reason="optional SciPy benchmark oracle is not installed")

from benchmarks.compare_scipy import compare_pair
from benchmarks.scipy_backend import SCIPY_MATCHING_BACKEND
from pdf_differences.matching import _context_vectors, match_entities
from pdf_differences.matching_algorithms import (
    NATIVE_MATCHING_BACKEND,
    minimum_cost_assignment,
    nearest_neighbors,
    radius_neighbors,
)
from pdf_differences.models import Entity, EntityKind


def _geometry(entity_id: str, signature: str, point: tuple[float, float]) -> Entity:
    x, y = point
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=(x - 0.02, y - 0.015, x + 0.02, y + 0.015),
        anchor=point,
        content_signature=signature,
        shape_signature=signature,
        style_signature="style",
        op_histogram=(("l", 4),),
        primitive_count=4,
        path_length=0.14,
    )


def _match_signature(result) -> tuple[object, ...]:
    return (
        tuple((match.old.id, match.new.id, match.tier, match.score) for match in result.matches),
        tuple(entity.id for entity in result.unmatched_old),
        tuple(entity.id for entity in result.unmatched_new),
    )


def test_rectangular_assignment_objective_matches_scipy():
    from scipy.optimize import linear_sum_assignment

    random = np.random.default_rng(314159)
    for shape in ((1, 8), (8, 1), (7, 11), (11, 7), (24, 24)):
        for _ in range(10):
            matrix = random.normal(size=shape)
            native_rows, native_columns = minimum_cost_assignment(matrix)
            scipy_rows, scipy_columns = linear_sum_assignment(matrix)
            assert matrix[native_rows, native_columns].sum() == pytest.approx(
                matrix[scipy_rows, scipy_columns].sum(), abs=1e-10
            )


def test_neighbor_primitives_match_scipy_on_unique_random_points():
    random = np.random.default_rng(271828)
    points = random.random((250, 2))
    queries = random.random((60, 2))

    native_distances, native_indices = nearest_neighbors(points, 7)
    scipy_distances, scipy_indices = SCIPY_MATCHING_BACKEND.nearest_neighbors(points, 7)
    assert np.allclose(native_distances, scipy_distances, rtol=0.0, atol=2e-8)
    assert np.array_equal(native_indices, scipy_indices)

    native_radius = radius_neighbors(points, queries, 0.125)
    scipy_radius = SCIPY_MATCHING_BACKEND.radius_neighbors(points, queries, 0.125)
    assert all(
        np.array_equal(native, scipy)
        for native, scipy in zip(native_radius, scipy_radius, strict=True)
    )


def test_context_explicitly_excludes_self_for_duplicate_anchors():
    entities = (
        _geometry("duplicate-a", "a", (0.2, 0.2)),
        _geometry("duplicate-b", "b", (0.2, 0.2)),
        _geometry("nearby", "c", (0.3, 0.2)),
        _geometry("farther", "d", (0.5, 0.2)),
    )

    native = _context_vectors(entities, NATIVE_MATCHING_BACKEND)
    scipy = _context_vectors(entities, SCIPY_MATCHING_BACKEND)

    assert native == scipy


def test_native_and_scipy_component_backends_choose_same_unique_assignment():
    row_count = 4
    column_count = 4
    edges = [
        (0, 0, 0.10),
        (0, 1, 0.21),
        (1, 0, 0.11),
        (1, 1, 0.30),
        (2, 2, 0.12),
        (2, 3, 0.28),
        (3, 2, 0.27),
        (3, 3, 0.13),
    ]
    arguments = (row_count, column_count, edges, 0.3200001, 1)

    native = NATIVE_MATCHING_BACKEND.assign_component(*arguments)
    scipy = SCIPY_MATCHING_BACKEND.assign_component(*arguments)

    assert np.array_equal(native[0], scipy[0])
    assert np.array_equal(native[1], scipy[1])


def test_crowded_synthetic_entity_matching_matches_scipy():
    old: list[Entity] = []
    new: list[Entity] = []
    for index in range(30):
        point = (0.1 + (index % 6) * 0.14, 0.1 + (index // 6) * 0.16)
        signature = f"shape-{index}"
        old.append(_geometry(f"old-{index:02}", signature, point))
        if index < 8:
            revised_point = point
        elif index < 28:
            revised_point = (point[0] + 0.035, point[1] + (0.025 if index % 2 else -0.025))
        else:
            continue
        new.append(_geometry(f"new-{index:02}", signature, revised_point))
    new.extend(
        (
            _geometry("new-extra-0", "extra-0", (0.04, 0.92)),
            _geometry("new-extra-1", "extra-1", (0.94, 0.92)),
        )
    )

    native = match_entities(old, new, backend=NATIVE_MATCHING_BACKEND)
    scipy = match_entities(old, new, backend=SCIPY_MATCHING_BACKEND)

    assert _match_signature(native) == _match_signature(scipy)


def test_committed_pdf_matching_and_detection_match_scipy():
    sample = Path(__file__).resolve().parents[1] / "samples" / "mechanical_pair"

    report = compare_pair("sample", sample / "baseline.pdf", sample / "revision.pdf", repeat=1)

    assert report["all_matching_equal"]
    assert report["all_detection_equal"]
