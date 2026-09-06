from __future__ import annotations

import itertools

import numpy as np
import pytest

from pdf_differences.matching_algorithms import (
    _dense_component_assignment,
    _sparse_component_assignment,
    minimum_cost_assignment,
    nearest_neighbors,
    radius_neighbors,
)


def _brute_force_cost(matrix: np.ndarray) -> float:
    rows, columns = matrix.shape
    if not rows or not columns:
        return 0.0
    if rows <= columns:
        return min(
            sum(float(matrix[row, column]) for row, column in enumerate(assignment))
            for assignment in itertools.permutations(range(columns), rows)
        )
    return min(
        sum(float(matrix[row, column]) for column, row in enumerate(assignment))
        for assignment in itertools.permutations(range(rows), columns)
    )


def _component_cost(
    result: tuple[np.ndarray, np.ndarray],
    costs: dict[tuple[int, int], float],
    unmatched_cost: float,
) -> float:
    return sum(
        costs.get((int(row), int(column)), unmatched_cost)
        for row, column in zip(*result, strict=True)
    )


@pytest.mark.parametrize("shape", [(0, 0), (0, 4), (4, 0), (1, 5), (5, 1), (3, 5), (5, 3), (4, 4)])
def test_assignment_matches_exhaustive_oracle(shape: tuple[int, int]):
    random = np.random.default_rng(0xCAD + shape[0] * 10 + shape[1])
    for _ in range(5):
        matrix = random.normal(size=shape)

        rows, columns = minimum_cost_assignment(matrix)

        assert np.array_equal(rows, np.sort(rows))
        assert len(rows) == min(shape)
        assert len(set(rows.tolist())) == len(rows)
        assert len(set(columns.tolist())) == len(columns)
        actual = float(matrix[rows, columns].sum()) if len(rows) else 0.0
        assert actual == pytest.approx(_brute_force_cost(matrix), abs=1e-11)


def test_assignment_rejects_invalid_or_nonfinite_costs():
    with pytest.raises(ValueError, match="two-dimensional"):
        minimum_cost_assignment(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        minimum_cost_assignment(np.asarray([[0.0, np.nan]]))
    with pytest.raises(ValueError, match="finite"):
        minimum_cost_assignment(np.asarray([[0.0, np.inf]]))


def test_nearest_neighbors_match_direct_distance_order():
    random = np.random.default_rng(1234)
    points = random.random((75, 2))

    distances, indices = nearest_neighbors(points, 7)

    expected_distances = np.sqrt(np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=2))
    expected_indices = np.argsort(expected_distances, axis=1, kind="stable")[:, :7]
    expected = np.take_along_axis(expected_distances, expected_indices, axis=1)
    assert np.array_equal(indices, expected_indices)
    assert np.allclose(distances, expected, rtol=0.0, atol=2e-8)


def test_nearest_neighbors_keep_each_duplicate_query_as_its_own_first_result():
    points = np.asarray([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0]])

    distances, indices = nearest_neighbors(points, 3)

    assert np.array_equal(indices[:, 0], np.arange(3))
    assert np.array_equal(distances[:, 0], np.zeros(3))


def test_spatial_queries_reject_invalid_inputs():
    with pytest.raises(ValueError, match="shape"):
        nearest_neighbors(np.zeros((2, 3)), 1)
    with pytest.raises(ValueError, match="finite"):
        nearest_neighbors(np.asarray([[np.nan, 0.0]]), 1)
    with pytest.raises(ValueError, match="non-negative"):
        radius_neighbors(np.zeros((1, 2)), np.zeros((1, 2)), -0.1)
    with pytest.raises(ValueError, match="finite"):
        radius_neighbors(np.zeros((1, 2)), np.zeros((1, 2)), np.inf)

    empty = radius_neighbors(np.empty((0, 2)), np.zeros((2, 2)), 0.1)
    assert len(empty) == 2
    assert all(not len(indices) for indices in empty)


def test_radius_neighbors_are_sorted_inclusive_and_handle_zero_radius():
    points = np.asarray([[0.0, 0.0], [0.3, 0.4], [0.5, 0.0], [0.5000001, 0.0]])
    queries = np.asarray([[0.0, 0.0], [0.3, 0.4]])

    neighbors = radius_neighbors(points, queries, 0.5)
    exact = radius_neighbors(points, np.asarray([[0.5, 0.0]]), 0.0)

    assert [indices.tolist() for indices in neighbors] == [[0, 1, 2], [0, 1, 2, 3]]
    assert [indices.tolist() for indices in exact] == [[2]]


def test_radius_neighbors_match_vectorized_oracle_on_random_points():
    random = np.random.default_rng(5678)
    points = random.uniform(-1.0, 2.0, size=(150, 2))
    queries = random.uniform(-1.0, 2.0, size=(35, 2))
    radius = 0.22

    actual = radius_neighbors(points, queries, radius)

    for query, indices in zip(queries, actual, strict=True):
        squared = np.sum((points - query) ** 2, axis=1)
        expected = np.flatnonzero(squared <= radius * radius)
        assert np.array_equal(indices, expected)


def test_sparse_component_solver_has_same_objective_as_dense_solver():
    random = np.random.default_rng(9012)
    unmatched_cost = 0.3200001
    for row_count in range(1, 16):
        real_column_count = row_count + 3
        for _ in range(5):
            edges = [
                (row, column, float(random.uniform(0.01, 0.31)))
                for row in range(row_count)
                for column in range(real_column_count)
                if random.random() < 0.2
            ]
            dense = _dense_component_assignment(row_count, real_column_count, edges, unmatched_cost)
            sparse = _sparse_component_assignment(
                row_count, real_column_count, edges, unmatched_cost
            )
            costs = {(row, column): cost for row, column, cost in edges}

            assert _component_cost(sparse, costs, unmatched_cost) == pytest.approx(
                _component_cost(dense, costs, unmatched_cost), abs=1e-11
            )
            assert len(set(sparse[1].tolist())) == row_count


@pytest.mark.parametrize("solver", [_dense_component_assignment, _sparse_component_assignment])
def test_component_solvers_reject_duplicate_edges(solver):
    with pytest.raises(ValueError, match="unique"):
        solver(1, 1, [(0, 0, 0.1), (0, 0, 0.2)], 0.3)


@pytest.mark.parametrize("solver", [_dense_component_assignment, _sparse_component_assignment])
def test_component_solvers_validate_dimensions_indices_and_costs(solver):
    with pytest.raises(ValueError, match="dimensions"):
        solver(-1, 1, [], 0.3)
    with pytest.raises(ValueError, match="outside"):
        solver(1, 1, [(1, 0, 0.1)], 0.3)
    with pytest.raises(ValueError, match="edge costs"):
        solver(1, 1, [(0, 0, -0.1)], 0.3)
    with pytest.raises(ValueError, match="unmatched cost"):
        solver(1, 1, [], np.nan)
