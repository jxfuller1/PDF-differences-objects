"""SciPy implementation retained solely as a matching benchmark oracle.

This reproduces the three SciPy primitives used before the production matcher
was made self-contained. Importing this module requires the optional
``benchmark`` dependency group; the application never imports it.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from pdf_differences.matching_algorithms import AssignmentEdge, _validated_component_edges

LEGACY_SCIPY_SPARSE_THRESHOLD = 300
SCIPY_VERSION = scipy.__version__


@dataclass(frozen=True, slots=True)
class ScipyMatchingBackend:
    """Legacy SciPy-backed primitives with the original dense/sparse switch."""

    def nearest_neighbors(self, points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        coordinates = np.asarray(points, dtype=np.float64)
        if not len(coordinates) or count <= 0:
            shape = (len(coordinates), 0)
            return np.empty(shape, dtype=np.float64), np.empty(shape, dtype=np.int64)
        distances, indices = cKDTree(coordinates).query(
            coordinates, k=min(int(count), len(coordinates))
        )
        if np.ndim(distances) == 1:
            distances = distances[:, None]
            indices = indices[:, None]
        return np.asarray(distances), np.asarray(indices, dtype=np.int64)

    def radius_neighbors(
        self, points: np.ndarray, queries: np.ndarray, radius: float
    ) -> tuple[np.ndarray, ...]:
        coordinates = np.asarray(points, dtype=np.float64)
        query_coordinates = np.asarray(queries, dtype=np.float64)
        if not len(query_coordinates):
            return ()
        if not len(coordinates):
            return tuple(np.empty(0, dtype=np.int64) for _ in query_coordinates)
        neighbors = cKDTree(coordinates).query_ball_point(query_coordinates, float(radius))
        return tuple(np.asarray(sorted(indices), dtype=np.int64) for indices in neighbors)

    def assign_component(
        self,
        row_count: int,
        real_column_count: int,
        edges: Iterable[AssignmentEdge],
        unmatched_cost: float,
        sparse_threshold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge_list = _validated_component_edges(row_count, real_column_count, edges, unmatched_cost)
        legacy_threshold = min(sparse_threshold, LEGACY_SCIPY_SPARSE_THRESHOLD)
        if max(row_count, real_column_count) > legacy_threshold:
            matrix_rows = [edge[0] for edge in edge_list]
            matrix_columns = [edge[1] for edge in edge_list]
            matrix_values = [edge[2] for edge in edge_list]
            for row in range(row_count):
                matrix_rows.append(row)
                matrix_columns.append(real_column_count + row)
                matrix_values.append(unmatched_cost)
            sparse_cost = coo_matrix(
                (matrix_values, (matrix_rows, matrix_columns)),
                shape=(row_count, real_column_count + row_count),
            ).tocsr()
            return min_weight_full_bipartite_matching(sparse_cost)

        cost = np.full((row_count, real_column_count + row_count), 2.0, dtype=np.float64)
        for row in range(row_count):
            cost[row, real_column_count + row] = unmatched_cost
        for row, column, value in edge_list:
            cost[row, column] = value
        return linear_sum_assignment(cost)


SCIPY_MATCHING_BACKEND = ScipyMatchingBackend()
