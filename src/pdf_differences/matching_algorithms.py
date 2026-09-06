"""Small deterministic matching primitives implemented with NumPy and Python.

The application only needs two-dimensional neighbor queries and rectangular
minimum-cost assignment. Keeping those focused operations here avoids pulling
SciPy into production while leaving the higher-level matching policy unchanged.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

AssignmentEdge = tuple[int, int, float]


class MatchingBackend(Protocol):
    """Primitive operations used by the entity-matching cascade."""

    def nearest_neighbors(
        self, points: np.ndarray, count: int
    ) -> tuple[np.ndarray, np.ndarray]: ...

    def radius_neighbors(
        self, points: np.ndarray, queries: np.ndarray, radius: float
    ) -> tuple[np.ndarray, ...]: ...

    def assign_component(
        self,
        row_count: int,
        real_column_count: int,
        edges: Iterable[AssignmentEdge],
        unmatched_cost: float,
        sparse_threshold: int,
    ) -> tuple[np.ndarray, np.ndarray]: ...


def _point_array(points: np.ndarray) -> np.ndarray:
    result = np.asarray(points, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2:
        raise ValueError("points must be a two-dimensional array with shape (n, 2)")
    if not np.all(np.isfinite(result)):
        raise ValueError("points must contain only finite coordinates")
    return result


def nearest_neighbors(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Return each point's nearest neighbors, including the point itself.

    Distances are evaluated in bounded-size NumPy blocks so a dense ``n x n``
    array is never retained. Only the requested nearest entries are sorted.
    """

    coordinates = _point_array(points)
    point_count = len(coordinates)
    requested = min(max(int(count), 0), point_count)
    if point_count == 0 or requested == 0:
        shape = (point_count, 0)
        return np.empty(shape, dtype=np.float64), np.empty(shape, dtype=np.int64)

    distances = np.empty((point_count, requested), dtype=np.float64)
    indices = np.empty((point_count, requested), dtype=np.int64)
    block_size = max(1, min(point_count, (32 * 1024 * 1024) // (8 * point_count)))

    for start in range(0, point_count, block_size):
        stop = min(point_count, start + block_size)
        block = coordinates[start:stop]
        squared = block[:, 0, None] - coordinates[None, :, 0]
        np.square(squared, out=squared)
        vertical = block[:, 1, None] - coordinates[None, :, 1]
        np.square(vertical, out=vertical)
        squared += vertical

        # Make self-selection deterministic even when multiple entities share
        # exactly the same anchor. The caller explicitly removes each self index.
        local_rows = np.arange(stop - start)
        own_columns = np.arange(start, stop)
        squared[local_rows, own_columns] = -1.0

        if requested == point_count:
            selected = np.broadcast_to(
                np.arange(point_count, dtype=np.int64), (stop - start, point_count)
            ).copy()
        else:
            selected = np.argpartition(squared, requested - 1, axis=1)[:, :requested]
        selected_squared = np.take_along_axis(squared, selected, axis=1)
        order = np.lexsort((selected, selected_squared), axis=1)
        selected = np.take_along_axis(selected, order, axis=1)
        selected_squared = np.take_along_axis(selected_squared, order, axis=1)

        indices[start:stop] = selected
        distances[start:stop] = np.sqrt(np.maximum(selected_squared, 0.0))

    return distances, indices


def radius_neighbors(
    points: np.ndarray, queries: np.ndarray, radius: float
) -> tuple[np.ndarray, ...]:
    """Return sorted reference indices within an inclusive Euclidean radius.

    A uniform spatial hash is well suited to normalized two-dimensional PDF
    coordinates: each query inspects at most the surrounding nine cells before
    applying an exact squared-distance check.
    """

    coordinates = _point_array(points)
    query_coordinates = _point_array(queries)
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("radius must be a finite non-negative value")
    if not len(query_coordinates):
        return ()
    if not len(coordinates):
        return tuple(np.empty(0, dtype=np.int64) for _ in query_coordinates)
    if radius == 0.0:
        return tuple(
            np.flatnonzero(np.all(coordinates == query, axis=1)).astype(np.int64, copy=False)
            for query in query_coordinates
        )

    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    cells = np.floor(coordinates / radius).astype(np.int64)
    for index, cell in enumerate(cells):
        buckets[(int(cell[0]), int(cell[1]))].append(index)

    squared_radius = radius * radius
    output: list[np.ndarray] = []
    for query in query_coordinates:
        query_cell = np.floor(query / radius).astype(np.int64)
        nearby: list[int] = []
        for x_offset in (-1, 0, 1):
            for y_offset in (-1, 0, 1):
                nearby.extend(
                    buckets.get(
                        (int(query_cell[0]) + x_offset, int(query_cell[1]) + y_offset),
                        (),
                    )
                )
        if not nearby:
            output.append(np.empty(0, dtype=np.int64))
            continue
        candidate_indices = np.asarray(sorted(nearby), dtype=np.int64)
        delta = coordinates[candidate_indices] - query
        squared_distances = np.einsum("ij,ij->i", delta, delta)
        output.append(candidate_indices[squared_distances <= squared_radius])
    return tuple(output)


def minimum_cost_assignment(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve a finite rectangular linear-sum assignment problem.

    This is a primal-dual shortest-augmenting-path form of the Hungarian
    algorithm. It runs in ``O(min(r, c)^2 * max(r, c))`` time and returns row
    indices in ascending order, matching the contract used by SciPy's dense
    assignment function.
    """

    matrix = np.asarray(cost, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("cost must be a two-dimensional matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("cost must contain only finite values")
    row_count, column_count = matrix.shape
    if row_count == 0 or column_count == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty.copy()
    if row_count > column_count:
        transposed_rows, transposed_columns = minimum_cost_assignment(matrix.T)
        rows = transposed_columns
        columns = transposed_rows
        order = np.argsort(rows, kind="stable")
        return rows[order], columns[order]

    row_potentials = np.zeros(row_count + 1, dtype=np.float64)
    column_potentials = np.zeros(column_count + 1, dtype=np.float64)
    row_for_column = np.zeros(column_count + 1, dtype=np.int64)
    predecessor = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        row_for_column[0] = row
        minimum_slack = np.full(column_count + 1, np.inf, dtype=np.float64)
        visited = np.zeros(column_count + 1, dtype=bool)
        column = 0

        while True:
            visited[column] = True
            active_row = int(row_for_column[column])
            available = np.flatnonzero(~visited[1:]) + 1
            reduced = (
                matrix[active_row - 1, available - 1]
                - row_potentials[active_row]
                - column_potentials[available]
            )
            improved = reduced < minimum_slack[available]
            improved_columns = available[improved]
            minimum_slack[improved_columns] = reduced[improved]
            predecessor[improved_columns] = column

            available_slack = minimum_slack[available]
            next_position = int(np.argmin(available_slack))
            delta = float(available_slack[next_position])
            next_column = int(available[next_position])

            visited_columns = np.flatnonzero(visited)
            row_potentials[row_for_column[visited_columns]] += delta
            column_potentials[visited_columns] -= delta
            minimum_slack[available] -= delta
            column = next_column
            if row_for_column[column] == 0:
                break

        while column:
            previous = int(predecessor[column])
            row_for_column[column] = row_for_column[previous]
            column = previous

    assigned_columns = np.empty(row_count, dtype=np.int64)
    for column in range(1, column_count + 1):
        assigned_row = int(row_for_column[column])
        if assigned_row:
            assigned_columns[assigned_row - 1] = column - 1
    return np.arange(row_count, dtype=np.int64), assigned_columns


def _validated_component_edges(
    row_count: int,
    real_column_count: int,
    edges: Iterable[AssignmentEdge],
    unmatched_cost: float,
) -> tuple[AssignmentEdge, ...]:
    if row_count < 0 or real_column_count < 0:
        raise ValueError("assignment dimensions must be non-negative")
    if not math.isfinite(unmatched_cost) or unmatched_cost < 0.0:
        raise ValueError("unmatched cost must be finite and non-negative")
    output: list[AssignmentEdge] = []
    seen: set[tuple[int, int]] = set()
    for row, column, value in edges:
        if not 0 <= row < row_count or not 0 <= column < real_column_count:
            raise ValueError("assignment edge index is outside the component")
        if (row, column) in seen:
            raise ValueError("assignment edges must contain unique row/column pairs")
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("assignment edge costs must be finite and non-negative")
        seen.add((row, column))
        output.append((row, column, float(value)))
    return tuple(output)


def _dense_component_assignment(
    row_count: int,
    real_column_count: int,
    edges: Iterable[AssignmentEdge],
    unmatched_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    edge_list = _validated_component_edges(row_count, real_column_count, edges, unmatched_cost)
    highest_valid_cost = max((value for _, _, value in edge_list), default=unmatched_cost)
    forbidden_cost = max(2.0, unmatched_cost + 1.0, highest_valid_cost + 1.0)
    cost = np.full((row_count, real_column_count + row_count), forbidden_cost, dtype=np.float64)
    for row in range(row_count):
        cost[row, real_column_count + row] = unmatched_cost
    for row, column, value in edge_list:
        cost[row, column] = value
    return minimum_cost_assignment(cost)


@dataclass(slots=True)
class _FlowEdge:
    target: int
    reverse_index: int
    capacity: int
    cost: float


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, cost: float) -> int:
    source_index = len(graph[source])
    target_index = len(graph[target])
    graph[source].append(_FlowEdge(target, target_index, 1, cost))
    graph[target].append(_FlowEdge(source, source_index, 0, -cost))
    return source_index


def _sparse_component_assignment(
    row_count: int,
    real_column_count: int,
    edges: Iterable[AssignmentEdge],
    unmatched_cost: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a component as sparse min-cost flow with private dummy columns."""

    edge_list = _validated_component_edges(row_count, real_column_count, edges, unmatched_cost)

    source = 0
    row_offset = 1
    column_offset = row_offset + row_count
    column_count = real_column_count + row_count
    sink = column_offset + column_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    assignment_edges: list[list[tuple[int, int]]] = [[] for _ in range(row_count)]

    for row in range(row_count):
        _add_flow_edge(graph, source, row_offset + row, 0.0)
    for row, column, value in sorted(edge_list):
        edge_index = _add_flow_edge(graph, row_offset + row, column_offset + column, float(value))
        assignment_edges[row].append((column, edge_index))
    for row in range(row_count):
        dummy_column = real_column_count + row
        edge_index = _add_flow_edge(
            graph, row_offset + row, column_offset + dummy_column, unmatched_cost
        )
        assignment_edges[row].append((dummy_column, edge_index))
    for column in range(column_count):
        _add_flow_edge(graph, column_offset + column, sink, 0.0)

    node_count = len(graph)
    potentials = np.zeros(node_count, dtype=np.float64)
    for _ in range(row_count):
        distances = np.full(node_count, np.inf, dtype=np.float64)
        previous_nodes = np.full(node_count, -1, dtype=np.int64)
        previous_edges = np.full(node_count, -1, dtype=np.int64)
        distances[source] = 0.0
        queue: list[tuple[float, int]] = [(0.0, source)]

        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances[node]:
                continue
            for edge_index, edge in enumerate(graph[node]):
                if not edge.capacity:
                    continue
                reduced_cost = edge.cost + potentials[node] - potentials[edge.target]
                if -1e-12 < reduced_cost < 0.0:
                    reduced_cost = 0.0
                candidate_distance = distance + reduced_cost
                if candidate_distance < distances[edge.target] - 1e-15:
                    distances[edge.target] = candidate_distance
                    previous_nodes[edge.target] = node
                    previous_edges[edge.target] = edge_index
                    heapq.heappush(queue, (candidate_distance, edge.target))

        if not math.isfinite(float(distances[sink])):
            raise ValueError("assignment graph has no full matching")
        reachable = np.isfinite(distances)
        potentials[reachable] += distances[reachable]

        node = sink
        while node != source:
            previous_node = int(previous_nodes[node])
            edge_index = int(previous_edges[node])
            edge = graph[previous_node][edge_index]
            edge.capacity = 0
            graph[node][edge.reverse_index].capacity = 1
            node = previous_node

    columns = np.empty(row_count, dtype=np.int64)
    for row, references in enumerate(assignment_edges):
        row_node = row_offset + row
        selected = [column for column, index in references if graph[row_node][index].capacity == 0]
        if len(selected) != 1:
            raise RuntimeError("internal assignment flow is inconsistent")
        columns[row] = selected[0]
    return np.arange(row_count, dtype=np.int64), columns


@dataclass(frozen=True, slots=True)
class NativeMatchingBackend:
    """NumPy/Python implementation used by production comparisons."""

    def nearest_neighbors(self, points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
        return nearest_neighbors(points, count)

    def radius_neighbors(
        self, points: np.ndarray, queries: np.ndarray, radius: float
    ) -> tuple[np.ndarray, ...]:
        return radius_neighbors(points, queries, radius)

    def assign_component(
        self,
        row_count: int,
        real_column_count: int,
        edges: Iterable[AssignmentEdge],
        unmatched_cost: float,
        sparse_threshold: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        edge_list = tuple(edges)
        if max(row_count, real_column_count) > sparse_threshold:
            return _sparse_component_assignment(
                row_count, real_column_count, edge_list, unmatched_cost
            )
        return _dense_component_assignment(row_count, real_column_count, edge_list, unmatched_cost)


NATIVE_MATCHING_BACKEND = NativeMatchingBackend()
