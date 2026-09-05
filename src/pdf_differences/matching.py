"""Three-tier entity matching with no neural-network or PyTorch dependency.

The cascade adopts CADMorph's conceptual ordering—exact, attribute, then
ambiguous assignment—but implements the final tier with deterministic,
handcrafted vector/text/context features. No CADMorph source is included.
"""

from __future__ import annotations

import math
from collections import defaultdict, deque
from collections.abc import Iterable
from difflib import SequenceMatcher

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching
from scipy.spatial import cKDTree

from .alignment import registered_distance, transform_bbox, transform_point
from .config import SETTINGS, ComparisonSettings
from .mechanical import classify_text
from .models import BBox, Entity, EntityKind, EntityMatch, MatchResult, MatchTier, Transform


def _ratio(first: float, second: float) -> float:
    if max(abs(first), abs(second)) < 1e-10:
        return 1.0
    return min(abs(first), abs(second)) / max(abs(first), abs(second))


def _text_similarity(first: str, second: str) -> float:
    return SequenceMatcher(None, first.casefold(), second.casefold(), autojunk=False).ratio()


def _histogram_similarity(first: Entity, second: Entity) -> float:
    left, right = dict(first.op_histogram), dict(second.op_histogram)
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    overlap = sum(min(left.get(key, 0), right.get(key, 0)) for key in keys)
    total = sum(max(left.get(key, 0), right.get(key, 0)) for key in keys)
    return overlap / total if total else 1.0


def _bbox_iou(first: BBox, second: BBox) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 1e-12 else float(first == second)


def _aspect(entity: Entity) -> float:
    if entity.height < 1e-8:
        return 1_000.0 if entity.width > 1e-8 else 1.0
    return min(1_000.0, entity.width / entity.height)


def _context_vectors(entities: tuple[Entity, ...]) -> dict[str, tuple[float, ...]]:
    """Summarize each entity's six nearest neighbors without learned embeddings."""

    if not entities:
        return {}
    if len(entities) == 1:
        return {entities[0].id: (0.0, 0.0, 0.0, 0.0)}
    points = np.asarray([entity.anchor for entity in entities], dtype=np.float64)
    tree = cKDTree(points)
    count = min(7, len(entities))
    distances, indices = tree.query(points, k=count)
    if count == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    output: dict[str, tuple[float, ...]] = {}
    for row, entity in enumerate(entities):
        neighbor_indices = [int(index) for index in indices[row][1:]]
        neighbor_distances = [float(value) for value in distances[row][1:]]
        divisor = max(1, len(neighbor_indices))
        text_share = (
            sum(entities[index].kind == EntityKind.TEXT for index in neighbor_indices) / divisor
        )
        mean_distance = sum(neighbor_distances) / divisor
        spread = (
            math.sqrt(sum((value - mean_distance) ** 2 for value in neighbor_distances) / divisor)
            if neighbor_distances
            else 0.0
        )
        mean_area = sum(entities[index].area for index in neighbor_indices) / divisor
        output[entity.id] = (text_share, 1.0 - text_share, mean_distance, spread + mean_area)
    return output


def _context_similarity(first: tuple[float, ...], second: tuple[float, ...]) -> float:
    components = []
    for left, right in zip(first, second, strict=True):
        scale = max(abs(left), abs(right), 0.02)
        components.append(max(0.0, 1.0 - abs(left - right) / scale))
    return sum(components) / len(components) if components else 1.0


def _position_score(distance: float, radius: float) -> float:
    return max(0.0, 1.0 - distance / max(radius, 1e-9))


def attribute_score(
    old: Entity,
    new: Entity,
    transform: Transform,
    old_page: tuple[Entity, ...],
    new_page: tuple[Entity, ...],
    settings: ComparisonSettings = SETTINGS,
) -> float:
    """Score a possible in-place edit; kind mismatches are never candidates."""

    if old.kind != new.kind:
        return 0.0
    distance = registered_distance(old, new, transform)
    if distance > settings.attribute_position_tolerance:
        return 0.0
    position = _position_score(distance, settings.attribute_position_tolerance)
    style = float(old.style_signature == new.style_signature)
    aligned_bbox = transform_bbox(old.bbox, transform)
    if old.kind == EntityKind.TEXT:
        old_category = classify_text(old.text, old.bbox, old_page)
        new_category = classify_text(new.text, new.bbox, new_page)
        category = float(old_category == new_category)
        size = _ratio(old.font_size * transform.scale, new.font_size)
        content = _text_similarity(old.text_normalized, new.text_normalized)
        return 0.48 * position + 0.22 * content + 0.13 * category + 0.09 * size + 0.08 * style

    histogram = _histogram_similarity(old, new)
    overlap = _bbox_iou(aligned_bbox, new.bbox)
    aspect = _ratio(_aspect(old), _aspect(new))
    return 0.34 * position + 0.24 * overlap + 0.20 * histogram + 0.12 * aspect + 0.10 * style


def structural_score(
    old: Entity,
    new: Entity,
    distance: float,
    old_context: dict[str, tuple[float, ...]],
    new_context: dict[str, tuple[float, ...]],
    old_page: tuple[Entity, ...],
    new_page: tuple[Entity, ...],
    settings: ComparisonSettings = SETTINGS,
) -> float:
    """Deterministic substitute for a learned embedding similarity."""

    if old.kind != new.kind or distance > settings.structural_search_radius:
        return 0.0
    context = _context_similarity(old_context[old.id], new_context[new.id])
    position = _position_score(distance, settings.structural_search_radius)
    style = float(old.style_signature == new.style_signature)
    if old.kind == EntityKind.TEXT:
        content = _text_similarity(old.text_normalized, new.text_normalized)
        category = float(
            classify_text(old.text, old.bbox, old_page)
            == classify_text(new.text, new.bbox, new_page)
        )
        size = _ratio(old.font_size, new.font_size)
        return 0.36 * content + 0.17 * category + 0.12 * size + 0.18 * context + 0.17 * position

    histogram = _histogram_similarity(old, new)
    aspect = _ratio(_aspect(old), _aspect(new))
    area = _ratio(old.area, new.area)
    length = _ratio(old.path_length, new.path_length)
    return (
        0.25 * histogram
        + 0.16 * aspect
        + 0.13 * area
        + 0.15 * length
        + 0.08 * style
        + 0.12 * context
        + 0.11 * position
    )


def _greedy_assign(
    candidates: list[tuple[float, float, int, int]],
    old: tuple[Entity, ...],
    new: tuple[Entity, ...],
    tier: MatchTier,
) -> tuple[list[EntityMatch], set[int], set[int]]:
    matches: list[EntityMatch] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for score, distance, old_index, new_index in sorted(
        candidates,
        key=lambda item: (-item[0], item[1], old[item[2]].id, new[item[3]].id),
    ):
        if old_index in used_old or new_index in used_new:
            continue
        used_old.add(old_index)
        used_new.add(new_index)
        matches.append(EntityMatch(old[old_index], new[new_index], tier, round(score, 6), distance))
    return matches, used_old, used_new


def _candidate_components(
    candidates: dict[tuple[int, int], tuple[float, float]],
) -> list[tuple[list[int], list[int]]]:
    old_to_new: dict[int, set[int]] = defaultdict(set)
    new_to_old: dict[int, set[int]] = defaultdict(set)
    for old_index, new_index in candidates:
        old_to_new[old_index].add(new_index)
        new_to_old[new_index].add(old_index)

    components: list[tuple[list[int], list[int]]] = []
    unseen_old = set(old_to_new)
    while unseen_old:
        seed = min(unseen_old)
        queue: deque[tuple[str, int]] = deque([("old", seed)])
        old_nodes: set[int] = set()
        new_nodes: set[int] = set()
        while queue:
            side, index = queue.popleft()
            if side == "old":
                if index in old_nodes:
                    continue
                old_nodes.add(index)
                unseen_old.discard(index)
                queue.extend(("new", neighbor) for neighbor in sorted(old_to_new[index]))
            else:
                if index in new_nodes:
                    continue
                new_nodes.add(index)
                queue.extend(("old", neighbor) for neighbor in sorted(new_to_old[index]))
        components.append((sorted(old_nodes), sorted(new_nodes)))
    return components


def _hungarian_assign(
    candidates: dict[tuple[int, int], tuple[float, float]],
    old: tuple[Entity, ...],
    new: tuple[Entity, ...],
    minimum: float,
    settings: ComparisonSettings,
) -> tuple[list[EntityMatch], set[int], set[int]]:
    matches: list[EntityMatch] = []
    used_old: set[int] = set()
    used_new: set[int] = set()
    for old_nodes, new_nodes in _candidate_components(candidates):
        old_position = {value: index for index, value in enumerate(old_nodes)}
        new_position = {value: index for index, value in enumerate(new_nodes)}
        # Each old row gets a private dummy column, allowing a principled unmatched choice.
        unmatched_cost = 1.0 - minimum + 1e-7
        component_candidates: list[tuple[int, int, float]] = []
        for (old_index, new_index), (score, _) in candidates.items():
            if old_index not in old_position or new_index not in new_position:
                continue
            row, column = old_position[old_index], new_position[new_index]
            tie_break = (row * max(1, len(new_nodes)) + column) * 1e-12
            # Sparse graph routines treat stored zeroes as missing edges.
            component_candidates.append((row, column, max(1e-12, 1.0 - score + tie_break)))

        if max(len(old_nodes), len(new_nodes)) > settings.sparse_assignment_threshold:
            matrix_rows = [item[0] for item in component_candidates]
            matrix_columns = [item[1] for item in component_candidates]
            matrix_values = [item[2] for item in component_candidates]
            for row in range(len(old_nodes)):
                matrix_rows.append(row)
                matrix_columns.append(len(new_nodes) + row)
                matrix_values.append(unmatched_cost)
            sparse_cost = coo_matrix(
                (matrix_values, (matrix_rows, matrix_columns)),
                shape=(len(old_nodes), len(new_nodes) + len(old_nodes)),
            ).tocsr()
            rows, columns = min_weight_full_bipartite_matching(sparse_cost)
        else:
            cost = np.full(
                (len(old_nodes), len(new_nodes) + len(old_nodes)),
                2.0,
                dtype=np.float64,
            )
            for row in range(len(old_nodes)):
                cost[row, len(new_nodes) + row] = unmatched_cost
            for row, column, value in component_candidates:
                cost[row, column] = value
            rows, columns = linear_sum_assignment(cost)

        for row, column in zip(rows, columns, strict=True):
            if column >= len(new_nodes):
                continue
            old_index, new_index = old_nodes[int(row)], new_nodes[int(column)]
            candidate = candidates.get((old_index, new_index))
            if candidate is None or candidate[0] < minimum:
                continue
            score, distance = candidate
            used_old.add(old_index)
            used_new.add(new_index)
            matches.append(
                EntityMatch(
                    old[old_index],
                    new[new_index],
                    MatchTier.STRUCTURAL,
                    round(score, 6),
                    distance,
                )
            )
    return matches, used_old, used_new


def _spatial_candidates(
    old: tuple[Entity, ...],
    new: tuple[Entity, ...],
    transform: Transform,
    radius: float,
) -> Iterable[tuple[int, int, float]]:
    if not old or not new:
        return ()
    new_points = np.asarray([entity.anchor for entity in new], dtype=np.float64)
    tree = cKDTree(new_points)
    output: list[tuple[int, int, float]] = []
    for old_index, entity in enumerate(old):
        point = transform_point(entity.anchor, transform)
        for new_index in sorted(tree.query_ball_point(point, radius)):
            candidate = new[int(new_index)]
            if entity.kind != candidate.kind:
                continue
            output.append((old_index, int(new_index), math.dist(point, candidate.anchor)))
    return output


def match_entities(
    old_entities: Iterable[Entity],
    new_entities: Iterable[Entity],
    transform: Transform | None = None,
    settings: ComparisonSettings = SETTINGS,
) -> MatchResult:
    """Match two page inventories using exact, attribute, and structural tiers."""

    old, new = tuple(old_entities), tuple(new_entities)
    transform = transform or Transform()
    if not old or not new:
        return MatchResult((), old, new)

    all_matches: list[EntityMatch] = []
    used_old: set[int] = set()
    used_new: set[int] = set()

    # Tier 1: identical full content at the aligned position.
    new_by_signature: dict[tuple[EntityKind, str], list[int]] = defaultdict(list)
    for new_index, entity in enumerate(new):
        new_by_signature[(entity.kind, entity.content_signature)].append(new_index)
    exact_candidates: list[tuple[float, float, int, int]] = []
    for old_index, entity in enumerate(old):
        for new_index in new_by_signature.get((entity.kind, entity.content_signature), []):
            distance = registered_distance(entity, new[new_index], transform)
            if distance <= settings.exact_position_tolerance:
                exact_candidates.append((1.0, distance, old_index, new_index))
    assigned, exact_old, exact_new = _greedy_assign(exact_candidates, old, new, MatchTier.EXACT)
    all_matches.extend(assigned)
    used_old.update(exact_old)
    used_new.update(exact_new)

    remaining_old_indices = [index for index in range(len(old)) if index not in used_old]
    remaining_new_indices = [index for index in range(len(new)) if index not in used_new]
    rem_old = tuple(old[index] for index in remaining_old_indices)
    rem_new = tuple(new[index] for index in remaining_new_indices)

    # Tier 2: deterministic in-place attribute/geometry edits.
    attribute_candidates: list[tuple[float, float, int, int]] = []
    for old_index, new_index, distance in _spatial_candidates(
        rem_old, rem_new, transform, settings.attribute_position_tolerance
    ):
        score = attribute_score(
            rem_old[old_index], rem_new[new_index], transform, old, new, settings
        )
        if score >= settings.attribute_min_score:
            attribute_candidates.append((score, distance, old_index, new_index))
    assigned, attribute_old, attribute_new = _greedy_assign(
        attribute_candidates, rem_old, rem_new, MatchTier.ATTRIBUTE
    )
    all_matches.extend(assigned)

    rem2_old_indices = [index for index in range(len(rem_old)) if index not in attribute_old]
    rem2_new_indices = [index for index in range(len(rem_new)) if index not in attribute_new]
    rem2_old = tuple(rem_old[index] for index in rem2_old_indices)
    rem2_new = tuple(rem_new[index] for index in rem2_new_indices)

    # Tier 3: ambiguous/moved entities, globally assigned within sparse components.
    old_context = _context_vectors(old)
    new_context = _context_vectors(new)
    structural_candidates: dict[tuple[int, int], tuple[float, float]] = {}
    for old_index, new_index, distance in _spatial_candidates(
        rem2_old, rem2_new, transform, settings.structural_search_radius
    ):
        score = structural_score(
            rem2_old[old_index],
            rem2_new[new_index],
            distance,
            old_context,
            new_context,
            old,
            new,
            settings,
        )
        if score >= settings.structural_min_score:
            structural_candidates[(old_index, new_index)] = (score, distance)
    assigned, structural_old, structural_new = _hungarian_assign(
        structural_candidates,
        rem2_old,
        rem2_new,
        settings.structural_min_score,
        settings,
    )
    all_matches.extend(assigned)

    unmatched_old = tuple(
        rem2_old[index] for index in range(len(rem2_old)) if index not in structural_old
    )
    unmatched_new = tuple(
        rem2_new[index] for index in range(len(rem2_new)) if index not in structural_new
    )
    return MatchResult(
        tuple(sorted(all_matches, key=lambda match: (match.old.id, match.new.id))),
        tuple(sorted(unmatched_old, key=lambda entity: entity.id)),
        tuple(sorted(unmatched_new, key=lambda entity: entity.id)),
    )
