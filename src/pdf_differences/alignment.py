"""Deterministic vector registration from unique text and geometry anchors."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Iterable

import numpy as np

from .config import SETTINGS, ComparisonSettings
from .models import AlignmentResult, BBox, Entity, EntityKind, Point, Transform

AnchorPair = tuple[Point, Point]


def transform_point(point: Point, transform: Transform) -> Point:
    cosine = math.cos(transform.rotation_radians)
    sine = math.sin(transform.rotation_radians)
    x, y = point
    return (
        transform.scale * (cosine * x - sine * y) + transform.tx,
        transform.scale * (sine * x + cosine * y) + transform.ty,
    )


def transform_bbox(bbox: BBox, transform: Transform) -> BBox:
    x0, y0, x1, y1 = bbox
    corners = (
        transform_point((x0, y0), transform),
        transform_point((x1, y0), transform),
        transform_point((x1, y1), transform),
        transform_point((x0, y1), transform),
    )
    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def registered_distance(old: Entity, new: Entity, transform: Transform) -> float:
    return math.dist(transform_point(old.anchor, transform), new.anchor)


def _unique_map(
    entities: Iterable[Entity],
    key: Callable[[Entity], str | None],
) -> dict[str, Entity]:
    keyed = [(candidate, key(candidate)) for candidate in entities]
    counts = Counter(value for _, value in keyed if value)
    return {
        value: candidate for candidate, value in keyed if value is not None and counts[value] == 1
    }


def build_anchor_pairs(old: Iterable[Entity], new: Iterable[Entity]) -> tuple[AnchorPair, ...]:
    """Pair only unique, unchanged attributes so repeated symbols cannot skew a fit."""

    old_items, new_items = tuple(old), tuple(new)
    old_text = _unique_map(
        (item for item in old_items if item.kind == EntityKind.TEXT),
        lambda item: item.text_normalized if len(item.text_normalized) >= 2 else None,
    )
    new_text = _unique_map(
        (item for item in new_items if item.kind == EntityKind.TEXT),
        lambda item: item.text_normalized if len(item.text_normalized) >= 2 else None,
    )
    old_geometry = _unique_map(
        (item for item in old_items if item.kind == EntityKind.GEOMETRY),
        lambda item: item.shape_signature,
    )
    new_geometry = _unique_map(
        (item for item in new_items if item.kind == EntityKind.GEOMETRY),
        lambda item: item.shape_signature,
    )

    pairs: list[AnchorPair] = []
    for key in sorted(set(old_text) & set(new_text)):
        pairs.append((old_text[key].anchor, new_text[key].anchor))
    for key in sorted(set(old_geometry) & set(new_geometry)):
        pairs.append((old_geometry[key].anchor, new_geometry[key].anchor))
    return tuple(pairs)


def _two_point_transform(first: AnchorPair, second: AnchorPair) -> Transform | None:
    old_a, new_a = first
    old_b, new_b = second
    old_dx, old_dy = old_b[0] - old_a[0], old_b[1] - old_a[1]
    new_dx, new_dy = new_b[0] - new_a[0], new_b[1] - new_a[1]
    old_length = math.hypot(old_dx, old_dy)
    new_length = math.hypot(new_dx, new_dy)
    if old_length < 1e-8 or new_length < 1e-8:
        return None
    scale = new_length / old_length
    if not 0.5 <= scale <= 2.0:
        return None
    rotation = math.atan2(new_dy, new_dx) - math.atan2(old_dy, old_dx)
    cosine, sine = math.cos(rotation), math.sin(rotation)
    mapped_x = scale * (cosine * old_a[0] - sine * old_a[1])
    mapped_y = scale * (sine * old_a[0] + cosine * old_a[1])
    return Transform(scale, rotation, new_a[0] - mapped_x, new_a[1] - mapped_y)


def _residuals(pairs: tuple[AnchorPair, ...], transform: Transform) -> np.ndarray:
    return np.asarray(
        [
            math.dist(transform_point(old_point, transform), new_point)
            for old_point, new_point in pairs
        ],
        dtype=np.float64,
    )


def _refine(pairs: tuple[AnchorPair, ...], inlier_indices: np.ndarray) -> Transform:
    """Least-squares 2-D similarity fit (rotation, uniform scale, translation)."""

    old = np.asarray([pairs[int(index)][0] for index in inlier_indices], dtype=np.float64)
    new = np.asarray([pairs[int(index)][1] for index in inlier_indices], dtype=np.float64)
    old_center = old.mean(axis=0)
    new_center = new.mean(axis=0)
    old_zero = old - old_center
    new_zero = new - new_center
    covariance = old_zero.T @ new_zero
    left, singular, right = np.linalg.svd(covariance)
    rotation_matrix = right.T @ left.T
    if np.linalg.det(rotation_matrix) < 0:
        right[-1, :] *= -1
        rotation_matrix = right.T @ left.T
    denominator = float(np.sum(old_zero * old_zero))
    scale = float(np.sum(singular) / denominator) if denominator > 1e-12 else 1.0
    translation = new_center - scale * (rotation_matrix @ old_center)
    rotation = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
    return Transform(scale, rotation, float(translation[0]), float(translation[1]))


def _candidate_pairs(count: int, limit: int) -> list[tuple[int, int]]:
    total = count * (count - 1) // 2
    if total <= limit:
        return [(first, second) for first in range(count) for second in range(first + 1, count)]

    # A cycle has independence number floor(count / 2): any strict majority of
    # anchors therefore contains at least one tested adjacent pair. Because an
    # accepted fit requires a 55% inlier majority, this prevents a fixed sample
    # cap from missing every all-inlier hypothesis on a dense page. The only
    # exception is a geometrically degenerate pair at the same location.
    selected = {tuple(sorted((index, (index + 1) % count))) for index in range(count)}
    target = min(total, max(limit, count))

    # A local RNG keeps the sample deterministic without changing process-wide state.
    rng = random.Random(0xCAD5EED)
    while len(selected) < target:
        first, second = sorted(rng.sample(range(count), 2))
        selected.add((first, second))
    return sorted(selected)


def estimate_alignment(
    old: Iterable[Entity],
    new: Iterable[Entity],
    settings: ComparisonSettings = SETTINGS,
) -> AlignmentResult:
    """Estimate an old-to-new transform without pixels or learned features."""

    pairs = build_anchor_pairs(old, new)
    identity = Transform()
    if not pairs:
        return AlignmentResult(
            status="identity-unverified",
            transform=identity,
            note="No unique stable anchors were available; normalized page coordinates were used.",
        )
    if len(pairs) == 1:
        old_point, new_point = pairs[0]
        transform = Transform(tx=new_point[0] - old_point[0], ty=new_point[1] - old_point[1])
        return AlignmentResult(
            status="translation-only",
            transform=transform,
            anchor_count=1,
            inlier_count=1,
            inlier_ratio=1.0,
            note="One unique anchor was available; only translation could be estimated.",
        )

    best_transform: Transform | None = None
    best_inliers = np.asarray([], dtype=np.int64)
    best_rms = float("inf")
    for first, second in _candidate_pairs(len(pairs), settings.alignment_max_hypotheses):
        candidate = _two_point_transform(pairs[first], pairs[second])
        if candidate is None:
            continue
        residuals = _residuals(pairs, candidate)
        inliers = np.flatnonzero(residuals <= settings.alignment_inlier_tolerance)
        if len(inliers) < 2:
            continue
        rms = float(np.sqrt(np.mean(np.square(residuals[inliers]))))
        rank = (len(inliers), -rms, -abs(candidate.scale - 1.0))
        best_rank = (
            len(best_inliers),
            -best_rms,
            -abs(best_transform.scale - 1.0) if best_transform is not None else float("-inf"),
        )
        if rank > best_rank:
            best_transform, best_inliers, best_rms = candidate, inliers, rms

    if best_transform is None:
        return AlignmentResult(
            status="failed",
            transform=identity,
            anchor_count=len(pairs),
            note="Stable anchors existed, but no valid similarity transform could be fitted.",
        )

    refined = _refine(pairs, best_inliers)
    refined_residuals = _residuals(pairs, refined)
    refined_inliers = np.flatnonzero(refined_residuals <= settings.alignment_inlier_tolerance)
    if len(refined_inliers) >= 2:
        refined = _refine(pairs, refined_inliers)
        refined_residuals = _residuals(pairs, refined)
        refined_inliers = np.flatnonzero(refined_residuals <= settings.alignment_inlier_tolerance)

    rms = (
        float(np.sqrt(np.mean(np.square(refined_residuals[refined_inliers]))))
        if len(refined_inliers)
        else float("inf")
    )
    ratio = len(refined_inliers) / len(pairs)
    accepted = ratio >= settings.alignment_min_inlier_ratio and rms <= settings.alignment_max_rms
    return AlignmentResult(
        status="aligned" if accepted else "failed",
        transform=refined if accepted else identity,
        anchor_count=len(pairs),
        inlier_count=len(refined_inliers),
        inlier_ratio=round(ratio, 6),
        rms_residual=round(rms, 8) if math.isfinite(rms) else 1.0,
        note=(
            "Alignment accepted from deterministic vector/text anchors."
            if accepted
            else "The drawings could not be registered reliably; comparison was declined."
        ),
    )
