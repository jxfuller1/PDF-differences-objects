from __future__ import annotations

import pytest

from pdf_differences.alignment import estimate_alignment, transform_point
from pdf_differences.config import ComparisonSettings
from pdf_differences.models import Entity, EntityKind, Transform


def _entity(entity_id: str, text: str, point: tuple[float, float]) -> Entity:
    x, y = point
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.TEXT,
        bbox=(x, y, x + 0.03, y + 0.01),
        anchor=point,
        content_signature=text,
        shape_signature="text-span",
        style_signature="style",
        text=text,
        text_normalized=text.casefold(),
        font_size=0.01,
    )


def test_recovers_similarity_transform_from_unique_text_anchors():
    expected = Transform(scale=1.03, rotation_radians=0.025, tx=0.018, ty=-0.011)
    points = ((0.1, 0.1), (0.8, 0.12), (0.2, 0.75), (0.75, 0.7))
    old = tuple(_entity(f"o{i}", f"ANCHOR {i}", point) for i, point in enumerate(points))
    new = tuple(
        _entity(f"n{i}", f"ANCHOR {i}", transform_point(point, expected))
        for i, point in enumerate(points)
    )
    result = estimate_alignment(old, new)
    assert result.status == "aligned"
    assert result.inlier_ratio == 1.0
    assert result.transform.scale == pytest.approx(expected.scale, abs=1e-7)
    assert result.transform.rotation_radians == pytest.approx(expected.rotation_radians, abs=1e-7)
    assert result.transform.tx == pytest.approx(expected.tx, abs=1e-7)
    assert result.transform.ty == pytest.approx(expected.ty, abs=1e-7)


def test_single_anchor_uses_page_frame_instead_of_unverified_translation():
    old = (_entity("a", "UNIQUE", (0.2, 0.3)),)
    new = (_entity("b", "UNIQUE", (0.25, 0.28)),)
    result = estimate_alignment(old, new)
    assert result.status == "identity-unverified"
    assert result.anchor_count == 1
    assert result.inlier_count == 0
    assert result.transform == Transform()
    assert "page frames were overlaid" in result.note


def test_inconsistent_anchor_correspondences_fail_instead_of_guessing():
    points = ((0.08, 0.12), (0.82, 0.17), (0.23, 0.81), (0.68, 0.62))
    old = tuple(_entity(f"o{i}", f"ANCHOR {i}", point) for i, point in enumerate(points))
    scrambled = (points[1], points[0], points[2], (0.91, 0.91))
    new = tuple(_entity(f"n{i}", f"ANCHOR {i}", point) for i, point in enumerate(scrambled))
    result = estimate_alignment(old, new)
    assert result.status == "failed"


def test_dense_alignment_tests_a_majority_cover_even_with_small_sample_limit():
    expected = Transform(scale=1.01, rotation_radians=0.012, tx=0.014, ty=-0.009)
    points = tuple((0.06 + (index % 8) * 0.105, 0.08 + (index // 8) * 0.17) for index in range(40))
    old = tuple(_entity(f"o{i}", f"ANCHOR {i:02}", point) for i, point in enumerate(points))
    new = tuple(
        _entity(
            f"n{i}",
            f"ANCHOR {i:02}",
            (
                transform_point(point, expected)
                if i < 22
                else (
                    0.04 + ((i * 37) % 89) / 100.0,
                    0.03 + ((i * 53) % 83) / 100.0,
                )
            ),
        )
        for i, point in enumerate(points)
    )
    settings = ComparisonSettings(alignment_max_hypotheses=2)

    result = estimate_alignment(old, new, settings)

    assert result.status == "aligned"
    assert result.inlier_count == 22
    assert result.inlier_ratio == pytest.approx(0.55)
    assert result.transform.tx == pytest.approx(expected.tx, abs=1e-7)
    assert result.transform.ty == pytest.approx(expected.ty, abs=1e-7)
