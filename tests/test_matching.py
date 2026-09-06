from __future__ import annotations

from pdf_differences.config import ComparisonSettings
from pdf_differences.matching import _hungarian_assign, match_entities
from pdf_differences.models import Entity, EntityKind, MatchTier


def _text(entity_id: str, text: str, point=(0.2, 0.2), *, width: float = 0.08) -> Entity:
    x, y = point
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.TEXT,
        bbox=(x, y, x + width, y + 0.02),
        anchor=point,
        content_signature=f"text:{text}",
        shape_signature="text-span",
        style_signature="style",
        text=text,
        text_normalized=text.casefold(),
        font_size=0.01,
    )


def _geometry(entity_id: str, point=(0.2, 0.2), signature="shape") -> Entity:
    x, y = point
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=(x - 0.03, y - 0.02, x + 0.03, y + 0.02),
        anchor=point,
        content_signature=f"geometry:{signature}",
        shape_signature=signature,
        style_signature="style",
        op_histogram=(("l", 4),),
        primitive_count=4,
        path_length=0.2,
    )


def _vector_glyph(entity_id: str, point=(0.2, 0.2)) -> Entity:
    x, y = point
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=(x, y, x + 0.008, y + 0.014),
        anchor=(x + 0.004, y + 0.007),
        content_signature="outlined-diameter-glyph",
        shape_signature="outlined-diameter-glyph",
        style_signature="style",
        op_histogram=(("l", 48),),
        primitive_count=48,
        path_length=0.04,
    )


def test_exact_tier_claims_unchanged_entity_first():
    result = match_entities((_text("old", "NOTE A"),), (_text("new", "NOTE A"),))
    assert len(result.matches) == 1
    assert result.matches[0].tier == MatchTier.EXACT


def test_attribute_tier_pairs_dimension_value_change_in_place():
    result = match_entities((_text("old", "10 mm"),), (_text("new", "12 mm"),))
    assert len(result.matches) == 1
    assert result.matches[0].tier == MatchTier.ATTRIBUTE


def test_structural_tier_recovers_moved_geometry_without_ml():
    result = match_entities(
        (_geometry("old", (0.2, 0.2)),),
        (_geometry("new", (0.29, 0.2)),),
    )
    assert len(result.matches) == 1
    assert result.matches[0].tier == MatchTier.STRUCTURAL
    assert result.matches[0].score >= 0.68


def test_structural_tier_keeps_identical_moved_dimension_fragment_matchable():
    result = match_entities(
        (_text("old", ".8745", (0.50, 0.55)),),
        (_text("new", ".8745", (0.45, 0.50)),),
    )

    assert len(result.matches) == 1
    assert result.matches[0].tier == MatchTier.STRUCTURAL


def test_structural_tier_does_not_cross_match_changed_raw_dimension_fragments():
    old = (_text("old", ".8745", (0.50, 0.55)),)
    new = (_text("new", ".045", (0.45, 0.50)),)

    result = match_entities(old, new)

    assert not result.matches
    assert result.unmatched_old == old
    assert result.unmatched_new == new


def test_structural_tier_keeps_small_line_geometry_without_dimension_context_matchable():
    result = match_entities(
        (_vector_glyph("old", (0.50, 0.55)),),
        (_vector_glyph("new", (0.45, 0.50)),),
    )

    assert len(result.matches) == 1
    assert result.matches[0].tier == MatchTier.STRUCTURAL


def test_structural_tier_does_not_move_an_unparented_dimension_glyph():
    old_glyph = _vector_glyph("old-glyph", (0.49, 0.55))
    new_glyph = _vector_glyph("new-glyph", (0.44, 0.50))
    old = (old_glyph, _text("old-value", ".8745", (0.50, 0.55)))
    new = (new_glyph, _text("new-value", ".045", (0.45, 0.50)))

    result = match_entities(old, new)

    assert not result.matches
    assert old_glyph in result.unmatched_old
    assert new_glyph in result.unmatched_new


def test_structural_tier_keeps_text_bearing_revision_bubble_matchable():
    old_glyph = _vector_glyph("old-glyph", (0.49, 0.55))
    new_glyph = _vector_glyph("new-glyph", (0.44, 0.50))
    old = (
        old_glyph,
        _text("old-revision", "B1", (0.491, 0.550), width=0.006),
        _text("old-value", ".8745", (0.50, 0.55)),
    )
    new = (
        new_glyph,
        _text("new-revision", "B1", (0.441, 0.500), width=0.006),
        _text("new-value", ".8745", (0.45, 0.50)),
    )

    result = match_entities(old, new)

    glyph_match = next(match for match in result.matches if match.old.id == "old-glyph")
    assert glyph_match.new.id == "new-glyph"
    assert glyph_match.tier == MatchTier.STRUCTURAL


def test_far_lookalike_is_removed_and_added_not_false_move():
    result = match_entities(
        (_geometry("old", (0.1, 0.1)),),
        (_geometry("new", (0.8, 0.8)),),
    )
    assert not result.matches
    assert len(result.unmatched_old) == 1
    assert len(result.unmatched_new) == 1


def test_sparse_assignment_preserves_global_optimum_for_large_components():
    old = (_geometry("old-0"), _geometry("old-1"))
    new = (_geometry("new-0"), _geometry("new-1"))
    # A greedy choice takes 0->0 and strands row 1 with its weak edge. The
    # global optimum crosses the pairs: 0.80 + 0.85 > 0.90 + 0.68.
    candidates = {
        (0, 0): (0.90, 0.01),
        (0, 1): (0.80, 0.02),
        (1, 0): (0.85, 0.02),
        (1, 1): (0.68, 0.01),
    }
    settings = ComparisonSettings(sparse_assignment_threshold=1)

    matches, used_old, used_new = _hungarian_assign(candidates, old, new, 0.68, settings)

    assert {(match.old.id, match.new.id) for match in matches} == {
        ("old-0", "new-1"),
        ("old-1", "new-0"),
    }
    assert used_old == {0, 1}
    assert used_new == {0, 1}
