from __future__ import annotations

from pdf_differences.callouts import reconstruct_callouts
from pdf_differences.matching import match_entities
from pdf_differences.models import ChangeCategory, Entity, EntityKind, MatchTier


def _text(
    entity_id: str,
    text: str,
    point: tuple[float, float],
    *,
    width: float | None = None,
    height: float = 0.014,
) -> Entity:
    x, y = point
    text_width = width if width is not None else max(0.014, 0.0065 * len(text))
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.TEXT,
        bbox=(x, y, x + text_width, y + height),
        anchor=(x + text_width / 2.0, y + height / 2.0),
        content_signature=f"text:{text}",
        shape_signature="text-span",
        style_signature="callout-style",
        text=text,
        text_normalized=text.casefold(),
        font_size=0.011,
    )


def _callout(
    entity_id: str,
    text: str,
    point: tuple[float, float],
    *,
    attachments: tuple[tuple[float, float], ...] = (),
    structure: str = "dimension:linear",
) -> Entity:
    x, y = point
    width = max(0.05, 0.0065 * len(text))
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.TEXT,
        bbox=(x, y, x + width, y + 0.014),
        anchor=(x + width / 2.0, y + 0.007),
        content_signature=f"callout:{ChangeCategory.DIMENSION.value}:{structure}:{text.casefold()}",
        shape_signature="callout-shape",
        style_signature="callout-style",
        text=text,
        text_normalized=text.casefold(),
        font_size=0.011,
        callout_category=ChangeCategory.DIMENSION,
        callout_structure=structure,
        callout_member_ids=(entity_id,),
        callout_attachment_points=attachments,
    )


def _compound_frame(
    entity_id: str,
    bbox: tuple[float, float, float, float],
    *,
    primitive_count: int = 4,
) -> Entity:
    x0, y0, x1, y1 = bbox
    geometry_segments = []
    cell_width = (x1 - x0) / primitive_count
    for index in range(primitive_count):
        left = x0 + index * cell_width
        right = left + cell_width
        geometry_segments.extend(
            (
                ((left, y0), (right, y0)),
                ((right, y0), (right, y1)),
                ((right, y1), (left, y1)),
                ((left, y1), (left, y0)),
            )
        )
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=bbox,
        anchor=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        content_signature="compound-frame",
        shape_signature="compound-frame-shape",
        style_signature="frame-style",
        op_histogram=(("re", primitive_count),),
        primitive_count=primitive_count,
        path_length=0.5,
        geometry_segments=tuple(geometry_segments),
    )


def _multi_primitive_boxed_note(entity_id: str, bbox: tuple[float, float, float, float]) -> Entity:
    frame = _compound_frame(entity_id, bbox, primitive_count=1)
    x0, y0, x1, y1 = bbox
    return Entity(
        id=frame.id,
        page_index=frame.page_index,
        kind=frame.kind,
        bbox=frame.bbox,
        anchor=frame.anchor,
        content_signature=frame.content_signature,
        shape_signature=frame.shape_signature,
        style_signature=frame.style_signature,
        op_histogram=(("l", 1), ("re", 1)),
        primitive_count=2,
        path_length=frame.path_length,
        geometry_segments=(
            *frame.geometry_segments,
            ((x0 + 0.02, (y0 + y1) / 2.0), (x1 - 0.02, (y0 + y1) / 2.0)),
        ),
    )


def _vector_symbol(entity_id: str, bbox: tuple[float, float, float, float]) -> Entity:
    x0, y0, x1, y1 = bbox
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=bbox,
        anchor=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        content_signature="vector-symbol",
        shape_signature="vector-symbol-shape",
        style_signature="frame-style",
        op_histogram=(("c", 2),),
        primitive_count=2,
        path_length=0.04,
    )


def test_fragmented_dimension_reconstructs_once_and_skips_non_dimension_neighbors():
    entities = (
        _text("qty", "4X", (0.10, 0.20), width=0.018),
        _text("dia", "Ø", (0.13, 0.20), width=0.010),
        _text("nom", ".255", (0.15, 0.20), width=0.022),
        _text("tol", "+.010/-.000", (0.18, 0.20), width=0.053),
        _text("side", "BOTH SIDES", (0.245, 0.20), width=0.055),
        _text("section", "SECTION E-E", (0.31, 0.20), width=0.075),
        _text("scale", "SCALE 4:1", (0.395, 0.20), width=0.070),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    composite = composites[0]
    assert composite.callout_category == ChangeCategory.DIMENSION
    assert composite.text == "4X Ø .255 +.010/-.000 BOTH SIDES"
    assert composite.callout_member_ids == ("qty", "dia", "nom", "tol", "side")
    assert "section" in {entity.id for entity in result}
    assert "scale" in {entity.id for entity in result}
    assert {entity.id for entity in result if entity.callout_category is None} == {
        "section",
        "scale",
    }


def test_replacement_glyph_between_numbers_is_treated_as_plus_minus():
    result = reconstruct_callouts((_text("dimension", "4X R.405�.050", (0.10, 0.20), width=0.10),))
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].callout_category == ChangeCategory.DIMENSION
    assert composites[0].text == "4X R.405 ±.050"


def test_unit_only_singleton_is_not_promoted_without_structural_evidence():
    result = reconstruct_callouts((_text("note", "10 MM", (0.10, 0.20), width=0.06),))

    assert not any(entity.callout_category is not None for entity in result)
    assert result[0].id == "note"


def test_fragmented_nominal_and_unit_is_valid_dimension_grammar():
    result = reconstruct_callouts(
        (
            _text("nominal", "10", (0.10, 0.20), width=0.02),
            _text("unit", "MM", (0.13, 0.20), width=0.025),
        )
    )
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].callout_category == ChangeCategory.DIMENSION
    assert composites[0].callout_member_ids == ("nominal", "unit")


def test_nominal_dimension_edit_matches_as_one_callout_pair():
    old = (_callout("old", "4X Ø .255 +.010/-.000 BOTH SIDES", (0.20, 0.24)),)
    new = (_callout("new", "4X Ø .260 +.010/-.000 BOTH SIDES", (0.20, 0.24)),)

    result = match_entities(old, new)

    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.tier == MatchTier.ATTRIBUTE
    assert (match.old.id, match.new.id) == ("old", "new")
    assert not result.unmatched_old
    assert not result.unmatched_new


def test_adjacent_complete_dimensions_do_not_merge_on_proximity_alone():
    entities = (
        _text("qty-a", "4X", (0.10, 0.20), width=0.018),
        _text("dia-a", "Ø", (0.13, 0.20), width=0.010),
        _text("nom-a", ".255", (0.15, 0.20), width=0.022),
        _text("qty-b", "2X", (0.20, 0.20), width=0.018),
        _text("dia-b", "Ø", (0.23, 0.20), width=0.010),
        _text("nom-b", ".500", (0.25, 0.20), width=0.022),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 2
    assert {entity.text for entity in composites} == {"4X Ø .255", "2X Ø .500"}
    assert {frozenset(entity.callout_member_ids) for entity in composites} == {
        frozenset({"qty-a", "dia-a", "nom-a"}),
        frozenset({"qty-b", "dia-b", "nom-b"}),
    }


def test_compound_path_feature_control_frame_reconstructs_as_one_gdt_callout():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.40, 0.35)),
        _text("symbol", "POSITION", (0.11, 0.315), width=0.08, height=0.015),
        _text("tol", ".010", (0.21, 0.315), width=0.04, height=0.015),
        _text("datum-a", "A", (0.28, 0.315), width=0.015, height=0.015),
        _text("datum-b", "B", (0.33, 0.315), width=0.015, height=0.015),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].callout_category == ChangeCategory.GDT
    assert composites[0].bbox == (0.10, 0.30, 0.40, 0.35)
    assert set(composites[0].callout_member_ids) == {
        "frame",
        "symbol",
        "tol",
        "datum-a",
        "datum-b",
    }


def test_single_boxed_position_note_is_not_treated_as_a_control_frame():
    entities = (
        _compound_frame("box", (0.10, 0.30, 0.40, 0.35), primitive_count=1),
        _text("note", "POSITION .010", (0.12, 0.315), width=0.12, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"box", "note"}


def test_multi_primitive_box_without_vertical_cells_is_still_not_a_control_frame():
    entities = (
        _multi_primitive_boxed_note("box", (0.10, 0.30, 0.40, 0.35)),
        _text("note", "POSITION .010", (0.12, 0.315), width=0.12, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"box", "note"}


def test_vector_symbol_gdt_is_inferred_from_frame_tolerance_and_datums():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.40, 0.35)),
        _vector_symbol("position-symbol", (0.115, 0.31, 0.145, 0.34)),
        _text("tol", ".010", (0.18, 0.315), width=0.04, height=0.015),
        _text("modifier", "M", (0.24, 0.315), width=0.015, height=0.015),
        _text("datum-a", "B", (0.29, 0.315), width=0.015, height=0.015),
        _text("datum-b", "C", (0.34, 0.315), width=0.015, height=0.015),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].callout_category == ChangeCategory.GDT
    assert set(composites[0].callout_member_ids) == {
        "frame",
        "position-symbol",
        "tol",
        "modifier",
        "datum-a",
        "datum-b",
    }
    assert not any(entity.id == "position-symbol" for entity in result)


def test_boxed_decimal_and_one_letter_is_not_assumed_to_be_gdt():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.28, 0.35)),
        _text("value", ".010", (0.13, 0.315), width=0.04, height=0.015),
        _text("letter", "A", (0.22, 0.315), width=0.015, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"frame", "value", "letter"}


def test_touching_feature_control_frames_split_at_each_semantic_marker():
    cells = tuple(
        _compound_frame(f"cell-{index}", (0.10 + index * 0.05, 0.30, 0.15 + index * 0.05, 0.35))
        for index in range(8)
    )
    entities = (
        *cells,
        _text("position", "POSITION", (0.105, 0.315), width=0.04, height=0.015),
        _text("tol-a", ".010", (0.155, 0.315), width=0.035, height=0.015),
        _text("datum-a", "A", (0.215, 0.315), width=0.012, height=0.015),
        _text("datum-b", "B", (0.265, 0.315), width=0.012, height=0.015),
        _text("flatness", "FLATNESS", (0.305, 0.315), width=0.04, height=0.015),
        _text("tol-b", ".020", (0.355, 0.315), width=0.035, height=0.015),
        _text("datum-c", "C", (0.415, 0.315), width=0.012, height=0.015),
        _text("datum-d", "D", (0.465, 0.315), width=0.012, height=0.015),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category == ChangeCategory.GDT]

    assert len(composites) == 2
    assert {entity.text for entity in composites} == {
        "POSITION | .010 | A | B",
        "FLATNESS | .020 | C | D",
    }


def test_adjacent_frameless_feature_control_sequences_stay_separate():
    entities = (
        _text("position", "POSITION", (0.10, 0.30), width=0.055),
        _text("tol-a", ".010", (0.16, 0.30), width=0.03),
        _text("datum-a", "A", (0.195, 0.30), width=0.012),
        _text("datum-b", "B", (0.212, 0.30), width=0.012),
        _text("flatness", "FLATNESS", (0.235, 0.30), width=0.055),
        _text("tol-b", ".020", (0.295, 0.30), width=0.03),
        _text("datum-c", "C", (0.33, 0.30), width=0.012),
        _text("datum-d", "D", (0.347, 0.30), width=0.012),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category == ChangeCategory.GDT]

    assert len(composites) == 2
    assert {entity.text for entity in composites} == {
        "POSITION | .010 | A | B",
        "FLATNESS | .020 | C | D",
    }


def test_plain_material_condition_modifier_stays_inside_frameless_gdt():
    entities = (
        _text("position", "POSITION", (0.10, 0.30), width=0.055),
        _text("tolerance", ".010", (0.16, 0.30), width=0.03),
        _text("modifier", "M", (0.195, 0.30), width=0.012),
        _text("datum-a", "A", (0.212, 0.30), width=0.012),
        _text("datum-b", "B", (0.229, 0.30), width=0.012),
    )

    result = reconstruct_callouts(entities)
    composite = next(entity for entity in result if entity.callout_category == ChangeCategory.GDT)

    assert composite.text == "POSITION | .010 | M | A | B"
    assert set(composite.callout_member_ids) == {
        "position",
        "tolerance",
        "modifier",
        "datum-a",
        "datum-b",
    }


def test_modifier_word_cannot_start_a_false_frameless_gdt_callout():
    entities = (
        _text("modifier", "MMC", (0.10, 0.30), width=0.04),
        _text("value", ".010", (0.15, 0.30), width=0.03),
        _text("datum-a", "A", (0.185, 0.30), width=0.012),
        _text("datum-b", "B", (0.202, 0.30), width=0.012),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_wide_table_row_with_gdt_words_is_not_collapsed_into_one_callout():
    cells = tuple(
        _compound_frame(f"cell-{index}", (index * 0.10, 0.30, (index + 1) * 0.10, 0.35))
        for index in range(8)
    )
    entities = (
        *cells,
        _text("position", "POSITION", (0.01, 0.315), width=0.07, height=0.015),
        _text("value", ".010", (0.11, 0.315), width=0.04, height=0.015),
        _text("datum-a", "A", (0.21, 0.315), width=0.015, height=0.015),
        _text("datum-b", "B", (0.31, 0.315), width=0.015, height=0.015),
        _text("description", "INSPECTION TABLE", (0.51, 0.315), width=0.16, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_repeated_identical_callouts_match_by_attachment_when_positions_swap():
    old = (
        _callout("old-a", "Ø .255", (0.400, 0.24), attachments=((0.18, 0.30),)),
        _callout("old-b", "Ø .255", (0.412, 0.24), attachments=((0.82, 0.30),)),
    )
    new = (
        _callout("new-a", "Ø .255", (0.408, 0.24), attachments=((0.18, 0.30),)),
        _callout("new-b", "Ø .255", (0.404, 0.24), attachments=((0.82, 0.30),)),
    )

    result = match_entities(old, new)

    assert len(result.matches) == 2
    assert {(match.old.id, match.new.id) for match in result.matches} == {
        ("old-a", "new-a"),
        ("old-b", "new-b"),
    }
    assert all(match.tier == MatchTier.ATTRIBUTE for match in result.matches)
    assert not result.unmatched_old
    assert not result.unmatched_new


def test_repeated_identical_callouts_without_attachment_are_not_force_matched():
    old = (
        _callout("old-a", "Ø .255", (0.40, 0.24)),
        _callout("old-b", "Ø .255", (0.40, 0.24)),
    )
    new = (
        _callout("new-a", "Ø .255", (0.40, 0.24)),
        _callout("new-b", "Ø .255", (0.40, 0.24)),
    )

    result = match_entities(old, new)

    assert not result.matches
    assert {entity.id for entity in result.unmatched_old} == {"old-a", "old-b"}
    assert {entity.id for entity in result.unmatched_new} == {"new-a", "new-b"}


def test_similar_numbers_attached_to_different_features_are_not_cross_matched():
    old = (_callout("old", ".530", (0.40, 0.24), attachments=((0.10, 0.10),)),)
    new = (_callout("new", ".050", (0.40, 0.24), attachments=((0.80, 0.80),)),)

    result = match_entities(old, new)

    assert not result.matches
    assert result.unmatched_old == old
    assert result.unmatched_new == new
