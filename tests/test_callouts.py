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
    source_block_index: int = -1,
    source_line_index: int = -1,
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
        source_block_index=source_block_index,
        source_line_index=source_line_index,
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


def _geometry(
    entity_id: str,
    bbox: tuple[float, float, float, float],
    segments: tuple[tuple[tuple[float, float], tuple[float, float]], ...],
    *,
    op: str = "l",
) -> Entity:
    x0, y0, x1, y1 = bbox
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=bbox,
        anchor=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        content_signature="geometry",
        shape_signature="geometry-shape",
        style_signature="frame-style",
        op_histogram=((op, len(segments)),),
        primitive_count=len(segments),
        path_length=sum(
            ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5 for start, end in segments
        ),
        geometry_segments=segments,
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


def test_unsigned_radius_limit_stack_reconstructs_as_one_dimension():
    entities = (
        _text(
            "feature",
            "R",
            (0.112, 0.20),
            width=0.012,
            source_block_index=4,
            source_line_index=0,
        ),
        _text(
            "upper",
            ".055",
            (0.125, 0.185),
            width=0.030,
            source_block_index=4,
            source_line_index=0,
        ),
        _text("lower", ".045", (0.125, 0.215), width=0.030),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].text == "R .055 / .045"
    assert composites[0].callout_structure == "dimension:radius:limit"
    assert set(composites[0].callout_member_ids) == {"feature", "upper", "lower"}


def test_compact_vector_diameter_glyph_joins_numeric_limit_stack():
    segments = (
        ((0.108, 0.20), (0.112, 0.22)),
        ((0.112, 0.22), (0.116, 0.20)),
        ((0.116, 0.20), (0.108, 0.22)),
        ((0.108, 0.22), (0.116, 0.20)),
    ) * 4
    entities = (
        _geometry("diameter-glyph", (0.10, 0.20, 0.116, 0.22), segments),
        _text("upper", ".055", (0.125, 0.185), width=0.030),
        _text("lower", ".045", (0.125, 0.215), width=0.030),
        _geometry(
            "leader",
            (0.155, 0.207, 0.250, 0.300),
            (((0.155, 0.207), (0.250, 0.300)),),
        ),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].text == ".055 / .045"
    assert composites[0].callout_structure == "dimension:linear:limit:vector"
    assert set(composites[0].callout_member_ids) == {"diameter-glyph", "upper", "lower"}


def test_compact_vector_prefix_joins_one_qualified_dimension():
    segments = (
        ((0.108, 0.20), (0.112, 0.22)),
        ((0.112, 0.22), (0.116, 0.20)),
        ((0.116, 0.20), (0.108, 0.22)),
        ((0.108, 0.22), (0.116, 0.20)),
    ) * 4
    entities = (
        _geometry("diameter-glyph", (0.10, 0.20, 0.116, 0.22), segments),
        _text("dimension", ".141 THRU", (0.125, 0.203), width=0.075),
        _geometry(
            "leader",
            (0.200, 0.210, 0.300, 0.300),
            (((0.200, 0.210), (0.300, 0.300)),),
        ),
    )

    result = reconstruct_callouts(entities)
    composite = next(entity for entity in result if entity.callout_category is not None)

    assert composite.callout_structure == "dimension:linear:vector"
    assert set(composite.callout_member_ids) == {"diameter-glyph", "dimension"}
    assert composite.bbox[0] == 0.10


def test_simple_arrowhead_cannot_become_a_vector_dimension_prefix():
    arrow = _geometry(
        "arrow",
        (0.10, 0.20, 0.116, 0.22),
        (
            ((0.10, 0.21), (0.116, 0.20)),
            ((0.10, 0.21), (0.116, 0.22)),
            ((0.116, 0.20), (0.116, 0.22)),
        ),
    )
    entities = (arrow, _text("dimension", ".141 THRU", (0.125, 0.203), width=0.075))

    result = reconstruct_callouts(entities)
    composite = next(entity for entity in result if entity.callout_category is not None)

    assert composite.callout_member_ids == ("dimension",)
    assert any(entity.id == "arrow" for entity in result)


def test_dense_line_art_without_a_leader_cannot_become_a_vector_prefix():
    segments = tuple(
        ((0.100 + index * 0.001, 0.200), (0.101 + index * 0.001, 0.220)) for index in range(16)
    )
    artwork = _geometry("dense-artwork", (0.10, 0.20, 0.116, 0.22), segments)
    entities = (artwork, _text("dimension", ".141 THRU", (0.125, 0.203), width=0.075))

    result = reconstruct_callouts(entities)
    composite = next(entity for entity in result if entity.callout_category is not None)

    assert composite.callout_member_ids == ("dimension",)
    assert any(entity.id == "dense-artwork" for entity in result)


def test_text_bearing_revision_bubble_cannot_become_a_vector_prefix():
    segments = tuple(
        ((0.100 + index * 0.001, 0.200), (0.101 + index * 0.001, 0.220)) for index in range(16)
    )
    bubble = _geometry("revision-bubble", (0.10, 0.20, 0.116, 0.22), segments)
    entities = (
        bubble,
        _text("revision", "B1", (0.103, 0.203), width=0.010),
        _text("dimension", ".141 THRU", (0.125, 0.203), width=0.075),
        _geometry(
            "leader",
            (0.200, 0.210, 0.300, 0.300),
            (((0.200, 0.210), (0.300, 0.300)),),
        ),
    )

    result = reconstruct_callouts(entities)
    composite = next(entity for entity in result if entity.callout_category is not None)

    assert composite.callout_member_ids == ("dimension",)
    assert {entity.id for entity in result if entity.callout_category is None} >= {
        "revision-bubble",
        "revision",
    }


def test_unsigned_number_stack_without_feature_evidence_stays_raw():
    entities = (
        _text("upper", ".055", (0.125, 0.185), width=0.030),
        _text("lower", ".045", (0.125, 0.215), width=0.030),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"upper", "lower"}


def test_feature_and_unrelated_numeric_rows_without_source_or_leader_evidence_stay_raw():
    entities = (
        _text("feature", "R", (0.112, 0.20), width=0.012),
        _text("upper", ".055", (0.125, 0.185), width=0.030),
        _text("lower", ".045", (0.125, 0.215), width=0.030),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"feature", "upper", "lower"}


def test_adjacent_unsigned_limit_stacks_keep_separate_feature_ownership():
    entities = (
        _text(
            "radius",
            "R",
            (0.112, 0.20),
            width=0.012,
            source_block_index=4,
            source_line_index=0,
        ),
        _text(
            "radius-upper",
            ".055",
            (0.125, 0.185),
            width=0.030,
            source_block_index=4,
            source_line_index=0,
        ),
        _text("radius-lower", ".045", (0.125, 0.215), width=0.030),
        _text(
            "diameter",
            "DIA",
            (0.300, 0.20),
            width=0.024,
            source_block_index=7,
            source_line_index=0,
        ),
        _text(
            "diameter-upper",
            ".750",
            (0.325, 0.185),
            width=0.030,
            source_block_index=7,
            source_line_index=0,
        ),
        _text("diameter-lower", ".745", (0.325, 0.215), width=0.030),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert {entity.text for entity in composites} == {
        "R .055 / .045",
        "DIA .750 / .745",
    }
    assert len(composites) == 2


def test_one_vector_prefix_tied_between_callouts_remains_raw():
    segments = (
        ((0.108, 0.20), (0.112, 0.22)),
        ((0.112, 0.22), (0.116, 0.20)),
        ((0.116, 0.20), (0.108, 0.22)),
        ((0.108, 0.22), (0.116, 0.20)),
    ) * 4
    vector = _geometry("ambiguous-glyph", (0.10, 0.20, 0.116, 0.22), segments)
    entities = (
        vector,
        _text("first", ".141 THRU", (0.125, 0.203), width=0.075),
        _text("second", ".151 THRU", (0.125, 0.203), width=0.075),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 2
    assert all("ambiguous-glyph" not in entity.callout_member_ids for entity in composites)
    assert any(entity.id == "ambiguous-glyph" for entity in result)


def test_unrelated_limit_and_diameter_callouts_remain_additions_and_removals():
    def glyph(entity_id: str, x: float, y: float) -> Entity:
        segments = (
            ((x + 0.008, y), (x + 0.012, y + 0.020)),
            ((x + 0.012, y + 0.020), (x + 0.016, y)),
            ((x + 0.016, y), (x + 0.008, y + 0.020)),
            ((x + 0.008, y + 0.020), (x + 0.016, y)),
        ) * 4
        return _geometry(entity_id, (x, y, x + 0.016, y + 0.020), segments)

    old = reconstruct_callouts(
        (
            glyph("old-diameter", 0.49, 0.54),
            _text("old-upper", ".8745", (0.515, 0.525), width=0.034),
            _text("old-lower", ".8740", (0.515, 0.555), width=0.034),
            _geometry(
                "old-leader",
                (0.420, 0.552, 0.515, 0.620),
                (((0.515, 0.552), (0.420, 0.620)),),
            ),
        )
    )
    new = reconstruct_callouts(
        (
            _text(
                "radius",
                "R",
                (0.437, 0.49),
                width=0.012,
                source_block_index=4,
                source_line_index=0,
            ),
            _text(
                "radius-upper",
                ".055",
                (0.450, 0.475),
                width=0.030,
                source_block_index=4,
                source_line_index=0,
            ),
            _text("radius-lower", ".045", (0.450, 0.505), width=0.030),
            glyph("new-diameter", 0.54, 0.50),
            _text("new-dimension", ".141 THRU", (0.565, 0.503), width=0.075),
            _geometry(
                "new-leader",
                (0.640, 0.450, 0.700, 0.510),
                (((0.640, 0.510), (0.700, 0.450)),),
            ),
        )
    )

    result = match_entities(old, new)

    assert not result.matches
    assert {
        entity.text for entity in result.unmatched_old if entity.callout_category is not None
    } == {".8745 / .8740"}
    assert {
        entity.text for entity in result.unmatched_new if entity.callout_category is not None
    } == {
        "R .055 / .045",
        ".141 THRU",
    }


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


def test_nearby_stacked_dimensions_keep_separate_tolerance_hypotheses():
    entities = (
        _text("nom-b", "4.650", (0.0577, 0.3135), width=0.0227, height=0.0173),
        _text("plus-b", "+", (0.0804, 0.3065), width=0.0049, height=0.0173),
        _text("upper-b", ".000", (0.0853, 0.3065), width=0.0158, height=0.0173),
        _text("minus-b", "-", (0.0815, 0.3191), width=0.0027, height=0.0173),
        _text("lower-b", ".020", (0.0853, 0.3191), width=0.0158, height=0.0173),
        _text("radius-a", "2X R.500", (0.0843, 0.2741), width=0.0349, height=0.0173),
        _text("plus-a", "+", (0.1192, 0.2671), width=0.0049, height=0.0173),
        _text("upper-a", ".050", (0.1241, 0.2671), width=0.0158, height=0.0173),
        _text("minus-a", "-", (0.1203, 0.2797), width=0.0027, height=0.0173),
        _text("lower-a", ".000", (0.1241, 0.2797), width=0.0158, height=0.0173),
        _text("nearby", ".590", (0.1315, 0.3200), width=0.0204, height=0.0173),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 2
    assert {entity.text for entity in composites} == {
        "2X R.500 +.050 / -.000",
        "4.650 +.000 / -.020",
    }
    assert {frozenset(entity.callout_member_ids) for entity in composites} == {
        frozenset({"radius-a", "plus-a", "upper-a", "minus-a", "lower-a"}),
        frozenset({"nom-b", "plus-b", "upper-b", "minus-b", "lower-b"}),
    }
    assert any(entity.id == "nearby" for entity in result)


def test_equally_plausible_tolerance_stacks_remain_unassigned():
    entities = (
        _text("nominal", "4.000", (0.10, 0.20), width=0.030, height=0.020),
        _text("upper-stack", "+.010/-.000", (0.132, 0.180), width=0.065, height=0.020),
        _text("lower-stack", "+.020/-.000", (0.132, 0.220), width=0.065, height=0.020),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"nominal", "upper-stack", "lower-stack"}


def test_support_fragment_shared_by_two_roots_stays_raw():
    entities = (
        _text("shared-quantity", "2X", (0.10, 0.20), width=0.018),
        _text("first-nominal", ".250", (0.122, 0.20), width=0.028),
        _text("second-nominal", ".500", (0.126, 0.20), width=0.028),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {
        "shared-quantity",
        "first-nominal",
        "second-nominal",
    }


def test_dimension_core_with_embedded_tolerance_cannot_claim_a_second_stack():
    entities = (
        _text("core", "R.500±.010", (0.10, 0.20), width=0.070, height=0.020),
        _text("other-stack", "+.020/-.000", (0.172, 0.20), width=0.065, height=0.020),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    assert composites[0].text == "R.500±.010"
    assert composites[0].callout_member_ids == ("core",)
    assert any(entity.id == "other-stack" for entity in result)


def test_boxed_decimal_and_one_letter_is_not_assumed_to_be_gdt():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.28, 0.35)),
        _text("value", ".010", (0.13, 0.315), width=0.04, height=0.015),
        _text("letter", "A", (0.22, 0.315), width=0.015, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category is not None for entity in result)
    assert {entity.id for entity in result} == {"frame", "value", "letter"}


def test_mixed_frame_topology_accepts_one_datum_with_vector_feature_evidence():
    entities = (
        _compound_frame("feature-cell", (0.10, 0.30, 0.14, 0.35), primitive_count=1),
        _geometry(
            "rails",
            (0.10, 0.30, 0.24, 0.35),
            (
                ((0.10, 0.30), (0.24, 0.30)),
                ((0.10, 0.35), (0.24, 0.35)),
            ),
        ),
        _geometry(
            "walls",
            (0.18, 0.30, 0.24, 0.35),
            (
                ((0.18, 0.30), (0.18, 0.35)),
                ((0.24, 0.30), (0.24, 0.35)),
            ),
        ),
        _vector_symbol("parallelism-symbol", (0.108, 0.308, 0.132, 0.342)),
        _text("tol", ".010", (0.145, 0.315), width=0.030, height=0.015),
        _text("datum-a", "A", (0.198, 0.315), width=0.015, height=0.015),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category == ChangeCategory.GDT]

    assert len(composites) == 1
    assert composites[0].text == ".010 | A"
    assert {"parallelism-symbol", "tol", "datum-a"} <= set(composites[0].callout_member_ids)


def test_one_datum_frame_accepts_a_two_object_parallelism_symbol():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.25, 0.35), primitive_count=3),
        _geometry(
            "parallelism-symbol",
            (0.11, 0.308, 0.14, 0.342),
            (
                ((0.11, 0.342), (0.125, 0.308)),
                ((0.125, 0.342), (0.14, 0.308)),
            ),
        ),
        _text("tol", ".010", (0.155, 0.315), width=0.035, height=0.015),
        _text("datum-a", "A", (0.215, 0.315), width=0.012, height=0.015),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category == ChangeCategory.GDT]

    assert len(composites) == 1
    assert composites[0].text == ".010 | A"
    assert "parallelism-symbol" not in {entity.id for entity in result}


def test_one_datum_frame_rejects_a_single_incidental_line_as_feature_evidence():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.25, 0.35), primitive_count=3),
        _geometry(
            "incidental-line",
            (0.115, 0.325, 0.135, 0.325),
            (((0.115, 0.325), (0.135, 0.325)),),
        ),
        _text("tol", ".010", (0.155, 0.315), width=0.035, height=0.015),
        _text("datum-a", "A", (0.215, 0.315), width=0.012, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_one_datum_frame_rejects_an_empty_trailing_table_cell():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.30, 0.35), primitive_count=4),
        _vector_symbol("parallelism-symbol", (0.108, 0.308, 0.142, 0.342)),
        _text("tol", ".010", (0.155, 0.315), width=0.035, height=0.015),
        _text("datum-a", "A", (0.215, 0.315), width=0.012, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_one_datum_frame_rejects_a_disconnected_feature_cell():
    entities = (
        _compound_frame("feature-cell", (0.10, 0.30, 0.14, 0.35), primitive_count=1),
        _geometry(
            "remaining-frame",
            (0.141, 0.30, 0.241, 0.35),
            (
                ((0.141, 0.30), (0.241, 0.30)),
                ((0.141, 0.35), (0.241, 0.35)),
                ((0.141, 0.30), (0.141, 0.35)),
                ((0.191, 0.30), (0.191, 0.35)),
                ((0.241, 0.30), (0.241, 0.35)),
            ),
        ),
        _vector_symbol("parallelism-symbol", (0.108, 0.308, 0.132, 0.342)),
        _text("tol", ".010", (0.149, 0.315), width=0.030, height=0.015),
        _text("datum-a", "A", (0.209, 0.315), width=0.015, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_one_datum_frame_rejects_vector_geometry_touching_the_cell_border():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.25, 0.35), primitive_count=3),
        _vector_symbol("border-artifact", (0.10, 0.308, 0.145, 0.342)),
        _text("tol", ".010", (0.155, 0.315), width=0.035, height=0.015),
        _text("datum-a", "A", (0.215, 0.315), width=0.012, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_one_datum_frame_rejects_off_center_multi_stroke_artwork():
    entities = (
        _compound_frame("frame", (0.10, 0.30, 0.25, 0.35), primitive_count=3),
        _vector_symbol("off-center-artwork", (0.102, 0.315, 0.112, 0.335)),
        _text("tol", ".010", (0.155, 0.315), width=0.035, height=0.015),
        _text("datum-a", "A", (0.215, 0.315), width=0.012, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)


def test_touching_feature_control_frames_split_at_each_semantic_marker():
    cells = tuple(
        _compound_frame(
            f"cell-{index}",
            (0.10 + index * 0.05, 0.30, 0.15 + index * 0.05, 0.35),
            primitive_count=1,
        )
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
        _compound_frame(
            f"cell-{index}",
            (index * 0.10, 0.30, (index + 1) * 0.10, 0.35),
            primitive_count=1,
        )
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
