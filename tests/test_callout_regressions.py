from __future__ import annotations

from pdf_differences.callouts import reconstruct_callouts
from pdf_differences.models import ChangeCategory, Entity, EntityKind


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
        content_signature=f"geometry:{entity_id}",
        shape_signature="batched-geometry",
        style_signature="frame-style",
        op_histogram=((op, len(segments)),),
        primitive_count=len(segments),
        path_length=0.12 * len(segments),
        geometry_segments=segments,
    )


def _rectangle(entity_id: str, bbox: tuple[float, float, float, float]) -> Entity:
    x0, y0, x1, y1 = bbox
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.GEOMETRY,
        bbox=bbox,
        anchor=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
        content_signature=f"geometry:{entity_id}",
        shape_signature="rectangle",
        style_signature="title-block-style",
        op_histogram=(("re", 1),),
        primitive_count=1,
        path_length=0.08,
        geometry_segments=(
            ((x0, y0), (x1, y0)),
            ((x1, y0), (x1, y1)),
            ((x1, y1), (x0, y1)),
            ((x0, y1), (x0, y0)),
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
        content_signature=f"geometry:{entity_id}",
        shape_signature="vector-symbol",
        style_signature="frame-style",
        op_histogram=(("c", 2),),
        primitive_count=2,
        path_length=0.04,
        geometry_segments=(
            ((x0, (y0 + y1) / 2.0), ((x0 + x1) / 2.0, y0)),
            (((x0 + x1) / 2.0, y0), (x1, (y0 + y1) / 2.0)),
            ((x1, (y0 + y1) / 2.0), ((x0 + x1) / 2.0, y1)),
            (((x0 + x1) / 2.0, y1), (x0, (y0 + y1) / 2.0)),
        ),
    )


def test_fragmented_dimension_tokens_with_standalone_signs_form_one_dimension_callout() -> None:
    entities = (
        _text("qty", "4X", (0.10, 0.20), width=0.018),
        _text("dia", "Ø", (0.13, 0.20), width=0.010),
        _text("nom", ".255", (0.15, 0.20), width=0.022),
        _text("plus", "+", (0.18, 0.20), width=0.010),
        _text("upper", ".010", (0.192, 0.20), width=0.022),
        _text("minus", "-", (0.220, 0.20), width=0.010),
        _text("lower", ".000", (0.232, 0.20), width=0.022),
        _text("side", "BOTH SIDES", (0.265, 0.20), width=0.055),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category is not None]

    assert len(composites) == 1
    composite = composites[0]
    assert composite.callout_category == ChangeCategory.DIMENSION
    assert composite.text == "4X Ø .255 +.010 -.000 BOTH SIDES"
    assert composite.callout_member_ids == (
        "qty",
        "dia",
        "nom",
        "plus",
        "upper",
        "minus",
        "lower",
        "side",
    )


def test_batched_frame_geometry_splits_into_local_gdt_rows() -> None:
    walls = (0.10, 0.14, 0.20, 0.24, 0.28, 0.32)
    batched_h = _geometry(
        "frame-h",
        (0.10, 0.10, 0.70, 0.55),
        (
            ((0.10, 0.10), (0.32, 0.10)),
            ((0.10, 0.14), (0.32, 0.14)),
            ((0.10, 0.20), (0.32, 0.20)),
            ((0.10, 0.24), (0.32, 0.24)),
            ((0.60, 0.50), (0.70, 0.50)),
            ((0.60, 0.55), (0.70, 0.55)),
        ),
    )
    batched_v = _geometry(
        "frame-v",
        (0.10, 0.10, 0.32, 0.24),
        tuple(
            ((x, top), (x, bottom)) for top, bottom in ((0.10, 0.14), (0.20, 0.24)) for x in walls
        ),
    )

    entities = (
        batched_h,
        batched_v,
        _vector_symbol("vector-1", (0.108, 0.108, 0.132, 0.132)),
        _text("tol-1", ".010", (0.148, 0.112), width=0.040, height=0.016),
        _text("modifier-1", "M", (0.210, 0.112), width=0.016, height=0.016),
        _text("datum-b", "B", (0.250, 0.112), width=0.016, height=0.016),
        _text("datum-c", "C", (0.290, 0.112), width=0.016, height=0.016),
        _vector_symbol("vector-2", (0.108, 0.208, 0.132, 0.232)),
        _text("tol-2", ".020", (0.148, 0.212), width=0.040, height=0.016),
        _text("modifier-2", "M", (0.210, 0.212), width=0.016, height=0.016),
        _text("datum-a", "A", (0.250, 0.212), width=0.016, height=0.016),
        _text("datum-d", "D", (0.290, 0.212), width=0.016, height=0.016),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category == ChangeCategory.GDT]

    assert len(composites) == 2
    member_sets = [set(composite.callout_member_ids) for composite in composites]
    first_members = {"vector-1", "tol-1", "modifier-1", "datum-b", "datum-c"}
    second_members = {"vector-2", "tol-2", "modifier-2", "datum-a", "datum-d"}
    assert any(first_members <= ids for ids in member_sets)
    assert any(second_members <= ids for ids in member_sets)
    assert "frame-h" not in {entity.id for entity in result}
    assert "frame-v" not in {entity.id for entity in result}
    residuals = [entity for entity in result if entity.callout_category is None]
    assert len(residuals) == 1
    assert set(residuals[0].geometry_segments) == {
        ((0.60, 0.50), (0.70, 0.50)),
        ((0.60, 0.55), (0.70, 0.55)),
    }


def test_title_block_grids_do_not_promote_to_gdt() -> None:
    entities = (
        _rectangle("title-block", (0.68, 0.78, 0.96, 0.94)),
        _text("label", "POSITION", (0.70, 0.81), width=0.06, height=0.015),
        _text("value", "01.02", (0.79, 0.81), width=0.035, height=0.015),
        _text("rev", "REV A", (0.70, 0.86), width=0.04, height=0.015),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)
    assert {entity.id for entity in result} == {"title-block", "label", "value", "rev"}


def test_numeric_table_row_without_feature_symbol_does_not_promote_to_gdt() -> None:
    walls = (0.10, 0.16, 0.22, 0.28, 0.34)
    frame = _geometry(
        "grid",
        (0.10, 0.30, 0.34, 0.35),
        (
            ((0.10, 0.30), (0.34, 0.30)),
            ((0.10, 0.35), (0.34, 0.35)),
            *(tuple(((x, 0.30), (x, 0.35)) for x in walls)),
        ),
    )
    entities = (
        frame,
        _text("value", ".010", (0.11, 0.315), width=0.04),
        _text("material", "M", (0.175, 0.315), width=0.012),
        _text("column-b", "B", (0.235, 0.315), width=0.012),
        _text("column-c", "C", (0.295, 0.315), width=0.012),
    )

    result = reconstruct_callouts(entities)

    assert not any(entity.callout_category == ChangeCategory.GDT for entity in result)
    assert {entity.id for entity in result} == {
        "grid",
        "value",
        "material",
        "column-b",
        "column-c",
    }


def test_touching_batched_frames_can_share_continuous_rail_sources() -> None:
    walls = tuple(0.10 + 0.05 * index for index in range(9))
    rails = _geometry(
        "shared-rails",
        (0.10, 0.30, 0.50, 0.35),
        (
            ((0.10, 0.30), (0.50, 0.30)),
            ((0.10, 0.35), (0.50, 0.35)),
        ),
    )
    dividers = _geometry(
        "shared-walls",
        (0.10, 0.30, 0.50, 0.35),
        tuple(((x, 0.30), (x, 0.35)) for x in walls),
    )
    entities = (
        rails,
        dividers,
        _text("position", "POSITION", (0.105, 0.315), width=0.040),
        _text("tol-a", ".010", (0.155, 0.315), width=0.035),
        _text("datum-a", "A", (0.215, 0.315), width=0.012),
        _text("datum-b", "B", (0.265, 0.315), width=0.012),
        _text("flatness", "FLATNESS", (0.305, 0.315), width=0.040),
        _text("tol-b", ".020", (0.355, 0.315), width=0.035),
        _text("datum-c", "C", (0.415, 0.315), width=0.012),
        _text("datum-d", "D", (0.465, 0.315), width=0.012),
    )

    result = reconstruct_callouts(entities)
    composites = [entity for entity in result if entity.callout_category == ChangeCategory.GDT]

    assert {entity.text for entity in composites} == {
        "POSITION | .010 | A | B",
        "FLATNESS | .020 | C | D",
    }
