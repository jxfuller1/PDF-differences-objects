from __future__ import annotations

from pdf_differences.mechanical import classify_text, interpret_change
from pdf_differences.models import ChangeCategory, Entity, EntityKind


def _text(entity_id: str, text: str, x: float, y: float) -> Entity:
    bbox = (x, y, x + 0.08, y + 0.02)
    return Entity(
        id=entity_id,
        page_index=0,
        kind=EntityKind.TEXT,
        bbox=bbox,
        anchor=(x, y),
        content_signature=entity_id,
        shape_signature="text-span",
        style_signature="style",
        text=text,
        text_normalized=text.casefold(),
        font_size=0.01,
    )


def test_mechanical_categories_cover_requested_parser_branches():
    assert classify_text("Ø12.5 ±0.1 mm", (0.2, 0.2, 0.3, 0.23)) == ChangeCategory.DIMENSION
    assert classify_text("⌖ ⌀0.20 M | A | B", (0.2, 0.3, 0.4, 0.33)) == ChangeCategory.GDT
    assert classify_text("GENERAL NOTE: REMOVE BURRS", (0.2, 0.4, 0.5, 0.43)) == ChangeCategory.NOTE
    assert classify_text("REVISION B", (0.8, 0.9, 0.9, 0.93)) == ChangeCategory.REVISION
    assert classify_text("SCALE 1:8", (0.2, 0.2, 0.3, 0.23)) == ChangeCategory.OTHER


def test_revision_cell_uses_nearby_header_context():
    header = _text("header", "REV", 0.78, 0.87)
    assert classify_text("B", (0.79, 0.89, 0.81, 0.91), (header,)) == ChangeCategory.REVISION


def test_inspection_relevance_is_explicit_and_auditable():
    dimension = interpret_change(EntityKind.TEXT, "10 mm", "12 mm", (0.2, 0.2, 0.3, 0.23), ())
    admin_revision = interpret_change(EntityKind.TEXT, "REV A", "REV B", (0.8, 0.9, 0.9, 0.93), ())
    technical_note = interpret_change(
        EntityKind.TEXT,
        "NOTE: SURFACE FINISH 63",
        "NOTE: SURFACE FINISH 32",
        (0.2, 0.5, 0.5, 0.53),
        (),
    )
    assert dimension.relevant
    assert not admin_revision.relevant
    assert technical_note.relevant
    assert dimension.reason and admin_revision.reason and technical_note.reason


def test_geometry_near_dimension_inherits_dimension_category():
    annotation = _text("dim", "25 mm", 0.4, 0.4)
    interpreted = interpret_change(
        EntityKind.GEOMETRY, None, None, (0.42, 0.41, 0.45, 0.44), (annotation,)
    )
    assert interpreted.category == ChangeCategory.DIMENSION
    assert interpreted.relevant
