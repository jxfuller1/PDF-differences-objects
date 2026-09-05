"""Rule-based mechanical-drawing parsing and inspection relevance."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from pdf_differences.config import (
    ADMIN_REVISION_KEYWORDS,
    INSPECTION_KEYWORDS,
    SETTINGS,
    ComparisonSettings,
)
from pdf_differences.models import BBox, ChangeCategory, Entity, EntityKind

_GDT_SYMBOLS = frozenset("⌖⏥⏊∥⌭⌒⌓○⌀ⓂⓁⓈ")
_GDT_WORDS = re.compile(
    r"\b(?:ANGULARITY|CIRCULARITY|CONCENTRICITY|CYLINDRICITY|DATUM|FLATNESS|"
    r"MMC|LMC|PARALLELISM|PERPENDICULARITY|POSITION|PROFILE|RUNOUT|STRAIGHTNESS|"
    r"SYMMETRY|TOTAL\s+RUNOUT)\b",
    re.IGNORECASE,
)
_REVISION_WORDS = re.compile(r"\b(?:REV|REVISION|ECN|ECO|CHANGE\s+NOTICE)\b", re.IGNORECASE)
_NOTE_WORDS = re.compile(
    r"\b(?:GENERAL\s+NOTES?|NOTES?|UNLESS\s+OTHERWISE|DO\s+NOT|TYP(?:ICAL)?|"
    r"DEBURR|BREAK\s+(?:ALL\s+)?EDGES|REMOVE\s+BURRS|MATERIAL|FINISH|WELD|"
    r"SURFACE|HARDNESS|TORQUE|CLEAN)\b",
    re.IGNORECASE,
)
_NON_DIMENSION_CONTEXT = re.compile(
    r"\b(?:DRAWING|DWG|PAGE|PART|SCALE|SHEET|ZONE)\b",
    re.IGNORECASE,
)
_DIMENSION = re.compile(
    r"(?:^|\s)(?:\d+\s*[xX]\s*)?(?:R|SR|Ø|⌀|DIA\.?|M)?\s*"
    r"\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?\s*"
    r"(?:MM|CM|M|IN|FT|°|DEG|\"|')?"
    r"(?:\s*(?:±|\+/-|\+\s*\d|[-+]\s*\d|MAX\b|MIN\b|REF\b|TYP\b|THRU\b|"
    r"EQ\s+SP))?",
    re.IGNORECASE,
)
_REVISION_CELL = re.compile(
    r"^(?:[A-Z]{1,2}|\d{1,3}|\d{1,2}[-/.]\d{1,2}(?:[-/.]\d{2,4})?)$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class Interpretation:
    category: ChangeCategory
    relevant: bool
    reason: str


def _center(bbox: BBox) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _distance_to_bbox(entity: Entity, bbox: BBox) -> float:
    return math.dist(entity.centroid, _center(bbox))


def _nearby_text(
    bbox: BBox,
    entities: Iterable[Entity],
    radius: float,
) -> tuple[Entity, ...]:
    return tuple(
        sorted(
            (
                entity
                for entity in entities
                if entity.kind == EntityKind.TEXT and _distance_to_bbox(entity, bbox) <= radius
            ),
            key=lambda entity: (_distance_to_bbox(entity, bbox), entity.id),
        )
    )


def _in_title_block(bbox: BBox, settings: ComparisonSettings) -> bool:
    x, y = _center(bbox)
    return (x >= settings.title_block_x_min and y >= settings.title_block_y_min) or y >= 0.9


def _contains_keyword(text: str, keywords: frozenset[str]) -> bool:
    upper = text.upper()
    return any(re.search(rf"\b{re.escape(keyword)}\b", upper) for keyword in keywords)


def classify_text(
    text: str,
    bbox: BBox,
    page_entities: Iterable[Entity] = (),
    settings: ComparisonSettings = SETTINGS,
) -> ChangeCategory:
    """Classify a changed text slot using its payload, position, and nearby labels."""

    cleaned = " ".join(text.split())
    neighbors = _nearby_text(bbox, page_entities, max(0.1, settings.nearby_annotation_radius * 2))
    neighborhood = " ".join([cleaned, *(entity.text for entity in neighbors[:6])])

    if any(symbol in cleaned for symbol in _GDT_SYMBOLS) or _GDT_WORDS.search(cleaned):
        return ChangeCategory.GDT
    if _REVISION_WORDS.search(cleaned):
        return ChangeCategory.REVISION
    revision_values = [value.strip() for value in re.split(r"\s*->\s*", cleaned)]
    if _in_title_block(bbox, settings) and _REVISION_WORDS.search(neighborhood):
        if revision_values and all(_REVISION_CELL.fullmatch(value) for value in revision_values):
            return ChangeCategory.REVISION
    if _NOTE_WORDS.search(cleaned):
        return ChangeCategory.NOTE
    dimension_match = None if _NON_DIMENSION_CONTEXT.search(cleaned) else _DIMENSION.search(cleaned)
    if dimension_match and any(character.isdigit() for character in dimension_match.group(0)):
        # Bare integers in a title block are more likely sheet/revision metadata.
        token = dimension_match.group(0).strip()
        if not (_in_title_block(bbox, settings) and token.isdigit() and len(token) <= 4):
            return ChangeCategory.DIMENSION
    return ChangeCategory.OTHER


def interpret_change(
    kind: EntityKind,
    before_text: str | None,
    after_text: str | None,
    bbox: BBox,
    page_entities: Iterable[Entity],
    settings: ComparisonSettings = SETTINGS,
) -> Interpretation:
    """Assign the diagram's parser bucket and an auditable relevance decision."""

    combined = " -> ".join(value for value in (before_text, after_text) if value)
    if kind == EntityKind.TEXT:
        category = classify_text(combined, bbox, page_entities, settings)
    else:
        category = ChangeCategory.GEOMETRY
        nearby = _nearby_text(bbox, page_entities, settings.nearby_annotation_radius)
        for annotation in nearby:
            candidate = classify_text(annotation.text, annotation.bbox, page_entities, settings)
            if candidate in {ChangeCategory.DIMENSION, ChangeCategory.GDT}:
                category = candidate
                break

    if category == ChangeCategory.GEOMETRY:
        return Interpretation(
            category, True, "Vector geometry changed; physical form may be affected."
        )
    if category == ChangeCategory.DIMENSION:
        return Interpretation(category, True, "A measured requirement or tolerance changed.")
    if category == ChangeCategory.GDT:
        return Interpretation(
            category, True, "A GD&T control or datum-related requirement changed."
        )
    if category == ChangeCategory.NOTE:
        if _contains_keyword(combined, INSPECTION_KEYWORDS):
            return Interpretation(category, True, "The note contains an inspection-impact keyword.")
        return Interpretation(
            category, False, "General note change has no configured inspection keyword."
        )
    if category == ChangeCategory.REVISION:
        technical_words = INSPECTION_KEYWORDS - ADMIN_REVISION_KEYWORDS
        if _contains_keyword(combined, technical_words):
            return Interpretation(
                category, True, "Revision text describes a technical inspection impact."
            )
        return Interpretation(
            category, False, "Administrative revision metadata is not itself inspectable."
        )
    return Interpretation(
        category, False, "Unclassified text change has no deterministic inspection rule."
    )
