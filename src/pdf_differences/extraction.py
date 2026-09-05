"""Extract exact vector paths and positioned text spans with PyMuPDF."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import pymupdf as fitz

from pdf_differences.models import BBox, Entity, EntityKind, Point

_ROUND = 6
_WHITE_THRESHOLD = 0.995
_VISIBLE_OPACITY = 0.001


def normalize_text(text: str) -> str:
    """Normalize Unicode and runs of whitespace, preserving meaningful case."""

    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _digest(*parts: object, length: int = 20) -> str:
    material = "|".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


def _norm_point(point: Any, width: float, height: float) -> Point:
    return (_clamp(float(point.x) / width), _clamp(float(point.y) / height))


def _norm_bbox(rect: Any, width: float, height: float) -> BBox:
    x0, y0, x1, y1 = (float(value) for value in rect)
    left, right = sorted((_clamp(x0 / width), _clamp(x1 / width)))
    top, bottom = sorted((_clamp(y0 / height), _clamp(y1 / height)))
    return (left, top, right, bottom)


def _color(value: Any) -> str:
    if value is None:
        return "none"
    if isinstance(value, int):
        return f"#{value:06x}"
    try:
        return "#" + "".join(f"{round(float(channel) * 255):02x}" for channel in value)
    except TypeError:
        return str(value)


def _opacity(value: Any) -> float:
    return 1.0 if value is None else float(value)


def _is_white(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, int):
        return value & 0xFFFFFF == 0xFFFFFF
    try:
        return all(float(channel) >= _WHITE_THRESHOLD for channel in value)
    except TypeError:
        return False


def _has_visible_paint(drawing: dict[str, Any]) -> bool:
    """Exclude transparent or white-on-white paths that do not mark a CAD sheet."""

    stroke = drawing.get("color")
    fill = drawing.get("fill")
    visible_stroke = (
        stroke is not None
        and _opacity(drawing.get("stroke_opacity")) > _VISIBLE_OPACITY
        and not _is_white(stroke)
    )
    visible_fill = (
        fill is not None
        and _opacity(drawing.get("fill_opacity")) > _VISIBLE_OPACITY
        and not _is_white(fill)
    )
    return visible_stroke or visible_fill


def _points_for_item(item: tuple, width: float, height: float) -> tuple[Point, ...]:
    operation = item[0]
    if operation == "l":
        return (_norm_point(item[1], width, height), _norm_point(item[2], width, height))
    if operation == "c":
        return tuple(_norm_point(point, width, height) for point in item[1:5])
    if operation == "re":
        rect = item[1]
        raw = (
            fitz.Point(rect.x0, rect.y0),
            fitz.Point(rect.x1, rect.y0),
            fitz.Point(rect.x1, rect.y1),
            fitz.Point(rect.x0, rect.y1),
        )
        return tuple(_norm_point(point, width, height) for point in raw)
    if operation == "qu":
        quad = item[1]
        return tuple(
            _norm_point(point, width, height) for point in (quad.ul, quad.ur, quad.lr, quad.ll)
        )
    return ()


def _polyline_length(points: tuple[Point, ...], close: bool = False) -> float:
    if len(points) < 2:
        return 0.0
    pairs = list(zip(points, points[1:], strict=False))
    if close:
        pairs.append((points[-1], points[0]))
    return sum(math.dist(first, second) for first, second in pairs)


def _geometry_entity(
    page_index: int, source_index: int, drawing: dict[str, Any], page: fitz.Page
) -> Entity | None:
    if not _has_visible_paint(drawing):
        return None
    width = max(float(page.rect.width), 1.0)
    height = max(float(page.rect.height), 1.0)
    operations: list[str] = []
    encoded_items: list[str] = []
    all_points: list[Point] = []
    total_length = 0.0

    for raw_item in drawing.get("items", []):
        operation = str(raw_item[0])
        points = _points_for_item(raw_item, width, height)
        if not points:
            continue
        operations.append(operation)
        all_points.extend(points)
        total_length += _polyline_length(points, close=operation in {"re", "qu"})
        encoded_items.append(
            operation + ":" + ";".join(f"{x:.{_ROUND}f},{y:.{_ROUND}f}" for x, y in points)
        )

    if not all_points:
        return None

    bbox = _norm_bbox(drawing.get("rect", page.rect), width, height)
    cx = sum(point[0] for point in all_points) / len(all_points)
    cy = sum(point[1] for point in all_points) / len(all_points)
    relative_items: list[str] = []
    for encoded in encoded_items:
        operation, material = encoded.split(":", 1)
        relative_points = []
        for pair in material.split(";"):
            x_text, y_text = pair.split(",", 1)
            relative_points.append(
                f"{float(x_text) - cx:.{_ROUND}f},{float(y_text) - cy:.{_ROUND}f}"
            )
        relative_items.append(operation + ":" + ";".join(relative_points))

    style_material = (
        _color(drawing.get("color")),
        _color(drawing.get("fill")),
        round(float(drawing.get("width") or 0.0) / max(width, height), _ROUND),
        str(drawing.get("dashes") or ""),
        round(_opacity(drawing.get("stroke_opacity")), 4),
        round(_opacity(drawing.get("fill_opacity")), 4),
        bool(drawing.get("closePath")),
    )
    shape_signature = _digest(*relative_items)
    style_signature = _digest(*style_material)
    content_signature = _digest("geometry", shape_signature, style_signature)
    op_histogram = tuple(sorted(Counter(operations).items()))
    seed_id = _digest(
        page_index, source_index, content_signature, *(round(v, _ROUND) for v in bbox), length=16
    )
    return Entity(
        id=f"p{page_index + 1}-g-{seed_id}",
        page_index=page_index,
        kind=EntityKind.GEOMETRY,
        bbox=bbox,
        anchor=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        content_signature=content_signature,
        shape_signature=shape_signature,
        style_signature=style_signature,
        op_histogram=op_histogram,
        primitive_count=len(operations),
        path_length=round(total_length, _ROUND),
    )


def _text_entities(page: fitz.Page, page_index: int) -> Iterable[Entity]:
    width = max(float(page.rect.width), 1.0)
    height = max(float(page.rect.height), 1.0)
    source_index = 0
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = normalize_text(span.get("text") or "")
                if not text:
                    continue
                bbox = _norm_bbox(span["bbox"], width, height)
                origin = span.get("origin")
                if origin is not None:
                    anchor = (_clamp(float(origin[0]) / width), _clamp(float(origin[1]) / height))
                else:
                    anchor = (bbox[0], (bbox[1] + bbox[3]) / 2.0)
                font_name = str(span.get("font") or "")
                relative_size = float(span.get("size") or 0.0) / max(width, height)
                style_signature = _digest(
                    font_name,
                    int(span.get("flags") or 0),
                    _color(span.get("color")),
                    round(relative_size, _ROUND),
                )
                content_signature = _digest("text", text, style_signature)
                seed_id = _digest(
                    page_index,
                    source_index,
                    content_signature,
                    round(anchor[0], _ROUND),
                    round(anchor[1], _ROUND),
                    length=16,
                )
                yield Entity(
                    id=f"p{page_index + 1}-t-{seed_id}",
                    page_index=page_index,
                    kind=EntityKind.TEXT,
                    bbox=bbox,
                    anchor=anchor,
                    content_signature=content_signature,
                    shape_signature="text-span",
                    style_signature=style_signature,
                    text=text,
                    text_normalized=text.casefold(),
                    font_name=font_name,
                    font_size=round(relative_size, _ROUND),
                )
                source_index += 1


def _deduplicate_ids(entities: list[Entity]) -> list[Entity]:
    """Give collocated duplicate display-list objects stable ordinal suffixes."""

    ordered = sorted(entities, key=lambda item: (item.kind.value, item.bbox, item.id))
    seen: Counter[str] = Counter()
    output: list[Entity] = []
    for entity in ordered:
        ordinal = seen[entity.id]
        seen[entity.id] += 1
        output.append(entity if ordinal == 0 else replace(entity, id=f"{entity.id}-{ordinal}"))
    return output


def extract_page_entities(page: fitz.Page, page_index: int) -> tuple[Entity, ...]:
    """Return a canonical entity inventory; no page image is ever created."""

    entities = list(_text_entities(page, page_index))
    for source_index, drawing in enumerate(page.get_drawings()):
        entity = _geometry_entity(page_index, source_index, drawing, page)
        if entity is not None:
            entities.append(entity)
    return tuple(_deduplicate_ids(entities))
