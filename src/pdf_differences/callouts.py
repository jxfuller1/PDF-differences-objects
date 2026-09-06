"""Reconstruct high-confidence mechanical callouts from raw PDF entities.

PDF display lists usually expose annotation fragments rather than dimensions or
feature-control frames.  This module creates composite entities only when
drafting grammar, frame containment, or an unambiguous leader attachment proves
that the fragments belong together.  Distance is used to nominate possible
relationships; it is never sufficient to approve a group by itself.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterable

from pdf_differences.config import SETTINGS, ComparisonSettings
from pdf_differences.models import BBox, ChangeCategory, Entity, EntityKind, Point

_NUMBER = r"(?:\d+(?:[.,]\d*)?|[.,]\d+)"
_NUMBER_RE = re.compile(rf"^\(?{_NUMBER}(?:°|DEG|MM|CM|IN|FT|\"|')?\)?$", re.IGNORECASE)
_NUMBER_SEARCH = re.compile(_NUMBER)
_SIGNED_TOLERANCE_RE = re.compile(
    rf"^(?:±|\+/-|[+\-])\s*{_NUMBER}(?:\s*/\s*[+\-]?\s*{_NUMBER})?$",
    re.IGNORECASE,
)
_QUANTITY_RE = re.compile(r"^\d+\s*[X×]$", re.IGNORECASE)
_FEATURE_ONLY_RE = re.compile(r"^(?:Ø|⌀|DIA\.?|R|SR)$", re.IGNORECASE)
_FEATURE_SEARCH = re.compile(
    r"(?:Ø|⌀|\bDIA\.?|\bSR(?=\s*[.,]?\d)|\bR(?=\s*[.,]?\d))",
    re.IGNORECASE,
)
_QUALIFIER_RE = re.compile(
    r"^(?:BOTH\s+SIDES|THRU|THROUGH|TYP(?:ICAL)?|REF(?:ERENCE)?|"
    r"MAX(?:IMUM)?|MIN(?:IMUM)?|EQ\s+SP|PLACES?)$",
    re.IGNORECASE,
)
_QUALIFIER_SEARCH = re.compile(
    r"\b(?:BOTH\s+SIDES|THRU|THROUGH|TYP(?:ICAL)?|REF(?:ERENCE)?|"
    r"MAX(?:IMUM)?|MIN(?:IMUM)?|EQ\s+SP|PLACES?)\b",
    re.IGNORECASE,
)
_UNIT_RE = re.compile(r"^(?:MM|CM|IN|FT|DEG|°|\"|')$", re.IGNORECASE)
_NON_DIMENSION_CONTEXT = re.compile(
    r"\b(?:SECTION|SCALE|DRAWING|DWG|PAGE|PART|SHEET|ZONE|REV|REVISION|NOTE)\b",
    re.IGNORECASE,
)
_GDT_FEATURE_SYMBOLS = frozenset("⌖⏥⏊∥⌭⌒⌓○⌀")
_GDT_MODIFIER_SYMBOLS = frozenset("ⓂⓁⓈ")
_GDT_SYMBOLS = _GDT_FEATURE_SYMBOLS | _GDT_MODIFIER_SYMBOLS
_GDT_FEATURE_WORDS = re.compile(
    r"\b(?:ANGULARITY|CIRCULARITY|CONCENTRICITY|CYLINDRICITY|DATUM|FLATNESS|"
    r"PARALLELISM|PERPENDICULARITY|POSITION|PROFILE|RUNOUT|STRAIGHTNESS|"
    r"SYMMETRY|TOTAL\s+RUNOUT)\b",
    re.IGNORECASE,
)
_GDT_MODIFIER_WORDS = re.compile(r"\b(?:LMC|MMC|RFS)\b", re.IGNORECASE)
_GDT_DECIMAL = r"(?:\d+[.,]\d*|[.,]\d+)"
_GDT_DECIMAL_SEARCH = re.compile(_GDT_DECIMAL)
_GDT_TOLERANCE_CELL = re.compile(
    rf"^(?:Ø|⌀|DIA\.?)?\s*{_GDT_DECIMAL}"
    rf"(?:\s*(?:±|\+/-|[+\-])\s*{_NUMBER})?\s*[ⓂⓁⓈ]?$",
    re.IGNORECASE,
)
_GDT_DATUM_CELL = re.compile(r"^[A-Z](?:\s*[ⓂⓁⓈ])?$", re.IGNORECASE)


def _digest(*parts: object, length: int = 20) -> str:
    material = "|".join(str(part) for part in parts)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:length]


def _clean(text: str) -> str:
    cleaned = text.replace("−", "-").replace("⌀", "Ø")
    # Custom CAD fonts commonly map ± to an unavailable glyph. Only interpret
    # the replacement character as ± when it separates two numeric values;
    # elsewhere it may represent a degree, diameter, or GD&T symbol.
    cleaned = re.sub(r"(?<=\d)�(?=\s*[.,]?\d)", " ±", cleaned)
    return " ".join(cleaned.split())


def _bbox_union(entities: Iterable[Entity]) -> BBox:
    material = tuple(entities)
    return (
        min(entity.bbox[0] for entity in material),
        min(entity.bbox[1] for entity in material),
        max(entity.bbox[2] for entity in material),
        max(entity.bbox[3] for entity in material),
    )


def _bbox_contains(box: BBox, point: Point, tolerance: float = 0.0) -> bool:
    return (
        box[0] - tolerance <= point[0] <= box[2] + tolerance
        and box[1] - tolerance <= point[1] <= box[3] + tolerance
    )


def _bbox_contains_bbox(outer: BBox, inner: BBox, tolerance: float = 0.0) -> bool:
    return (
        outer[0] - tolerance <= inner[0]
        and outer[1] - tolerance <= inner[1]
        and outer[2] + tolerance >= inner[2]
        and outer[3] + tolerance >= inner[3]
    )


def _point_bbox_distance(point: Point, box: BBox) -> float:
    dx = max(box[0] - point[0], 0.0, point[0] - box[2])
    dy = max(box[1] - point[1], 0.0, point[1] - box[3])
    return math.hypot(dx, dy)


def _height(entity: Entity) -> float:
    return max(entity.height, entity.font_size, 1e-5)


def _same_direction(first: Entity, second: Entity) -> bool:
    dot = sum(
        left * right
        for left, right in zip(first.writing_direction, second.writing_direction, strict=True)
    )
    return dot >= 0.98


def _fragment_role(text: str) -> str:
    cleaned = _clean(text)
    if not cleaned or _NON_DIMENSION_CONTEXT.search(cleaned):
        return "other"
    if (
        any(symbol in cleaned for symbol in _GDT_SYMBOLS)
        or _GDT_FEATURE_WORDS.search(cleaned)
        or _GDT_MODIFIER_WORDS.search(cleaned)
    ):
        return "gdt"
    if _QUALIFIER_RE.fullmatch(cleaned):
        return "qualifier"
    if _QUANTITY_RE.fullmatch(cleaned):
        return "prefix"
    if _FEATURE_ONLY_RE.fullmatch(cleaned):
        return "feature"
    if _SIGNED_TOLERANCE_RE.fullmatch(cleaned):
        return "tolerance"
    if _NUMBER_RE.fullmatch(cleaned):
        return "nominal"
    if _UNIT_RE.fullmatch(cleaned):
        return "unit"
    if _dimension_profile(cleaned, has_attachment=False) is not None:
        return "core"
    return "other"


def _dimension_profile(
    text: str,
    *,
    has_attachment: bool,
    member_count: int = 1,
) -> str | None:
    cleaned = _clean(text)
    upper = cleaned.upper()
    if not cleaned or _NON_DIMENSION_CONTEXT.search(cleaned) or ":" in cleaned:
        return None
    if (
        any(symbol in cleaned for symbol in _GDT_SYMBOLS)
        or _GDT_FEATURE_WORDS.search(cleaned)
        or _GDT_MODIFIER_WORDS.search(cleaned)
    ):
        return None
    if not _NUMBER_SEARCH.search(cleaned):
        return None

    quantities = re.findall(r"\b\d+\s*[X×]\b", upper)
    features = _FEATURE_SEARCH.findall(cleaned)
    signed_tolerances = re.findall(rf"(?:±|\+/-|[+\-])\s*{_NUMBER}", cleaned)
    qualifiers = _QUALIFIER_SEARCH.findall(cleaned)
    units = re.findall(r"(?:\b(?:MM|CM|IN|FT|DEG)\b|°|\")", upper)

    material_without_roles = re.sub(r"\b\d+\s*[X×]\b", " ", upper)
    material_without_roles = re.sub(rf"(?:±|\+/-|[+\-])\s*{_NUMBER}", " ", material_without_roles)
    unsigned_numbers = _NUMBER_SEARCH.findall(material_without_roles)
    if len(features) > 1 or len(unsigned_numbers) > 1:
        return None
    if not unsigned_numbers:
        return None

    semantic_roles = bool(quantities or features or signed_tolerances or qualifiers)
    strong = semantic_roles or bool(units and member_count > 1) or has_attachment
    if not strong and not has_attachment:
        return None
    if "Ø" in cleaned or "⌀" in cleaned or re.search(r"\bDIA\.?", upper):
        family = "diameter"
    elif re.search(r"\bSR(?=\s*[.,]?\d)", upper):
        family = "spherical-radius"
    elif re.search(r"\bR(?=\s*[.,]?\d)", upper):
        family = "radius"
    elif "°" in cleaned or "DEG" in upper:
        family = "angular"
    else:
        family = "linear"
    return f"dimension:{family}"


_INLINE_TRANSITIONS = {
    "prefix": frozenset({"feature", "nominal", "core"}),
    "feature": frozenset({"nominal", "core"}),
    "nominal": frozenset({"tolerance", "qualifier", "unit"}),
    "core": frozenset({"tolerance", "qualifier", "unit"}),
    "tolerance": frozenset({"tolerance", "qualifier", "unit"}),
}


def _inline_dimension_edge(
    first: Entity,
    second: Entity,
    settings: ComparisonSettings,
) -> bool:
    if not _same_direction(first, second):
        return False
    left, right = sorted((first, second), key=lambda entity: (entity.bbox[0], entity.id))
    left_role = _fragment_role(left.text)
    right_role = _fragment_role(right.text)
    if right_role not in _INLINE_TRANSITIONS.get(left_role, ()):
        return False
    scale = max(_height(left), _height(right))
    baseline_delta = abs(left.anchor[1] - right.anchor[1])
    gap = max(0.0, right.bbox[0] - left.bbox[2])
    same_pdf_line = (
        left.source_block_index >= 0
        and left.source_block_index == right.source_block_index
        and left.source_line_index == right.source_line_index
    )
    baseline_limit = settings.callout_baseline_tolerance_factor * scale
    gap_limit = settings.callout_inline_gap_factor * scale * (1.5 if same_pdf_line else 1.0)
    return baseline_delta <= baseline_limit and gap <= gap_limit


def _stacked_dimension_edge(
    first: Entity,
    second: Entity,
    settings: ComparisonSettings,
) -> bool:
    first_role = _fragment_role(first.text)
    second_role = _fragment_role(second.text)
    roles = {first_role, second_role}
    if "tolerance" not in roles or not roles <= {"core", "nominal", "tolerance"}:
        return False
    scale = max(_height(first), _height(second))
    vertical_gap = max(0.0, max(first.bbox[1], second.bbox[1]) - min(first.bbox[3], second.bbox[3]))
    if vertical_gap > settings.callout_stacked_gap_factor * scale:
        return False
    horizontal_gap = max(
        0.0,
        max(first.bbox[0], second.bbox[0]) - min(first.bbox[2], second.bbox[2]),
    )
    if first_role == second_role == "tolerance":
        left_alignment = abs(first.bbox[0] - second.bbox[0])
        return left_alignment <= scale and horizontal_gap <= scale
    base = first if first_role in {"core", "nominal"} else second
    tolerance = second if base is first else first
    return (
        tolerance.bbox[0] >= base.bbox[0] - scale
        and tolerance.bbox[0] - base.bbox[2] <= settings.callout_inline_gap_factor * scale
    )


def _components(
    nodes: tuple[Entity, ...], edges: dict[str, set[str]]
) -> tuple[tuple[Entity, ...], ...]:
    by_id = {entity.id: entity for entity in nodes}
    unseen = set(by_id)
    output: list[tuple[Entity, ...]] = []
    while unseen:
        seed = min(unseen)
        queue = deque([seed])
        component: list[Entity] = []
        while queue:
            entity_id = queue.popleft()
            if entity_id not in unseen:
                continue
            unseen.remove(entity_id)
            component.append(by_id[entity_id])
            queue.extend(sorted(edges.get(entity_id, ())))
        output.append(tuple(component))
    return tuple(output)


def _reading_lines(entities: Iterable[Entity]) -> tuple[tuple[Entity, ...], ...]:
    ordered = sorted(entities, key=lambda entity: (entity.anchor[1], entity.bbox[0], entity.id))
    lines: list[list[Entity]] = []
    for entity in ordered:
        for line in lines:
            scale = max(_height(entity), *(_height(member) for member in line))
            baseline = sum(member.anchor[1] for member in line) / len(line)
            if abs(entity.anchor[1] - baseline) <= 0.5 * scale:
                line.append(entity)
                break
        else:
            lines.append([entity])
    return tuple(
        tuple(sorted(line, key=lambda entity: (entity.bbox[0], entity.id))) for line in lines
    )


def _dimension_text(entities: Iterable[Entity]) -> str:
    return " / ".join(
        " ".join(_clean(entity.text) for entity in line) for line in _reading_lines(entities)
    )


def _attachment_points(
    bbox: BBox,
    text_height: float,
    geometry: Iterable[Entity],
    settings: ComparisonSettings,
    competing_bboxes: Iterable[BBox] = (),
) -> tuple[Point, ...]:
    tolerance = max(0.0015, text_height * settings.callout_leader_touch_factor)
    ambiguity_margin = max(
        0.00075,
        text_height * settings.callout_attachment_ambiguity_factor,
    )
    competitors = tuple(competing_bboxes)
    candidates: list[tuple[float, Point]] = []
    for entity in geometry:
        # Cell frames and other closed rectangular paths can sit close to a
        # dimension without being its leader. Treating their corners as
        # attachment points makes repeated callouts easier to cross-match.
        if _is_rectangle(entity):
            continue
        for first, second in entity.geometry_segments:
            first_distance = _point_bbox_distance(first, bbox)
            second_distance = _point_bbox_distance(second, bbox)
            if first_distance <= tolerance < second_distance:
                near_distance, near_point, far_point = first_distance, first, second
            elif second_distance <= tolerance < first_distance:
                near_distance, near_point, far_point = second_distance, second, first
            else:
                continue
            if any(
                _point_bbox_distance(near_point, other) <= near_distance + ambiguity_margin
                for other in competitors
            ):
                continue
            candidates.append((near_distance, far_point))
    candidates.sort(key=lambda item: (item[0], round(item[1][0], 8), round(item[1][1], 8)))
    output: list[Point] = []
    for _, point in candidates:
        if all(math.dist(point, existing) > tolerance for existing in output):
            output.append(point)
        if len(output) == 4:
            break
    return tuple(output)


def _is_rectangle(entity: Entity) -> bool:
    histogram = dict(entity.op_histogram)
    if entity.width <= 1e-6 or entity.height <= 1e-6:
        return False
    if histogram.get("re", 0) or histogram.get("qu", 0):
        return True
    if len(entity.geometry_segments) < 4:
        return False
    tolerance = max(1e-5, min(entity.width, entity.height) * 0.04)
    for first, second in entity.geometry_segments:
        horizontal = abs(first[1] - second[1]) <= tolerance
        vertical = abs(first[0] - second[0]) <= tolerance
        if not horizontal and not vertical:
            return False
    corners = (
        (entity.bbox[0], entity.bbox[1]),
        (entity.bbox[2], entity.bbox[1]),
        (entity.bbox[2], entity.bbox[3]),
        (entity.bbox[0], entity.bbox[3]),
    )
    endpoints = tuple(point for segment in entity.geometry_segments for point in segment)
    if not all(
        any(math.dist(corner, point) <= tolerance for point in endpoints) for corner in corners
    ):
        return False
    return all(
        (
            abs(first[0] - entity.bbox[0]) <= tolerance
            and abs(second[0] - entity.bbox[0]) <= tolerance
        )
        or (
            abs(first[0] - entity.bbox[2]) <= tolerance
            and abs(second[0] - entity.bbox[2]) <= tolerance
        )
        or (
            abs(first[1] - entity.bbox[1]) <= tolerance
            and abs(second[1] - entity.bbox[1]) <= tolerance
        )
        or (
            abs(first[1] - entity.bbox[3]) <= tolerance
            and abs(second[1] - entity.bbox[3]) <= tolerance
        )
        for first, second in entity.geometry_segments
    )


def _has_internal_frame_divider(entity: Entity, tolerance: float) -> bool:
    """Return whether one compound path visibly contains adjoining cells."""

    minimum_length = entity.height * 0.75
    for first, second in entity.geometry_segments:
        vertical = abs(first[0] - second[0]) <= tolerance
        interior = entity.bbox[0] + tolerance < first[0] < entity.bbox[2] - tolerance
        if vertical and interior and abs(first[1] - second[1]) >= minimum_length:
            return True
    return False


def _frame_neighbors(first: Entity, second: Entity, tolerance: float) -> bool:
    vertical_overlap = max(
        0.0,
        min(first.bbox[3], second.bbox[3]) - max(first.bbox[1], second.bbox[1]),
    )
    height = min(first.height, second.height)
    same_row = height > 0.0 and vertical_overlap / height >= 0.8
    horizontal_gap = min(abs(first.bbox[2] - second.bbox[0]), abs(second.bbox[2] - first.bbox[0]))
    return same_row and horizontal_gap <= tolerance


def _has_gdt_marker(text: str) -> bool:
    return any(symbol in text for symbol in _GDT_FEATURE_SYMBOLS) or bool(
        _GDT_FEATURE_WORDS.search(text)
    )


def _valid_gdt_text(text: str) -> bool:
    return _has_gdt_marker(text) and bool(_GDT_DECIMAL_SEARCH.search(text))


def _looks_like_gdt_sequence(parts: Iterable[str]) -> bool:
    """Recognize a framed FCF whose custom-font symbol did not extract as text."""

    cleaned = tuple(_clean(part) for part in parts if _clean(part))
    for tolerance_index, part in enumerate(cleaned):
        if not _GDT_TOLERANCE_CELL.fullmatch(part):
            continue
        # A feature-control tolerance is followed by multiple datum or
        # material-condition cells. Requiring that topology avoids treating an
        # arbitrary boxed decimal as GD&T.
        suffix = cleaned[tolerance_index + 1 :]
        if sum(bool(_GDT_DATUM_CELL.fullmatch(value)) for value in suffix) >= 2:
            return True
    return False


def _cell_text(
    cell: Entity,
    contained: Iterable[Entity],
    tolerance: float,
) -> str:
    return " ".join(
        _clean(entity.text)
        for entity in sorted(contained, key=lambda item: (item.bbox[0], item.id))
        if _bbox_contains(cell.bbox, entity.centroid, tolerance)
    )


def _gdt_starts(
    cells: tuple[Entity, ...],
    contained: tuple[Entity, ...],
    tolerance: float,
) -> tuple[int, ...]:
    known = tuple(
        index
        for index, cell in enumerate(cells)
        if _has_gdt_marker(_cell_text(cell, contained, tolerance))
    )
    if known:
        return known

    cell_parts = tuple(_cell_text(cell, contained, tolerance) for cell in cells)
    inferred: list[int] = []
    for tolerance_index, part in enumerate(cell_parts):
        if not _GDT_TOLERANCE_CELL.fullmatch(part):
            continue
        suffix = tuple(value for value in cell_parts[tolerance_index + 1 :] if value)
        if sum(bool(_GDT_DATUM_CELL.fullmatch(value)) for value in suffix) >= 2:
            inferred.append(max(0, tolerance_index - 1))
    if inferred:
        return tuple(dict.fromkeys(inferred))

    # One PDF path may contain every cell, leaving only one aggregate bbox.
    # Multiple rectangle/line primitives plus tolerance→datum text order still
    # provide frame topology even when the symbol is a vector outline.
    if len(cells) == 1 and _has_internal_frame_divider(cells[0], tolerance):
        ordered_text = tuple(
            entity.text for entity in sorted(contained, key=lambda item: (item.bbox[0], item.id))
        )
        if _looks_like_gdt_sequence(ordered_text):
            return (0,)
    return ()


def _combined_histogram(entities: Iterable[Entity]) -> tuple[tuple[str, int], ...]:
    counts: Counter[str] = Counter()
    for entity in entities:
        counts.update(dict(entity.op_histogram))
    return tuple(sorted(counts.items()))


def _composite_callout(
    text_members: Iterable[Entity],
    geometry_members: Iterable[Entity],
    category: ChangeCategory,
    structure: str,
    text: str,
    attachments: tuple[Point, ...],
) -> Entity:
    text_items = tuple(sorted(text_members, key=lambda entity: (entity.bbox, entity.id)))
    geometry_items = tuple(sorted(geometry_members, key=lambda entity: (entity.bbox, entity.id)))
    members = (*text_items, *geometry_items)
    bbox = _bbox_union(members)
    width = max(bbox[2] - bbox[0], 1e-9)
    height = max(bbox[3] - bbox[1], 1e-9)
    layout_material = tuple(
        (
            entity.kind.value,
            _fragment_role(entity.text)
            if entity.kind == EntityKind.TEXT
            else entity.shape_signature,
            round((entity.bbox[0] - bbox[0]) / width, 5),
            round((entity.bbox[1] - bbox[1]) / height, 5),
            round((entity.bbox[2] - bbox[0]) / width, 5),
            round((entity.bbox[3] - bbox[1]) / height, 5),
        )
        for entity in members
    )
    style_signature = _digest(*(entity.style_signature for entity in members))
    shape_signature = _digest(structure, *layout_material)
    content_signature = _digest(
        "callout",
        category.value,
        _clean(text).casefold(),
        shape_signature,
        style_signature,
    )
    page_index = members[0].page_index
    seed = _digest(
        page_index,
        category.value,
        content_signature,
        *(round(value, 6) for value in bbox),
        length=16,
    )
    font_sizes = sorted(entity.font_size for entity in text_items if entity.font_size > 0.0)
    font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 0.0
    path_length = sum(entity.path_length for entity in geometry_items)
    primitive_count = sum(entity.primitive_count for entity in geometry_items)
    return Entity(
        id=f"p{page_index + 1}-c-{seed}",
        page_index=page_index,
        kind=EntityKind.TEXT,
        bbox=bbox,
        anchor=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        content_signature=content_signature,
        shape_signature=shape_signature,
        style_signature=style_signature,
        text=_clean(text),
        text_normalized=_clean(text).casefold(),
        font_name="callout",
        font_size=font_size,
        op_histogram=_combined_histogram(geometry_items),
        primitive_count=primitive_count,
        path_length=round(path_length, 6),
        geometry_segments=tuple(
            segment for entity in geometry_items for segment in entity.geometry_segments
        ),
        callout_category=category,
        callout_structure=structure,
        callout_member_ids=tuple(entity.id for entity in members),
        callout_attachment_points=attachments,
    )


def _gdt_groups(
    text_entities: tuple[Entity, ...],
    geometry_entities: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[str, ...]], ...]:
    rectangles = tuple(
        entity
        for entity in geometry_entities
        if _is_rectangle(entity)
        and entity.height <= settings.callout_max_frame_height
        and entity.width <= settings.callout_max_frame_width
    )
    framed_text_ids = frozenset(
        text.id
        for text in text_entities
        if any(
            _bbox_contains(frame.bbox, text.centroid, settings.callout_frame_edge_tolerance)
            for frame in rectangles
        )
    )
    edges: dict[str, set[str]] = defaultdict(set)
    for index, first in enumerate(rectangles):
        for second in rectangles[index + 1 :]:
            if _frame_neighbors(first, second, settings.callout_frame_edge_tolerance):
                edges[first.id].add(second.id)
                edges[second.id].add(first.id)

    output: list[tuple[Entity, tuple[str, ...]]] = []
    used_text: set[str] = set()
    used_geometry: set[str] = set()
    for frames in _components(rectangles, edges):
        frame_bbox = _bbox_union(frames)
        tolerance = settings.callout_frame_edge_tolerance
        contained = tuple(
            entity
            for entity in text_entities
            if entity.id not in used_text and _bbox_contains(frame_bbox, entity.centroid, tolerance)
        )
        if not contained:
            continue
        cells = sorted(frames, key=lambda entity: (entity.bbox[0], entity.bbox[1], entity.id))
        starts = list(_gdt_starts(tuple(cells), contained, tolerance))
        if not starts:
            combined = " | ".join(
                entity.text for entity in sorted(contained, key=lambda item: item.bbox)
            )
            if len(frames) == 1 and _valid_gdt_text(combined):
                starts = [0]
            else:
                continue
        for position, start in enumerate(starts):
            stop = starts[position + 1] if position + 1 < len(starts) else len(cells)
            group_frames = tuple(cells[start:stop])
            group_bbox = _bbox_union(group_frames)
            if group_bbox[2] - group_bbox[0] > settings.callout_max_frame_width:
                continue
            frame_topology = len(group_frames) >= 2 or any(
                _has_internal_frame_divider(frame, tolerance) for frame in group_frames
            )
            if not frame_topology:
                continue
            group_text = tuple(
                entity
                for entity in contained
                if entity.id not in used_text
                and _bbox_contains(group_bbox, entity.centroid, tolerance)
            )
            ordered_cells: list[str] = []
            for cell in group_frames:
                cell_text = _cell_text(cell, group_text, tolerance)
                if cell_text:
                    ordered_cells.append(cell_text)
            combined = " | ".join(ordered_cells)
            # Some CAD exporters emit the entire feature-control frame as one
            # compound path instead of one path per cell. Strong GD&T grammar
            # plus containment is sufficient in that representation.
            if not (
                _valid_gdt_text(combined)
                or _looks_like_gdt_sequence(ordered_cells)
                or (
                    len(group_frames) == 1
                    and _has_internal_frame_divider(group_frames[0], tolerance)
                    and _looks_like_gdt_sequence(
                        entity.text
                        for entity in sorted(
                            group_text,
                            key=lambda item: (item.bbox[0], item.id),
                        )
                    )
                )
            ):
                continue
            enclosed_geometry = tuple(
                entity
                for entity in geometry_entities
                if entity.id not in used_geometry
                and entity not in group_frames
                and _bbox_contains_bbox(group_bbox, entity.bbox, tolerance)
            )
            group_geometry = (*group_frames, *enclosed_geometry)
            attachments = _attachment_points(
                group_bbox,
                max((_height(entity) for entity in group_text), default=0.01),
                (entity for entity in geometry_entities if entity not in group_geometry),
                settings,
            )
            composite = _composite_callout(
                group_text,
                group_geometry,
                ChangeCategory.GDT,
                f"gdt:{len(group_frames)}",
                combined,
                attachments,
            )
            member_ids = tuple(entity.id for entity in (*group_text, *group_geometry))
            output.append((composite, member_ids))
            used_text.update(entity.id for entity in group_text)
            used_geometry.update(entity.id for entity in group_geometry)

    # Frameless but explicit feature-control text remains safe to reconstruct.
    ordered_text = tuple(sorted(text_entities, key=lambda entity: (entity.bbox[0], entity.id)))
    for marker in sorted(text_entities, key=lambda entity: (entity.bbox, entity.id)):
        if (
            marker.id in used_text
            or marker.id in framed_text_ids
            or not _has_gdt_marker(marker.text)
        ):
            continue
        row = [marker]
        current = marker
        has_tolerance = bool(_GDT_DECIMAL_SEARCH.search(marker.text))
        for candidate in ordered_text:
            if (
                candidate.id in used_text
                or candidate.id in framed_text_ids
                or candidate.id == marker.id
            ):
                continue
            scale = max(_height(current), _height(candidate))
            gap = candidate.bbox[0] - current.bbox[2]
            if candidate.bbox[0] < current.bbox[0]:
                continue
            if (
                abs(candidate.anchor[1] - marker.anchor[1])
                > settings.callout_baseline_tolerance_factor * scale
                or gap < 0.0
                or gap > settings.callout_inline_gap_factor * scale
            ):
                continue
            if _has_gdt_marker(candidate.text):
                break
            cleaned = _clean(candidate.text)
            if not has_tolerance:
                if not _GDT_TOLERANCE_CELL.fullmatch(cleaned):
                    break
                has_tolerance = True
            elif not (_GDT_DATUM_CELL.fullmatch(cleaned) or cleaned in _GDT_MODIFIER_SYMBOLS):
                break
            row.append(candidate)
            current = candidate
        text = " | ".join(_clean(entity.text) for entity in row)
        if not _valid_gdt_text(text):
            continue
        bbox = _bbox_union(row)
        attachments = _attachment_points(
            bbox,
            max(_height(entity) for entity in row),
            geometry_entities,
            settings,
        )
        composite = _composite_callout(
            row,
            (),
            ChangeCategory.GDT,
            "gdt:text",
            text,
            attachments,
        )
        member_ids = tuple(entity.id for entity in row)
        output.append((composite, member_ids))
        used_text.update(member_ids)

    return tuple(output)


def _dimension_groups(
    text_entities: tuple[Entity, ...],
    geometry_entities: tuple[Entity, ...],
    excluded_ids: frozenset[str],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[str, ...]], ...]:
    candidates = tuple(
        entity
        for entity in text_entities
        if entity.id not in excluded_ids and _fragment_role(entity.text) not in {"other", "gdt"}
    )
    edges: dict[str, set[str]] = defaultdict(set)
    for index, first in enumerate(candidates):
        for second in candidates[index + 1 :]:
            if _inline_dimension_edge(first, second, settings) or _stacked_dimension_edge(
                first, second, settings
            ):
                edges[first.id].add(second.id)
                edges[second.id].add(first.id)

    components = _components(candidates, edges)
    component_boxes = tuple(_bbox_union(component) for component in components)
    output: list[tuple[Entity, tuple[str, ...]]] = []
    for index, component in enumerate(components):
        bbox = _bbox_union(component)
        height = max(_height(entity) for entity in component)
        attachments = _attachment_points(
            bbox,
            height,
            geometry_entities,
            settings,
            (
                other_bbox
                for other_index, other_bbox in enumerate(component_boxes)
                if other_index != index
            ),
        )
        text = _dimension_text(component)
        structure = _dimension_profile(
            text,
            has_attachment=bool(attachments),
            member_count=len(component),
        )
        if structure is None:
            continue
        enclosed_symbols = tuple(
            entity
            for entity in geometry_entities
            if entity.id not in excluded_ids
            and not _is_rectangle(entity)
            and entity.height <= height * 1.75
            and entity.width <= height * 1.75
            and _bbox_contains_bbox(bbox, entity.bbox, max(0.0005, height * 0.15))
            and not any(
                other_index != index
                and _bbox_contains_bbox(
                    other_bbox,
                    entity.bbox,
                    max(0.0005, height * 0.15),
                )
                for other_index, other_bbox in enumerate(component_boxes)
            )
        )
        composite = _composite_callout(
            component,
            enclosed_symbols,
            ChangeCategory.DIMENSION,
            structure,
            text,
            attachments,
        )
        output.append(
            (
                composite,
                tuple(entity.id for entity in (*component, *enclosed_symbols)),
            )
        )
    return tuple(output)


def reconstruct_callouts(
    entities: Iterable[Entity],
    settings: ComparisonSettings = SETTINGS,
) -> tuple[Entity, ...]:
    """Replace confidently recognized callout members with composite entities.

    Unrecognized or ambiguous material remains byte-for-byte equivalent at the
    entity level, which makes conservative under-grouping safer than a false
    semantic merge.
    """

    raw = tuple(entities)
    text_entities = tuple(entity for entity in raw if entity.kind == EntityKind.TEXT)
    geometry_entities = tuple(entity for entity in raw if entity.kind == EntityKind.GEOMETRY)

    grouped: list[Entity] = []
    consumed: set[str] = set()
    for composite, member_ids in _gdt_groups(text_entities, geometry_entities, settings):
        if consumed.isdisjoint(member_ids):
            grouped.append(composite)
            consumed.update(member_ids)
    for composite, member_ids in _dimension_groups(
        text_entities,
        geometry_entities,
        frozenset(consumed),
        settings,
    ):
        if consumed.isdisjoint(member_ids):
            grouped.append(composite)
            consumed.update(member_ids)

    output = [entity for entity in raw if entity.id not in consumed]
    output.extend(grouped)
    return tuple(sorted(output, key=lambda entity: (entity.kind.value, entity.bbox, entity.id)))
