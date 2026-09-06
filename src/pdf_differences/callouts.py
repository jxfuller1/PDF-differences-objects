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
from dataclasses import dataclass

from pdf_differences.config import SETTINGS, ComparisonSettings
from pdf_differences.models import BBox, ChangeCategory, Entity, EntityKind, Point

_NUMBER = r"(?:\d+(?:[.,]\d*)?|[.,]\d+)"
_NUMBER_RE = re.compile(rf"^\(?{_NUMBER}(?:°|DEG|MM|CM|IN|FT|\"|')?\)?$", re.IGNORECASE)
_NUMBER_SEARCH = re.compile(_NUMBER)
_SIGNED_TOLERANCE_RE = re.compile(
    rf"^(?:±|\+/-|[+\-])\s*{_NUMBER}(?:\s*/\s*[+\-]?\s*{_NUMBER})?$",
    re.IGNORECASE,
)
_SIGNED_TOLERANCE_SEARCH = re.compile(rf"(?:±|\+/-|[+\-])\s*{_NUMBER}", re.IGNORECASE)
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


def _leaf_member_ids(entity: Entity) -> tuple[str, ...]:
    """Return original inventory IDs represented by a temporary fragment."""

    return entity.callout_member_ids or (entity.id,)


def _flatten_member_ids(entities: Iterable[Entity]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(member_id for entity in entities for member_id in _leaf_member_ids(entity))
    )


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


def _tolerance_polarity(entity: Entity) -> str:
    cleaned = _clean(entity.text)
    if "/" in cleaned or cleaned.startswith("±"):
        return ""
    if cleaned.startswith("+") and not cleaned.startswith("+/-"):
        return "+"
    if cleaned.startswith("-"):
        return "-"
    return ""


def _tolerance_pair_score(
    first: Entity,
    second: Entity,
    settings: ComparisonSettings,
) -> float | None:
    """Score an upper/lower or inline plus/minus tolerance pair.

    A spatially close tolerance is not enough: the pair must have opposite
    signs and either share a baseline in reading order or form a left-aligned
    stack. This keeps neighboring dimensions from becoming a transitive
    tolerance component.
    """

    if {_tolerance_polarity(first), _tolerance_polarity(second)} != {"+", "-"}:
        return None
    if not _same_direction(first, second):
        return None
    scale = max(_height(first), _height(second))
    baseline_delta = abs(first.anchor[1] - second.anchor[1])
    horizontal_gap = max(
        0.0,
        max(first.bbox[0], second.bbox[0]) - min(first.bbox[2], second.bbox[2]),
    )
    scores: list[float] = []
    if (
        baseline_delta <= settings.callout_baseline_tolerance_factor * scale
        and horizontal_gap <= settings.callout_inline_gap_factor * scale
    ):
        scores.append((baseline_delta + horizontal_gap) / scale)

    vertical_gap = max(
        0.0,
        max(first.bbox[1], second.bbox[1]) - min(first.bbox[3], second.bbox[3]),
    )
    left_alignment = abs(first.bbox[0] - second.bbox[0])
    if (
        left_alignment <= settings.callout_tolerance_pair_alignment_factor * scale
        and vertical_gap <= settings.callout_stacked_gap_factor * scale
    ):
        scores.append((left_alignment + vertical_gap) / scale)
    return min(scores) if scores else None


def _clear_best(
    candidates: Iterable[tuple[float, str]],
    *,
    ambiguity_margin: float,
) -> str | None:
    ordered = sorted(candidates)
    if not ordered:
        return None
    if len(ordered) > 1 and ordered[1][0] - ordered[0][0] <= ambiguity_margin:
        return None
    return ordered[0][1]


def _tolerance_units(
    tolerance_entities: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, ...], ...]:
    """Build non-transitive, mutually owned tolerance stacks."""

    pair_scores: dict[tuple[str, str], float] = {}
    by_id = {entity.id: entity for entity in tolerance_entities}
    for index, first in enumerate(tolerance_entities):
        for second in tolerance_entities[index + 1 :]:
            score = _tolerance_pair_score(first, second, settings)
            if score is not None:
                pair_scores[(first.id, second.id)] = score

    choices: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for (first_id, second_id), score in pair_scores.items():
        choices[first_id].append((score, second_id))
        choices[second_id].append((score, first_id))
    best = {
        entity_id: _clear_best(
            values,
            ambiguity_margin=settings.callout_hypothesis_ambiguity_margin,
        )
        for entity_id, values in choices.items()
    }

    used: set[str] = set()
    units: list[tuple[Entity, ...]] = []
    for entity in sorted(tolerance_entities, key=lambda item: (item.bbox, item.id)):
        if entity.id in used:
            continue
        partner_id = best.get(entity.id)
        if partner_id is not None and best.get(partner_id) == entity.id:
            members = tuple(
                sorted((entity, by_id[partner_id]), key=lambda item: (item.bbox, item.id))
            )
            units.append(members)
            used.update((entity.id, partner_id))
            continue
        units.append((entity,))
        used.add(entity.id)
    return tuple(units)


def _tolerance_unit_score(
    root: Entity,
    unit: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> float | None:
    unit_bbox = _bbox_union(unit)
    scale = max(_height(root), *(_height(entity) for entity in unit))
    vertical_gap = max(
        0.0,
        max(root.bbox[1], unit_bbox[1]) - min(root.bbox[3], unit_bbox[3]),
    )
    if vertical_gap > settings.callout_stacked_gap_factor * scale:
        return None
    if unit_bbox[0] < root.bbox[0] - 0.5 * scale:
        return None
    if unit_bbox[0] - root.bbox[2] > settings.callout_inline_gap_factor * scale:
        return None
    horizontal_alignment = abs(unit_bbox[0] - root.bbox[2]) / scale
    root_center_y = 0.5 * (root.bbox[1] + root.bbox[3])
    unit_center_y = 0.5 * (unit_bbox[1] + unit_bbox[3])
    vertical_alignment = abs(unit_center_y - root_center_y) / scale
    return math.hypot(horizontal_alignment, vertical_alignment)


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


def _merged_text_fragment(parts: Iterable[Entity], text: str) -> Entity:
    members = tuple(sorted(parts, key=lambda entity: (entity.bbox[0], entity.id)))
    bbox = _bbox_union(members)
    page_index = members[0].page_index
    member_ids = _flatten_member_ids(members)
    style_signature = _digest(*(entity.style_signature for entity in members))
    content_signature = _digest("text-fragment", _clean(text).casefold(), style_signature)
    seed = _digest(page_index, *member_ids, content_signature, length=16)
    font_sizes = sorted(entity.font_size for entity in members if entity.font_size > 0.0)
    font_size = font_sizes[len(font_sizes) // 2] if font_sizes else 0.0
    block_indices = {entity.source_block_index for entity in members}
    line_indices = {entity.source_line_index for entity in members}
    return Entity(
        id=f"p{page_index + 1}-f-{seed}",
        page_index=page_index,
        kind=EntityKind.TEXT,
        bbox=bbox,
        anchor=(bbox[0], sum(entity.anchor[1] for entity in members) / len(members)),
        content_signature=content_signature,
        shape_signature="text-fragment",
        style_signature=style_signature,
        text=_clean(text),
        text_normalized=_clean(text).casefold(),
        font_name="fragment",
        font_size=font_size,
        source_block_index=block_indices.pop() if len(block_indices) == 1 else -1,
        source_line_index=line_indices.pop() if len(line_indices) == 1 else -1,
        writing_direction=members[0].writing_direction,
        callout_member_ids=member_ids,
    )


def _signed_tolerance_fragments(
    text_entities: Iterable[Entity],
    settings: ComparisonSettings,
) -> tuple[Entity, ...]:
    """Join CAD-exported standalone signs to their numeric tolerance spans."""

    material = tuple(text_entities)
    signs = tuple(entity for entity in material if _clean(entity.text) in {"+", "-"})
    numbers = tuple(entity for entity in material if _NUMBER_RE.fullmatch(_clean(entity.text)))
    used: set[str] = set()
    fragments: list[Entity] = []
    for sign in sorted(signs, key=lambda entity: (entity.bbox, entity.id)):
        candidates: list[tuple[float, Entity]] = []
        for number in numbers:
            if number.id in used or not _same_direction(sign, number):
                continue
            scale = max(_height(sign), _height(number))
            if not (0.5 <= _height(sign) / _height(number) <= 2.0):
                continue
            if number.bbox[0] < sign.bbox[0]:
                continue
            gap = max(0.0, number.bbox[0] - sign.bbox[2])
            baseline_delta = abs(sign.anchor[1] - number.anchor[1])
            if (
                gap > 0.75 * settings.callout_inline_gap_factor * scale
                or baseline_delta > settings.callout_baseline_tolerance_factor * scale
            ):
                continue
            candidates.append((gap + baseline_delta, number))
        candidates.sort(key=lambda item: (item[0], item[1].bbox, item[1].id))
        if not candidates:
            continue
        ambiguity = max(0.0005, 0.15 * max(_height(sign), _height(candidates[0][1])))
        if len(candidates) > 1 and candidates[1][0] - candidates[0][0] <= ambiguity:
            continue
        number = candidates[0][1]
        used.update((sign.id, number.id))
        fragments.append(
            _merged_text_fragment((sign, number), f"{_clean(sign.text)}{_clean(number.text)}")
        )

    output = [entity for entity in material if entity.id not in used]
    output.extend(fragments)
    return tuple(sorted(output, key=lambda entity: (entity.bbox, entity.id)))


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


def _segment_bbox(segment: tuple[Point, Point]) -> BBox:
    first, second = segment
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[0], second[0]),
        max(first[1], second[1]),
    )


def _point_segment_distance(point: Point, segment: tuple[Point, Point]) -> float:
    first, second = segment
    dx = second[0] - first[0]
    dy = second[1] - first[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-20:
        return math.dist(point, first)
    projection = ((point[0] - first[0]) * dx + (point[1] - first[1]) * dy) / denominator
    projection = min(1.0, max(0.0, projection))
    nearest = (first[0] + projection * dx, first[1] + projection * dy)
    return math.dist(point, nearest)


def _segments_touch(
    first: tuple[Point, Point],
    second: tuple[Point, Point],
    tolerance: float,
) -> bool:
    first_bbox = _segment_bbox(first)
    second_bbox = _segment_bbox(second)
    if (
        first_bbox[2] + tolerance < second_bbox[0]
        or second_bbox[2] + tolerance < first_bbox[0]
        or first_bbox[3] + tolerance < second_bbox[1]
        or second_bbox[3] + tolerance < first_bbox[1]
    ):
        return False

    def cross(origin: Point, left: Point, right: Point) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    first_a, first_b = first
    second_a, second_b = second
    first_side_a = cross(first_a, first_b, second_a)
    first_side_b = cross(first_a, first_b, second_b)
    second_side_a = cross(second_a, second_b, first_a)
    second_side_b = cross(second_a, second_b, first_b)
    if first_side_a * first_side_b <= 0.0 and second_side_a * second_side_b <= 0.0:
        return True
    return (
        min(
            _point_segment_distance(first_a, second),
            _point_segment_distance(first_b, second),
            _point_segment_distance(second_a, first),
            _point_segment_distance(second_b, first),
        )
        <= tolerance
    )


def _segment_components(
    segments: tuple[tuple[Point, Point], ...],
    tolerance: float,
) -> tuple[tuple[int, ...], ...]:
    if len(segments) < 2:
        return (tuple(range(len(segments))),)
    cell_size = max(0.01, tolerance * 8.0)
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    edges: dict[int, set[int]] = defaultdict(set)
    for index, segment in enumerate(segments):
        x0, y0, x1, y1 = _segment_bbox(segment)
        x_start = math.floor((x0 - tolerance) / cell_size)
        x_stop = math.floor((x1 + tolerance) / cell_size)
        y_start = math.floor((y0 - tolerance) / cell_size)
        y_stop = math.floor((y1 + tolerance) / cell_size)
        keys = tuple((x, y) for x in range(x_start, x_stop + 1) for y in range(y_start, y_stop + 1))
        possible = {other_index for key in keys for other_index in buckets.get(key, ())}
        for other_index in possible:
            if _segments_touch(segment, segments[other_index], tolerance):
                edges[index].add(other_index)
                edges[other_index].add(index)
        for key in keys:
            buckets[key].append(index)

    unseen = set(range(len(segments)))
    output: list[tuple[int, ...]] = []
    while unseen:
        seed = min(unseen)
        queue = deque((seed,))
        component: list[int] = []
        while queue:
            index = queue.popleft()
            if index not in unseen:
                continue
            unseen.remove(index)
            component.append(index)
            queue.extend(sorted(edges.get(index, ())))
        output.append(tuple(sorted(component)))
    return tuple(output)


def _geometry_component_entity(
    parent: Entity,
    component_indices: tuple[int, ...],
) -> Entity:
    segments = tuple(parent.geometry_segments[index] for index in component_indices)
    points = tuple(point for segment in segments for point in segment)
    bbox = (
        min(point[0] for point in points),
        min(point[1] for point in points),
        max(point[0] for point in points),
        max(point[1] for point in points),
    )
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    relative_segments = tuple(
        sorted(
            (
                round(first[0] - cx, 6),
                round(first[1] - cy, 6),
                round(second[0] - cx, 6),
                round(second[1] - cy, 6),
            )
            for first, second in segments
        )
    )
    shape_signature = _digest("geometry-component", *relative_segments)
    content_signature = _digest("geometry", shape_signature, parent.style_signature)
    seed = _digest(parent.id, *component_indices, content_signature, length=12)
    return Entity(
        id=f"{parent.id}-c-{seed}",
        page_index=parent.page_index,
        kind=EntityKind.GEOMETRY,
        bbox=bbox,
        anchor=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
        content_signature=content_signature,
        shape_signature=shape_signature,
        style_signature=parent.style_signature,
        op_histogram=(("l", len(segments)),),
        primitive_count=len(segments),
        path_length=round(sum(math.dist(*segment) for segment in segments), 6),
        geometry_segments=segments,
    )


def _geometry_residual_entity(parent: Entity, children: Iterable[Entity]) -> Entity:
    """Recombine unclaimed local components into one deterministic residual."""

    remaining = Counter(segment for child in children for segment in child.geometry_segments)
    component_indices: list[int] = []
    for index, segment in enumerate(parent.geometry_segments):
        if remaining[segment] <= 0:
            continue
        component_indices.append(index)
        remaining[segment] -= 1
    return _geometry_component_entity(parent, tuple(component_indices))


def _local_geometry_entities(
    entities: Iterable[Entity],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, ...], dict[str, tuple[Entity, ...]]]:
    """Split disconnected straight-line batches without changing local paths."""

    output: list[Entity] = []
    children_by_parent: dict[str, tuple[Entity, ...]] = {}
    for entity in entities:
        operations = set(dict(entity.op_histogram))
        if (
            len(entity.geometry_segments) < 2
            or not operations
            or not operations <= {"l", "re", "qu"}
        ):
            output.append(entity)
            children_by_parent[entity.id] = (entity,)
            continue
        components = _segment_components(
            entity.geometry_segments,
            settings.callout_segment_connect_tolerance,
        )
        if len(components) == 1:
            output.append(entity)
            children_by_parent[entity.id] = (entity,)
            continue
        children = tuple(_geometry_component_entity(entity, component) for component in components)
        output.extend(children)
        children_by_parent[entity.id] = children
    return (
        tuple(sorted(output, key=lambda entity: (entity.bbox, entity.id))),
        children_by_parent,
    )


@dataclass(frozen=True, slots=True)
class _LineCoverage:
    fixed: float
    lo: float
    hi: float
    owner_ids: tuple[str, ...]


def _merge_line_coverages(
    values: Iterable[tuple[float, float, float, str]],
    tolerance: float,
) -> tuple[_LineCoverage, ...]:
    ordered = sorted(values)
    coordinate_groups: list[list[tuple[float, float, float, str]]] = []
    for value in ordered:
        if not coordinate_groups:
            coordinate_groups.append([value])
            continue
        fixed = sum(item[0] for item in coordinate_groups[-1]) / len(coordinate_groups[-1])
        if abs(value[0] - fixed) <= tolerance:
            coordinate_groups[-1].append(value)
        else:
            coordinate_groups.append([value])

    output: list[_LineCoverage] = []
    for group in coordinate_groups:
        fixed = sum(item[0] for item in group) / len(group)
        intervals = sorted((lo, hi, owner_id) for _, lo, hi, owner_id in group)
        current_lo, current_hi, first_owner = intervals[0]
        owners = {first_owner}
        for lo, hi, owner_id in intervals[1:]:
            if lo <= current_hi + tolerance:
                current_hi = max(current_hi, hi)
                owners.add(owner_id)
                continue
            output.append(_LineCoverage(fixed, current_lo, current_hi, tuple(sorted(owners))))
            current_lo, current_hi, owners = lo, hi, {owner_id}
        output.append(_LineCoverage(fixed, current_lo, current_hi, tuple(sorted(owners))))
    return tuple(output)


def _axis_coverages(
    geometry_entities: Iterable[Entity],
    tolerance: float,
) -> tuple[tuple[_LineCoverage, ...], tuple[_LineCoverage, ...]]:
    horizontal: list[tuple[float, float, float, str]] = []
    vertical: list[tuple[float, float, float, str]] = []
    for entity in geometry_entities:
        for first, second in entity.geometry_segments:
            dx = abs(first[0] - second[0])
            dy = abs(first[1] - second[1])
            if dy <= tolerance and dx > 2.0 * tolerance:
                horizontal.append(
                    (
                        0.5 * (first[1] + second[1]),
                        min(first[0], second[0]),
                        max(first[0], second[0]),
                        entity.id,
                    )
                )
            elif dx <= tolerance and dy > 2.0 * tolerance:
                vertical.append(
                    (
                        0.5 * (first[0] + second[0]),
                        min(first[1], second[1]),
                        max(first[1], second[1]),
                        entity.id,
                    )
                )
    return (
        _merge_line_coverages(horizontal, tolerance),
        _merge_line_coverages(vertical, tolerance),
    )


def _coverage_owners(
    coverages: Iterable[_LineCoverage],
    fixed: float,
    lo: float,
    hi: float,
    tolerance: float,
) -> tuple[str, ...]:
    owners: set[str] = set()
    for coverage in coverages:
        if (
            abs(coverage.fixed - fixed) <= tolerance
            and coverage.lo <= lo + tolerance
            and coverage.hi >= hi - tolerance
        ):
            owners.update(coverage.owner_ids)
    return tuple(sorted(owners))


def _frame_cells_from_segments(
    geometry_entities: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[Entity, ...]:
    tolerance = settings.callout_segment_connect_tolerance
    horizontal, vertical = _axis_coverages(geometry_entities, tolerance)
    if not horizontal or not vertical:
        return ()

    bands: set[tuple[float, float]] = set()
    for wall in vertical:
        crossings = sorted(
            {
                rail.fixed
                for rail in horizontal
                if wall.lo - tolerance <= rail.fixed <= wall.hi + tolerance
                and rail.lo - tolerance <= wall.fixed <= rail.hi + tolerance
            }
        )
        for top, bottom in zip(crossings, crossings[1:], strict=False):
            height = bottom - top
            if settings.callout_min_frame_height <= height <= settings.callout_max_frame_height:
                bands.add((top, bottom))

    source_by_id = {entity.id: entity for entity in geometry_entities}
    cells: dict[tuple[float, float, float, float], Entity] = {}
    for top, bottom in sorted(bands):
        walls = tuple(
            wall
            for wall in vertical
            if wall.lo <= top + tolerance and wall.hi >= bottom - tolerance
        )
        for left, right in zip(walls, walls[1:], strict=False):
            width = right.fixed - left.fixed
            if not (
                settings.callout_min_frame_cell_width <= width <= settings.callout_max_frame_width
            ):
                continue
            top_owners = _coverage_owners(horizontal, top, left.fixed, right.fixed, tolerance)
            bottom_owners = _coverage_owners(horizontal, bottom, left.fixed, right.fixed, tolerance)
            if not top_owners or not bottom_owners:
                continue
            owner_ids = tuple(
                sorted(
                    set(left.owner_ids)
                    | set(right.owner_ids)
                    | set(top_owners)
                    | set(bottom_owners)
                )
            )
            raw_bbox = (left.fixed, top, right.fixed, bottom)
            key = tuple(round(value, 7) for value in raw_bbox)
            bbox = (key[0], key[1], key[2], key[3])
            style_signature = _digest(
                *(source_by_id[owner_id].style_signature for owner_id in owner_ids)
            )
            shape_signature = _digest("frame-cell", round(width, 6), round(bottom - top, 6))
            content_signature = _digest("geometry", shape_signature, style_signature)
            page_index = source_by_id[owner_ids[0]].page_index
            cell = Entity(
                id=f"p{page_index + 1}-fc-{_digest(*key, *owner_ids, length=14)}",
                page_index=page_index,
                kind=EntityKind.GEOMETRY,
                bbox=bbox,
                anchor=((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0),
                content_signature=content_signature,
                shape_signature=shape_signature,
                style_signature=style_signature,
                op_histogram=(("l", 4),),
                primitive_count=4,
                path_length=round(2.0 * (width + bottom - top), 6),
                geometry_segments=(
                    ((bbox[0], bbox[1]), (bbox[2], bbox[1])),
                    ((bbox[2], bbox[1]), (bbox[2], bbox[3])),
                    ((bbox[2], bbox[3]), (bbox[0], bbox[3])),
                    ((bbox[0], bbox[3]), (bbox[0], bbox[1])),
                ),
                callout_member_ids=owner_ids,
            )
            previous = cells.get(key)
            if previous is None or cell.id < previous.id:
                cells[key] = cell
    return tuple(sorted(cells.values(), key=lambda entity: (entity.bbox, entity.id)))


def _same_frame_band(first: Entity, second: Entity, tolerance: float) -> bool:
    return (
        abs(first.bbox[1] - second.bbox[1]) <= tolerance
        and abs(first.bbox[3] - second.bbox[3]) <= tolerance
    )


def _unified_frame_cells(
    geometry_entities: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[Entity, ...]:
    """Combine explicit rectangles with cells reconstructed from all segments.

    Explicit rectangle paths are also fed into topology reconstruction so a
    cell emitted on its own can supply a missing wall to rails emitted in a
    different drawing record. Reconstructed cells replace equivalent explicit
    boxes and partition compound multi-cell rectangles.
    """

    explicit = tuple(
        entity
        for entity in geometry_entities
        if _is_rectangle(entity)
        and entity.height <= settings.callout_max_frame_height
        and entity.width <= settings.callout_max_frame_width
    )
    inferred = _frame_cells_from_segments(geometry_entities, settings)
    tolerance = settings.callout_segment_connect_tolerance
    output = list(inferred)
    for rectangle in explicit:
        same_band = tuple(
            cell
            for cell in inferred
            if _same_frame_band(rectangle, cell, tolerance)
            and _bbox_contains_bbox(rectangle.bbox, cell.bbox, tolerance)
        )
        if same_band:
            covered_left = min(cell.bbox[0] for cell in same_band)
            covered_right = max(cell.bbox[2] for cell in same_band)
            if (
                covered_left <= rectangle.bbox[0] + tolerance
                and covered_right >= rectangle.bbox[2] - tolerance
            ):
                continue
        output.append(rectangle)
    return tuple(sorted(output, key=lambda entity: (entity.bbox, entity.id)))


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


def _looks_like_gdt_sequence(parts: Iterable[str], *, minimum_datums: int = 2) -> bool:
    """Recognize a framed FCF whose custom-font symbol did not extract as text."""

    cleaned = tuple(_clean(part) for part in parts if _clean(part))
    for tolerance_index, part in enumerate(cleaned):
        if not _GDT_TOLERANCE_CELL.fullmatch(part):
            continue
        # A feature-control tolerance is followed by multiple datum or
        # material-condition cells. Requiring that topology avoids treating an
        # arbitrary boxed decimal as GD&T.
        suffix = cleaned[tolerance_index + 1 :]
        allowed_suffix = tuple(
            bool(_GDT_DATUM_CELL.fullmatch(value))
            or value in _GDT_MODIFIER_SYMBOLS
            or bool(_GDT_MODIFIER_WORDS.fullmatch(value))
            for value in suffix
        )
        datum_count = sum(bool(_GDT_DATUM_CELL.fullmatch(value)) for value in suffix)
        if datum_count >= minimum_datums and all(allowed_suffix):
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

    # Custom CAD fonts frequently outline the feature symbol and place text
    # very close to a divider. In that case tolerant cell assignment can put a
    # glyph in two cells. Use the unambiguous left-to-right semantic sequence,
    # then locate its tolerance in the reconstructed frame topology.
    ordered_entities = tuple(
        sorted(contained, key=lambda entity: (entity.centroid[0], entity.bbox[1], entity.id))
    )
    if _looks_like_gdt_sequence(entity.text for entity in ordered_entities):
        for entity in ordered_entities:
            if not _GDT_TOLERANCE_CELL.fullmatch(_clean(entity.text)):
                continue
            containing = [
                index
                for index, cell in enumerate(cells)
                if _bbox_contains(cell.bbox, entity.centroid, 0.0)
            ]
            if not containing:
                containing = [
                    min(
                        range(len(cells)),
                        key=lambda index: (
                            abs(cells[index].centroid[0] - entity.centroid[0]),
                            index,
                        ),
                    )
                ]
            tolerance_index = min(
                containing,
                key=lambda index: (
                    abs(cells[index].centroid[0] - entity.centroid[0]),
                    index,
                ),
            )
            return (max(0, tolerance_index - 1),)

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


def _has_genuine_vector_feature(
    geometry: Iterable[Entity],
    cell: Entity,
    tolerance: float,
    settings: ComparisonSettings,
) -> bool:
    """Require a compact, centered multi-stroke mark inside the feature cell."""

    inset = min(0.25 * tolerance, 0.05 * min(cell.width, cell.height))
    inner = (
        cell.bbox[0] + inset,
        cell.bbox[1] + inset,
        cell.bbox[2] - inset,
        cell.bbox[3] - inset,
    )
    candidates = tuple(
        entity
        for entity in geometry
        if not _is_rectangle(entity) and _bbox_contains_bbox(inner, entity.bbox)
    )
    if sum(entity.primitive_count for entity in candidates) < 2:
        return False
    bbox = _bbox_union(candidates)
    width_fill = (bbox[2] - bbox[0]) / cell.width
    height_fill = (bbox[3] - bbox[1]) / cell.height
    if not (
        settings.callout_vector_feature_min_fill
        <= width_fill
        <= settings.callout_vector_feature_max_fill
        and settings.callout_vector_feature_min_fill
        <= height_fill
        <= settings.callout_vector_feature_max_fill
    ):
        return False
    center_x = abs(0.5 * (bbox[0] + bbox[2]) - cell.centroid[0]) / cell.width
    center_y = abs(0.5 * (bbox[1] + bbox[3]) - cell.centroid[1]) / cell.height
    return max(center_x, center_y) <= settings.callout_vector_feature_max_center_offset


def _one_datum_vector_frame(
    cells: tuple[Entity, ...],
    contained: tuple[Entity, ...],
    geometry: tuple[Entity, ...],
    tolerance: float,
    settings: ComparisonSettings,
) -> bool:
    """Recognize only a complete vector-feature | tolerance | datum frame."""

    if len(cells) < 3:
        return False
    topology_tolerance = settings.callout_segment_connect_tolerance
    if any(
        not _same_frame_band(left, right, topology_tolerance)
        or abs(left.bbox[2] - right.bbox[0]) > topology_tolerance
        for left, right in zip(cells, cells[1:], strict=False)
    ):
        return False
    cell_parts = tuple(_cell_text(cell, contained, tolerance) for cell in cells)
    if cell_parts[0] or not _GDT_TOLERANCE_CELL.fullmatch(cell_parts[1]):
        return False
    suffix = cell_parts[2:]
    if not suffix or any(not value for value in suffix):
        return False
    allowed_suffix = tuple(
        bool(_GDT_DATUM_CELL.fullmatch(value))
        or value in _GDT_MODIFIER_SYMBOLS
        or bool(_GDT_MODIFIER_WORDS.fullmatch(value))
        for value in suffix
    )
    if not all(allowed_suffix):
        return False
    if sum(bool(_GDT_DATUM_CELL.fullmatch(value)) for value in suffix) != 1:
        return False
    return _has_genuine_vector_feature(geometry, cells[0], tolerance, settings)


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
        callout_member_ids=_flatten_member_ids(members),
        callout_attachment_points=attachments,
    )


def _gdt_groups(
    text_entities: tuple[Entity, ...],
    geometry_entities: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[str, ...]], ...]:
    rectangles = _unified_frame_cells(geometry_entities, settings)
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
        component_geometry = tuple(
            entity
            for entity in geometry_entities
            if entity.id not in used_geometry and _bbox_contains(frame_bbox, entity.bbox, tolerance)
        )
        contained = tuple(
            entity
            for entity in text_entities
            if entity.id not in used_text and _bbox_contains(frame_bbox, entity.centroid, tolerance)
        )
        if not contained:
            continue
        cells = sorted(frames, key=lambda entity: (entity.bbox[0], entity.bbox[1], entity.id))
        starts = list(_gdt_starts(tuple(cells), contained, tolerance))
        single_datum_vector = not starts and _one_datum_vector_frame(
            tuple(cells),
            contained,
            component_geometry,
            tolerance,
            settings,
        )
        if single_datum_vector:
            starts = [0]
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
            if len(group_frames) > settings.callout_max_frame_cells:
                continue
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
            frame_source_ids = frozenset(_flatten_member_ids(group_frames))
            if not frame_source_ids:
                continue
            enclosed_geometry = tuple(
                entity
                for entity in component_geometry
                if entity.id not in frame_source_ids
                and _bbox_contains_bbox(group_bbox, entity.bbox, tolerance)
            )
            ordered_cells = [
                cell_text
                for cell in group_frames
                if (cell_text := _cell_text(cell, group_text, tolerance))
            ]
            ordered_group_text = tuple(
                sorted(
                    group_text,
                    key=lambda entity: (entity.centroid[0], entity.bbox[1], entity.id),
                )
            )
            combined = " | ".join(_clean(entity.text) for entity in ordered_group_text)
            # Some CAD exporters emit the entire feature-control frame as one
            # compound path instead of one path per cell. Strong GD&T grammar
            # plus containment is sufficient in that representation.
            sequence_minimum_datums = 1 if single_datum_vector else 2
            if not (
                _valid_gdt_text(combined)
                or _looks_like_gdt_sequence(
                    ordered_cells,
                    minimum_datums=sequence_minimum_datums,
                )
                or _looks_like_gdt_sequence(
                    (entity.text for entity in ordered_group_text),
                    minimum_datums=sequence_minimum_datums,
                )
                or (
                    len(group_frames) == 1
                    and _has_internal_frame_divider(group_frames[0], tolerance)
                    and _looks_like_gdt_sequence(
                        (
                            entity.text
                            for entity in sorted(
                                group_text,
                                key=lambda item: (item.bbox[0], item.id),
                            )
                        ),
                        minimum_datums=sequence_minimum_datums,
                    )
                )
            ):
                continue
            has_text_feature = any(_has_gdt_marker(entity.text) for entity in group_text)
            has_vector_feature = _has_genuine_vector_feature(
                enclosed_geometry,
                group_frames[0],
                tolerance,
                settings,
            )
            if not has_text_feature and not has_vector_feature:
                continue
            group_geometry = (*group_frames, *enclosed_geometry)
            attachments = _attachment_points(
                group_bbox,
                max((_height(entity) for entity in group_text), default=0.01),
                (
                    entity
                    for entity in geometry_entities
                    if entity.id not in frame_source_ids and entity not in enclosed_geometry
                ),
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
            member_ids = _flatten_member_ids((*group_text, *group_geometry))
            output.append((composite, member_ids))
            used_text.update(_flatten_member_ids(group_text))
            used_geometry.update(frame_source_ids)
            used_geometry.update(_flatten_member_ids(enclosed_geometry))

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
        if not _valid_gdt_text(text) or not _looks_like_gdt_sequence(entity.text for entity in row):
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
        member_ids = _flatten_member_ids(row)
        output.append((composite, member_ids))
        used_text.update(member_ids)

    return tuple(output)


def _dimension_base_hypotheses(
    candidates: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[Entity, ...]], ...]:
    """Build one root-owned hypothesis per nominal/core fragment."""

    material = tuple(entity for entity in candidates if _fragment_role(entity.text) != "tolerance")
    edges: dict[str, set[str]] = defaultdict(set)
    for index, first in enumerate(material):
        for second in material[index + 1 :]:
            if _inline_dimension_edge(first, second, settings):
                edges[first.id].add(second.id)
                edges[second.id].add(first.id)

    output: list[tuple[Entity, tuple[Entity, ...]]] = []
    for component in _components(material, edges):
        roots = tuple(
            entity for entity in component if _fragment_role(entity.text) in {"core", "nominal"}
        )
        if len(roots) == 1:
            output.append((roots[0], component))
            continue
        # A support fragment that reaches multiple roots is ambiguous. It is
        # left raw while the roots remain independent hypotheses.
        output.extend((root, (root,)) for root in roots)
    return tuple(sorted(output, key=lambda item: (item[0].bbox, item[0].id)))


def _assign_tolerance_units(
    hypotheses: tuple[tuple[Entity, tuple[Entity, ...]], ...],
    tolerances: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[Entity, ...]], ...]:
    units = _tolerance_units(tolerances, settings)
    unit_by_key = {"\x1f".join(sorted(entity.id for entity in unit)): unit for unit in units}
    root_choices: dict[str, list[tuple[float, str]]] = defaultdict(list)
    unit_choices: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for root, _ in hypotheses:
        if _SIGNED_TOLERANCE_SEARCH.search(_clean(root.text)):
            continue
        for unit_key, unit in unit_by_key.items():
            score = _tolerance_unit_score(root, unit, settings)
            if score is None:
                continue
            root_choices[root.id].append((score, unit_key))
            unit_choices[unit_key].append((score, root.id))

    best_unit = {
        root_id: _clear_best(
            values,
            ambiguity_margin=settings.callout_hypothesis_ambiguity_margin,
        )
        for root_id, values in root_choices.items()
    }
    best_root = {
        unit_key: _clear_best(
            values,
            ambiguity_margin=settings.callout_hypothesis_ambiguity_margin,
        )
        for unit_key, values in unit_choices.items()
    }
    output: list[tuple[Entity, tuple[Entity, ...]]] = []
    for root, members in hypotheses:
        unit_key = best_unit.get(root.id)
        if unit_key is None or best_root.get(unit_key) != root.id:
            output.append((root, members))
            continue
        output.append((root, (*members, *unit_by_key[unit_key])))
    return tuple(output)


def _attach_dimension_suffixes(
    hypotheses: tuple[tuple[Entity, tuple[Entity, ...]], ...],
    candidates: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[Entity, ...]], ...]:
    owned = {entity.id for _, members in hypotheses for entity in members}
    suffixes = tuple(
        entity
        for entity in candidates
        if entity.id not in owned and _fragment_role(entity.text) in {"qualifier", "unit"}
    )
    additions: dict[str, list[Entity]] = defaultdict(list)
    for suffix in suffixes:
        choices: list[tuple[float, str]] = []
        for root, members in hypotheses:
            predecessors = tuple(
                member for member in members if _inline_dimension_edge(member, suffix, settings)
            )
            if not predecessors:
                continue
            scale = max(_height(suffix), *(_height(member) for member in predecessors))
            score = min(
                max(0.0, suffix.bbox[0] - member.bbox[2]) / scale
                + abs(suffix.anchor[1] - member.anchor[1]) / scale
                for member in predecessors
            )
            choices.append((score, root.id))
        root_id = _clear_best(
            choices,
            ambiguity_margin=settings.callout_hypothesis_ambiguity_margin,
        )
        if root_id is not None:
            additions[root_id].append(suffix)
    return tuple((root, (*members, *additions.get(root.id, ()))) for root, members in hypotheses)


def _dimension_groups(
    text_entities: tuple[Entity, ...],
    geometry_entities: tuple[Entity, ...],
    excluded_ids: frozenset[str],
    settings: ComparisonSettings,
) -> tuple[tuple[Entity, tuple[str, ...]], ...]:
    fragments = _signed_tolerance_fragments(
        (entity for entity in text_entities if excluded_ids.isdisjoint(_leaf_member_ids(entity))),
        settings,
    )
    candidates = tuple(
        entity
        for entity in fragments
        if excluded_ids.isdisjoint(_leaf_member_ids(entity))
        and _fragment_role(entity.text) not in {"other", "gdt"}
    )
    tolerances = tuple(
        entity for entity in candidates if _fragment_role(entity.text) == "tolerance"
    )
    hypotheses = _dimension_base_hypotheses(candidates, settings)
    hypotheses = _assign_tolerance_units(hypotheses, tolerances, settings)
    hypotheses = _attach_dimension_suffixes(hypotheses, candidates, settings)
    components = tuple(members for _, members in hypotheses)
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
                _flatten_member_ids((*component, *enclosed_symbols)),
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
    original_geometry = tuple(entity for entity in raw if entity.kind == EntityKind.GEOMETRY)
    geometry_entities, children_by_parent = _local_geometry_entities(
        original_geometry,
        settings,
    )

    grouped: list[Entity] = []
    consumed: set[str] = set()
    for composite, member_ids in _gdt_groups(text_entities, geometry_entities, settings):
        # A CAD exporter may use one continuous rail entity for two touching
        # frames. _gdt_groups owns text/vector members uniquely but permits that
        # shared frame source to support both semantic composites.
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

    output = [entity for entity in text_entities if entity.id not in consumed]
    for parent in original_geometry:
        children = children_by_parent[parent.id]
        remaining = tuple(child for child in children if child.id not in consumed)
        if len(remaining) == len(children):
            output.append(parent)
        elif remaining:
            output.append(_geometry_residual_entity(parent, remaining))
    output.extend(grouped)
    return tuple(sorted(output, key=lambda entity: (entity.kind.value, entity.bbox, entity.id)))
