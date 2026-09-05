"""Serializable domain models for extraction, matching, and reporting."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum, StrEnum
from pathlib import Path
from typing import Any

BBox = tuple[float, float, float, float]
Point = tuple[float, float]


class EntityKind(StrEnum):
    GEOMETRY = "geometry"
    TEXT = "text"


class ChangeType(StrEnum):
    ADDED = "added"
    REMOVED = "removed"
    MOVED = "moved"
    MODIFIED = "modified"


class MatchTier(StrEnum):
    EXACT = "exact"
    ATTRIBUTE = "attribute"
    STRUCTURAL = "structural"


class ChangeCategory(StrEnum):
    DIMENSION = "DIMENSION"
    GDT = "GD&T"
    NOTE = "NOTE"
    REVISION = "REVISION"
    GEOMETRY = "GEOMETRY"
    OTHER = "OTHER"


@dataclass(frozen=True, slots=True)
class Entity:
    """One exact text span or vector path extracted from a PDF display list."""

    id: str
    page_index: int
    kind: EntityKind
    bbox: BBox
    anchor: Point
    content_signature: str
    shape_signature: str
    style_signature: str
    text: str = ""
    text_normalized: str = ""
    font_name: str = ""
    font_size: float = 0.0
    op_histogram: tuple[tuple[str, int], ...] = ()
    primitive_count: int = 0
    path_length: float = 0.0

    @property
    def centroid(self) -> Point:
        x0, y0, x1, y1 = self.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox[3] - self.bbox[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class Transform:
    """Similarity transform mapping old normalized coordinates into the new frame."""

    scale: float = 1.0
    rotation_radians: float = 0.0
    tx: float = 0.0
    ty: float = 0.0


@dataclass(frozen=True, slots=True)
class AlignmentResult:
    status: str
    transform: Transform = Transform()
    anchor_count: int = 0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    rms_residual: float = 0.0
    note: str = ""


@dataclass(frozen=True, slots=True)
class EntityMatch:
    old: Entity
    new: Entity
    tier: MatchTier
    score: float
    registered_distance: float


@dataclass(frozen=True, slots=True)
class MatchResult:
    matches: tuple[EntityMatch, ...]
    unmatched_old: tuple[Entity, ...]
    unmatched_new: tuple[Entity, ...]


@dataclass(frozen=True, slots=True)
class Change:
    id: str
    page_index: int
    change_type: ChangeType
    category: ChangeCategory
    inspection_relevant: bool
    relevance_reason: str
    bbox: BBox
    old_bbox: BBox | None
    label: str
    detail: str
    modification_kinds: tuple[str, ...] = ()
    old_entity_id: str | None = None
    new_entity_id: str | None = None
    before_text: str | None = None
    after_text: str | None = None
    match_tier: MatchTier | None = None
    similarity_score: float | None = None

    @property
    def area(self) -> float:
        return max(0.0, self.bbox[2] - self.bbox[0]) * max(0.0, self.bbox[3] - self.bbox[1])


@dataclass(frozen=True, slots=True)
class PageResult:
    page_index: int
    old_width: float
    old_height: float
    new_width: float
    new_height: float
    alignment: AlignmentResult
    entity_count_old: int
    entity_count_new: int
    changes: tuple[Change, ...]
    affected_area_fraction: float = 0.0
    relevant_area_fraction: float = 0.0
    notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    old_file: str
    new_file: str
    pages: tuple[PageResult, ...]
    summary: str
    notes: tuple[str, ...] = ()
    engine_version: str = "0.2.0"
    analysis_mode: str = "vector-and-text-entities-only"

    @property
    def changes(self) -> tuple[Change, ...]:
        return tuple(change for page in self.pages for change in page.changes)

    @property
    def relevant_changes(self) -> tuple[Change, ...]:
        return tuple(change for change in self.changes if change.inspection_relevant)

    @property
    def counts(self) -> dict[str, int]:
        return {
            change_type.value: sum(c.change_type == change_type for c in self.changes)
            for change_type in ChangeType
        }

    @property
    def mean_affected_area_fraction(self) -> float:
        return (
            sum(page.affected_area_fraction for page in self.pages) / len(self.pages)
            if self.pages
            else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        return _plain(self)


def _plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


@dataclass(slots=True)
class ProgressEvent:
    stage: str
    fraction: float
    message: str = field(default="")
