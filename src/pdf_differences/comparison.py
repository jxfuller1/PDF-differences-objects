"""End-to-end vector-native PDF comparison orchestration."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pymupdf as fitz

from pdf_differences.alignment import estimate_alignment, transform_bbox
from pdf_differences.callouts import reconstruct_callouts
from pdf_differences.config import SETTINGS, ComparisonSettings
from pdf_differences.errors import AlignmentError, ComparisonCancelled
from pdf_differences.extraction import extract_page_entities
from pdf_differences.matching import match_entities
from pdf_differences.mechanical import Interpretation, interpret_change
from pdf_differences.models import (
    AlignmentResult,
    BBox,
    Change,
    ChangeCategory,
    ChangeType,
    ComparisonResult,
    Entity,
    EntityKind,
    EntityMatch,
    MatchTier,
    PageResult,
    ProgressEvent,
    Transform,
)
from pdf_differences.validation import PdfCapabilities, validate_pdf

ProgressCallback = Callable[[ProgressEvent], None]
CancelCallback = Callable[[], bool]


def _emit(callback: ProgressCallback | None, stage: str, fraction: float, message: str) -> None:
    if callback is not None:
        callback(ProgressEvent(stage, max(0.0, min(1.0, fraction)), message))


def _check_cancel(cancelled: CancelCallback | None) -> None:
    if cancelled is not None and cancelled():
        raise ComparisonCancelled("Comparison cancelled.")


def _change_id(page_index: int, change_type: ChangeType, old_id: str, new_id: str) -> str:
    material = f"{page_index}|{change_type.value}|{old_id}|{new_id}"
    return "chg-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:14]


def _entity_label(entity: Entity) -> str:
    if entity.kind == EntityKind.TEXT:
        return entity.text
    noun = "primitive" if entity.primitive_count == 1 else "primitives"
    return f"Vector geometry ({entity.primitive_count} {noun})"


def _interpret(
    kind: EntityKind,
    before_text: str | None,
    after_text: str | None,
    bbox: BBox,
    context: tuple[Entity, ...],
    settings: ComparisonSettings,
    category_hint: ChangeCategory | None = None,
) -> Interpretation:
    return interpret_change(
        kind,
        before_text,
        after_text,
        bbox,
        context,
        settings,
        category_hint=category_hint,
    )


def _unmatched_change(
    entity: Entity,
    change_type: ChangeType,
    transform: Transform,
    context: tuple[Entity, ...],
    settings: ComparisonSettings,
) -> Change:
    removed = change_type == ChangeType.REMOVED
    display_bbox = transform_bbox(entity.bbox, transform) if removed else entity.bbox
    before_text = entity.text if removed and entity.kind == EntityKind.TEXT else None
    after_text = entity.text if not removed and entity.kind == EntityKind.TEXT else None
    interpretation = _interpret(
        entity.kind,
        before_text,
        after_text,
        entity.bbox,
        context,
        settings,
        entity.callout_category,
    )
    action = "removed" if removed else "added"
    return Change(
        id=_change_id(
            entity.page_index,
            change_type,
            entity.id if removed else "",
            entity.id if not removed else "",
        ),
        page_index=entity.page_index,
        change_type=change_type,
        category=interpretation.category,
        inspection_relevant=interpretation.relevant,
        relevance_reason=interpretation.reason,
        bbox=display_bbox,
        old_bbox=entity.bbox if removed else None,
        label=_entity_label(entity),
        detail=f"{_entity_label(entity)} {action}.",
        old_entity_id=entity.id if removed else None,
        new_entity_id=entity.id if not removed else None,
        before_text=before_text,
        after_text=after_text,
        old_member_entity_ids=entity.callout_member_ids if removed else (),
        new_member_entity_ids=entity.callout_member_ids if not removed else (),
    )


def _modification_kinds(match: EntityMatch, settings: ComparisonSettings) -> tuple[str, ...]:
    kinds: list[str] = []
    if match.old.kind == EntityKind.TEXT:
        if match.registered_distance > settings.moved_tolerance:
            kinds.append("position")
        if match.old.text != match.new.text:
            kinds.append("text")
        if match.old.font_size != match.new.font_size or match.old.font_name != match.new.font_name:
            kinds.append("text-style")
        if (
            match.old.callout_category is not None
            and match.old.shape_signature != match.new.shape_signature
        ):
            kinds.append("callout-structure")
    else:
        geometry_changed = match.old.shape_signature != match.new.shape_signature
        if geometry_changed:
            kinds.append("geometry")
        size_unchanged = abs(
            match.old.width * match.old.height - match.new.width * match.new.height
        ) <= 0.02 * max(match.old.area, match.new.area, 1e-9)
        if match.registered_distance > settings.moved_tolerance and (
            not geometry_changed or size_unchanged
        ):
            kinds.insert(0, "position")
    if match.old.style_signature != match.new.style_signature and "text-style" not in kinds:
        kinds.append("style")
    return tuple(kinds)


def _matched_change(
    match: EntityMatch,
    context: tuple[Entity, ...],
    transform: Transform,
    settings: ComparisonSettings,
) -> Change | None:
    if match.tier == MatchTier.EXACT:
        return None
    kinds = _modification_kinds(match, settings)
    if not kinds:
        return None
    change_type = ChangeType.MOVED if kinds == ("position",) else ChangeType.MODIFIED
    before_text = match.old.text if match.old.kind == EntityKind.TEXT else None
    after_text = match.new.text if match.new.kind == EntityKind.TEXT else None
    interpretation = _interpret(
        match.new.kind,
        before_text,
        after_text,
        match.new.bbox,
        context,
        settings,
        match.new.callout_category or match.old.callout_category,
    )
    if before_text is not None or after_text is not None:
        detail = f"{before_text!r} -> {after_text!r} ({', '.join(kinds)})"
    else:
        detail = "Vector entity changed: " + ", ".join(kinds) + "."
    display_bbox = match.new.bbox
    if match.old.callout_category is not None or match.new.callout_category is not None:
        aligned_old_bbox = transform_bbox(match.old.bbox, transform)
        display_bbox = (
            min(aligned_old_bbox[0], match.new.bbox[0]),
            min(aligned_old_bbox[1], match.new.bbox[1]),
            max(aligned_old_bbox[2], match.new.bbox[2]),
            max(aligned_old_bbox[3], match.new.bbox[3]),
        )
    return Change(
        id=_change_id(match.old.page_index, change_type, match.old.id, match.new.id),
        page_index=match.old.page_index,
        change_type=change_type,
        category=interpretation.category,
        inspection_relevant=interpretation.relevant,
        relevance_reason=interpretation.reason,
        bbox=display_bbox,
        old_bbox=match.old.bbox,
        label=_entity_label(match.new),
        detail=detail,
        modification_kinds=kinds,
        old_entity_id=match.old.id,
        new_entity_id=match.new.id,
        before_text=before_text,
        after_text=after_text,
        match_tier=match.tier,
        similarity_score=match.score,
        old_member_entity_ids=match.old.callout_member_ids,
        new_member_entity_ids=match.new.callout_member_ids,
    )


def _changes_from_match(
    matches,
    old_context: tuple[Entity, ...],
    new_context: tuple[Entity, ...],
    transform: Transform,
    settings: ComparisonSettings,
) -> tuple[Change, ...]:
    changes = [
        change
        for match in matches.matches
        if (change := _matched_change(match, new_context, transform, settings)) is not None
    ]
    changes.extend(
        _unmatched_change(entity, ChangeType.REMOVED, transform, old_context, settings)
        for entity in matches.unmatched_old
    )
    changes.extend(
        _unmatched_change(entity, ChangeType.ADDED, transform, new_context, settings)
        for entity in matches.unmatched_new
    )
    return tuple(
        sorted(
            changes,
            key=lambda change: (
                change.page_index,
                change.bbox[1],
                change.bbox[0],
                change.change_type.value,
                change.id,
            ),
        )
    )


def _page_notes(capabilities: PdfCapabilities, page_index: int, role: str) -> tuple[str, ...]:
    if page_index >= capabilities.page_count:
        return ()
    page = capabilities.pages[page_index]
    notes: list[str] = []
    if page.text_span_count and not page.drawing_count:
        notes.append(
            f"{role} page {page_index + 1} has a text layer but no vector paths; "
            "only text can be compared."
        )
    if page.image_count:
        notes.append(
            f"{role} page {page_index + 1} contains {page.image_count} embedded image "
            "object(s); they were ignored because image comparison is disabled."
        )
    return tuple(notes)


def _union_area(boxes: tuple[BBox, ...]) -> float:
    """Exact union area for normalized axis-aligned change boxes."""

    if not boxes:
        return 0.0
    x_edges = sorted({max(0.0, min(1.0, value)) for box in boxes for value in (box[0], box[2])})
    total = 0.0
    for left, right in zip(x_edges, x_edges[1:], strict=False):
        if right <= left:
            continue
        midpoint = (left + right) / 2.0
        intervals = sorted(
            (max(0.0, box[1]), min(1.0, box[3]))
            for box in boxes
            if box[0] <= midpoint <= box[2] and box[3] > box[1]
        )
        covered = 0.0
        if intervals:
            start, end = intervals[0]
            for next_start, next_end in intervals[1:]:
                if next_start <= end:
                    end = max(end, next_end)
                else:
                    covered += max(0.0, end - start)
                    start, end = next_start, next_end
            covered += max(0.0, end - start)
        total += (right - left) * covered
    return min(1.0, max(0.0, total))


def _summary(pages: tuple[PageResult, ...]) -> str:
    changes = tuple(change for page in pages for change in page.changes)
    if not changes:
        return "No structured vector or text-layer differences were detected."
    counts = Counter(change.change_type.value for change in changes)
    categories = Counter(change.category.value for change in changes if change.inspection_relevant)
    relevant = sum(change.inspection_relevant for change in changes)
    type_text = ", ".join(
        f"{counts[kind]} {kind}"
        for kind in ("added", "removed", "moved", "modified")
        if counts[kind]
    )
    if categories:
        category_text = ", ".join(f"{count} {name}" for name, count in sorted(categories.items()))
        relevance_text = f"{relevant} are inspection-relevant ({category_text})."
    else:
        relevance_text = "None matched the configured deterministic inspection-relevance rules."
    affected = sum(page.affected_area_fraction for page in pages) / max(1, len(pages))
    return (
        f"Detected {len(changes)} structured changes: {type_text}. {relevance_text} "
        f"Change boxes cover {affected * 100:.2f}% of the average compared page area."
    )


def compare_pdfs(
    old_path: str | Path,
    new_path: str | Path,
    *,
    settings: ComparisonSettings = SETTINGS,
    progress: ProgressCallback | None = None,
    cancelled: CancelCallback | None = None,
) -> ComparisonResult:
    """Compare every page by index and return a deterministic structured report."""

    _emit(progress, "validation", 0.01, "Validating native PDF structure")
    old_capabilities = validate_pdf(old_path, settings)
    _check_cancel(cancelled)
    new_capabilities = validate_pdf(new_path, settings)
    _check_cancel(cancelled)

    old_document = fitz.open(old_capabilities.path)
    new_document = fitz.open(new_capabilities.path)
    try:
        total_pages = max(old_document.page_count, new_document.page_count)
        page_results: list[PageResult] = []
        overall_notes: list[str] = []
        if old_document.page_count != new_document.page_count:
            overall_notes.append(
                "Page counts differ "
                f"({old_document.page_count} old vs {new_document.page_count} new); "
                "extra pages are treated as wholly removed or added."
            )

        for page_index in range(total_pages):
            _check_cancel(cancelled)
            fraction = 0.05 + 0.9 * (page_index / max(total_pages, 1))
            _emit(
                progress,
                "extraction",
                fraction,
                f"Extracting page {page_index + 1} of {total_pages}",
            )
            old_page = (
                old_document.load_page(page_index) if page_index < old_document.page_count else None
            )
            new_page = (
                new_document.load_page(page_index) if page_index < new_document.page_count else None
            )
            old_entities = (
                extract_page_entities(old_page, page_index) if old_page is not None else ()
            )
            new_entities = (
                extract_page_entities(new_page, page_index) if new_page is not None else ()
            )
            notes = [
                *(_page_notes(old_capabilities, page_index, "Old")),
                *(_page_notes(new_capabilities, page_index, "New")),
            ]

            if old_page is None or new_page is None:
                alignment = AlignmentResult(
                    status="not-applicable",
                    note=(
                        "Page exists in only one revision; all structured entities are "
                        "page additions/removals."
                    ),
                )
                transform = Transform()
            elif not old_entities or not new_entities:
                alignment = AlignmentResult(
                    status="identity-unverified",
                    note="One or both pages are blank; alignment is not applicable.",
                )
                transform = Transform()
            else:
                _emit(progress, "alignment", fraction + 0.015, f"Aligning page {page_index + 1}")
                alignment = estimate_alignment(old_entities, new_entities, settings)
                if alignment.status == "failed":
                    raise AlignmentError(
                        f"Page {page_index + 1} could not be aligned reliably. "
                        "Confirm that both files are revisions of the same drawing sheet."
                    )
                transform = alignment.transform

            _check_cancel(cancelled)
            _emit(progress, "matching", fraction + 0.03, f"Tier-matching page {page_index + 1}")
            old_matching_entities = reconstruct_callouts(old_entities, settings)
            new_matching_entities = reconstruct_callouts(new_entities, settings)
            outcome = match_entities(
                old_matching_entities,
                new_matching_entities,
                transform,
                settings,
            )
            changes = _changes_from_match(
                outcome,
                old_matching_entities,
                new_matching_entities,
                transform,
                settings,
            )
            if alignment.note:
                notes.append(alignment.note)
            page_results.append(
                PageResult(
                    page_index=page_index,
                    old_width=float(old_page.rect.width) if old_page is not None else 0.0,
                    old_height=float(old_page.rect.height) if old_page is not None else 0.0,
                    new_width=float(new_page.rect.width) if new_page is not None else 0.0,
                    new_height=float(new_page.rect.height) if new_page is not None else 0.0,
                    alignment=alignment,
                    entity_count_old=len(old_entities),
                    entity_count_new=len(new_entities),
                    changes=changes,
                    affected_area_fraction=round(
                        _union_area(tuple(change.bbox for change in changes)), 8
                    ),
                    relevant_area_fraction=round(
                        _union_area(
                            tuple(change.bbox for change in changes if change.inspection_relevant)
                        ),
                        8,
                    ),
                    notes=tuple(notes),
                )
            )

        pages = tuple(page_results)
        _emit(progress, "complete", 1.0, "Comparison complete")
        return ComparisonResult(
            old_file=old_capabilities.path.name,
            new_file=new_capabilities.path.name,
            pages=pages,
            summary=_summary(pages),
            notes=tuple(overall_notes),
        )
    finally:
        old_document.close()
        new_document.close()
