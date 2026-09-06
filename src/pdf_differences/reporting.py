"""Deterministic JSON/CSV exports and vector markup PDF generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pymupdf as fitz

from pdf_differences.models import BBox, Change, ChangeType, ComparisonResult

_COLORS: dict[ChangeType, tuple[float, float, float]] = {
    ChangeType.ADDED: (0.12, 0.63, 0.29),
    ChangeType.REMOVED: (0.86, 0.18, 0.18),
    ChangeType.MOVED: (0.95, 0.58, 0.10),
    ChangeType.MODIFIED: (0.10, 0.42, 0.89),
}


def _prepare_output(path: str | Path, overwrite: bool) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {destination}")
    return destination


def export_json(result: ComparisonResult, path: str | Path, *, overwrite: bool = False) -> Path:
    destination = _prepare_output(path, overwrite)
    destination.write_text(
        json.dumps(result.to_dict(), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def export_csv(result: ComparisonResult, path: str | Path, *, overwrite: bool = False) -> Path:
    destination = _prepare_output(path, overwrite)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "page",
                "change_type",
                "category",
                "inspection_relevant",
                "relevance_reason",
                "label",
                "detail",
                "match_tier",
                "similarity_score",
                "bbox_x0",
                "bbox_y0",
                "bbox_x1",
                "bbox_y1",
                "old_entity_id",
                "new_entity_id",
                "old_member_entity_ids",
                "new_member_entity_ids",
            )
        )
        for change in result.changes:
            writer.writerow(
                (
                    change.page_index + 1,
                    change.change_type.value,
                    change.category.value,
                    change.inspection_relevant,
                    change.relevance_reason,
                    change.label,
                    change.detail,
                    change.match_tier.value if change.match_tier else "",
                    change.similarity_score if change.similarity_score is not None else "",
                    *change.bbox,
                    change.old_entity_id or "",
                    change.new_entity_id or "",
                    "|".join(change.old_member_entity_ids),
                    "|".join(change.new_member_entity_ids),
                )
            )
    return destination


def _page_rect(bbox: BBox, page: fitz.Page) -> fitz.Rect:
    return fitz.Rect(
        bbox[0] * page.rect.width,
        bbox[1] * page.rect.height,
        bbox[2] * page.rect.width,
        bbox[3] * page.rect.height,
    )


def _draw_change(page: fitz.Page, change: Change) -> None:
    color = _COLORS[change.change_type]
    rectangle = _page_rect(change.bbox, page)
    pad = max(2.0, min(page.rect.width, page.rect.height) * 0.0025)
    rectangle = fitz.Rect(
        max(page.rect.x0, rectangle.x0 - pad),
        max(page.rect.y0, rectangle.y0 - pad),
        min(page.rect.x1, rectangle.x1 + pad),
        min(page.rect.y1, rectangle.y1 + pad),
    )
    if rectangle.width < pad * 2:
        rectangle.x1 = min(page.rect.x1, rectangle.x0 + pad * 2)
    if rectangle.height < pad * 2:
        rectangle.y1 = min(page.rect.y1, rectangle.y0 + pad * 2)
    page.draw_rect(
        rectangle,
        color=color,
        fill=color,
        fill_opacity=0.10,
        width=max(0.8, page.rect.height / 800.0),
        overlay=True,
    )
    label = f"{change.change_type.value.upper()} · {change.category.value}"
    label_y = max(page.rect.y0 + 8.0, rectangle.y0 - 2.0)
    page.insert_text(
        fitz.Point(rectangle.x0, label_y),
        label,
        fontsize=max(6.0, min(10.0, page.rect.height / 80.0)),
        color=color,
        overlay=True,
    )


def export_annotated_pdf(
    result: ComparisonResult,
    new_pdf: str | Path,
    path: str | Path,
    *,
    relevant_only: bool = False,
    overwrite: bool = False,
) -> Path:
    """Overlay vector annotations on a copy of the new revision.

    Rendering is for human review only. The analysis that produced ``result``
    has already completed entirely from structured PDF entities.
    """

    source = Path(new_pdf).expanduser().resolve()
    destination = _prepare_output(path, overwrite)
    if source == destination:
        raise ValueError("Annotated output must not overwrite the source PDF.")
    document = fitz.open(source)
    try:
        for change in result.changes:
            if relevant_only and not change.inspection_relevant:
                continue
            if change.page_index >= document.page_count:
                continue  # old-only pages do not exist in the new revision copy
            _draw_change(document.load_page(change.page_index), change)
        metadata = document.metadata or {}
        metadata["subject"] = "Structured revision markup generated by PDF Differences Objects"
        document.set_metadata(metadata)
        document.save(destination, garbage=4, deflate=True)
    finally:
        document.close()
    return destination
