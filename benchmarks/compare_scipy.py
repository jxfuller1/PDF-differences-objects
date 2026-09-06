"""Compare native matching output and timing with the retained SciPy oracle."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymupdf as fitz

from benchmarks.scipy_backend import (
    LEGACY_SCIPY_SPARSE_THRESHOLD,
    SCIPY_MATCHING_BACKEND,
    SCIPY_VERSION,
)
from pdf_differences.alignment import estimate_alignment
from pdf_differences.callouts import reconstruct_callouts
from pdf_differences.comparison import _changes_from_match
from pdf_differences.config import SETTINGS
from pdf_differences.extraction import extract_page_entities
from pdf_differences.matching import match_entities
from pdf_differences.matching_algorithms import NATIVE_MATCHING_BACKEND, MatchingBackend
from pdf_differences.models import Entity, MatchResult, Transform


@dataclass(frozen=True, slots=True)
class PageInput:
    page_index: int
    old_entities: tuple[Entity, ...]
    new_entities: tuple[Entity, ...]
    transform: Transform
    alignment_status: str


def _page_inputs(old_path: Path, new_path: Path) -> tuple[PageInput, ...]:
    old_document = fitz.open(old_path)
    new_document = fitz.open(new_path)
    try:
        pages: list[PageInput] = []
        for page_index in range(max(old_document.page_count, new_document.page_count)):
            raw_old_entities = (
                extract_page_entities(old_document.load_page(page_index), page_index)
                if page_index < old_document.page_count
                else ()
            )
            raw_new_entities = (
                extract_page_entities(new_document.load_page(page_index), page_index)
                if page_index < new_document.page_count
                else ()
            )
            if raw_old_entities and raw_new_entities:
                alignment = estimate_alignment(raw_old_entities, raw_new_entities)
                if alignment.status == "failed":
                    raise RuntimeError(
                        f"Page {page_index + 1} of {old_path.name}/{new_path.name} "
                        "did not pass alignment"
                    )
                transform = alignment.transform
                status = alignment.status
            else:
                transform = Transform()
                status = "not-applicable"
            old_entities = reconstruct_callouts(raw_old_entities, SETTINGS)
            new_entities = reconstruct_callouts(raw_new_entities, SETTINGS)
            pages.append(
                PageInput(
                    page_index,
                    old_entities,
                    new_entities,
                    transform,
                    status,
                )
            )
        return tuple(pages)
    finally:
        old_document.close()
        new_document.close()


def _timed_match(
    page: PageInput, backend: MatchingBackend, repeat: int
) -> tuple[MatchResult, list[float]]:
    result = match_entities(
        page.old_entities,
        page.new_entities,
        page.transform,
        SETTINGS,
        backend=backend,
    )
    samples: list[float] = []
    for _ in range(repeat):
        started = time.perf_counter()
        result = match_entities(
            page.old_entities,
            page.new_entities,
            page.transform,
            SETTINGS,
            backend=backend,
        )
        samples.append(time.perf_counter() - started)
    return result, samples


def _match_signature(result: MatchResult) -> tuple[object, ...]:
    return (
        tuple(
            (
                match.old.id,
                match.new.id,
                match.tier.value,
                match.score,
                round(match.registered_distance, 12),
            )
            for match in result.matches
        ),
        tuple(entity.id for entity in result.unmatched_old),
        tuple(entity.id for entity in result.unmatched_new),
    )


def _detection_signature(page: PageInput, result: MatchResult) -> tuple[object, ...]:
    changes = _changes_from_match(
        result,
        page.old_entities,
        page.new_entities,
        page.transform,
        SETTINGS,
    )
    return tuple(
        (
            change.id,
            change.change_type.value,
            change.category.value,
            change.inspection_relevant,
            tuple(round(value, 12) for value in change.bbox),
            change.old_entity_id,
            change.new_entity_id,
            change.match_tier.value if change.match_tier else None,
            change.similarity_score,
        )
        for change in changes
    )


def compare_pair(label: str, old_path: Path, new_path: Path, repeat: int) -> dict[str, object]:
    page_reports: list[dict[str, object]] = []
    for page in _page_inputs(old_path, new_path):
        native_result, native_times = _timed_match(page, NATIVE_MATCHING_BACKEND, repeat)
        scipy_result, scipy_times = _timed_match(page, SCIPY_MATCHING_BACKEND, repeat)
        native_median = statistics.median(native_times)
        scipy_median = statistics.median(scipy_times)
        native_match_signature = _match_signature(native_result)
        scipy_match_signature = _match_signature(scipy_result)
        native_detection_signature = _detection_signature(page, native_result)
        scipy_detection_signature = _detection_signature(page, scipy_result)
        page_reports.append(
            {
                "page": page.page_index + 1,
                "alignment": page.alignment_status,
                "old_entities": len(page.old_entities),
                "new_entities": len(page.new_entities),
                "native_median_seconds": native_median,
                "scipy_median_seconds": scipy_median,
                "native_to_scipy_ratio": (native_median / scipy_median if scipy_median else None),
                "matching_equal": native_match_signature == scipy_match_signature,
                "detection_equal": native_detection_signature == scipy_detection_signature,
                "native_match_count": len(native_result.matches),
                "scipy_match_count": len(scipy_result.matches),
                "native_change_count": len(native_detection_signature),
                "scipy_change_count": len(scipy_detection_signature),
            }
        )
    return {
        "label": label,
        "old": str(old_path.resolve()),
        "new": str(new_path.resolve()),
        "repeat": repeat,
        "environment": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": np.__version__,
            "scipy": SCIPY_VERSION,
            "platform": sys.platform,
        },
        "native_sparse_threshold": SETTINGS.sparse_assignment_threshold,
        "legacy_scipy_sparse_threshold": LEGACY_SCIPY_SPARSE_THRESHOLD,
        "pages": page_reports,
        "all_matching_equal": all(page["matching_equal"] for page in page_reports),
        "all_detection_equal": all(page["detection_equal"] for page in page_reports),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        nargs=3,
        action="append",
        metavar=("LABEL", "OLD_PDF", "NEW_PDF"),
        help="PDF pair to compare; may be supplied more than once",
    )
    parser.add_argument("--repeat", type=int, default=7, help="timed runs per page/backend")
    parser.add_argument("--json", action="store_true", help="emit complete JSON output")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    root = Path(__file__).resolve().parents[1]
    pairs = arguments.pair or [
        (
            "sample",
            str(root / "samples" / "mechanical_pair" / "baseline.pdf"),
            str(root / "samples" / "mechanical_pair" / "revision.pdf"),
        )
    ]
    reports = [
        compare_pair(label, Path(old_path), Path(new_path), arguments.repeat)
        for label, old_path, new_path in pairs
    ]
    if arguments.json:
        print(json.dumps(reports, indent=2, sort_keys=True))
    else:
        for report in reports:
            print(f"{report['label']}: {report['old']} -> {report['new']}")
            for page in report["pages"]:
                ratio = page["native_to_scipy_ratio"]
                ratio_text = f"{ratio:.2f}x" if ratio is not None else "n/a"
                print(
                    f"  page {page['page']}: {page['old_entities']} -> "
                    f"{page['new_entities']} entities, native "
                    f"{page['native_median_seconds'] * 1000:.3f} ms, SciPy "
                    f"{page['scipy_median_seconds'] * 1000:.3f} ms ({ratio_text}), "
                    f"matching={'equal' if page['matching_equal'] else 'DIFFERENT'}, "
                    f"detection={'equal' if page['detection_equal'] else 'DIFFERENT'}"
                )
    return (
        0
        if all(report["all_matching_equal"] and report["all_detection_equal"] for report in reports)
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
