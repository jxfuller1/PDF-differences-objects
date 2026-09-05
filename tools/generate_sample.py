"""Generate a small, fully vector-native mechanical drawing comparison pair."""

from __future__ import annotations

import argparse
from pathlib import Path

import pymupdf as fitz


def _drawing(path: Path, *, revised: bool) -> None:
    document = fitz.open()
    page = document.new_page(width=792, height=612)

    # Border, title block, and a simplified machined plate.
    page.draw_rect(fitz.Rect(24, 24, 768, 588), width=1.0)
    page.draw_rect(fitz.Rect(500, 480, 768, 588), width=1.0)
    page.draw_line(fitz.Point(500, 520), fitz.Point(768, 520), width=0.7)
    page.draw_line(fitz.Point(620, 480), fitz.Point(620, 588), width=0.7)
    outline_right = 445 if revised else 430
    page.draw_rect(fitz.Rect(155, 145, outline_right, 345), width=1.8)
    page.draw_circle(fitz.Point(220, 245), 28, width=1.4)
    page.draw_circle(fitz.Point(365, 245), 28, width=1.4)
    if revised:
        page.draw_circle(fitz.Point(292, 245), 18, width=1.4)

    page.insert_text(fitz.Point(160, 112), "PLATE, MOUNTING", fontsize=16)
    page.insert_text(
        fitz.Point(160, 382), "WIDTH 55.0 ±0.1 mm" if revised else "WIDTH 50.0 ±0.1 mm", fontsize=11
    )
    page.insert_text(
        fitz.Point(160, 410),
        "POSITION | DIA 0.10 | A | B" if revised else "POSITION | DIA 0.20 | A | B",
        fontsize=10,
    )
    page.insert_text(
        fitz.Point(160, 438),
        "NOTE: SURFACE FINISH 32 Ra" if revised else "NOTE: SURFACE FINISH 63 Ra",
        fontsize=10,
    )
    page.insert_text(fitz.Point(510, 502), "PART: PD-1001", fontsize=10)
    page.insert_text(fitz.Point(510, 542), "MATERIAL: 6061-T6", fontsize=9)
    page.insert_text(fitz.Point(634, 542), "REVISION", fontsize=9)
    page.insert_text(fitz.Point(675, 570), "B" if revised else "A", fontsize=12)

    document.set_metadata(
        {
            "title": "PDF Differences Objects vector sample",
            "subject": "Synthetic mechanical drawing; no raster or OCR content",
        }
    )
    document.save(path, deflate=True)
    document.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, nargs="?", default=Path("samples/mechanical_pair"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _drawing(args.output / "baseline.pdf", revised=False)
    _drawing(args.output / "revision.pdf", revised=True)
    print(f"Wrote vector sample pair to {args.output.resolve()}")


if __name__ == "__main__":
    main()
