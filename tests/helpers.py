from __future__ import annotations

from pathlib import Path

import pymupdf as fitz


def make_drawing_pdf(
    path: Path,
    *,
    offset: tuple[float, float] = (0.0, 0.0),
    dimension: str = "10.0 ±0.1 mm",
    extra_line: bool = False,
    pages: int = 1,
) -> Path:
    document = fitz.open()
    for page_index in range(pages):
        page = document.new_page(width=600, height=400)
        dx, dy = offset
        page.draw_rect(fitz.Rect(80 + dx, 70 + dy, 280 + dx, 230 + dy), width=1.5)
        page.draw_line(
            fitz.Point(330 + dx, 100 + dy),
            fitz.Point(445 + dx, 185 + dy),
            width=1.0,
        )
        page.insert_text(fitz.Point(95 + dx, 285 + dy), dimension, fontsize=12)
        page.insert_text(fitz.Point(390 + dx, 340 + dy), f"PART A{page_index + 1}", fontsize=10)
        if extra_line:
            page.draw_line(
                fitz.Point(470 + dx, 260 + dy),
                fitz.Point(550 + dx, 330 + dy),
                width=1.5,
            )
    document.save(path)
    document.close()
    return path


def make_text_only_pdf(path: Path) -> Path:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    page.insert_text(fitz.Point(40, 60), "GENERAL NOTE", fontsize=12)
    document.save(path)
    document.close()
    return path


def make_raster_only_pdf(
    path: Path,
    *,
    image_rect: tuple[float, float, float, float] | None = None,
) -> Path:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
    pixmap.clear_with(225)
    page.insert_image(fitz.Rect(image_rect) if image_rect is not None else page.rect, pixmap=pixmap)
    document.save(path)
    document.close()
    return path


def make_text_layer_over_image_pdf(path: Path, text: str = "SEARCHABLE NOTE") -> Path:
    return make_image_dominant_pdf(path, overlay_texts=((30, 40, text, 10),))


def make_hidden_text_over_image_pdf(path: Path, text: str = "HIDDEN NOTE") -> Path:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
    pixmap.clear_with(225)
    page.insert_image(page.rect, pixmap=pixmap)
    page.insert_text(fitz.Point(30, 40), text, fontsize=10, render_mode=3)
    document.save(path)
    document.close()
    return path


def make_image_dominant_pdf(
    path: Path,
    *,
    image_rect: tuple[float, float, float, float] | None = None,
    overlay_texts: tuple[tuple[float, float, str, float], ...] = (),
    overlay_rects: tuple[tuple[float, float, float, float], ...] = (),
    overlay_lines: tuple[tuple[float, float, float, float], ...] = (),
) -> Path:
    document = fitz.open()
    page = document.new_page(width=300, height=200)
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 8, 8), False)
    pixmap.clear_with(225)
    page.insert_image(fitz.Rect(image_rect) if image_rect is not None else page.rect, pixmap=pixmap)
    for left, top, right, bottom in overlay_rects:
        page.draw_rect(fitz.Rect(left, top, right, bottom), width=1.25)
    for x0, y0, x1, y1 in overlay_lines:
        page.draw_line(fitz.Point(x0, y0), fitz.Point(x1, y1), width=0.25)
    for x, y, text, fontsize in overlay_texts:
        page.insert_text(fitz.Point(x, y), text, fontsize=fontsize)
    document.save(path)
    document.close()
    return path
