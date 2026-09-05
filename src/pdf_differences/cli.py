"""Headless command-line entry point for batch and CI usage."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pdf_differences.comparison import compare_pdfs
from pdf_differences.errors import PdfDifferencesError
from pdf_differences.reporting import export_annotated_pdf, export_csv, export_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdf-differences-cli",
        description="Compare vector/text-layer CAD PDFs without OCR, pixels, or PyTorch.",
    )
    parser.add_argument("old_pdf", type=Path, help="Previous drawing revision")
    parser.add_argument("new_pdf", type=Path, help="New drawing revision")
    parser.add_argument("--json", type=Path, dest="json_path", help="Write structured JSON")
    parser.add_argument("--csv", type=Path, dest="csv_path", help="Write a flat change list")
    parser.add_argument("--annotated", type=Path, help="Write a marked-up copy of the new PDF")
    parser.add_argument(
        "--relevant-only",
        action="store_true",
        help="Limit PDF markup to inspection-relevant changes",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing output files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    def progress(event) -> None:
        print(f"[{event.stage}] {event.message}", file=sys.stderr)

    try:
        result = compare_pdfs(args.old_pdf, args.new_pdf, progress=progress)
        if args.json_path:
            export_json(result, args.json_path, overwrite=args.force)
        if args.csv_path:
            export_csv(result, args.csv_path, overwrite=args.force)
        if args.annotated:
            export_annotated_pdf(
                result,
                args.new_pdf,
                args.annotated,
                relevant_only=args.relevant_only,
                overwrite=args.force,
            )
    except (PdfDifferencesError, FileExistsError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(result.summary)
    for note in result.notes:
        print(f"note: {note}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
