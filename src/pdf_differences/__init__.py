"""Vector-native mechanical drawing comparison.

This is a substantially modified CaD-Track derivative. See ``NOTICE``.
"""

from pdf_differences.comparison import compare_pdfs
from pdf_differences.models import ComparisonResult

__all__ = ["ComparisonResult", "compare_pdfs"]
__version__ = "0.2.0"
