"""User-facing comparison errors."""


class PdfDifferencesError(Exception):
    """Base exception for expected, displayable failures."""


class PdfValidationError(PdfDifferencesError):
    """The selected input cannot be analyzed safely."""


class RasterPdfError(PdfValidationError):
    """A page has only raster content and no supported structured entities."""


class AlignmentError(PdfDifferencesError):
    """Two populated pages cannot be registered reliably."""


class ComparisonCancelled(PdfDifferencesError):
    """A desktop comparison was cancelled between processing stages."""
