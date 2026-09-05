"""User-editable appearance and animation settings for the PDF overlay viewer."""

from __future__ import annotations

from PyQt6.QtCore import QEasingCurve

# Page tint colors used between the Old and New slider endpoints.
OLD_PAGE_TINT_RGB = (220, 35, 35)
NEW_PAGE_TINT_RGB = (25, 90, 230)

# Change-region colors. These are intentionally independent of the page tints.
REMOVED_REGION_RGB = (220, 35, 35)
ADDED_REGION_RGB = (25, 90, 230)
OTHER_REGION_RGB = (170, 80, 180)

# One complete 0 -> peak -> 0 blink cycle, in milliseconds.
BLINK_DURATION_MS = 1100
BLINK_START_STRENGTH = 0.0
BLINK_PEAK_POSITION = 0.5
BLINK_PEAK_STRENGTH = 1.0
BLINK_END_STRENGTH = 0.0
BLINK_EASING_CURVE = QEasingCurve.Type.InOutSine
BLINK_LOOP_COUNT = -1  # -1 repeats until the Blink checkbox is cleared.

# Region border and fill values at the dim and bright points of each blink.
# Alpha values use Qt's 0 (transparent) through 255 (opaque) range.
REGION_BORDER_ALPHA_MIN = 110
REGION_BORDER_ALPHA_MAX = 255
REGION_FILL_ALPHA_MIN = 16
REGION_FILL_ALPHA_MAX = 70
REGION_BORDER_WIDTH_MIN = 1.5
REGION_BORDER_WIDTH_MAX = 3.5

# Extra emphasis applied to the region selected from either the table or viewer.
SELECTED_REGION_BORDER_ALPHA = 255
SELECTED_REGION_BORDER_WIDTH_BONUS = 1.5
SELECTED_REGION_FILL_ALPHA_MIN = 72
