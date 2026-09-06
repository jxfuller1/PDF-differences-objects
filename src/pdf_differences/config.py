"""Central, deterministic comparison thresholds and inspection rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ComparisonSettings:
    """Tunable values expressed in normalized page coordinates."""

    max_file_size_mb: int = 100

    # Alignment. Coordinates are normalized, so a page diagonal is sqrt(2).
    alignment_inlier_tolerance: float = 0.006
    alignment_min_inlier_ratio: float = 0.55
    alignment_max_rms: float = 0.004
    alignment_max_hypotheses: int = 400
    # Reject otherwise-consistent transforms that move the normalized page
    # origin implausibly far from the other page's top-left corner.
    alignment_max_origin_shift: float = 0.15

    # CADMorph-inspired deterministic matching cascade.
    exact_position_tolerance: float = 0.0025
    attribute_position_tolerance: float = 0.018
    structural_search_radius: float = 0.16
    moved_tolerance: float = 0.006
    attribute_min_score: float = 0.58
    structural_min_score: float = 0.68
    # Dense Hungarian matrices are convenient for small ambiguity groups. Larger
    # groups use SciPy's sparse global bipartite solver with the same objective.
    sparse_assignment_threshold: int = 300

    # Text/geometry interpretation.
    nearby_annotation_radius: float = 0.035
    title_block_x_min: float = 0.62
    title_block_y_min: float = 0.76
    preview_dpi: int = 125


SETTINGS = ComparisonSettings()


INSPECTION_KEYWORDS = frozenset(
    {
        "ACCEPT",
        "CHAMFER",
        "CHECK",
        "CLEAN",
        "COATING",
        "CRITICAL",
        "DIAMETER",
        "FINISH",
        "FLATNESS",
        "HARDNESS",
        "INSPECT",
        "MATERIAL",
        "MEASURE",
        "PARALLEL",
        "PERPENDICULAR",
        "POSITION",
        "PROFILE",
        "ROUGHNESS",
        "SURFACE",
        "THREAD",
        "TOLERANCE",
        "TORQUE",
        "VERIFY",
        "WELD",
    }
)

ADMIN_REVISION_KEYWORDS = frozenset(
    {"APPROVED", "CHECKED", "DATE", "DRAWN", "ISSUED", "RELEASED", "REV", "REVISION"}
)
