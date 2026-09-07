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
    # Small, textless line-built CAD-font glyphs (diameter, degree, datum
    # symbols, and similar marks) repeat throughout drawings and have no safe
    # identity when detached from a reconstructed callout. Keep their loose
    # move matching disabled; exact and in-place attribute matching remain.
    structural_glyph_max_extent: float = 0.035
    structural_glyph_min_primitives: int = 12
    # Above this component size, sparse min-cost flow bounds assignment memory.
    # Dense NumPy-backed Hungarian assignment is faster below the threshold.
    sparse_assignment_threshold: int = 1200

    # Semantic callout reconstruction. Spatial tests only nominate candidates;
    # dimension grammar, frame containment, or leader attachment must approve
    # every group.
    callout_inline_gap_factor: float = 2.75
    callout_baseline_tolerance_factor: float = 0.55
    callout_stacked_gap_factor: float = 2.25
    callout_tolerance_pair_alignment_factor: float = 0.75
    callout_limit_width_ratio: float = 0.75
    callout_hypothesis_ambiguity_margin: float = 0.20
    callout_frame_edge_tolerance: float = 0.003
    # Much tighter than frame containment: this only reconciles coordinate
    # noise while rebuilding local line topology from CAD drawing records.
    callout_segment_connect_tolerance: float = 0.0002
    callout_min_frame_height: float = 0.003
    callout_min_frame_cell_width: float = 0.002
    callout_max_frame_height: float = 0.08
    callout_max_frame_width: float = 0.55
    callout_max_frame_cells: int = 12
    callout_vector_feature_min_fill: float = 0.15
    callout_vector_feature_max_fill: float = 0.92
    callout_vector_feature_max_center_offset: float = 0.25
    callout_leader_touch_factor: float = 1.25
    callout_attachment_ambiguity_factor: float = 0.35
    callout_attachment_match_tolerance: float = 0.035
    callout_match_ambiguity_margin: float = 0.055

    # Text/geometry interpretation.
    nearby_annotation_radius: float = 0.035
    note_letter_threshold: int = 15
    title_block_x_min: float = 0.62
    title_block_y_min: float = 0.76
    preview_dpi: int = 125

    # Validation heuristic for effectively scanned pages. A page is rejected
    # when one embedded image dominates the page area and the remaining
    # structured content is sparse or occupies too little of the page.
    scanned_page_min_image_coverage: float = 0.85
    scanned_page_max_structured_entities: int = 5
    scanned_page_max_structured_coverage: float = 0.02


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
