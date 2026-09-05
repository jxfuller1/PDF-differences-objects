# Architecture

## Analysis boundary

The analysis boundary is intentionally strict:

```text
PDF display list / text layer ──> analysis
rendered pixels               -X-> analysis
```

`pymupdf.Page.get_drawings()` supplies path objects and
`pymupdf.Page.get_text("dict")` supplies positioned spans. The GUI later calls
`get_pixmap()` for a disposable human preview. No function in extraction,
alignment, matching, mechanical parsing, or comparison accepts image data.

Raster-only pages are rejected in `validation.py`. A searchable, text-only page
is accepted with an explicit warning because its text entities are still exact;
geometry changes on that page cannot be observed.

## Components

| Module | Responsibility |
| --- | --- |
| `validation.py` | PDF/header/size/encryption checks and per-page capability inventory |
| `extraction.py` | Canonical text and vector entities with stable hashes |
| `alignment.py` | Unique-anchor construction and deterministic robust similarity fit |
| `matching.py` | Exact, attribute, and structural one-to-one matching cascade |
| `comparison.py` | Page orchestration, change typing, traceability, and summaries |
| `mechanical.py` | Mechanical parser buckets and inspection-relevance reasons |
| `reporting.py` | JSON, CSV, and vector annotation exports |
| `ui/` | Background worker, synchronized PDF previews, filtering, and export workflow |

## Entity representation

Text and geometry share normalized `[0, 1]` page coordinates. Each entity has a
stable ID plus three different signatures:

- `content_signature`: content and style; used by exact matching;
- `shape_signature`: translation-invariant path geometry; used as an alignment
  anchor and a geometry-change test;
- `style_signature`: stroke/fill/width or font/color/size attributes.

Geometry also stores its PDF operation histogram, primitive count, path length,
bounds, and anchor. Text stores its exact payload, Unicode/whitespace-normalized
payload, font, relative font size, insertion anchor, and bounds.

Stable signatures and canonical sorting make repeated comparisons deterministic.
The code never assigns random UUIDs to extracted entities or changes.

## Alignment

Unique text payloads and unique translation-invariant geometry signatures create
old/new anchor pairs. When there are at least two pairs, the engine generates a
repeatable set of two-anchor transform hypotheses. Its deterministic cycle cover
guarantees that any strict majority of distinct-location anchors contributes an
all-inlier pair; additional pairs are sampled with a local fixed-seed generator.
Each hypothesis is scored by inlier count and residual, and the best consensus is
refined by a least-squares 2-D similarity fit.

Statuses are explicit:

- `aligned`: transform passed inlier and RMS thresholds;
- `translation-only`: only one unique anchor existed;
- `identity-unverified`: no unique anchor or a blank side;
- `not-applicable`: a page exists in only one revision;
- `failed`: populated sheets had anchors but no trustworthy registration; the
  comparison stops with an actionable error.

## Matching cascade

The matcher maintains one-to-one ownership across every tier.

### Tier 1 — exact

An index over entity kind and full content signature yields candidates. Only a
candidate inside the registered-position epsilon is accepted. Duplicates are
resolved nearest-first with stable ID tie-breaks.

### Tier 2 — attribute

A spatial tree exposes only same-kind candidates in a small in-place radius.
Text scores combine registered position, payload similarity, mechanical class,
font size, and style. Geometry scores combine registered position, aligned-box
overlap, operation histogram, aspect, and style. Accepted pairs use a stable
greedy one-to-one assignment.

### Tier 3 — structural

The remaining sparse candidate graph uses deterministic features rather than a
Siamese encoder:

- text similarity, mechanical category, font size, local entity context, and
  registered distance;
- path-operation overlap, aspect/area/path-length agreement, style, local entity
  context, and registered distance.

Each connected candidate component is solved with the Hungarian algorithm and
private dummy columns, so leaving an entity unmatched is cheaper than forcing a
below-threshold pair. Small components use a dense cost matrix; large components
use a sparse minimum-weight full bipartite solver with the same global objective,
which bounds memory without degrading to greedy assignment. Candidates outside
the structural radius are never paired, which is the moved-versus-removed/added
guard.

The returned numeric value is named a *similarity score*, not a statistical
confidence: it is a transparent heuristic, not a calibrated probability.

## Change interpretation

Unmatched old/new entities become removals/additions. A non-exact pair is field-
diffed for position, text, text style, geometry, and drawing style. A pure local
translation becomes `moved`; any payload/shape/style change becomes `modified`
and can also retain `position` in `modification_kinds`.

The mechanical parser then examines changed text, title-block position, nearby
labels, and nearby annotations. It assigns one parser category and an auditable
inspection-relevance decision. Geometry adjacent to a recognized dimension or
GD&T annotation inherits that category.

## Multi-page behavior

Pages are paired by zero-based index and processed independently. Extra pages
are treated as entirely added or removed. Each page retains its own transform,
entity counts, notes, and exact dimensions. Reordered pages are not inferred.

## Desktop threading and display

The PyQt6 main window owns a `QThread` worker. Progress is emitted only at safe
stage boundaries; cancellation is checked between validation, extraction,
alignment, and matching stages. Selecting a change synchronizes both page
previews and focuses its old/new bounding boxes. Preview documents are opened,
rendered, and closed per load so file handles do not leak.

## Outputs

All outputs derive from the same immutable `ComparisonResult`:

- JSON preserves the complete nested page/change schema;
- CSV flattens one row per change, including relevance reason and match tier;
- the annotated PDF draws colored vector rectangles on a copy of the new PDF.

Color convention: green added, red removed, orange moved, and blue modified.
