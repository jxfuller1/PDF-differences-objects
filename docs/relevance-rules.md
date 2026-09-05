# Mechanical parser and inspection relevance

The relevance layer is intentionally deterministic. It makes no claim to infer
engineering intent, and every decision includes a human-readable reason in the
UI and exports.

| Category | Typical evidence | Default relevance |
| --- | --- | --- |
| `DIMENSION` | measurement, diameter/radius, units, plus/minus, min/max, reference, typical, or through modifier | Relevant |
| `GD&T` | common Unicode symbols or words such as position, flatness, datum, profile, runout, MMC/LMC | Relevant |
| `NOTE` | `NOTE`, general-note phrasing, material/finish/weld/deburr language | Relevant only with an inspection-impact keyword |
| `REVISION` | REV/REVISION/ECN/ECO text or a nearby title-block revision cell | Ignored if administrative; relevant when technical inspection words occur |
| `GEOMETRY` | vector path added, removed, reshaped, or moved | Relevant |
| `OTHER` | text that matches none of the rules | Ignored pending human review |

The configured inspection-impact vocabulary lives in
`src/pdf_differences/config.py`. It includes measurement and verification terms,
material/finish/surface/hardness/coating requirements, common GD&T concepts,
weld/thread/torque/cleanliness language, and similar cues.

## Context rules

- A short title-block value such as `A -> B` becomes `REVISION` only when it is
  near a `REV` or `REVISION` label.
- A changed geometry path near a recognized dimension or GD&T span inherits
  that annotation category.
- Revision letters, dates, drawing/checking approvals, and issue metadata stay
  visible in the complete change report even when the relevance filter ignores
  them.

## Tuning

Tune keyword sets and spatial thresholds only with representative labeled
drawing pairs. A useful calibration set should include:

1. true dimension, tolerance, datum, material, finish, and geometry changes;
2. revision-table edits that do and do not affect inspection;
3. repeated symbols and dimensions near one another;
4. custom CAD fonts and their extracted Unicode payloads;
5. global export shifts, scale changes, and small rotations;
6. deliberately unrelated sheets that must be declined.

Measure both missed relevant changes and noisy false positives. The UI's full
report and explicit reasons are designed to make that review practical.

## Safety posture

Inspection relevance is triage, not authorization to accept, reject, release,
or manufacture a part. Keep a qualified drawing reviewer in the loop, especially
for custom fonts, company-specific notation, crowded annotation fields, and
changes classified as `OTHER`.
