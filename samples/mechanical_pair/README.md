# Vector mechanical sample

Both PDFs are generated directly with vector paths and positioned text. The
revision demonstrates:

- a plate-width geometry edit;
- an added center hole;
- a `50.0` to `55.0 mm` dimension edit;
- a position-tolerance edit;
- a surface-finish note edit;
- an administrative `REVISION A` to `B` edit.

The first five are inspection-relevant under the default rules. The revision
letter remains in the complete report but is ignored by the relevance filter.

Regenerate both files from the repository root:

```powershell
python tools/generate_sample.py
```
