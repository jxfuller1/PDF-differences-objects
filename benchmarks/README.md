# Matching benchmark

`compare_scipy.py` keeps the application's former SciPy operations as an
optional oracle. SciPy is not imported by, or installed with, the production
package.

Install the benchmark dependencies and run the committed PDF pair:

```powershell
python -m pip install -e ".[dev,benchmark]"
python -m benchmarks.compare_scipy --repeat 7
```

Supply any number of real-world pairs as `LABEL OLD_PDF NEW_PDF` triples:

```powershell
python -m benchmarks.compare_scipy --repeat 7 `
  --pair drawing-a C:\drawings\old-a.pdf C:\drawings\new-a.pdf `
  --pair drawing-b C:\drawings\old-b.pdf C:\drawings\new-b.pdf
```

Add `--json` for machine-readable results. Each page reports median native and
SciPy matching time after an untimed warm-up. PDF validation, extraction, and
alignment are intentionally outside the timed region. `matching_equal` compares
entity IDs, tiers, scores, distances, and unmatched IDs. `detection_equal`
compares the change IDs, types, categories, relevance, regions, entity IDs,
tiers, and scores produced from those matches. JSON output also records the
Python, NumPy, SciPy, and platform versions for reproducibility.

The process exits with status 1 if any page differs, making the command useful
as a regression check. Equal-cost tie assignments are not a stable SciPy API
guarantee, so synthetic solver tests compare objective values when ties are
deliberately present.

The report also records both large-component cutoffs. The retained SciPy oracle
uses the former cutoff of 300 entities. The native backend uses its current
configured cutoff (1,200 by default), because its vectorized dense solver is
faster at those intermediate sizes while still using bounded-memory sparse flow
for unusually large components. Both paths optimize the same objective.
