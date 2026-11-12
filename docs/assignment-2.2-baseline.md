# Assignment 2.2 - Baseline and source review

## Repository arrangement

Assignment 2.2 may be completed on a personal
branch of the original repository; a separate fork is not required.

- Repository: https://github.com/Jiahe-SHI/COMP5004-Assignment-2.1-Programming-Task-1
- Development branch: assignment-2.2
- Upstream branch: origin/assignment-2.2
- Original main baseline: 5fe1d3f2b3c3a0c4e74abe62030e5c1098b31425
- Baseline commit subject: Update README
- Keep Assignment 2.2 changes on the personal branch.

## Verified original baseline

Evidence: recorded local environment and regression-test output.

- Python: 3.12.6
- NumPy: 1.26.4
- NumCompute: 0.1.0
- Interpreter: this checkout's .venv/Scripts/python.exe
- Package import: this checkout's numcompute/__init__.py
- pip check: No broken requirements found.
- Test command: python -m unittest discover -s tests -v
- Result: 497 tests, 0.609 seconds, OK.
- Working tree after tests: clean.

This is the original package baseline, not evidence that the new
streaming functionality has been implemented.

## Source review

| Module | Reusable foundation | Required changes |
| --- | --- | --- |
| io | True chunk iteration, final partial chunk, independent arrays | CSV parsing with quoted fields; aligned feature/target extraction |
| preprocessing | Shape validation, independent outputs, constant-column handling | Incremental state, mean imputation, vectorised category encoding |
| stats | Batch-merge count/mean/M2; vectorised histogram counting | Per-feature state, cumulative quantiles and histogram, update_stats |
| metrics | Classification formulas and validation policies | Vectorised label mapping/counting; cumulative/rolling state; AUC |
| pipeline | Named-step validation, composition, duplicate-object detection | Compatible signatures, single buffer owner, paired snapshots |

Specific findings:

- OneHotEncoder validates, discovers and encodes categories using
  sample-level Python loops.
- confusion_matrix discovers labels using list membership and increments
  counts in a loop over sample pairs.
- Pipeline and FeatureUnion pass y positionally to fitting methods,
  rejecting compatible external methods that accept only X.
- Existing StreamingStats accepts one-dimensional batches.
- Existing Pipeline changes component state in place and does not roll
  back child state after failure.

Review covered the current local source. No performance measurements
were made during this review.

## Package skeleton

- numcompute_stream/: module placeholders, without implemented APIs.
- tests/stream/: separate discovery location for new tests.
- docs/assignment-2.2-design.md: initial design contract.

The existing setuptools include pattern, numcompute*, matches both
numcompute and numcompute_stream. Verify the editable installation after
creating the new package.

## Verification commands

Run from the repository root using the project virtual environment.

Original regression suite, including any discoverable new tests:
    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

Assignment 2.2 tests only:
    .\.venv\Scripts\python.exe -m unittest discover -s tests/stream -t . -v

At the skeleton stage the new suite contains zero tests. This is expected
and is not evidence of functional correctness.

## Verification status

- Repository branch and original baseline recorded.
- Environment and dependency checks passed.
- Source review completed; initial design contract recorded.
- Editable installation refreshed successfully.
- All scaffold modules imported from this checkout.
- New test discovery: 0 tests; no functional tests implemented yet.
- Original regression suite after scaffolding: 497 tests, 0.568 seconds, OK.
- Streaming functionality remains unimplemented.
