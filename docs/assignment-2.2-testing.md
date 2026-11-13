# Assignment 2.2 - Batch classifier testing

## Scope

Implemented functionality includes shared validation, label encoding,
Gini and entropy, vectorised split search, recursive batch fitting,
batch labels and class probabilities.

Buffered partial_fit and the remaining streaming components are pending.

## Environment

- Python 3.12.6
- NumPy 1.26.4
- Project interpreter: .venv/Scripts/python.exe

## Test coverage

- Shape, type, parameter and missing-value validation.
- Integer and string labels, including large integers.
- Hand-calculated impurity and split examples.
- Independent exhaustive split-search comparison.
- Depth, sample-count, purity and gain stopping rules.
- Seeded feature selection and global RNG isolation.
- Failed-fit recovery and successful model replacement.
- Batch prediction versus scalar traversal.
- Class-probability alignment, empty inputs and output ownership.
- Cross-dtype threshold comparisons and numeric extremes.
- Training-row permutations and chunked prediction equivalence.
- Read-only, noncontiguous inputs and conflicting duplicate observations.

## Verified results

- User checkout before the final boundary additions:
  90 streaming-package tests passed in 0.440 seconds.
- Temporary verification copy with six additional boundary tests:
  497 original tests plus 96 streaming-package tests;
  593 tests passed in 0.792 seconds.
- Final full regression in the user checkout:
  593 tests passed in 0.677 seconds on 2026-11-13,
  comprising 497 original tests and 96 streaming-package tests.
- Commit and push: pending.

The temporary-copy result is separate from final checkout verification.
Test execution times are not performance benchmark results.

## Reproduction

Streaming-package tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests/stream -t . -v

Full regression, including original tests:

    .\.venv\Scripts\python.exe -m unittest discover -s tests -v

The original integration tests require demo/sample_data.csv.

## Known limits

- No buffered incremental training has been implemented yet.
- Recursive construction remains subject to Python's recursion limit.
- Split evaluation uses temporary arrays proportional to input size.
- Numerical-zero and gain-tie tolerance is 1e-12.
- Benchmark speedups and streaming performance have not been measured.