# Assignment 2.2 - Decision tree testing

## Scope

Implemented functionality includes shared validation, label encoding,
Gini and entropy, vectorised split search, recursive batch fitting,
batch prediction, whole-buffer retraining, explicit retry and training
state counters.

Incremental preprocessing, statistics, metrics, Pipeline, StreamTrainer,
streaming CSV integration and visualisation remain pending.

## Environment

- Python 3.12.6
- NumPy 1.26.4
- Project interpreter: .venv/Scripts/python.exe

## Coverage

The original 96 streaming-package tests cover validation, impurity,
split-search reference comparisons, stopping rules, seeded feature
selection, numerical boundaries, failed batch fits and batch predictions.

Buffered training adds 33 tests:

| File | Tests | Focus |
| --- | ---: | --- |
| test_tree_buffer.py | 7 | Validation, independent copies and pending state |
| test_tree_partial_fit.py | 9 | Thresholds, overshoot, multiple rounds and dtype handling |
| test_tree_lifecycle.py | 5 | Warm-up, active probabilities and direct-fit reset |
| test_tree_recovery.py | 6 | Retry without duplication and caller input mutation |
| test_tree_state.py | 6 | Successful-event counters and failure stability |

Subtests are not counted as additional test methods.

## Verified results

### Batch classifier baseline

- Final user-checkout regression on 2026-11-13:
  593 tests passed in 0.677 seconds.
- This comprised 497 original tests and 96 streaming-package tests.
- Commit b4b720d was committed and pushed on assignment-2.2.

### Buffered classifier verification

- Final state-specific user-checkout run:
  6 tests passed in 0.004 seconds.
- Final streaming-package user-checkout run:
  129 tests passed in 0.505 seconds.
- Full regression in a temporary verification copy:
  626 tests passed in 0.838 seconds, comprising 497 original tests
  and 129 streaming-package tests.
- Final full regression in the user checkout on 2026-11-13:
  626 tests passed in 0.813 seconds, comprising 497 original tests
  and 129 streaming-package tests.
- Buffered-training commit and push: pending.

Temporary-copy verification is separate from user-checkout verification.
Test execution times are not performance benchmarks.

## Debugging evidence

- Initial label-family detection treated all object arrays as strings.
  Validated integer lists and string lists can both have object dtype.
  The correction inspects the first value of an already validated,
  nonempty homogeneous object array.
- A duplicated threshold block in retry_pending_fit trained and cleared
  the buffer before reaching the counter update. Removing the duplicate
  restored one successful-event increment per buffered training.
- The corresponding buffer and state regression tests now pass.

## Reproduction

Streaming-package tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/stream -t . -v
```

Full regression:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The original integration tests require demo/sample_data.csv.

## Known limits

- Recursive construction is subject to Python's recursion limit.
- Split search allocates temporary arrays proportional to input size.
- Numerical-zero and gain-tie tolerance is 1e-12.
- Buffer size is a trigger, not a strict memory limit.
- Potentially lossy integer-to-floating feature promotion is rejected
  conservatively; retry alone does not resolve incompatible input dtypes.
- No automatic final-buffer flush is performed.
- End-to-end streaming evaluation and performance benchmarks remain pending.