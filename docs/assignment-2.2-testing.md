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

## Incremental StandardScaler verification

Implemented a shared NaN-aware moment-merging helper and an incremental
StandardScaler with fit, partial_fit, transform and fit_transform.
StandardScaler is exported from the package root.

The scaler maintains per-feature valid counts, means and centred sums
of squares. Population variance uses ddof=0. Constant columns use scale 1.
Unobserved columns have mean/variance NaN and scale 1, and can recover
when subsequent chunks contain valid observations.

Validation performed in the original working checkout:

- Manual moment-merging checks: PASS.
- Manual StandardScaler lifecycle and transformation checks: PASS.
- Dedicated scaler suite with warnings treated as errors:
  16 tests, 0.019 seconds, OK.
- Complete streaming-package suite:
  145 tests, 0.241 seconds, OK.

Commands:

    python -W error -m unittest tests.stream.test_standard_scaler -v
    python -m unittest discover -s tests/stream -t . -v

The 16 new tests cover hand-calculated statistics, unequal chunks,
singleton updates, large-offset data, missing-column recovery, reset
behaviour, invalid-input and overflow recovery, input/output ownership,
snapshot independence, read-only noncontiguous inputs, empty transforms
and per-feature state storage.

These timings are test execution times, not performance benchmarks.
The full original-plus-streaming regression suite has not yet been
rerun in this checkout after these changes.

Incremental imputation, categorical encoding and the public streaming
statistics API remain pending. Commit and push are pending.

## Incremental Imputer verification

Implemented Imputer with mean and constant strategies, supporting fit,
partial_fit, transform and fit_transform. Imputer is exported from the
package root.

Mean imputation uses cumulative original nonmissing observations.
Unobserved columns use the finite fill_value fallback, default 0, and
switch to observed means when valid data arrives. Constant imputation
always uses fill_value and does not compute moments.

transform preserves observed values and does not update learning state.
Changing strategy or fill_value after fitting requires a new fit.
Invalid learning input or mean-calculation overflow preserves prior
learned state.

Validation performed in the original working checkout:

- Manual imputation and lifecycle checks: PASS.
- Dedicated imputer suite with warnings treated as errors:
  18 tests, 0.014 seconds, OK.
- Complete streaming-package suite:
  163 tests, 0.182 seconds, OK.

Commands:

    python -W error -m unittest tests.stream.test_imputer -v
    python -m unittest discover -s tests/stream -t . -v

Coverage includes unequal chunks against NumPy batch means, default and
custom fallbacks, missing-column recovery, constant imputation, exclusion
of transformed values from learning, reset behaviour, parameter changes,
invalid-input and overflow recovery, input/output ownership, independent
snapshots, read-only noncontiguous inputs and empty transforms.

Mean imputation shares the float64 moment helper and its numerical
limits. Raw sample history is not retained.

These timings are test execution times, not performance benchmarks.
The full original-plus-streaming regression suite has not yet been
rerun in this checkout after these changes.

Incremental categorical encoding and the public streaming statistics
API remain pending. Commit and push are pending.

## Incremental OneHotEncoder verification

Implemented categorical input validation and batch factorisation in
_categorical.py, and OneHotEncoder in preprocessing.py. OneHotEncoder
is exported from the package root.

Supported methods are fit, partial_fit, transform and fit_transform.
New categories append without reordering existing categories within each
feature. Expanding an earlier feature can shift later output blocks.
Missing values have a distinct learned category.

Unknown values either produce an all-zero feature block (default ignore)
or raise an error. transform does not learn. Failed updates preserve
previous learned state.

Validation performed in the original working checkout:

- Manual categorical input checks: PASS.
- Manual OneHotEncoder behaviour checks: PASS.
- Dedicated encoder suite with warnings treated as errors:
  22 tests, 0.012 seconds, OK.
- Complete streaming-package suite:
  185 tests, 0.190 seconds, OK.

Commands:

    python -W error -m unittest tests.stream.test_one_hot_encoder -v
    python -m unittest discover -s tests/stream -t . -v

Coverage includes stable category indices, shifting feature blocks,
unknown-value policies, missing categories, numeric equivalence, large
integers, input validation, failed-update atomicity, reset behaviour,
input/output ownership, independent snapshots and empty transforms.

Chunked and batch outputs are compared after aligning columns by category
meaning: their learned category orders can legitimately differ.

Dense output memory grows with sample count times total learned
categories. Stored category state can grow with new categories.
No performance benchmark has been completed for this encoder.

The full original-plus-streaming regression suite has not yet been
rerun in this checkout after these changes. The public streaming
statistics API and Pipeline snapshot coordination remain pending.
Commit and push are pending.


## StreamingStats verification

Implemented cumulative moments, exact global quantiles and optional
fixed-edge per-feature histograms in StreamingStats. The class is
exported from the package root.

Validation performed in the original working checkout:

- Manual moment checks: PASS.
- Manual global quantile checks: PASS.
- Manual histogram boundary and accumulation checks: PASS.
- Dedicated statistics suite with warnings treated as errors:
  20 tests, 0.026 seconds, OK.
- Complete streaming-package suite:
  205 tests, 0.215 seconds, OK.

Commands:

    python -W error -m unittest tests.stream.test_streaming_stats -v
    python -m unittest discover -s tests/stream -t . -v

Coverage includes unequal chunks against NumPy batch references,
missing-column recovery, all-missing updates, quantile interpolation
and output shapes, histogram endpoints and out-of-range values,
invalid parameters, reset behaviour, input/output ownership,
independent snapshots and read-only noncontiguous inputs.

Failure recovery includes invalid input, numerical overflow and a
simulated histogram failure on the second feature. The latter verifies
that partially prepared candidate state is not published.

Exact quantiles retain valid history; memory is not constant.
These test timings are not performance benchmark results.

The full original-plus-streaming regression suite has not yet been
rerun in this checkout after these changes. Cross-component verification
and final documentation review remain pending. Commit and push are pending.

## Preprocessing integration and full regression

Added six cross-component tests covering shared observed statistics,
finite numeric preprocessing for tree input, mixed feature blocks,
frozen numeric preprocessing/tree predictions, categorical layout
changes and query operations that do not update learning.

These tests compose components explicitly. They do not implement or
validate the future automatic Pipeline coordination protocol.

Results from the original working checkout:

| Check | Tests | Time | Result |
| --- | ---: | ---: | --- |
| Preprocessing integration, warnings treated as errors | 6 | 0.008 s | OK |
| Complete streaming-package suite | 211 | 0.206 s | OK |
| Full original-plus-streaming regression | 708 | 0.835 s | OK |

Commands:

    python -W error -m unittest tests.stream.test_preprocessing_integration -v
    python -m unittest discover -s tests/stream -t . -v
    python -m unittest discover -s tests -v

The full suite comprises 497 original tests, 129 previous streaming tests
and 82 new tests: scaler 16, imputer 18, encoder 22, statistics 20 and
integration 6. subTest cases are not counted as separate test methods.

Earlier entries stating that the full regression had not yet been rerun
describe historical checkpoints. The full regression above now verifies
the combined implementation.

These execution times are not performance benchmark results.
A trailing blank line in the design document was removed after
git diff --check reported it. Subsequent checking reported only Git
line-ending conversion warnings.

Final documentation review, staging, commit and push are pending.