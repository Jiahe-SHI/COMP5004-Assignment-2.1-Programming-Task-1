# Assignment 2.2 - Implementation testing

## Scope

Implemented functionality includes shared validation, batch and buffered
decision trees, incremental preprocessing, StreamingStats, cumulative
and rolling classification metrics, cumulative and rolling binary AUC,
Pipeline and StreamTrainer.

Pipeline and trainer coverage includes paired preprocessing/model
publication, external-transformer interoperability, evaluation before
learning, automatic binary probability alignment, event logs and scoped
NumPy-array memory accounting.

Streaming CSV integration, visualisation, the streaming demonstration
and performance benchmarks remain pending.

Verification sections below are chronological checkpoints. Statements
such as "pending" or "not yet rerun" inside earlier entries describe the
state at that checkpoint, not the current implementation.

The buffered-tree work was committed and pushed as 5a4d6cc.
Incremental preprocessing and statistics were committed and pushed as
be0694e. Cumulative and rolling metrics were committed and pushed as
25d2a8b. The current development branch is Jiahe-assignment-2.2.

Pipeline and StreamTrainer code verification has passed. Documentation
review, staging, commit and push for these changes remain pending.

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

## Cumulative and rolling metrics verification (2026-11-14)

Implemented cumulative confusion counts, per-class classification metrics,
macro and support-weighted summaries, binary ROC AUC, and sample-based
rolling classification/AUC windows.

All six public metrics APIs are exported from numcompute_stream.
Tests use unittest and NumPy; no external machine-learning metric library
is required.

### Added coverage

| File | Tests | Focus |
| --- | ---: | --- |
| test_streaming_confusion_matrix.py | 7 | Stable labels, counts, ownership and invalid updates |
| test_streaming_classification_metrics.py | 7 | Hand calculations, unequal chunks, zero division and reset |
| test_binary_roc_auc.py | 8 | Ordering, ties, undefined results and integer score precision |
| test_streaming_roc_auc.py | 8 | Cumulative history, cross-chunk comparisons and failure recovery |
| test_rolling_classification_metrics.py | 8 | Partial eviction, oversized chunks and retained class mappings |
| test_rolling_roc_auc.py | 8 | Window AUC, class disappearance, bounded history and reset |
| test_metrics_integration.py | 6 | Partitioning, tree probabilities, snapshots and candidate failures |
| Total | 52 | |

subTest cases do not count as additional test methods.

### Recorded user-checkout results

Dedicated suites were run with warnings treated as errors.

| Check | Tests | Time | Result |
| --- | ---: | ---: | --- |
| Cumulative confusion matrix | 7 | 0.155 s | OK |
| Cumulative classification metrics | 7 | 0.329 s | OK |
| Binary AUC, after test-data correction | 8 | 0.129 s | OK |
| Cumulative AUC | 8 | 0.007 s | OK |
| Rolling classification metrics | 8 | 0.195 s | OK |
| Rolling AUC | 8 | 0.004 s | OK |
| Metrics integration | 6 | 0.090 s | OK |
| Final streaming-package suite | 263 | 0.231 s | OK |
| Full original-plus-streaming regression | 760 | 0.821 s | OK |

The full suite comprises 497 original tests and 263 streaming-package
tests. The new package previously contained 211 tests; metrics added 52.
These times are test execution measurements, not performance benchmarks.

### Debugging evidence

Two initial binary AUC tests expected 0.75 for labels [0, 1, 0, 1] and
scores [0.1, 0.4, 0.35, 0.8]. Both positive scores exceeded both negative
scores, so the implementation correctly returned 1.0.

The positive score 0.4 was changed to 0.3 in the direct example and its
noncontiguous-array counterpart. Three of four positive-negative pairs
then receive full credit, giving the intended AUC of 0.75.
No AUC algorithm change was needed. The corrected eight-test suite and
subsequent full regression passed.

git diff --check reported only LF-to-CRLF conversion warnings for
numcompute_stream/__init__.py and numcompute_stream/metrics.py at the
code-verification checkpoint. No whitespace errors were reported.

### Integration boundaries and limitations

- Partition tests compare cumulative and rolling results across unequal
  chunks, singleton updates and oversized chunks.
- Scalar confusion counts and pairwise AUC provide independent test
  references on small datasets.
- Tree tests select positive probability by classes_, including a positive
  class in column zero and a positive class absent after refitting.
- The absent-positive policy is demonstrated by a test-local helper;
  StreamTrainer and Pipeline are not implemented by these tests.
- Snapshot tests verify independent metric copies.
- Simulated recount/allocation failures verify rolling state preservation.
- Cumulative AUC retains full history; rolling AUC retains at most W pairs.
- Rolling classification retains historical classes, so its class table and
  dense matrix may grow even though its sample window is bounded.
- Float64 score storage and integer exactness checks remain documented
  constraints of stateful AUC.
- Performance benchmarks and long-stream memory measurements remain pending.

### Reproduction

```powershell
python -W error -m unittest tests.stream.test_streaming_confusion_matrix -v
python -W error -m unittest tests.stream.test_streaming_classification_metrics -v
python -W error -m unittest tests.stream.test_binary_roc_auc -v
python -W error -m unittest tests.stream.test_streaming_roc_auc -v
python -W error -m unittest tests.stream.test_rolling_classification_metrics -v
python -W error -m unittest tests.stream.test_rolling_roc_auc -v
python -W error -m unittest tests.stream.test_metrics_integration -v
python -m unittest discover -s tests/stream -t . -v
python -m unittest discover -s tests -v
git diff --check
```

Metrics documentation review, staging, commit and push are pending.

## Pipeline and StreamTrainer verification

Implemented Pipeline raw-data buffering and paired publication of fresh
preprocessing/tree candidates. Implemented StreamTrainer evaluation before
learning, cumulative and rolling metrics, binary probability alignment,
event logging and retained-array memory accounting.

Pipeline and StreamTrainer are exported from numcompute_stream.
Tests use unittest and NumPy without an external machine-learning library.

### Added coverage

| File | Tests | Focus |
| --- | ---: | --- |
| test_pipeline_components.py | 5 | Component validation and configuration capture |
| test_pipeline_buffer.py | 8 | Buffer thresholds, ownership and training lifecycle |
| test_pipeline_snapshots.py | 8 | Paired preprocessing/tree state and failure recovery |
| test_pipeline_dispatch.py | 6 | Compatible fitting signatures and exception propagation |
| test_pipeline_external.py | 8 | External clone protocol and transformer interoperability |
| test_pipeline_boundaries.py | 8 | Input boundaries, schema and numerical handling |
| test_stream_trainer.py | 8 | Warm-up, evaluation, learning and metric state |
| test_stream_trainer_auc.py | 6 | Positive probability alignment and binary AUC |
| test_stream_trainer_logs.py | 8 | Event records, timing and failed learning |
| test_stream_trainer_memory.py | 7 | Array accounting, ownership and measurement failures |
| test_stream_integration.py | 4 | Unequal chunks, changing categories, retries and independent trainers |
| Total | 76 | |

subTest cases do not count as additional test methods.

### Recorded user-checkout result

After correcting the categorical integration test expectation, the full
original-plus-streaming regression completed successfully:

| Check | Tests | Time | Result |
| --- | ---: | ---: | --- |
| Full original-plus-streaming regression | 836 | 0.836 s | OK |

The suite comprises 497 original tests and 339 streaming-package tests.
The previous streaming suite contained 263 tests; this work adds 76.

The full regression includes all four new streaming integration tests.
Its execution time is not a performance benchmark.

git diff --check reported only LF-to-CRLF conversion warnings at the
code-verification checkpoint. No whitespace errors were reported.

### Integration evidence

- Unequal numeric chunks exercise warm-up, threshold overshoot, missing
  value preprocessing and multiple model publications.
- Recorded predictions from before learning provide the classification
  reference; a small independent pairwise calculation provides the AUC
  reference.
- Changing categorical data exercises paired encoder/tree replacement
  and evaluation against the previously published representation.
- A simulated training failure preserves accepted buffer data and the
  active model. Retrying training does not repeat successful scoring.
- Separate trainer instances retain independent model, evaluation and
  log state.
- External-transformer tests exercise minimal fitting signatures and
  internal TypeError propagation without a second fitting call.
- Logging and memory tests verify accounting behaviour, rather than
  claiming process-wide memory or benchmark performance.

### Debugging evidence

The changing-category integration test initially expected cumulative
classes [0, 2, 1, 3], while the implementation returned [0, 1, 2, 3].

The first evaluated update introduced true labels 0 and 2 together with
predicted label 1. Newly observed labels from truth and predictions are
combined and appended in sorted order within that update. Therefore,
[0, 1, 2, 3] is the correct accumulated class order.

Only the test expectation was corrected. No classification metric
algorithm change was required. The subsequent full regression passed.

### Boundaries and limitations

- The caller must score each chunk before learning it; ordering is not
  automatically enforced.
- Repeated successful score_chunk calls count as separate evaluations.
  There is no automatic deduplication.
- Warm-up rows count as received scoring rows but are excluded from
  evaluated-sample denominators.
- Pipeline preprocessing is rebuilt from each eligible training buffer,
  rather than accumulating statistics across all historical buffers.
- External clone independence and non-learning transform behaviour are
  protocol requirements; identity and shape checks cannot prove them.
- A training failure after chunk acceptance retains that chunk. Retry
  learning without submitting the same chunk again.
- Buffer size is a trigger, not a hard memory cap. Partial final buffers
  are not automatically flushed.
- training_seconds includes buffering and any triggered training; it is
  not an isolated tree-building measurement.
- Memory accounting measures reachable NumPy array nbytes, excluding
  Python-object overhead, logs, temporaries, process RSS and peak memory.
- Exact cumulative AUC retains history, and candidate metric copies
  duplicate that history during scoring. Event logs and class tables
  can also grow.
- Streaming CSV integration, visualisation, the streaming demonstration,
  performance benchmarks and long-stream memory measurements remain
  pending.

### Reproduction

Run from the repository root:

```powershell
python -W error -m unittest tests.stream.test_stream_integration -v
python -m unittest discover -s tests/stream -t . -v
python -m unittest discover -s tests -v
git diff --check
```

These commands describe how to reproduce the checks. The recorded timing
above belongs to the verified full-regression run.

Documentation review, staging, commit and push for Pipeline and
StreamTrainer remain pending.
