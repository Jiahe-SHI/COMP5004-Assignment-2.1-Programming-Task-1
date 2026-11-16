# Assignment 2.2 - Design contract

Batch and standalone buffered trees, incremental preprocessing,
StreamingStats, cumulative and rolling classification metrics, binary
ROC AUC, Pipeline and StreamTrainer are implemented and tested.

Pipeline coordinates raw-data buffering and paired publication of
preprocessing and tree state. StreamTrainer separates evaluation from
learning and provides probability alignment, event logs and scoped
NumPy-array memory accounting.

Streaming CSV input, CSV-to-trainer integration, three plotting interfaces
and the streaming demonstration notebook are implemented and verified.
Performance benchmarks, systematic buffer-size timing comparisons and
long-stream memory experiments remain pending.

## Inputs and state

- X: shape (n_samples, n_features); y: shape (n_samples,).
- Labels: homogeneous integers or homogeneous strings.
- Reject boolean, floating, mixed-type and missing labels.
- Preserve integer labels without conversion through float.
- Each successful tree fit creates a sorted class table from that training
  dataset. Buffered retraining does not retain absent historical classes.
- Cumulative and rolling classification metric class tables append new
  classes without rearranging existing entries. Rolling tables retain
  classes after their samples leave the window. This differs from the
  tree's per-training class table.
- fit resets learning; partial_fit accumulates within its documented scope.
- Reject empty training data and zero-feature training inputs.
- A fitted predictor accepts zero rows with the correct feature count.
- Validate before state changes; prediction does not update training state.

## Preprocessing

- Numeric NaN denotes missing data; reject infinity.
- StandardScaler tracks per-feature valid count, mean and M2; ddof=0.
- Constant columns use scale 1.
- Unobserved columns have count 0, mean/variance NaN and scale 1.
- Imputer supports mean and constant strategies.
- Mean imputation uses observed values only; unobserved columns use an
  explicit finite fallback, default 0.
- OneHotEncoder supports incremental categories and error/ignore handling
  of unknown values; default ignore.
- Missing categories are represented explicitly.
- Category indices within a feature remain stable; output offsets for
  later features may change when an earlier feature gains categories.
- Trees require finite numeric inputs after preprocessing.

## Decision tree

- Numeric threshold splits support Gini (default) and entropy.
- Root depth is 0; max_depth=0 gives a root leaf.
- Stop when no positive information gain is available.
- Gain ties use fixed feature and split-position ordering.
- Leaf vote ties follow classes_ order.
- max_features supports None, valid positive integers, sqrt and log2.
- Use a local random generator controlled by random_state.
- predict_proba returns columns aligned to the active model's classes_.

## Standalone tree buffering

- Copy incoming chunks into the pending buffer.
- At size >= buffer_size, train once using the entire pending buffer.
- Replace the model and clear the buffer only after successful training.
- Preserve the old model and pending data after a training failure.
- Before first successful training, prediction raises an explicit error.
- Between retraining events, use the last successful model.
- Successful fit clears previous pending data.
- Do not automatically flush a partial final buffer.

## Pipeline buffering and preprocessing scope

Pipeline accepts uniquely named steps with distinct component instances.
The final component must be the built-in DecisionTreeClassifier.
Intermediate components must provide callable partial_fit and transform.

The constructor captures independent unfitted configuration templates.
Training does not fit the original supplied component instances.

Pipeline is the sole raw-data buffer owner when a tree is composed.
It uses the captured final tree's buffer_size and invokes candidate_tree.fit,
rather than buffering the same data through tree.partial_fit.

At retraining:

1. Create a fresh, unfitted candidate preprocessing chain.
2. On the entire raw buffer, call partial_fit then transform for each step.
3. Fit a candidate tree on the final representation.
4. Publish the preprocessing chain and tree together only after success.
5. Clear the raw buffer after successful publication.

Standalone preprocessors accumulate across successive partial_fit calls.
Pipeline preprocessing is rebuilt for each retraining buffer: its
statistics describe that training window, not all historical observations.
The active prediction chain remains frozen between retraining events.

This scope difference prevents downstream accumulators from mixing
changing upstream representations.

### Inputs and transformer interoperability

- Numeric raw input may contain NaN, but not infinity. The final tree
  still requires finite numeric features.
- A pipeline beginning with the built-in OneHotEncoder accepts categorical
  raw input and preserves per-feature string/numeric category kinds.
- Automatic routing of mixed numeric and categorical column groups is
  not implemented.
- Raw feature count and label family must remain compatible across
  partial_fit calls. Successful direct fit may establish a new schema.
- External transformers must implement clone_unfitted(), returning a
  distinct instance of the same type with equivalent configuration.
- The external implementation must ensure that the clone is unfitted,
  does not share learned mutable state and does not mutate its source.
  Identity/type checks cannot prove these semantic guarantees.
- Signature inspection supports X-only partial_fit methods and compatible
  positional-only, keyword or **kwargs handling of y.
- Internal TypeError exceptions propagate without a second fitting call.
- Transformer calls receive copied arrays. Outputs must be two-dimensional,
  preserve row count and contain at least one feature; outputs are copied.
- External transform methods must preserve row order and must not learn.
  Shape checks alone cannot verify those requirements.

### State and failure recovery

- partial_fit validates and copies a chunk before accepting it.
- At size >= buffer_size, train once on the entire pending buffer,
  including any threshold overshoot.
- Validation failure rejects the chunk. Candidate-training failure after
  acceptance retains the chunk and the previous active chain.
- retry_pending_fit retries eligible pending data without appending.
  Empty or below-threshold buffers are not trained.
- There is no automatic final-buffer flush.
- Direct fit trains immediately. Success replaces the active chain and
  clears pending data; failure preserves the previous state.
- Before the first successful fit, prediction raises RuntimeError.
- n_samples_buffered_ reports pending raw rows; n_samples_fit_ reports
  the latest successful training size.
- n_features_in_ describes raw input features, not encoded feature width.
- classes_ describes the active tree and returns an independent copy.
- n_retrains_ counts successful buffered publications, including retries.
  Direct fit does not increment or reset it.
- model_version_ counts every successful publication, including direct fit.
  Failed training and prediction do not advance these counters.
- Numeric buffer concatenation conservatively rejects potentially lossy
  integer-to-floating promotion. Retrying cannot resolve incompatible
  data without addressing its cause.
- buffer_size is a training trigger, not a strict memory limit.

## Statistics and metrics

- Per-feature means/variances use stable block merging.
- Initial quantiles are exact and retain observed history; memory grows.
- Histograms use fixed user-provided edges, include the last endpoint,
  and exclude out-of-range values.
- Confusion matrix rows are truth, columns are predictions.
- Compute cumulative classification metrics from counts, not chunk means.
- Default zero-division value is 0 with explicit configuration.
- Rolling windows contain the latest W evaluated samples and support
  partial removal across chunk boundaries.
- Initial AUC scope is binary with explicit positive class and scores.
- Undefined AUC returns NaN.
- Exact cumulative AUC retains label/score history; rolling AUC retains
  only window data.
- With no evaluated samples, overall metrics return NaN.
- reset clears learned counts/history while retaining configuration and
  explicitly configured classes.
- Multiclass classification is supported; multiclass AUC is not promised.

## StreamTrainer

StreamTrainer accepts a built-in DecisionTreeClassifier or Pipeline.
It retains the supplied model by reference; learning updates that model.
Concurrent or external mutation is outside the coordination contract.

### Evaluation and learning

- score_chunk validates input, predicts and updates evaluation metrics.
  It does not train the model.
- fit_chunk delegates to model.partial_fit without updating evaluation
  metrics.
- Call score_chunk before fit_chunk for each incoming chunk. This ordering
  is the caller's responsibility and is not automatically enforced.
- Before a model is fitted, scoring records a warm-up event without
  predictions or metric updates.
- n_chunks_scored_ and n_samples_scored_ include successful warm-up calls.
  n_samples_evaluated_ excludes warm-up rows.
- Repeated successful score_chunk calls are separate evaluation events.
  There is no automatic deduplication.
- Candidate metric copies are updated before coordinated publication.
  Failed scoring preserves previous evaluation state and adds no log.
- retry_pending_fit retries learning without appending or scoring again.
  A training failure does not undo an earlier successful scoring event.
- metrics_result returns cumulative and rolling classification results,
  plus cumulative and rolling AUC results when enabled.

### Binary probability alignment

- positive_label=None disables AUC and reports None for AUC results.
- With an explicit positive_label, scores use the active model's matching
  classes_ probability column, not a fixed column index.
- If the positive class is absent from the active model, scores are zero.
- Enabled AUC is NaN when either class is absent from evaluated data.
- Binary label compatibility is checked using model classes and true
  labels; stateful AUC also retains its established negative label.
- Multiclass classification is supported when AUC is disabled.
  Multiclass AUC is unsupported.

### Event logs and timing

- logs_ returns independent copies of retained event records.
- Successful scoring records warm-up status, chunk/cumulative/rolling
  accuracy, optional AUC, scoring counts and model version.
- Successful and failed learning calls record buffer state, retraining
  count, model version, elapsed time and success status.
- Learning failures record the exception type and re-raise the exception.
- A retry is a separate learning event, not another scoring event.
- score_seconds covers scoring and metric work.
- training_seconds covers the model update or retry, including buffering
  and any triggered training. It is not isolated tree-building time.
- Memory traversal and log insertion are outside these timed regions.
- Model operations performed directly outside the trainer are not logged.

### Memory accounting and computational limits

- memory_result reports reachable NumPy array nbytes for model and
  evaluation state using the reachable_ndarray_nbytes method.
- buffer_array_bytes is the pending-buffer subset of the total.
- The same array object is counted once. Distinct overlapping views are
  counted separately by their logical nbytes.
- Array base storage and objects referenced inside object arrays are not
  recursively measured through those arrays.
- Python-object overhead, logs, temporary allocations, native allocations,
  process RSS and peak memory are excluded.
- log_events reports event count, not the memory occupied by logs.
- Log-time measurement failures are recorded as unavailable and do not
  replace the original training exception.
- Exact cumulative AUC retains full evaluation history. Candidate metric
  deepcopy duplicates that history during scoring.
- Classification class tables and event logs may grow over time.
- Memory traversal and metric queries also require work; neither bounded
  total memory nor constant-time updates are promised.

## Verification scope

Current tests cover independent numerical references, unequal chunks,
component snapshots, categorical expansion, standalone and Pipeline
training recovery, rolling-window eviction, tied AUC scores and automatic
probability-column alignment.

External-transformer tests cover clone validation, compatible method
signatures, copied inputs and internal exception propagation. Trainer
integration tests cover evaluation before learning, changing categories,
retry without repeated scoring and independent trainer instances.

Timing and memory tests verify logging and accounting behaviour.
They do not establish performance or long-stream memory bounds.

- Vectorised versus independent reference outputs.
- Unequal chunk sizes versus full-batch statistics.
- Minimal external method signatures and internal exception propagation.
- Snapshot independence and changing categorical output dimensions.
- Buffer threshold overshoot and failed-training recovery.
- Rolling-window partial eviction and AUC tied-score handling.
- Stable label mapping, including large integer labels.


## Split search numerical policy (2026-11-13)

- Candidate splits occur only between adjacent distinct feature values.
- Class counts and impurity are evaluated in batches for each feature.
- Gains <= 1e-12 are treated as numerical zero.
- Candidates within absolute tolerance 1e-12 of the global maximum gain
  are tied. Prefer the smallest feature index, then the earliest boundary.
- Integer thresholds use exact Python integer midpoint arithmetic.
- Floating thresholds use half-sums to avoid overflow and fall back to
  the lower endpoint if rounding fails to preserve lower <= threshold < upper.
- The partition rule is X[:, feature] <= threshold.
- Recursive tree construction supplies feature subsets selected by its
  local random generator.
- Temporary split-search storage is O(n_samples * n_classes +
  n_samples * n_candidate_features).


## Batch tree construction policy (2026-11-13)

- Root depth is zero. max_depth=0 creates a root leaf.
- Stop on pure labels, insufficient samples, the depth limit, or no
  candidate gain greater than the documented numerical tolerance.
- min_samples_split controls the parent node's eligibility to split;
  it is not a minimum leaf size.
- Leaf vote ties follow the sorted classes_ order.
- Every node retains counts in the complete fitted class order.
- max_features=None uses all features. Positive integers specify a count.
  sqrt and log2 round down, with a minimum of one feature.
- Each eligible node samples features without replacement using a local
  random generator. Selected features are searched in ascending order.
- A selected subset with no useful split produces a leaf; no replacement
  features are sampled.
- Each fit recreates the local generator. An integer seed reproduces the
  tree for identical input and configuration without changing global RNG state.
- fit validates configuration again, builds a complete candidate tree,
  then publishes learned attributes. Failure preserves prior learned state.
- Successful fit replaces the class table and feature schema using the
  new training data. Training arrays are not retained.
- Batch fit trains immediately regardless of buffer_size and clears pending
  data after success. Buffered partial_fit uses the whole pending buffer
  once its sample count reaches the threshold.
- Recursive construction is subject to Python's recursion limit.

## Batch prediction policy (2026-11-13)

- predict returns original labels with shape (n_samples,), preserving
  the fitted class-table dtype and input row order.
- predict_proba returns unsmoothed leaf class frequencies with shape
  (n_samples, n_classes_). Columns follow classes_.
- Traversal uses a stack of nodes and sample-index arrays. Routing and
  output assignment operate on batches, without per-sample Python traversal.
- Prediction before successful fitting raises RuntimeError.
- Empty prediction input is supported only with the fitted feature count.
- Prediction rejects NaN, infinity and invalid input shapes or types.
- Prediction does not update model state; returned arrays are independent.
- Training and prediction share a threshold comparison helper. Integer
  bounds and downward floating threshold adjustment prevent dtype
  conversion from moving a value to the wrong side of a stored threshold.
- The package exports DecisionTreeClassifier with batch fitting, buffered
  partial_fit and explicit retry_pending_fit.

## Buffered tree state and recovery policy (2026-11-13)

- Chunks are validated and copied before buffer publication. Numeric feature
  count and integer/string label family must match pending data and any
  active model. New class values are allowed.
- Buffer storage is a list of independent chunks. Concatenation occurs at
  training time, rather than copying the whole buffer on every arrival.
- At size >= buffer_size, train once on all pending rows. A single oversized
  chunk also causes only one training event.
- Successful fit publishes the new tree and clears pending data. Failed
  fitting preserves the old tree and pending chunks.
- Validation failure rejects the incoming chunk. Failure after acceptance
  retains it; callers must not resubmit the same chunk.
- retry_pending_fit validates current configuration and retries eligible
  pending data without appending. Empty or below-threshold buffers are
  not trained. It does not force a final partial-buffer flush.
- Incoming array mutation cannot alter retained training data.
- Signed/unsigned integer labels use object storage when necessary to avoid
  floating conversion. Feature promotion to floating storage conservatively
  rejects integers outside that dtype's consecutive exact-integer range.
- n_samples_buffered_ is the pending row count. n_samples_fit_ describes
  the latest successful training dataset and is absent before first fit.
- n_retrains_ counts lifetime successful buffered fits, including the first
  and successful retries. Direct fit does not increment or reset it.
- model_version_ counts lifetime successful model publications, including
  direct fits. Both counters start at zero and are exposed as read-only
  properties. Failed fits and prediction do not advance them.
- Direct fit resets learning data and pending schema, but lifetime counters
  remain cumulative. A new instance starts new counters.
- Each fitted class table belongs to its training window. Prediction and
  probability columns remain unchanged until a new model is published.
- buffer_size is a trigger, not a memory cap. Pending chunks, concatenated
  training arrays and tree-building temporaries can coexist.
- Pipeline owns its own buffer and invokes candidate_tree.fit.
  Pipeline maintains its own publication and buffered-retraining counters;
  StreamTrainer reads the outer model's counters.

## Incremental one-hot encoding contract

The following rules are implemented by OneHotEncoder and covered by
22 dedicated tests. Additional Pipeline tests verify paired encoder/model
publication; independent encoder deepcopy behaviour is also tested.

- fit resets learned categories; partial_fit extends them.
- Existing category indices within each input feature remain unchanged.
- Newly observed ordinary categories are sorted deterministically within
  each update and appended after existing categories.
- Missing values are represented by a distinct category, not a reserved
  ordinary string. A missing category is added only when observed during
  learning; when newly added, it follows that update's ordinary additions.
- Output blocks follow the original input feature order.
- Expanding an earlier feature can shift the output offsets of later
  features, even though their internal category indices remain stable.
- transform does not learn or expand categories.
- handle_unknown defaults to ignore: an unknown value produces zeros
  throughout that feature's output block.
- handle_unknown=error rejects unknown values.
- An unlearned missing category follows the same unknown-value policy.
- Training requires a nonempty two-dimensional input. Incremental updates
  retain the original feature count.
- Output is a new dense float64 matrix.
- Conversion may loop over features, but must avoid Python loops that
  encode individual samples.
- Dense output memory grows with sample count times total learned
  categories. Category state can grow as new categories arrive.
- Input validation and candidate category construction must complete
  before publishing updated learned state.

Pipeline preserves a matching encoder/model snapshot by rebuilding and
publishing both together. An independently updated encoder must not be
paired directly with a tree trained on an earlier output layout.

Input category types:

- Each feature accepts strings or real numeric categories, optionally
  containing None or numeric NaN as missing values.
- Strings and numbers cannot be mixed within one feature.
- Equal numeric values such as 1 and 1.0 share a category.
- Boolean, complex, infinite and unsupported object values are rejected.
- Ordinary strings such as "None" and "nan" are not missing values.
- A feature's learned string/numeric kind is retained during incremental
  updates and transformation. An entirely missing feature can acquire
  its ordinary category kind in a later update.
- Heterogeneous input should use nested Python lists or object arrays.
  String coercion performed before calling the encoder cannot be undone.
- Object input requires scalar type inspection during validation.
  Category factorisation, lookup and output assignment use NumPy batch
  operations rather than Python loops encoding individual samples.


## Streaming descriptive statistics

StreamingStats implements update_stats(X_chunk), result(), quantile(q)
and histogram(), with reset() clearing learned state.

- Each instance maintains independent state; update_stats is an instance
  method, not a module-level global accumulator.
- Input shape is (samples, features). Updates require nonempty numeric
  input with a consistent feature count.
- NaN is excluded independently per feature; infinity is rejected.
- result returns independent count, mean and population variance arrays
  of shape (n_features,), plus n_rows.
- n_rows includes entirely missing rows; count records valid observations
  separately for each feature.
- Unobserved features have count 0 and mean/variance NaN.
- Moments use the shared float64 block-merging helper with ddof=0.

Quantiles:

- quantile(q) uses all valid observations received since reset.
- It retains independent float64 copies of valid observations by feature
  and chunk, then concatenates them when queried.
- Quantiles use linear interpolation and are exact over retained values,
  subject to float64 representation and arithmetic limits.
- Scalar q returns (n_features,); a sequence of k probabilities returns
  (k, n_features). Probabilities must be finite and within [0, 1].
- Unobserved features return NaN without all-NaN reduction warnings.
- Persistent history memory grows with valid observations and retained
  chunk references. Queries require additional temporary memory.
- Updates copy historical reference lists, not historical sample arrays;
  reference-management work therefore also grows with retained chunk count.
- This is a global-history implementation, not a sliding window.

Histograms:

- Optional bin_edges are copied at construction and shared across features.
- Edges must be finite and strictly increasing after float64 conversion.
- Counts have shape (n_features, n_bins).
- Bins are left-closed and right-open, except the final bin includes its
  right endpoint.
- NaN and out-of-range values are excluded from histogram counts.
  Out-of-range valid values still contribute to moments and quantiles.
- Histogram state uses O(n_features * n_bins) additional memory.
- histogram returns independent count and edge arrays.

State and failure behaviour:

- Queries do not learn or expose internal mutable arrays.
- Updates prepare moments, history and histogram counts before publication.
- Validation, numerical or histogram-calculation failure preserves prior
  accumulated state.
- reset clears observations and feature schema while retaining bin edges.
- result and quantile require a successful update. histogram additionally
  requires bin_edges configured at construction.

## Classification metrics and ROC AUC contract

Implemented public APIs:

| API | Methods or arguments | Scope |
| --- | --- | --- |
| StreamingConfusionMatrix | update, result, reset | Cumulative counts |
| StreamingClassificationMetrics | update, result, reset | Cumulative classification |
| RollingClassificationMetrics | window_size; update, result, reset | Latest W samples |
| binary_roc_auc | y_true, y_score, positive_label | One supplied dataset |
| StreamingROCAUC | positive_label; update, result, reset | Full evaluation history |
| RollingROCAUC | window_size, positive_label; update, result, reset | Latest W samples |

Classification:

- Labels are homogeneous integers or strings, consistently across updates.
- Initial classes must be unique and nonempty when explicitly supplied.
  They define an initial order, not a closed whitelist.
- New labels from truth or predictions append in sorted order per update.
- Confusion counts use NumPy index mapping and bincount. Object-label
  validation may inspect individual elements; the implementation is not
  claimed to eliminate every Python-level scalar operation.
- Rows represent truth and columns represent predictions.
- Counts are int64. Updates exceeding the cumulative int64 sample-count
  capacity of StreamingConfusionMatrix are rejected.
- Accuracy and per-class metrics derive from counts, not chunk averages.
- F1 uses 2 * TP / (true support + predicted support).
- Macro averages include the complete current class table.
- Weighted averages use true support.
- zero_division is numeric 0 or 1; booleans are rejected.
- With no evaluated samples, accuracy and aggregate metrics are NaN.
  Per-class arrays contain NaN for configured classes, or are empty when
  no classes exist. Confusion counts remain zero.
- Returned arrays are independent of internal state.

Windows and lifecycle:

- window_size is a positive integer, excluding booleans.
- Windows contain the latest W samples, including partial chunk eviction.
- The entire incoming chunk is validated before truncation.
- Oversized chunks retain only their final W rows.
- Rolling classification keeps historical class mappings until reset;
  absent historical classes still affect macro averages.
- Rolling n_samples is window occupancy; n_samples_seen and the property
  n_samples_seen_ count all successful input since reset.
- reset retains configuration and explicitly supplied initial classes.
- Updates prepare candidates before publishing state.
- Tests cover invalid-update recovery, independent deepcopy snapshots,
  simulated rolling recount failure and rolling AUC allocation failure.

Binary AUC:

- positive_label is required; larger scores indicate the positive class.
- Only one other label is permitted. Stateful AUC fixes that negative
  label until reset, even after it leaves a rolling window.
- Scores must be one-dimensional, finite and real numeric values.
- Stateful updates reject empty chunks. The standalone function accepts
  empty input and returns NaN after validating its arguments.
- Missing either class yields NaN; equal scores receive half credit.
- Score groups are sorted and counted without a pairwise sample matrix.
- Stateful score storage is float64. Integer scores must convert exactly;
  higher-precision floating scores follow float64 rounding.
- The standalone function preserves typed integer score ordering.
  Array-like inputs follow NumPy dtype inference.
- AUC is exact over retained scores subject to representation and arithmetic
  limits. Multiclass AUC is unsupported.
- Probability columns must be aligned using the active model's classes_.
  StreamTrainer performs this alignment automatically and uses zero
  positive probability when the positive class is absent.

Memory and work:

- Cumulative classification retains O(K^2) counts, not raw sample history.
- Rolling classification retains O(W) sample pairs plus class mappings
  and O(K^2) counts. K can grow with observed classes.
- Rolling classification validates/counts the incoming chunk and recounts
  the retained window; updates are not constant-time.
- Exact cumulative AUC retains all label/score chunks. Updates also copy
  the history's chunk-reference lists.
- Rolling AUC retains one owned window of at most W label/score pairs;
  oversized input prefixes are not retained through array views.
- AUC queries concatenate and sort retained samples, using O(N log N)
  time and O(N) auxiliary memory, with N bounded by W for rolling AUC.
- Update temporaries depend on incoming chunk size.
- No benchmark or long-stream memory measurement is established by these
  unit tests.

## Streaming CSV input contract (2026-11-16)

- iter_csv_chunks is exported from numcompute_stream and is separate
  from the original numcompute.io reader.
- Use standard-library csv parsing with a required header and strict
  CSV syntax checking. Quoted fields are supported.
- target_column explicitly identifies the label column.
- feature_columns optionally selects and orders numeric feature columns.
  By default, select all columns except the target. Reject duplicate,
  unknown or empty selections and target inclusion.
- Header names are matched exactly. Reject duplicate or blank names.
- chunk_size must be a positive integer; boolean values are rejected.
- Yield independent X/y arrays. X is float64 with shape
  (n_rows, n_features); y is one-dimensional.
- label_dtype is "str" or "int". Integer labels are parsed directly as
  Python integers and stored in object arrays without conversion through
  float. Integer syntax is [+-]?[0-9]+.
- Strip surrounding whitespace from data fields and missing markers.
  Missing-marker matching is case-sensitive.
- Configured missing feature markers become NaN. Reject other non-finite
  feature values and missing labels. Float64 feature conversion may
  round large integer values.
- Reject blank data rows and incorrect field counts. Parsing errors
  include useful line/column context where available. For records that
  span physical lines, csv.reader.line_num identifies the ending line.
- Read incrementally rather than loading the complete file. Yield the
  final short chunk; do not produce an extra empty chunk.
- An empty file is invalid. A valid header-only file yields no chunks.
- Validation and file opening occur when iteration begins.
- A later parsing failure does not invalidate previously yielded chunks.
  The current incomplete chunk is not yielded; downstream training
  already performed on earlier chunks is not rolled back.
- Exhaust or close the generator to release its file handle.
- The reader does not impute, scale or automatically route categorical
  feature columns.

## Plotting contracts (2026-11-16)

The public functions are:

- plot_metric_over_time(metric_values, title, ylabel)
- compare_models(metric1, metric2, labels)
- plot_predictions_vs_ground_truth(y_true, y_pred)

Common behaviour:

- Return (fig, ax); support optional ax, save_path and show arguments.
- Import pyplot lazily and do not select a backend inside the module.
- Create a figure when ax is omitted; otherwise append artists without
  clearing existing axes. Titles, labels and other documented axis
  settings may change.
- Output parent directories must already exist.
- Successful figures belong to the caller, who should close them when
  finished. Close a newly created figure if drawing, saving or showing
  fails. Keep caller-supplied figures open; do not promise rollback of
  modifications to their axes.
- Tests use Agg. The demonstration notebook uses inline display.

Metric curves:

- Require non-empty, one-dimensional real numeric series.
- Preserve NaN values as gaps, including all-NaN warm-up series.
  Reject infinity and non-numeric, boolean or complex array dtypes.
- Default coordinates are 1 through n. Explicit x_values must be finite,
  strictly increasing and match the series length.
- NumPy input dtype inference occurs before numeric dtype validation;
  the checks do not recover original element types after coercion.

Model comparisons:

- Require equal-length series at shared coordinates.
- Require two distinct, non-empty string labels after whitespace stripping.
- The legend describes the two newly added comparison lines.
- Callers must ensure metric meaning and evaluation intervals agree.
  The plotting function cannot establish experimental comparability.

Prediction plots:

- Use the shared homogeneous integer/string label contract. Require
  non-empty, equal-length true/predicted arrays of the same label family.
- Map the sorted union of true and predicted classes to categorical
  positions. Preserve integer label precision when building the mapping.
- Draw true labels as open circles and predictions as crosses.
- Default sample positions are 1 through n. Explicit sample_indices must
  be finite, strictly increasing and match the label count.
- Set y-axis ticks and limits for the current class mapping. Existing
  artists are meaningful only if they use a compatible mapping.
- Exclude warm-up samples without predictions before calling.

## Streaming demonstration protocol (2026-11-16)

- demo/stream_demo.ipynb uses the supplied synthetic classification CSV.
  Generation rules and the original generation environment are recorded
  in demo/data/stream_classification_source.md.
- The dataset contains 1,200 rows, three numeric features and integer
  binary labels. CSV row order simulates arrival order; no real
  timestamps or intentional concept drift are claimed.
- Read chunks of 128 rows. Use Imputer, StandardScaler and a decision
  tree with max_depth=5 and random_state=2026.
- Score each chunk before learning from it. Preserve pre-learning
  predictions and original row positions for plotting and accuracy
  checks; do not recompute historical predictions with the final model.
- The buffer-300 run first trains on chunk 3, evaluates 816 rows,
  retrains on chunks 3/6/9 with 384 rows each and retains 48 final rows.
- The independent buffer-600 run first trains on chunk 5 using 640 rows,
  evaluates 560 rows and retains 560 final rows.
- Compare chunk accuracy only on common evaluation chunks 6 through 10.
  Preserve earlier positions as NaN. The last chunk contains 48 rows,
  so an unweighted average of chunk accuracies is not overall accuracy.
- Show the first 50 evaluated samples from the buffer-300 run, at data
  row positions 385 through 434, without selection by correctness.
- Compute raw-data statistics in a separate CSV pass. Include warm-up
  rows in descriptive statistics without feeding these full-stream
  statistics back into earlier model predictions.
- Display all three specified plot types and save PNG files under
  demo/figures. Repeated execution overwrites those generated images.
- Recreate experiment state in the relevant code cells. Restarting the
  kernel and running all nine code cells was verified in VS Code.
- Demonstration arrays, statistics state and the second trainer are
  outside the original trainer's reported array-memory scope.
- Demonstration accuracy and timings do not replace formal benchmarks
  or establish real-world predictive performance.

## Benchmark measurement contracts

- Loop references are benchmark-only implementations; production APIs
  and model behaviour are unchanged.
- Split comparisons use the same observed-sample partitions, criteria
  and global gain-tie tolerance of 1e-12.
- Confusion timing compares complete batch operations, including fresh
  streaming metric construction and result retrieval; it does not
  isolate the counting kernel.
- Buffer comparisons fix data order, preprocessing, tree parameters
  and chunk size. Accuracy is computed from sample counts, with warm-up
  excluded and a common evaluation interval reported separately.
- A retraining duration covers the entire triggering
  Pipeline.partial_fit call, including preprocessing and publication.
- Memory diagnostics count reachable ndarray storage. Python overhead,
  temporary allocations and process peak RAM are outside this measure.
- Exact cumulative AUC, exact-quantile history and event logs can grow.
  A bounded training buffer does not make the whole framework bounded.
- Final partial buffers are retained; experiments do not force a flush.

Measured evidence and reproduction:
[Streaming benchmarks](assignment-2.2-benchmarks.md).
