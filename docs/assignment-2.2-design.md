# Assignment 2.2 - Design contract

Batch and standalone buffered tree behaviour is implemented and tested. Preprocessing, statistics, metrics, Pipeline and StreamTrainer sections describe planned behaviour.

## Inputs and state

- X: shape (n_samples, n_features); y: shape (n_samples,).
- Labels: homogeneous integers or homogeneous strings.
- Reject boolean, floating, mixed-type and missing labels.
- Preserve integer labels without conversion through float.
- Each successful tree fit creates a sorted class table from that training
  dataset. Buffered retraining does not retain absent historical classes.
- Planned cumulative metric class tables append new classes without
  rearranging existing entries; this differs from the tree's class table.
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

The Pipeline is the sole raw-data buffer owner when a tree is composed.
It uses the final tree's buffer_size and invokes candidate_tree.fit at
the threshold, rather than buffering the same data through tree.partial_fit.

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

This deliberate scope difference prevents downstream accumulators from
mixing changing upstream representations.

External incremental transformers need compatible partial_fit/transform
methods and an explicit way to create an unfitted independent instance.
The reset/configuration-copy protocol must be specified before external
component integration; arbitrary objects are not assumed safely cloneable.

Inspect method signatures before dispatch. Support X-only fitting methods
and compatible y arguments. Never catch an internal TypeError and retry
as if it were a signature mismatch.

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

- score_chunk predicts and updates metrics; it does not train.
- fit_chunk learns; it does not update prediction metrics.
- Evaluate each chunk before learning it.
- Warm-up samples are excluded from evaluation denominators.
- Logs distinguish chunk, cumulative and rolling metrics, buffer counts,
  retraining events and timing.
- Memory measurements state their scope; array bytes are not process RSS.

## Planned focused verification

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
- Planned Pipeline composition owns its own buffer and invokes tree.fit;
  its future retraining logs must not treat the standalone tree's buffered
  counter as a Pipeline-wide event counter.