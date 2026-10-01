# NumCompute API Contract

Status: implementation baseline
Version: 0.1

This document defines the shared rules for implementation.
It does not indicate that the computational functions are implemented.

## 1. Dependencies

- Computational code uses only the Python standard library and NumPy.
- Tests use unittest and numpy.testing.
- setuptools is a build dependency, not a computational dependency.
- Jupyter is a tool for running the demo notebook.

## 2. Input shapes

- Tabular data X has shape (n_samples, n_features).
- Preprocessors require 2D input; they do not automatically reshape 1D input.
- Classification labels have shape (n_samples,).
- True and predicted classification labels must have equal lengths.
- MSE accepts 1D or 2D numeric inputs with exactly matching shapes.
- Sorting of values, ranking, Top-k, quickselect and binary search use 1D input.
- Multi-key sorting uses a 2D table.
- grad accepts a 1D input and returns a gradient with the same shape.
- jacobian accepts a 1D input and a function returning a 1D output.
- Jacobian output shape is (n_outputs, n_inputs).

## 3. Numeric types and input preservation

- Numeric calculations use float64 where conversion is needed.
- Convert integer inputs before operations such as squaring.
- Categorical data is handled separately from numeric data.
- Functions do not modify the caller's input arrays.
- Non-contiguous arrays, such as slices, must be supported.
- Integer parameters such as k and batch_size reject boolean values.

## 4. Axis conventions

- axis=None reduces over all elements.
- axis=0 reduces over rows, producing per-column results for 2D input.
- axis=1 reduces over columns, producing per-row results for 2D input.
- Statistical reductions support a single integer axis, including negative axes.
- Tuple axes are outside the baseline scope.

## 5. Errors

- TypeError: unsupported input or parameter type.
- ValueError: invalid shape, value, range or parameter option.
- RuntimeError: transform or predict is called before fitting.
- FileNotFoundError: an input file does not exist.
- Error messages must explain the problem clearly.

## 6. NaN handling

Statistical reductions, quantile, percentile and rank support:

- nan_policy="propagate": return NaN for affected results.
- nan_policy="omit": ignore NaN values when calculating.
- nan_policy="raise": reject inputs containing NaN.

The default is "propagate".

For statistical reductions, propagation applies separately to each
reduction slice.

For 1D ranking:
- propagate: any NaN makes all output ranks NaN.
- omit: rank valid elements and leave original NaN positions as NaN.
- raise: reject an input containing NaN.

Scalers:
- Ignore NaN when fitting each column.
- Preserve NaN during transformation.
- Reject a training column containing only NaN.
- Reject positive and negative infinity.

Constant imputation:
- Fill NaN using a specified finite constant.
- Allow a column containing only NaN.

Top-k and quickselect:
- Reject NaN.
- Allow positive and negative infinity.

Binary search:
- The caller provides an ascending, NaN-free array.
- Reject a NaN target.
- Allow positive and negative infinity.
- Do not scan the whole input to validate these preconditions.

Metrics and numerical differentiation:
- Reject non-finite numeric input.
- String classification labels are validated as categories.

Special-value rules for mathematical functions are specified separately.

## 7. Empty inputs

- Sorting and ranking may return empty arrays.
- Top-k allows an empty input only when k=0.
- Binary search on an empty array returns (0, False).
- quickselect, preprocessing fit and metrics reject empty samples.
- Statistical reductions return NaN when no valid values remain.
- Variance and standard deviation return NaN when
  valid_count <= ddof.
- ddof must be a non-negative integer.
- Histogram and batching have separate empty-input rules.

## 8. Fitted transformers

All built-in preprocessors provide:

- fit(X, y=None)
- transform(X)
- fit_transform(X, y=None)

Rules:
- fit returns self.
- Repeated fit replaces previous fitted state.
- transform checks fitted state and feature count.
- fit_transform is equivalent to fit followed by transform.
- Training parameters are learned only from training data.
- Transforming test data must not refit the transformer.

## 9. Documentation and tests

Each public function or method documents:
- Parameters and accepted types.
- Input and output shapes.
- Return values.
- Errors and edge cases.
- Time and additional space complexity.

Tests use hand-calculated examples, analytic derivatives,
simple independent reference implementations or NumPy references.

Floating-point comparisons use appropriate tolerances.

The final project requires at least 20 meaningful functional tests.
Package setup checks do not replace functional tests for computational modules.

## 10. Changes to this contract

When an interface changes, update the corresponding:
- Implementation.
- Tests.
- API documentation.
- README examples.
- Demo notebook.

Sections 11–21 define the baseline module interfaces.
Any later changes must be recorded and reflected in the
implementation, tests and documentation.

## 11. Sorting, selection and ranking

### Stable sorting

stable_sort(values, descending=False, return_indices=False)

- Accept a 1D numeric array.
- Reject NaN; allow positive and negative infinity.
- Equal values retain their original order, including descending sort.
- Return sorted values by default.
- If return_indices=True, return (sorted_values, original_indices).

### Multi-key sorting

multi_key_sort(X, columns, return_indices=False)

- Accept a 2D numeric table.
- columns lists keys from primary to secondary.
- All keys use ascending order in the baseline implementation.
- Equal key combinations retain their original row order.
- Reject NaN and invalid, repeated or empty column selections.
- Return sorted rows, or (sorted_rows, original_row_indices).

### Top-k

topk(values, k, largest=True, return_indices=True)

- k must be an integer satisfying 0 <= k <= len(values).
- largest=True returns values in descending order.
- largest=False returns values in ascending order.
- Ties are resolved by original index, from smallest to largest.
- This rule applies both to selection at the boundary and output order.
- Return (selected_values, original_indices) by default.
- If return_indices=False, return only selected_values.
- k=0 returns empty outputs.

### Quickselect

quickselect(values, k)

- Return the zero-based kth smallest value.
- Require 0 <= k < len(values).
- Work on a copy of the input.
- Implement the selection algorithm manually.
- Do not replace it with NumPy partition or a full sort.

### Binary search

binary_search(values, target)

- Input must already be sorted in ascending order.
- Return (left_insertion_index, found).
- Repeated values use the first matching index.
- Reject NaN in the target.
- Sorted, NaN-free input is a caller precondition.
- Do not scan the full array on each call, preserving O(log n) search.

### Ranking

rank(values, method="average", nan_policy="propagate")

- Ranks start at 1.
- average: tied values receive their average rank.
- dense: tied values share a rank, without gaps between groups.
- ordinal: ties receive successive ranks in original input order.
- Return a float64 array with the same shape as the input.

### Percentiles and quantiles

percentile(values, q, method="linear",
           axis=None, nan_policy="propagate")

quantile(values, q, axis=None,
         method="linear", nan_policy="propagate")

- percentile q is in [0, 100].
- quantile q is in [0, 1].
- q is a finite scalar or a 1D array.
- Supported methods: linear, lower, higher and midpoint.
- A scalar q adds no output dimension.
- A 1D q adds a leading dimension of length len(q).
- Remaining dimensions follow the reduction axis.
- Both functions share the same interpolation logic.

## 12. Preprocessing

### StandardScaler()

- Use population standard deviation: ddof=0.
- Store mean_, scale_ and n_features_in_.
- For a constant training column, set scale_ to 1.
- Transform using (X - mean_) / scale_.

### MinMaxScaler(feature_range=(0, 1))

- Require finite bounds with lower < upper.
- Store data_min_, data_max_ and n_features_in_.
- A constant training column maps to the lower bound.
- For a zero training range, use an effective range of 1.
- Do not clip transformed new data by default.

### SimpleImputer(strategy="constant", fill_value=0.0)

- The baseline supports only the constant strategy.
- fill_value must be a finite numeric scalar.
- Fill NaN values and preserve other values.
- Mean and median strategies are optional extensions.

### OneHotEncoder(handle_unknown="error")

- Store categories_ and n_features_in_.
- Record categories in first-appearance order within each column.
- Concatenate outputs in original feature-column order.
- Return a dense float64 array.
- Each column contains strings or real numeric categories;
  do not mix strings and numbers within a column.
- Treat None and numeric NaN as one missing category.
- If missing values appear during fit, record that category.
- Reject positive and negative infinity.

Unknown categories:
- error: raise ValueError.
- ignore: output zeros for that feature's category block.
- A missing category not observed during fit is also unknown.

### ColumnSelector(columns, dtype=None)

- Select columns in the specified order and preserve 2D shape.
- Reject empty, duplicate, negative or out-of-range indices.
- dtype=None preserves the input dtype.
- An explicit dtype allows conversion for a numeric branch.
- Provide fit, transform and fit_transform.

All preprocessors:
- fit rejects zero samples or zero features.
- transform permits zero samples with the correct feature count.

## 13. Evaluation metrics

accuracy(y_true, y_pred)

mse(y_true, y_pred)

precision(y_true, y_pred, average="binary",
          pos_label=1, zero_division=0)

recall(y_true, y_pred, average="binary",
       pos_label=1, zero_division=0)

f1(y_true, y_pred, average="binary",
   pos_label=1, zero_division=0)

confusion_matrix(y_true, y_pred, labels=None)

Classification:
- Labels are homogeneous strings or finite real numbers.
- Reject missing labels and mismatched lengths.
- average supports binary, macro, micro and weighted.
- These averaging options are a project scope decision.
- binary permits at most two observed classes.
- For binary averaging, pos_label must be an observed class.
- zero_division accepts 0 or 1 and controls zero-denominator results.
- macro uses equal class weights.
- micro aggregates class counts before calculation.
- weighted uses true-label support as weights.
- Calculate per-class F1 directly from TP, FP and FN before averaging.

Confusion matrix:
- Rows represent true labels; columns represent predicted labels.
- Return (matrix, labels_array).
- Default label order is first appearance in y_true, then any
  additional labels in y_pred.
- Explicit labels must be unique and include all observed labels.
- Explicit labels may include classes with no observations.

MSE:
- Accept only numeric 1D or 2D inputs with identical shapes.
- Reject empty arrays and non-finite values.
- Return one scalar mean squared error over all elements.
- Do not allow broadcasting between different input shapes.

## 14. Pipeline and feature composition

Pipeline(steps)

- steps is a non-empty sequence of (name, object) pairs.
- Names must be unique.
- Intermediate steps provide fit and transform.
- The final step may be a Transformer or an Estimator.
- fit(X, y=None) fits and transforms intermediate steps,
  then fits the final step; it returns self.
- transform(X) is available when the final step supports transform.
- predict(X) is available when the final step supports predict.
- Unsupported operations raise TypeError.
- Calling transform or predict before fit raises RuntimeError.
- Prediction and transformation never refit any step.
- fit_transform is supported when the final step is a Transformer.

FeatureUnion(transformers)

- Accept a non-empty sequence of uniquely named Transformers.
- Fit every branch on the same input.
- Each transformed output must be 2D and preserve sample count.
- Concatenate outputs by columns, in declared branch order.
- Provide fit, transform and fit_transform.
- Use ColumnSelector inside branches for mixed numeric/category data.

Shared rules:
- Use the supplied step objects; do not automatically clone them.
- Do not reuse one fitted-state object in multiple branches or steps.
- Transformer and Estimator protocols use structural interfaces;
  users do not need to inherit from a specific base class.
- Compose means sequential composition through Pipeline.
- No separate Compose class is required by this baseline.

## 15. CSV input

load_csv(path, delimiter=",", dtype=float, skip_header=0,
         usecols=None, missing="fill", fill_value=None,
         missing_values=("", "NA", "NaN", "nan"))

iter_csv_chunks(path, chunk_size=1024, delimiter=",",
                dtype=float, skip_header=0, usecols=None,
                missing="fill", fill_value=None,
                missing_values=("", "NA", "NaN", "nan"))

Shared rules:
- Return homogeneous 2D NumPy arrays.
- A single row has shape (1, n_features).
- A single column has shape (n_samples, 1).
- Support UTF-8 files and a single-character delimiter.
- Baseline input contains no quoted fields or embedded newlines.
- Reject quoted fields and inconsistent field counts.
- Skip blank lines.
- skip_header counts physical lines at the beginning of the file.
- usecols contains unique, non-negative column indices;
  preserve the specified column order.
- Strip surrounding whitespace before checking missing tokens.
- Apply missing-value handling to selected columns.

Missing values:
- missing="skip": discard rows with missing selected values.
- missing="fill": replace missing selected values.
- With fill_value=None, floating-point data uses NaN and
  string data uses an empty string.
- Integer data with missing values requires an explicit,
  representable integer fill value.
- Reject unsupported missing strategies.

Types:
- Support homogeneous real numeric or string dtypes.
- Structured dtypes are outside the baseline scope.
- Reject conversions that silently truncate strings or
  convert non-integer values into integers.

Empty data:
- load_csv raises ValueError if no usable data rows remain.
- iter_csv_chunks raises ValueError during iteration if
  no usable data rows exist.

Chunking:
- chunk_size must be a positive integer.
- Read incrementally rather than loading the entire file first.
- Yield chunk_size usable rows per chunk, except the last chunk.
- Use consistent column counts and conversion rules across chunks.
- Combining chunks must reproduce the values and shape of load_csv.
- Invalid data encountered later may raise after earlier chunks
  have already been yielded.

## 16. Descriptive statistics

mean(X, axis=None, nan_policy="propagate")

median(X, axis=None, nan_policy="propagate")

variance(X, axis=None, ddof=0, nan_policy="propagate")

std(X, axis=None, ddof=0, nan_policy="propagate")

min(X, axis=None, nan_policy="propagate")

max(X, axis=None, nan_policy="propagate")

- Accept real numeric arrays.
- Reject positive and negative infinity.
- Handle NaN according to nan_policy.
- Support axis=None or one integer axis.
- Remove the reduced axis from the output shape.
- Return a scalar when no dimensions remain.
- ddof must be a non-negative integer.
- Empty reductions and reductions without valid values return NaN.
- Variance and std return NaN when valid_count <= ddof.
- Do not deliberately emit warnings for these documented cases.
- nan_policy="raise" rejects NaN before calculating.
- Use numerically stable variance calculations.

quantile follows Section 11 and the same statistical input rules.

### Histogram

histogram(values, bins=10, range=None, nan_policy="omit")

- Accept a 1D real numeric array.
- Reject positive and negative infinity.
- Support nan_policy="omit" or "raise".
- Reject "propagate": integer counts cannot represent NaN results.
- bins is a positive integer or a finite, strictly increasing
  1D array containing at least two bin edges.
- range, when used with integer bins, is a finite (lower, upper)
  pair with lower < upper.
- Explicit bin edges determine the range; reject an additional
  range argument in this case.
- Bins include the left edge and exclude the right edge,
  except that the final bin includes its right edge.
- Ignore values outside the selected range.
- Return (counts, edges), where counts contains integer counts.
- Empty data produces zero counts.
- For empty data with integer bins and no range, use (0, 1).
- For constant data without a range, try (value-0.5, value+0.5).
- If valid distinct edges cannot be represented, raise ValueError
  and require a suitable explicit range or edge array.

## 17. Streaming statistics

StreamingStats(nan_policy="propagate")

Public interface:
- update(values): update from a 1D numeric batch and return self.
- reset(): clear accumulated state and return self.
- count: number of non-NaN observations incorporated.
- mean: running mean.
- M2: sum of squared deviations from the running mean.
- variance(ddof=0): return running variance.
- std(ddof=0): return running standard deviation.

Rules:
- Use Welford updates or mathematically equivalent batch merging.
- Reject positive and negative infinity.
- Empty updates leave the state unchanged.
- Before any valid observations, count=0, mean=NaN and M2=0.
- Return NaN for variance/std when count <= ddof.
- ddof must be a non-negative integer.
- omit skips NaN values.
- raise validates the whole batch before changing state.
- propagate makes mean, M2, variance and std NaN after a NaN
  is observed, until reset; count still counts non-NaN observations.
- The baseline handles a 1D stream, not separate 2D feature streams.

## 18. Stable mathematical utilities

sigmoid(x)

- Accept real numeric input of any shape and preserve its shape.
- Avoid overflow in exponential calculations for finite inputs.
- NaN propagates.
- Positive infinity returns 1.
- Negative infinity returns 0.

softmax(x, axis=-1)

- Accept real numeric input with at least one dimension.
- Preserve the input shape.
- axis is one integer.
- Reject a zero-length reduction axis.
- For finite inputs, use a stable max-shift calculation.
- Process special values separately within each reduction slice:
  - Any NaN makes the whole slice NaN.
  - If positive infinities exist, divide probability equally
    among those positions and assign zero elsewhere.
  - A slice containing only negative infinity returns all NaN.
  - Otherwise, negative-infinity positions receive zero.

logsumexp(x, axis=None, keepdims=False)

- Accept real numeric arrays.
- axis is None or one integer.
- Use a numerically stable calculation.
- NaN propagates within each reduction slice.
- Positive infinity returns positive infinity unless NaN is present.
- An all-negative-infinity or empty slice returns negative infinity.
- keepdims=True retains reduced dimensions with size 1.

Shared numeric rules:
- Reject complex data and implicit conversion of numeric strings.
- Reject boolean data in numerical calculation functions.
- Return float64 numeric results.
- Floating-point range still limits representable results;
  stability does not guarantee every mathematical result is finite.

## 19. Distances and batching

euclidean_distances(X, Y=None)

- X has shape (n_samples, n_features).
- Y has shape (m_samples, n_features).
- Y=None means compare X with itself.
- Require finite real numeric values and equal feature counts.
- Require at least one feature.
- Allow zero samples.
- Return shape (n_samples, m_samples).
- Avoid avoidable cancellation when comparing nearby large values.
- Consider intermediate memory usage and use blocks if necessary.
- Do not modify either input.

batch_iter(X, batch_size=32, shuffle=False,
           random_state=None, drop_last=False)

- Accept an array with at least one dimension.
- Batch along the first dimension.
- batch_size must be a positive integer.
- Return an iterator yielding independent array copies.
- Empty input yields no batches.
- drop_last=False keeps a final smaller batch.
- drop_last=True discards a final incomplete batch.
- shuffle=False preserves sample order.
- random_state accepts None or a non-negative integer seed.
- Use a local NumPy random generator.
- Do not modify global random state.
- The same input and seed reproduce the shuffled order.

## 20. Numerical differentiation

grad(f, x, method="central", h=1e-5)

- x is a non-empty, finite 1D real numeric array.
- f returns a finite real scalar.
- Return a float64 gradient with the same shape as x.

jacobian(f, x, method="central", h=1e-5)

- x is a non-empty, finite 1D real numeric array.
- f returns a non-empty, finite 1D real numeric array.
- The output length must remain constant across evaluations.
- Return shape (n_outputs, n_inputs).

Shared rules:
- Support method="central" and method="forward".
- h is a positive, finite real scalar absolute step size.
- Reject boolean h.
- Each perturbed coordinate must remain finite and differ
  from the original floating-point value.
- Central differences use positive and negative perturbations.
- Forward differences reuse one baseline evaluation of f(x).
- Give f independent input copies.
- Do not modify the caller's x.
- Reject invalid function-output types, shapes and non-finite values.
- Document function-evaluation counts and include evaluation cost
  in the complexity description.
- Loops over input dimensions are allowed.
- Validate against functions with known analytical derivatives.

## 21. Benchmarking

benchmark(func, *args, warmup=1, repeats=5, **kwargs)

- warmup is a non-negative integer.
- repeats is a positive integer.
- Use time.perf_counter.
- Run warmup calls before measured calls.
- Record each measured duration in seconds.
- Return a dictionary containing:
  - times_seconds
  - median_seconds
  - min_seconds
- Propagate errors from the measured function.
- The caller verifies correctness before timing.
- Data generation and unrelated printing are outside timed calls.
- State whether timing includes fitting, allocation or copying.

environment_info()

- Return Python, NumPy, operating-system and CPU information.
- Use "unknown" when a detail cannot be obtained reliably.

Comparisons:
- Use fixed random seeds and identical input data.
- Compare loops and vectorised implementations for mean,
  standardisation and MSE.
- Label full-sort versus argpartition Top-k as an algorithm
  comparison, not a loop-versus-vectorisation comparison.
- Define speedup as baseline_time / vectorised_time.
- Report actual measurements; do not assume a speedup.
- Do not use performance thresholds as ordinary unit-test gates.

## 22. Shared internal validation helpers

These helpers are shared implementation dependencies, not package-level public APIs.

| Interface | Contract |
|---|---|
| `_as_float_array(x, *, name="x", ndim=None, finite=False)` | 实数数值输入转换为独立 `float64` 数组；支持非连续数组；拒绝布尔、复数、字符串及 object 数组；可检查维数与有限性；不规则形状报 `ValueError` |
| `_validate_integer(value, *, name, minimum=None)` | 接受 Python/NumPy 整数，拒绝布尔值；可检查最小值；返回 Python `int` |
| `_validate_axis(axis, ndim)` | 只接受单个整数轴，拒绝布尔值；检查范围并将负轴规范化为非负轴 |

Additional utility rules:
- batch_iter validates parameters when iteration starts.
- shuffle and drop_last must be booleans.
- random_state is validated even when shuffle=False.
- Callers must not modify the original input during iteration.
- Copies of object arrays retain references to contained Python objects; they are not deep copies.
