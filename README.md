# NumCompute

A modular scientific computing toolkit built with Python and NumPy for COMP 5004 Assignment 2.1. It provides tabular data loading, preprocessing, sorting and ranking, descriptive statistics, evaluation metrics, numerical differentiation, composable pipelines, and reproducible performance comparisons.

The original computational package uses the Python standard library and
NumPy. The Assignment 2.2 extension additionally uses matplotlib for
visualisation.

The original `numcompute` package was developed as a group project for
Assignment 2.1. Its documentation and contribution records are retained
below.

For Assignment 2.2, this repository is being extended individually with
the new `numcompute_stream` package on the `Jiahe-assignment-2.2` branch.
The Assignment 2.2 section later in this README documents the new
streaming components, their current implementation status and limitations.

## Requirements and installation

- Python **3.10 or later**.
- NumPy **1.24 or later**, installed automatically with the package.
- Matplotlib, installed automatically with the package for streaming visualisation.
- The original Assignment 2.1 integration benchmark used Python **3.12.6** and NumPy **1.26.4** on Windows 11. The verified Assignment 2.2 environment is recorded in the streaming section below.

Download or clone this repository and open a terminal in its root directory (the directory containing `pyproject.toml`).

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

If PowerShell prevents activation, use the environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -c "import numcompute; print(numcompute.__version__)"
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

After activation, check the installation:

```bash
python -c "import numcompute; print(numcompute.__version__)"

python -c "import numcompute_stream; print(numcompute_stream.__all__)"
```

Editable installation makes local package changes available without reinstalling. Run the commands below from the repository root unless a section explicitly changes directory.

## Package overview

The sections from this package overview through Team contributions
document the original Assignment 2.1 `numcompute` toolkit. Its API
conventions and limitations apply to that package. The Assignment 2.2
section below documents `numcompute_stream`, which provides separate
streaming APIs and different state-management rules.

| Module | Main functionality |
| --- | --- |
| `numcompute.io` | `load_csv`, `iter_csv_chunks`; delimiter, dtype, column selection and missing-value handling |
| `numcompute.preprocessing` | `SimpleImputer`, `StandardScaler`, `MinMaxScaler`, `OneHotEncoder`, `ColumnSelector` |
| `numcompute.sort_search` | `stable_sort`, `multi_key_sort`, `topk`, hand-written `quickselect`, `binary_search` |
| `numcompute.rank` | Average, dense and ordinal ranking; percentiles |
| `numcompute.stats` | Mean, median, variance, standard deviation, min/max, histogram, quantiles and Welford streaming statistics |
| `numcompute.metrics` | Accuracy, precision, recall, F1, confusion matrix and MSE |
| `numcompute.optim` | Forward and central finite-difference gradients and Jacobians |
| `numcompute.pipeline` | Sequential `Pipeline` and parallel feature concatenation with `FeatureUnion` |
| `numcompute.utils` | Stable sigmoid, softmax and logsumexp; Euclidean distances; batching |
| `numcompute.benchmarking` | Repeated timing and environment information |

Function and class docstrings describe parameters, shapes, exceptions and computational complexity. For example, run `help(numcompute.stats.mean)` after importing `numcompute.stats`.

## Quick start

### Load data and build a feature pipeline

This example uses the same numeric processing order as the notebook: **select columns → standardize → fill remaining NaNs with zero**. The scaler learns statistics from observed training values; zero in standardized space represents the training mean.

Run this Python code from the repository root:

```python
import numpy as np
from numcompute.io import load_csv
from numcompute.pipeline import Pipeline, FeatureUnion
from numcompute.preprocessing import (
    ColumnSelector, StandardScaler, SimpleImputer, OneHotEncoder,
)

path = "demo/data/sample.csv"
numeric = load_csv(path, skip_header=1, usecols=[0, 1, 2, 3])
city = load_csv(path, skip_header=1, usecols=[4], dtype=str)

# Convert the reader's empty strings into explicit missing categories.
city = np.where(city == "", None, city).astype(object)
table = np.empty((len(numeric), 5), dtype=object)
table[:, :4] = numeric
table[:, 4] = city[:, 0]

numeric_branch = Pipeline([
    ("select", ColumnSelector([0, 1, 2, 3], dtype=float)),
    ("scale", StandardScaler()),
    ("impute", SimpleImputer(strategy="constant", fill_value=0.0)),
])
category_branch = Pipeline([
    ("select", ColumnSelector([4])),
    ("encode", OneHotEncoder(handle_unknown="ignore")),
])
features = FeatureUnion([
    ("numeric", numeric_branch),
    ("city", category_branch),
])

X = features.fit_transform(table)
print(X.shape)                        # (30, 9)
print(bool(np.isfinite(X).all()))     # True

new_rows = np.array([
    [30, 170.0, 65.0, 50.0, "Sydney"],
    [40, np.nan, 80.0, 70.0, "Brisbane"],
], dtype=object)
print(features.transform(new_rows).shape)  # (2, 9)
```

Fit on training data, then call `transform` on new data to reuse the learned statistics and categories. An unseen city produces an all-zero categorical block when `handle_unknown="ignore"`. Use separate component instances for separate branches.

A `Pipeline` can also end in an estimator implementing `fit` and `predict`; call `pipeline.predict(X)` after fitting. No built-in predictive model is required or supplied. `FeatureUnion` applies each branch to the same input and concatenates its two-dimensional output by columns.

### Ranking, metrics and numerical derivatives

The following example is independent of the CSV example:

```python
import numpy as np
from numcompute import metrics, optim, rank, sort_search, stats

values = np.array([70.0, 85.0, 70.0, 90.0])
top_values, indices = sort_search.topk(values, 2)
print(top_values, indices)                    # [90. 85.] [3 1]
print(rank.rank(values, method="average"))    # [1.5 3.  1.5 4. ]
print(stats.mean(values))                     # 78.75
print(rank.percentile(values, 50))            # 77.5
print(sort_search.binary_search(np.sort(values), 85.0))  # (2, True)

y_true = [0, 1, 1, 0]
y_pred = [0, 1, 0, 0]
print(metrics.accuracy(y_true, y_pred))        # 0.75
print(metrics.f1(y_true, y_pred))              # approximately 0.6667
matrix, labels = metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
print(matrix)                                # [[2, 0], [1, 1]]
print(metrics.mse([1.0, 2.0], [1.0, 3.0]))    # 0.5

x = np.array([1.0, 2.0])
g = optim.grad(lambda z: np.sum(z ** 2), x, method="central", h=1e-5)
J = optim.jacobian(
    lambda z: np.array([z[0] + z[1], z[0] * z[1]]), x,
    method="central", h=1e-5,
)
np.testing.assert_allclose(g, [2.0, 4.0], rtol=1e-8, atol=1e-8)
np.testing.assert_allclose(J, [[1.0, 1.0], [2.0, 1.0]], rtol=1e-8, atol=1e-8)
print(g)                                     # approximately [2., 4.]
print(J.shape)                               # (2, 2)
```

## Run the demonstration notebook

With the virtual environment active, install the notebook tools and launch from `demo/`:

```bash
python -m pip install jupyterlab ipykernel
python -m ipykernel install --user --name numcompute --display-name "Python (NumCompute)"
cd demo
python -m jupyterlab quickstart.ipynb
```

Select **Python (NumCompute)** as the kernel, then **Restart Kernel and Run All Cells**. Execute Setup first. The notebook expects its working directory to be `demo/`, where `data/sample.csv` and `sample_data.csv` are available. If using VS Code, select the same virtual environment and ensure the notebook runs from that directory.

The notebook demonstrates CSV and chunked loading, mixed-data preprocessing, Pipeline/FeatureUnion, sorting and ranks, statistics, metrics, gradients/Jacobians, edge cases, and a small live benchmark. The main feature table has shape `(30, 9)` and contains no NaNs after preprocessing.

The notebook benchmark is an illustrative mean comparison with 2 warmups and 5 repetitions. Use the full runner below for the recorded multi-size experiments. Return to the repository root with `cd ..` before running tests or the benchmark runner.

## Tests

Run the complete suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

The original Assignment 2.1 suite contains **497 tests**, exceeding the assignment's minimum of 20. Tests use standard-library `unittest` and `numpy.testing`, with coverage for empty inputs, equal values, duplicates and ties, extreme `k`, NaNs, non-contiguous arrays, invalid types and shapes, fitted-state behavior, numerical derivatives, and benchmark result handling.

For a focused module check:

```bash
python -m unittest discover -s tests -p "test_metrics.py" -v
```

Floating-point checks use tolerances where appropriate. Passing tests provide evidence for the covered cases; they do not establish correctness for every possible input. Unit-test execution time is not a performance benchmark.

## Reproduce the benchmarks

From the repository root, run:

```bash
python -m benchmark.run_benchmarks
```

The runner verifies output equivalence before measuring and creates a new timestamped directory under `benchmark/results/` containing:

| File | Contents |
| --- | --- |
| `raw_timings.csv` | Individual measured call durations |
| `summary.csv` | Median/minimum times, speedups and equivalence flags |
| `environment.json` | Python/NumPy/OS/CPU, Git state, configuration and timing methodology |

The experiments use seed 42, float64 inputs, sizes 1,000 / 10,000 / 100,000, 3 warmup calls and 10 measured calls per implementation. Each timing sample measures one call with `time.perf_counter`. Input generation, equivalence checks and result export are outside timing; validation, allocation and copying within the measured function are included.

Mean, standardization and MSE compare Python loops with the package's NumPy-based implementation. Standardization includes both fitting and transformation on `(n, 10)` arrays with a constant first column. Top-k compares full sorting with argpartition-based selection; it is an **algorithm comparison**, not a Python-loop comparison. Ordinary numeric comparisons use `rtol=1e-10`, `atol=1e-12`; Top-k values and indices must match exactly.

### Recorded integration results

Source: [summary.csv](benchmark/results/run_20261024T040153_841771Z/summary.csv) and [environment.json](benchmark/results/run_20261024T040153_841771Z/environment.json), recorded on 24 October 2026. Environment: Python 3.12.6, NumPy 1.26.4, Windows 11; CPU identifier `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`.

Speedup = **baseline median time / candidate median time**. Values greater than 1 mean the candidate was faster in this run. All 15 cases passed equivalence checks.

| Comparison | n = 1,000 | n = 10,000 | n = 100,000 |
| --- | ---: | ---: | ---: |
| Mean: loop / vectorised | 2.75× | 13.20× | 7.99× |
| Standardization: loop / vectorised | 14.31× | 13.29× | 13.87× |
| MSE: loop / vectorised | 4.45× | 23.64× | 7.43× |
| Top-k: full sort / selection, k = 10 | 2.08× | 11.12× | 21.02× |
| Top-k: full sort / selection, k = n // 10 | 1.88× | 3.78× | 5.42× |

These are measured results, not guaranteed speedups. The runner measures the baseline before the candidate, and short single-call samples can be affected by system load and timer overhead. Results depend on hardware, versions, input size and `k`; consult the raw timings for variability. The notebook's live timing and other saved runs are separate experiments.

## API conventions and limitations

- **Shapes:** tabular preprocessing accepts `(n_samples, n_features)` arrays. Classification labels are one-dimensional. MSE requires identical, non-empty one- or two-dimensional input shapes and does not broadcast mismatched shapes. A gradient has shape `(n_inputs,)`; a Jacobian has shape `(n_outputs, n_inputs)`.
- **CSV subset:** readers use UTF-8 and a single-character delimiter and return homogeneous two-dimensional arrays. Quoted fields, embedded field newlines and structured dtypes are unsupported. Missing-value handling applies to selected columns; `missing="skip"` drops rows with missing selected values. Integer missing values need an explicit representable integer fill value.
- **Preprocessing:** `fit` returns `self`; `transform` requires a fitted object and matching feature count. StandardScaler uses population standard deviation (`ddof=0`) and scale 1 for constant columns. Scalers ignore NaNs during fitting, retain them during transformation, reject all-NaN training columns and reject infinities. MinMaxScaler does not clip new values.
- **Imputation and categories:** SimpleImputer supports only finite constant filling. OneHotEncoder produces dense output, records categories in first-appearance order, and supports unknown-category `error` or `ignore`. `None` and numeric NaN share a missing category; an empty string is an ordinary category unless explicitly converted.
- **NaNs:** statistics and ranking expose `nan_policy="propagate"`, `"omit"` or `"raise"`; check each function's default (histogram defaults to `omit`). Top-k and quickselect reject NaNs. Metrics and numerical differentiation reject invalid non-finite numeric inputs.
- **Ordering:** ranks start at 1; ordinal ties follow input order. Top-k breaks ties by original index, including boundary ties; `k=0` is supported. Quickselect uses a zero-based order statistic. Binary search returns `(leftmost_insertion_index, found)` and requires ascending, NaN-free input; it does not scan the array to verify that precondition.
- **Percentiles and metrics:** percentile `q` is in `[0, 100]`, quantile `q` in `[0, 1]`; supported interpolation methods are `linear`, `lower`, `higher`, `midpoint`. Precision/recall/F1 support binary, macro, micro and weighted averaging with explicit `pos_label`/`zero_division` options. Confusion matrices return `(matrix, labels)` with true classes in rows and predicted classes in columns.
- **Composition:** components are not cloned, and duplicate instances within a composition are rejected. Pipeline/FeatureUnion do not automatically copy inputs or roll back child state after a failed fit; custom components should avoid mutating shared input. Sequential composition is provided by Pipeline, with no separate Compose class.
- **Numerics:** stable activation formulas and input validation reduce common overflow problems, but finite float64 range and rounding still apply. Finite differences support `central` and `forward` methods with positive finite `h`; too-small or overflowing perturbations are rejected. Derivative quality depends on step size and function smoothness.
- **Scope:** mean/median imputation, ROC/AUC and line search are not implemented. The package does not depend on pandas, SciPy, scikit-learn or a machine-learning framework.

## Repository contents

| Path | Purpose |
| --- | --- |
| `numcompute/` | Installable computational package |
| `tests/` | Unit and integration tests |
| `demo/quickstart.ipynb` | End-to-end demonstration |
| `demo/data/sample.csv` | Main 30-row demonstration dataset |
| `demo/sample_data.csv` | Small dataset for missing-value and constant-column examples |
| `benchmark/baselines.py` | Reference loop and full-sort implementations |
| `benchmark/run_benchmarks.py` | Full experiment runner |
| `benchmark/results/` | Saved measurements and environment records |
| `docs/` | Supporting API and collaboration documentation |
| `pyproject.toml` | Build configuration and dependencies |
| `README.md` | Installation, usage, testing and benchmark guide |

The installed distribution includes both `numcompute` and
`numcompute_stream`. Benchmark scripts, notebooks and demonstration
data are repository resources. Keep the checkout available to run
the examples and full benchmarks.

## Team contributions

| Member | Contributions |
| --- | --- |
| Jiahe | Utilities, shared I/O work, preprocessing, statistics, branch review/integration and final project checks |
| Crisanto | Sorting, searching, ranking, percentiles and final report |
| Shubh | Numerical differentiation, pipelines, notebook demonstration and demo video |
| Lingqi | Shared I/O work, evaluation metrics and benchmarks |


## Assignment 2.2: streaming machine learning framework

This personal extension builds on the original NumCompute toolkit.
The original `numcompute` package is retained; new functionality is
provided by `numcompute_stream`.

Currently implemented:

- Shared feature, label and parameter validation.
- Numeric decision tree classification with Gini or entropy.
- Vectorised split evaluation and batch prediction.
- `fit`, `partial_fit`, `retry_pending_fit`, `predict` and `predict_proba`.
- Whole-buffer retraining, failure recovery and observable training state.
- Incremental StandardScaler, mean/constant Imputer and OneHotEncoder.
- StreamingStats with cumulative moments, exact global quantiles and
  optional fixed-edge histograms.
- Cross-component checks for preprocessing, tree inputs and independent
  copied snapshots.
- Cumulative confusion counts and classification metrics.
- Sample-based rolling classification metrics.
- Binary ROC AUC with cumulative and rolling evaluation state.
- Integration checks for chunk partitioning, dynamic labels, probability
  column alignment, independent snapshots and failed-update recovery.
- Incremental CSV reading with explicit feature/target selection,
  missing-value handling and row/column error reporting.
- CSV-to-Pipeline-to-StreamTrainer integration.
- Metric curves, aligned model comparisons and categorical prediction plots.
- A streaming demo notebook verified by restarting the kernel and running
  all cells, with inline figures and saved PNG outputs.

Buffered training is implemented for the standalone decision tree.
Batch `fit` trains immediately on all supplied rows. `partial_fit` copies
incoming chunks and trains once on the entire pending buffer when its
sample count reaches or exceeds `buffer_size`.

Pipeline and StreamTrainer are implemented, including paired
preprocessing/model publication, evaluation before learning, binary
probability alignment, event logs and retained-array memory accounting.

Streaming CSV integration, the three plotting interfaces and
demo/stream_demo.ipynb are implemented and verified. Formal vectorisation
and buffer-size benchmarks, retained-state growth checks and a longer
integrated stream have also been completed. The report has been prepared and the demonstration video recorded.
Clean-checkout reproduction has passed. Final deliverable placement,
links, access permissions and submission checks remain pending.

### Minimal example

```python
from numcompute_stream import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit([[0.0], [1.0], [2.0], [3.0]], ["low", "low", "high", "high"])

print(model.classes_.tolist())
# ['high', 'low']

print(model.predict([[0.5], [2.5]]).tolist())
# ['low', 'high']

print(model.predict_proba([[0.5], [2.5]]).tolist())
# [[0.0, 1.0], [1.0, 0.0]]
```

Probability columns follow `classes_`. Values are unsmoothed class
frequencies in each reached leaf.

### Input and state rules

- X has shape `(n_samples, n_features)` and contains finite real numbers.
- y has shape `(n_samples,)` and contains homogeneous integers or strings.
- Missing labels, boolean labels and floating labels are rejected.
- Trees reject NaN and infinity; missing features require preprocessing.
- Training requires at least one sample and one feature.
- Fitted models accept empty prediction input with the correct feature count.
- Successful fit replaces the learned model; failed fit preserves it.
- Prediction does not update training state.

### Buffered training example

```python
import numpy as np
from numcompute_stream import DecisionTreeClassifier

model = DecisionTreeClassifier(buffer_size=500, random_state=42)

for chunk_index in range(3):
    X = np.arange(200).reshape(-1, 1)
    y = (X[:, 0] >= 100).astype(int)
    model.partial_fit(X, y)
    print(model.n_samples_buffered_, model.n_retrains_)

# 200 0
# 400 0
# 0 1

print(model.n_samples_fit_)   # 600
print(model.model_version_)   # 1
print(model.predict([[0], [199]]).tolist())  # [0, 1]
```

The threshold is a trigger, not a strict memory limit. Overshoot is
included in one training event; no rows are carried into the next buffer
after successful training.

Before the first successful fit, both prediction methods raise RuntimeError.
Between training events, predictions and probability columns use the last
successful model. Each successful fit builds a sorted class table from its
own training data; absent classes are not retained automatically.

Incoming chunks must match the feature count and integer/string label
family of pending data and the active model. New class values are allowed.
A successful direct fit can establish a new schema and clears pending data.

Validation failures do not append a chunk. If training fails after
appending, the chunk remains buffered and the old model remains available.
After resolving the cause, call `model.retry_pending_fit()` rather than
resubmitting the same chunk. Retrying does not append samples or force
training below the current threshold.

| Attribute | Meaning |
| --- | --- |
| `is_fitted_` | Whether a successful model is available |
| `n_samples_buffered_` | Current pending sample count |
| `n_samples_fit_` | Rows used by the latest successful fit; absent before fitting |
| `n_retrains_` | Lifetime successful buffered fits, including the first |
| `model_version_` | Lifetime successful model publications, including direct fits |

The two lifetime counters start at zero. Direct fit advances the model
version but does not increment or reset the buffered-fit count. Failed
training, prediction and below-threshold updates do not advance either.

Buffers own independent copies of incoming arrays. Integer labels are
preserved when combining signed and unsigned types. When feature dtype
promotion requires floating storage, integer values outside that dtype's
consecutive exact-integer range are conservatively rejected. Such a failure
retains pending data; retrying without resolving the cause still fails.

### Incremental preprocessing and descriptive statistics

Import these components from `numcompute_stream`.

| Component | Public methods | Behaviour |
| --- | --- | --- |
| StandardScaler | fit, partial_fit, transform, fit_transform | Cumulative per-feature observed mean and population variance |
| Imputer | fit, partial_fit, transform, fit_transform | Observed-mean or constant filling; finite fallback for unobserved columns |
| OneHotEncoder | fit, partial_fit, transform, fit_transform | Incremental categories with unknown-value ignore/error policies |
| StreamingStats | update_stats, result, quantile, histogram, reset | Cumulative moments, exact global quantiles and optional fixed-edge histograms |

Preprocessor fit resets learning; partial_fit accumulates observations.
Transforms do not learn. Numeric NaN is ignored when estimating moments.
StandardScaler retains NaN during transformation and uses scale 1 for
constant or unobserved columns; Imputer supplies missing-value filling.

OneHotEncoder keeps existing category indices within each feature.
Expanding an earlier feature can shift later output blocks. None and
numeric NaN share a missing category, distinct from ordinary strings.
Dense output memory and category storage grow with data dimensions and
new categories.

StreamingStats.update_stats is an instance method. result returns
independent count, mean and population variance arrays plus total rows.
quantile uses complete retained valid history since reset, with linear
interpolation; its memory usage is not constant.
Optional bin_edges are fixed and shared across features. Histogram bins
include the final right endpoint and exclude NaN/out-of-range values.
reset clears learning and history while retaining configured bin edges.

Numeric statistics and imputation use float64 and retain finite-precision
and overflow limits. These APIs do not promise arbitrary extreme-number
support.

Pipeline rebuilds preprocessing and the decision tree together from each
eligible raw-data buffer. It publishes the complete fitted chain only
after successful training, preserving the previous prediction chain
when candidate training fails. External incremental transformers are
supported through compatible methods and an explicit clone_unfitted
protocol.

The verified streaming environment used Python 3.12.6, NumPy 2.5.5 and
matplotlib 3.11.2 on Windows. See the streaming Tests section below for
the latest regression totals and docs/assignment-2.2-testing.md for
commands and historical verification records. Test execution times
are not performance benchmarks.

### Cumulative and rolling metrics

The package exports StreamingConfusionMatrix,
StreamingClassificationMetrics, RollingClassificationMetrics,
binary_roc_auc, StreamingROCAUC and RollingROCAUC.

Stateful metrics provide update, result and reset. Updates require
nonempty one-dimensional inputs and return self. Queries do not learn.

```python
from numcompute_stream import (
    StreamingClassificationMetrics,
    RollingClassificationMetrics,
    StreamingROCAUC,
    RollingROCAUC,
)

cumulative = StreamingClassificationMetrics()
rolling = RollingClassificationMetrics(window_size=2)
auc = StreamingROCAUC(positive_label=1)
rolling_auc = RollingROCAUC(window_size=2, positive_label=1)

cumulative.update([0, 1], [0, 1])
rolling.update([0, 1], [0, 1])
auc.update([0, 1], [0.1, 0.2])
rolling_auc.update([0, 1], [0.1, 0.2])

cumulative.update([0, 1], [1, 1])
rolling.update([0, 1], [1, 1])
auc.update([0, 1], [0.8, 0.9])
rolling_auc.update([0, 1], [0.8, 0.9])

print(cumulative.result()["accuracy"])  # 0.75
print(rolling.result()["accuracy"])     # 0.5
print(auc.result()["auc"])              # 0.75
print(rolling_auc.result()["auc"])      # 1.0
```

Confusion matrix rows are true classes and columns are predictions.
Per-class precision, recall and F1 follow the returned class order.
Macro averages include every class in that table; weighted averages
use true-label support. zero_division accepts 0 or 1, defaulting to 0.
Without evaluated samples, scalar metrics and configured per-class
metrics return NaN.

Initial class order is preserved and new labels append. Rolling
classification retains observed classes even after their samples leave
the window, so absent classes still participate in macro averages.

Rolling windows contain the latest W evaluated samples. Their n_samples
describes the current window; n_samples_seen and n_samples_seen_ describe
all successful input since reset.

AUC requires an explicit positive label and finite scores that increase
towards that class. When supplying scores directly, select probability
columns using the current tree's classes_; do not assume the positive
class is column 1.

StreamTrainer performs this alignment automatically when positive_label
is configured. If the active model has no positive-class column, it
uses zero positive probability. AUC is disabled when positive_label is
None; enabled AUC is NaN when either class is missing from the evaluated
data. Multiclass AUC is unsupported.

Tied scores receive half credit. Missing either class yields NaN.
Stateful AUC retains its first negative label until reset, including
after that label leaves a rolling window. Multiclass AUC is unsupported.

Stateful AUC stores float64 scores and rejects integer scores that cannot
be represented exactly. Higher-precision floating scores follow float64
rounding. The standalone binary_roc_auc function preserves typed integer
scores during sorting.

Cumulative classification stores O(K^2) counts for K classes, without raw
sample history. Rolling classification retains at most W sample pairs
plus its growing class table and O(K^2) counts, and recounts the window
on update. Exact cumulative AUC retains all evaluated labels and scores;
its memory grows with history. Rolling AUC retains at most W sample pairs.
Update temporaries also depend on incoming chunk size. AUC queries sort
the retained data; these APIs do not promise constant-time updates or
queries.

### Buffered Pipeline and evaluation before learning

Import the streaming Pipeline from numcompute_stream. It is separate
from the original numcompute.pipeline.Pipeline.

```python
import numpy as np
from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    Pipeline,
    StandardScaler,
    StreamTrainer,
)

model = Pipeline([
    ("impute", Imputer(strategy="mean")),
    ("scale", StandardScaler()),
    ("tree", DecisionTreeClassifier(
        max_depth=2,
        buffer_size=4,
        random_state=42,
    )),
])
trainer = StreamTrainer(model, window_size=4, positive_label=1)

chunks = [
    (np.array([[0.0], [1.0]]), np.array([0, 0])),
    (np.array([[2.0], [3.0]]), np.array([1, 1])),
    (np.array([[0.5], [2.5]]), np.array([0, 1])),
]

for X_chunk, y_chunk in chunks:
    # Evaluate using only the previously published model.
    record = trainer.score_chunk(X_chunk, y_chunk)
    print(record["warm_up"], record["chunk_accuracy"])

    # Learning is a separate operation.
    trainer.fit_chunk(X_chunk, y_chunk)

print(trainer.n_samples_scored_)       # 6, including warm-up
print(trainer.n_samples_evaluated_)    # 2, excluding warm-up
print(model.n_retrains_)              # 1
print(model.n_samples_buffered_)       # 2
print(trainer.metrics_result()["cumulative"]["accuracy"])  # 1.0
```

The first two chunks are warm-up scoring events. The second learning
call reaches the four-row threshold and publishes the first fitted
chain. The third chunk is evaluated before being added to the next
training buffer. There is no automatic final-buffer flush.

Pipeline owns the raw-data buffer and trains the candidate tree with
fit. At each threshold, every preprocessing step is rebuilt from the
whole pending buffer. Its learned statistics describe that buffer,
rather than all historical observations. The active preprocessing
chain and tree remain paired and frozen between successful fits.

A failed candidate training attempt retains accepted pending data and
the previous active chain. After resolving the cause, call
trainer.retry_pending_fit() to retry learning without appending the
chunk or scoring it again. Repeated successful score_chunk calls are
counted as separate evaluation events; automatic deduplication and
automatic score-before-fit ordering are not provided.

Pipeline captures component configuration and creates fresh instances.
External transformers must provide partial_fit, transform and
clone_unfitted(). The clone must be a distinct instance of the same
type, with equivalent configuration and independent, unfitted state.
The external implementation is responsible for avoiding shared learned
state and source mutation. Signature inspection supports X-only and
compatible y arguments without retrying internal TypeError failures.

Numeric pipelines may accept NaN before imputation. A pipeline beginning
with the built-in OneHotEncoder accepts categorical input. Automatic
routing of mixed numeric and categorical column groups is not provided.

### Trainer logs and memory accounting

trainer.logs_ returns independent copies of event records. Successful
scoring calls and successful or failed learning calls are logged.
Learning failures record their exception type and are re-raised.
Failed scoring calls do not publish evaluation state or a log entry.

score_seconds measures the scoring operation, including metric work.
training_seconds measures the model update or retry call, including
buffer handling and any triggered training. It is not an isolated
tree-retraining time. Memory traversal and log insertion occur outside
these timings.

trainer.memory_result() reports reachable NumPy array nbytes for the
model and evaluation state. buffer_array_bytes is a subset of that
total, not an additional allocation. The same array object is counted
once; distinct overlapping views are counted separately.

This accounting excludes Python-object overhead, logs, temporary
allocations, process RSS and peak memory. It does not measure total
process memory. If log-time memory measurement fails, the event records
that measurement as unavailable.

Exact cumulative AUC retains all evaluated labels and scores, and
candidate metric copies duplicate this history during scoring.
Classification class tables and the event log can also grow. The
trainer therefore does not promise bounded memory or constant-time
updates.

### Streaming CSV input

Import the streaming reader from `numcompute_stream`; it is separate
from the original `numcompute.io` reader.

```python
from numcompute_stream import iter_csv_chunks

for X_chunk, y_chunk in iter_csv_chunks(
    "demo/data/stream_classification.csv",
    target_column="target",
    label_dtype="int",
    chunk_size=128,
):
    print(X_chunk.shape, y_chunk.shape)
```

The reader uses standard-library CSV parsing, requires a header and
yields the final short chunk. Features are numeric float64 arrays;
configured missing markers become NaN. Labels are strings or exact
Python integers in an object array. Missing labels and non-finite
non-missing features are rejected.

`feature_columns` optionally selects and orders features. By default,
all columns except the explicit target are selected. Categorical feature
columns require separate handling and are not automatically encoded.
Float64 conversion can round large integer feature values.

Validation occurs during iteration. If a later row fails, previously
yielded chunks remain valid; the incomplete current chunk is not yielded.
Exhaust or close the generator to release its file handle.

### Streaming visualisation

The package exports:

- `plot_metric_over_time(metric_values, title, ylabel)`
- `compare_models(metric1, metric2, labels)`
- `plot_predictions_vs_ground_truth(y_true, y_pred)`

All three return `(fig, ax)` and support optional `ax`, `save_path` and
`show` arguments. The caller owns successful figures and should close
them when finished. Output directories must already exist.

Metric plots preserve NaN gaps. Explicit `x_values` retain actual chunk
positions. Model comparisons require equal-length series at shared
positions; callers must ensure the metrics and evaluation intervals are
comparable.

Prediction plots map integer or string classes onto shared categorical
positions. Optional `sample_indices` retain original row positions.
Exclude warm-up rows without predictions before plotting.

The plotting module imports pyplot lazily and does not force a backend.
Automated plotting tests use Agg; the notebook uses inline display.

### Run the streaming demonstration

With the project virtual environment active, install notebook tools:

```powershell
python -m pip install ipython ipykernel nbformat nbclient
python -m ipykernel install --user --name numcompute-stream --display-name "Python (NumCompute Stream)"
python -m pip check
```

These are notebook development tools, separate from core package
dependencies. Open `demo/stream_demo.ipynb` in VS Code with Jupyter support,
select the project's virtual environment or registered kernel, restart
the kernel and run all cells. The notebook locates the repository from
its root or the `demo` directory.

The supplied CSV contains 1,200 synthetic samples and three numeric
features. Its generation rules and original generation environment are
documented in `demo/data/stream_classification_source.md`.

The notebook demonstrates:

- CSV chunks and paired imputation/scaling/tree training.
- Evaluation before learning, cumulative/rolling metrics and binary AUC.
- A common-interval comparison of buffer sizes 300 and 600.
- Recorded pre-learning predictions, raw-data statistics and event logs.
- Scoped array-storage accounting and its limitations.

The buffer-300 run evaluates 816 samples, retrains on chunks 3, 6 and 9
using 384 rows each, and retains a final 48-row buffer. The buffer-600
run evaluates 560 samples, trains once on 640 rows and retains 560 rows.
Neither run automatically flushes its final buffer.

The notebook displays and saves these figures under `demo/figures`:

- `stream_cumulative_accuracy.png`
- `stream_buffer_comparison.png`
- `stream_predictions_vs_truth.png`

Repeated runs overwrite those generated images. Restart-and-run-all
verification completed successfully with nine executed code cells and
`Demo completion checks: PASS`.

The verified notebook environment used IPython 9.18.1 and ipykernel 7.4.0;
nbformat 5.11.1 and nbclient 0.11.0 were installed. Verification was
performed through VS Code, not an automated nbclient execution.

Synthetic-data accuracy and notebook timings are demonstration results,
not evidence of real-world predictive performance or a formal benchmark.

### Tests

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/stream -t . -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The second command includes both original and streaming-package tests.
Latest verified regression: 406 streaming tests and 903 total tests
(497 original tests plus 406 streaming tests). The streaming suite
also passed with warnings treated as errors.

Fresh-checkout verification at commit `2ffa7c4` passed in a newly
created virtual environment: 406 streaming tests with warnings treated
as errors, 903 total tests, and all nine demo notebook code cells.
Both formal benchmark runners, retained-state and long-stream checks,
and plotting from the recorded benchmark results also passed.
See `docs/assignment-2.2-testing.md` for the verification record.

### Current limitations

- Gains <= 1e-12 are treated as numerical zero.
- Recursive construction is subject to Python's recursion limit.
- Split-search temporary memory grows with sample, class and feature counts.
- Feature subsampling may stop at a leaf when the selected subset has no
  useful split.
- In-memory and CSV streaming integration are tested; the streaming
  notebook and three plotting interfaces are verified.
- Formal vectorisation and buffer-size benchmarks, retained-state growth
  checks and a longer integrated stream have been verified. Results are
  specific to the recorded inputs and environment.
- Cumulative AUC, exact-quantile history and event logs grow with stream
  length. Array-storage diagnostics are not process RAM or peak memory.
- The demo uses synthetic data and does not establish real-world accuracy.

See `docs/assignment-2.2-design.md` for detailed numerical policies.

### Streaming benchmarks and long-stream checks

Run from the repository root with the project environment active:

```powershell
python -W error -m benchmark.run_stream_vectorization
python -W error -m benchmark.run_stream_buffers
python -W error -m benchmark.check_stream_state_growth
python -W error -m benchmark.check_stream_long_run
python -W error -m benchmark.plot_stream_results
```

The two timing scripts support --quick for smoke checks. Formal runs
save raw timings, configuration and source metadata in timestamped
directories. The plotting script uses explicitly selected recorded
results; it does not automatically plot the latest run.

See [benchmark methods, results and figures](docs/assignment-2.2-benchmarks.md).

Measured split-search speedups ranged from 2.309x to 7.473x.
The complete streaming confusion-matrix path was slower than the
batch loop reference at all measured sizes; these results are retained.

Buffer sizes 100, 300 and 600 were compared on the same prequential
evaluation interval. Retraining timings include the entire triggering
Pipeline.partial_fit call, not isolated tree fitting.

### Submission materials

The PDF report and demonstration video are submitted separately through
the course submission system. The video duration is 4 minutes 45 seconds.

The Assignment 2.2 implementation is on the
[Jiahe-assignment-2.2 branch](https://github.com/Jiahe-SHI/COMP5004-Assignment-2.1-Programming-Task-1/tree/Jiahe-assignment-2.2).

Fresh-checkout verification was performed at commit `2ffa7c4`.
Reproduction records are documented in
[the testing notes](docs/assignment-2.2-testing.md).
