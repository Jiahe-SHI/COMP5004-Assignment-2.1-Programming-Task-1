# NumCompute

A modular scientific computing toolkit built with Python and NumPy for COMP 5004 Assignment 2.1. It provides tabular data loading, preprocessing, sorting and ranking, descriptive statistics, evaluation metrics, numerical differentiation, composable pipelines, and reproducible performance comparisons.

The computational package uses only the Python standard library and NumPy.

## Requirements and installation

- Python **3.10 or later**.
- NumPy **1.24 or later**, installed automatically with the package.
- The recorded integration benchmark used Python **3.12.6** and NumPy **1.26.4** on Windows 11.

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
```

Editable installation makes local package changes available without reinstalling. Run the commands below from the repository root unless a section explicitly changes directory.

## Package overview

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

The `benchmark/` runner and `demo/` are repository resources; the installed distribution includes the `numcompute` package. Keep the checkout available to run the examples and full benchmarks.

## Team contributions

| Member | Contributions |
| --- | --- |
| Jiahe | Utilities, shared I/O work, preprocessing, statistics, branch review/integration and final project checks |
| Crisanto | Sorting, searching, ranking, percentiles and final report |
| Shubh | Numerical differentiation, pipelines, notebook demonstration and demo video |
| Lingqi | Shared I/O work, evaluation metrics and benchmarks |


## Assignment 2.2: batch and buffered decision tree

This personal extension builds on the original NumCompute toolkit.
The original `numcompute` package is retained; new functionality is
provided by `numcompute_stream`.

Currently implemented:

- Shared feature, label and parameter validation.
- Numeric decision tree classification with Gini or entropy.
- Vectorised split evaluation and batch prediction.
- `fit`, `partial_fit`, `retry_pending_fit`, `predict` and `predict_proba`.
- Whole-buffer retraining, failure recovery and observable training state.

Buffered training is implemented for the standalone decision tree.
Batch `fit` trains immediately on all supplied rows. `partial_fit` copies
incoming chunks and trains once on the entire pending buffer when its
sample count reaches or exceeds `buffer_size`.

Incremental preprocessing, statistics, metrics, Pipeline, StreamTrainer,
streaming CSV integration and visualisation remain pending.

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

### Tests

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests/stream -t . -v
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The second command includes both original and streaming-package tests.

### Current limitations

- Gains <= 1e-12 are treated as numerical zero.
- Recursive construction is subject to Python's recursion limit.
- Split-search temporary memory grows with sample, class and feature counts.
- Feature subsampling may stop at a leaf when the selected subset has no
  useful split.
- Performance benchmarks and end-to-end streaming evaluation remain pending.

See `docs/assignment-2.2-design.md` for detailed numerical policies.