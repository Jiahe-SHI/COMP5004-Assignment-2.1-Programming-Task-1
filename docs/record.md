# NumCompute — Team Scope and Responsibilities

## 1. Project scope

Build an installable scientific computing toolkit using only
Python standard-library modules and NumPy for computation.

| Module | Baseline scope |
|---|---|
| utils | Stable mathematical functions, distances, batching |
| io | Delimited-file loading, missing values, chunked reading |
| preprocessing | Scalers, OneHotEncoder, constant SimpleImputer, ColumnSelector |
| sort_search | Stable/multi-key sorting, Top-k, quickselect, binary search |
| rank | Average/dense/ordinal ranks and percentiles |
| stats | Descriptive statistics, quantiles, histogram, streaming statistics |
| metrics | Accuracy, precision, recall, F1, confusion matrix, MSE |
| optim | Forward/central gradients and Jacobians |
| pipeline | Transformer/Estimator protocols, Pipeline, FeatureUnion |
| benchmarking | Reproducible timing and performance comparisons |

Additional deliverables:
- Functional tests using unittest and numpy.testing.
- README with installation, usage and documented limitations.
- Demo CSV and an executable quickstart notebook.
- Packaging configuration.
- Actual benchmark results and environment information.

Scope decisions:
- Jacobians, chunked reading and streaming statistics are included.
- SimpleImputer initially supports constant filling only.
- Classification averaging supports binary, macro, micro and weighted.
- Compose is covered by sequential composition through Pipeline.
- ROC/AUC, line search and mean/median imputation are deferred extensions.
- No actual machine-learning model is required; use a Dummy Estimator
  to test the fit/predict protocol.

## 2. Proposed maintenance responsibilities

P1–P4 are role labels; actual member names remain to be recorded.
Shared responsibilities below are proposed for team confirmation.

| Role | Main modules | Shared maintenance |
|---|---|---|
| P1 | preprocessing, stats | utils and shared numeric validation |
| P2 | sort_search, rank | benchmarking and benchmark entry script |
| P3 | optim, pipeline | Packaging, API contract and integration checks |
| P4 | io, metrics | README, demo CSV and notebook integration |

Each member maintains tests and docstrings for their own modules.
Each member supplies demo examples and relevant benchmark comparisons.
The whole team reviews interface changes and final integration.

## 3. Development requirements

Declared compatibility targets:
- Python >= 3.10.
- NumPy >= 1.24.
- Verify the minimum-version combination before claiming tested compatibility.
- Use a project-local virtual environment (.venv).
- NumPy is the only third-party runtime dependency; setuptools is a build dependency.

Install and run tests from the project root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

## 4. Git rules

Track source code, tests, shared documentation and configuration.
Use docs/api_contract.md as the shared interface reference and the project plan for module scope and acceptance criteria.

Exclude:
- .venv/
- __pycache__/
- *.py[cod]
- *.egg-info/
- build/
- dist/
- .ipynb_checkpoints/
- .env

Review git status and staged changes before committing.
Use small commits with meaningful descriptions.
Do not overwrite teammates' work.

## 5. Information for team confirmation

- Complete original rubric for final requirement verification.
- Submission deadline, entry point and naming/archive rules.
- Whether any separate report or video is required.
- Actual member names and agreement on shared responsibilities.
- Minimum-version compatibility verification.
