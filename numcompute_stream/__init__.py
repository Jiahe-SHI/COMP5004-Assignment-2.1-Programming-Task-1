"""Streaming extension of the original NumCompute toolkit.

Available components include batch and buffered decision trees,
incremental numeric and categorical preprocessing, streaming descriptive
statistics, cumulative and rolling classification metrics, binary ROC AUC,
a buffered Pipeline with paired preprocessing/model snapshots, and
StreamTrainer evaluation, learning, event logs and array-storage accounting.

CSV chunk reading supports numeric features and classification labels,
with tested CSV-to-trainer integration. Matplotlib helpers provide metric
curves, model comparisons and categorical prediction plots.
The streaming demo notebook remains under development.
"""

from .tree import DecisionTreeClassifier
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .stats import StreamingStats
from .metrics import (
    StreamingConfusionMatrix,
    StreamingClassificationMetrics,
    binary_roc_auc,
    StreamingROCAUC,
    RollingClassificationMetrics,
    RollingROCAUC,
)
from .pipeline import Pipeline
from .stream import StreamTrainer
from .io import iter_csv_chunks
from .visualise import (
    plot_metric_over_time,
    compare_models,
    plot_predictions_vs_ground_truth,
)

__all__ = [
    "DecisionTreeClassifier",
    "StandardScaler",
    "Imputer",
    "OneHotEncoder",
    "StreamingStats",
    "StreamingConfusionMatrix",
    "StreamingClassificationMetrics",
    "binary_roc_auc",
    "StreamingROCAUC",
    "RollingClassificationMetrics",
    "RollingROCAUC",
    "Pipeline",
    "StreamTrainer",
    "iter_csv_chunks",
    "plot_metric_over_time",
    "compare_models",
    "plot_predictions_vs_ground_truth",
]
