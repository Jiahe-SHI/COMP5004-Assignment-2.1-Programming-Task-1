"""Streaming extension of the original NumCompute toolkit.

Available components include batch and buffered decision trees,
incremental numeric and categorical preprocessing, streaming descriptive
statistics, cumulative and rolling classification metrics, and
cumulative and rolling binary ROC AUC.
Pipeline, StreamTrainer, CSV integration and visualisation remain
under development.
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
]
