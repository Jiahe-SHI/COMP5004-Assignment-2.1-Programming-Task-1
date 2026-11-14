"""Streaming extension of the original NumCompute toolkit.

Available components include batch and buffered decision trees,
incremental numeric and categorical preprocessing, and streaming
descriptive statistics. Streaming metrics, Pipeline, StreamTrainer,
CSV integration and visualisation remain under development.
"""

from .tree import DecisionTreeClassifier
from .preprocessing import StandardScaler, Imputer, OneHotEncoder
from .stats import StreamingStats

__all__ = [
    "DecisionTreeClassifier",
    "StandardScaler",
    "Imputer",
    "OneHotEncoder",
    "StreamingStats",
]
