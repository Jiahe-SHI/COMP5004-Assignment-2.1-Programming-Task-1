"""Separate prequential evaluation from buffered model learning."""

from copy import deepcopy
import numpy as np
from ._validation import validate_numeric_X, validate_y
from .metrics import (
    RollingClassificationMetrics,
    StreamingClassificationMetrics,
    RollingROCAUC,
    StreamingROCAUC,
)
from .pipeline import Pipeline
from .tree import DecisionTreeClassifier
from time import perf_counter
from ._memory import array_storage_bytes


class StreamTrainer:
    """Coordinate evaluation and learning for a tree or Pipeline.

    score_chunk predicts and updates classification metrics without learning.
    fit_chunk learns without updating evaluation metrics.

    Call score_chunk before fit_chunk for each incoming chunk. Successful
    score_chunk calls are separate evaluation events; repeated calls are
    not automatically deduplicated.

    The supplied model is retained by reference. Training through this
    trainer updates that same model. Do not concurrently mutate it.

    Warm-up chunks count as scoring calls and received scoring rows, but
    are excluded from classification metric denominators.

    positive_label=None disables AUC. Otherwise, evaluate binary AUC
    using the active model's positive-class probability column.
    Missing positive classes receive zero scores. Undefined enabled AUC
    is NaN; disabled AUC is None.

    Candidate metric copies provide coordinated evaluation publication.
    Exact cumulative AUC retains history, and deepcopy duplicates that
    history during scoring. This implementation is not constant-memory.
    """

    def __init__(self, model, *, window_size=100, classes=None,
                 zero_division=0, positive_label=None,
                 ):
        if type(model) not in (DecisionTreeClassifier, Pipeline):
            raise TypeError("model must be a DecisionTreeClassifier or Pipeline.")

        cumulative = StreamingClassificationMetrics(
            classes=classes, zero_division=zero_division
        )
        rolling = RollingClassificationMetrics(
            window_size,
            classes=classes,
            zero_division=zero_division,
        )
        cumulative_auc = None
        rolling_auc = None

        if positive_label is not None:
            cumulative_auc = StreamingROCAUC(positive_label=positive_label)
            rolling_auc = RollingROCAUC(
                window_size, positive_label=positive_label
            )
            positive_label = cumulative_auc.positive_label

        self._model = model
        self._positive_label = positive_label

        # All evaluation components are published together.
        self._evaluation = (cumulative, rolling, 0, 0, cumulative_auc, rolling_auc)

        self._logs = []

    @property
    def model(self):
        return self._model

    @property
    def n_chunks_scored_(self):
        """Successful scoring calls, including warm-up."""
        return self._evaluation[2]

    @property
    def n_samples_scored_(self):
        """Rows in successful scoring calls, including warm-up."""
        return self._evaluation[3]

    @property
    def n_samples_evaluated_(self):
        """Rows actually predicted and included in cumulative metrics."""
        return self._evaluation[0].result()["n_samples"]

    @property
    def logs_(self):
        """Return independent event records in call order.

        Successful scores and all completed learning attempts are retained.
        Failed scores are omitted because their evaluation state is rolled
        back. Log memory grows with the number of recorded events.
        """
        return deepcopy(self._logs)

    def _model_status(self):
        """Capture the observable model state without training."""
        return {
            "is_fitted": self._model.is_fitted_,
            "buffer_samples": self._model.n_samples_buffered_,
            "retrain_count": self._model.n_retrains_,
            "model_version": self._model.model_version_,
        }

    def metrics_result(self):
        """Return independent classification and optional AUC results."""
        cumulative, rolling, _, _, cumulative_auc, rolling_auc = (
            self._evaluation
        )
        return {
            "cumulative": cumulative.result(),
            "rolling": rolling.result(),
            "cumulative_auc": (None if cumulative_auc is None else cumulative_auc.result()),
            "rolling_auc": (None if rolling_auc is None else rolling_auc.result()),
        }

    def _checked_score_data(self, X, y):
        """Validate scoring data without changing model or metric state."""
        if type(self._model) is Pipeline:
            # Reuse the Pipeline raw-input contract, including categorical
            # validation. This helper prepares data without publishing state.
            checked_X, checked_y, _, _ = (self._model._checked_training_data(X, y))
            return checked_X, checked_y

        expected = (
            self._model.n_features_in_
            if self._model.is_fitted_
            else None
        )
        checked_X = validate_numeric_X(X, n_features=expected)
        checked_y = validate_y(y, n_samples=checked_X.shape[0])
        return checked_X, checked_y

    def _positive_scores(self, X, y):
        """Align positive scores with the currently active class table.

        Reject a current model/evaluation pair containing more than one
        non-positive label. Stateful AUC additionally enforces its observed
        negative-label history.
        """
        classes = validate_y(self._model.classes_)
        combined = np.concatenate((classes.astype(object), y.astype(object)))
        combined = validate_y(combined)

        labels = np.unique(combined)
        negatives = [
            label for label in labels
            if label != self._positive_label
        ]
        if len(negatives) > 1:
            raise ValueError("AUC evaluation supports only one non-positive label.")

        # Validate label-family compatibility with the configured positive
        # label, including when that label is absent from the current model.
        validate_y(np.concatenate((
            combined,
            np.array([self._positive_label], dtype=object),
        )))

        positive_columns = [
            index for index, label in enumerate(classes)
            if label == self._positive_label
        ]
        if not positive_columns:
            return np.zeros(X.shape[0], dtype=float)

        probabilities = self._model.predict_proba(X)
        return probabilities[:, positive_columns[0]].copy()

    def _score_chunk_impl(self, X, y):
        """Evaluate once without training and publish metrics together.

        Any prediction, metric update or result-calculation failure leaves
        all evaluation metrics and scoring counters unchanged.

        Repeated successful calls are separate evaluation events.
        """
        checked_X, checked_y = self._checked_score_data(X, y)
        (
            cumulative, rolling, n_chunks, n_rows,
            cumulative_auc, rolling_auc,
        ) = self._evaluation

        warm_up = not self._model.is_fitted_
        chunk_accuracy = float("nan")
        next_cumulative_auc = cumulative_auc
        next_rolling_auc = rolling_auc

        if warm_up:
            next_cumulative = cumulative
            next_rolling = rolling
        else:
            predictions = validate_y(
                self._model.predict(checked_X),
                n_samples=checked_y.size,
            )
            scores = None
            if cumulative_auc is not None:
                scores = self._positive_scores(checked_X, checked_y)

            next_cumulative = deepcopy(cumulative)
            next_rolling = deepcopy(rolling)
            next_cumulative.update(checked_y, predictions)
            next_rolling.update(checked_y, predictions)

            if cumulative_auc is not None:
                next_cumulative_auc = deepcopy(cumulative_auc)
                next_rolling_auc = deepcopy(rolling_auc)
                next_cumulative_auc.update(checked_y, scores)
                next_rolling_auc.update(checked_y, scores)

            chunk_accuracy = float(np.mean(checked_y == predictions))

        cumulative_result = next_cumulative.result()
        rolling_result = next_rolling.result()
        cumulative_auc_result = (
            None if next_cumulative_auc is None
            else next_cumulative_auc.result()
        )
        rolling_auc_result = (
            None if next_rolling_auc is None
            else next_rolling_auc.result()
        )

        next_chunks = n_chunks + 1
        next_rows = n_rows + checked_y.size
        record = {
            "score_id": next_chunks,
            "chunk_size": checked_y.size,
            "warm_up": warm_up,
            "chunk_accuracy": chunk_accuracy,
            "cumulative_accuracy": cumulative_result["accuracy"],
            "rolling_accuracy": rolling_result["accuracy"],
            "cumulative_auc": (
                None if cumulative_auc_result is None
                else cumulative_auc_result["auc"]
            ),
            "rolling_auc": (
                None if rolling_auc_result is None
                else rolling_auc_result["auc"]
            ),
            "n_samples_scored": next_rows,
            "n_samples_evaluated": cumulative_result["n_samples"],
            "model_version": self._model.model_version_,
        }

        self._evaluation = (
            next_cumulative, next_rolling, next_chunks, next_rows,
            next_cumulative_auc, next_rolling_auc,
        )
        return record

    def score_chunk(self, X, y):
        """Score without learning and retain an independent event record.

        score_seconds includes validation, prediction, metric copying,
        updates and result calculation. It excludes log insertion.
        """
        previous_evaluation = self._evaluation
        started = perf_counter()

        try:
            record = self._score_chunk_impl(X, y)
            elapsed = perf_counter() - started
            status = self._model_status()

            record.update({
                "event_id": len(self._logs) + 1,
                "event": "score",
                "success": True,
                "score_seconds": elapsed,
                "buffer_samples": status["buffer_samples"],
                "retrain_count": status["retrain_count"],
            })

            # All record fields are scalar values. Keep a separate dict so
            # callers cannot modify retained history through the return value.
            record.update(self._memory_log_fields())
            stored_record = record.copy()
            self._logs.append(stored_record)
        except Exception:
            # A failed scoring operation must not advance metric state.
            self._evaluation = previous_evaluation
            raise

        return record

    def fit_chunk(self, X, y):
        """Learn without scoring and record the learning attempt.

        A failure after buffer acceptance retains that chunk. Resolve the
        cause and call retry_pending_fit rather than resubmitting it.
        """
        self._run_learning("fit", X, y)
        return self

    def retry_pending_fit(self):
        """Retry buffered training without scoring or adding samples."""
        self._run_learning("retry_fit")
        return self

    def _run_learning(self, event, X=None, y=None):
        """Record success/failure around the model's own state semantics."""
        before = self._model_status()
        record = {
            "event_id": len(self._logs) + 1,
            "event": event,
            "success": False,
            "error_type": None,
            "buffer_samples_before": before["buffer_samples"],
            "model_version_before": before["model_version"],
            "retrain_count_before": before["retrain_count"],
        }
        started = perf_counter()

        try:
            if event == "fit":
                self._model.partial_fit(X, y)
            else:
                self._model.retry_pending_fit()
        except Exception as error:
            elapsed = perf_counter() - started
            after = self._model_status()
            record.update({
                "error_type": type(error).__name__,
                "training_seconds": elapsed,
                "buffer_samples": after["buffer_samples"],
                "model_version": after["model_version"],
                "retrain_count": after["retrain_count"],
                "retrained": (
                    after["retrain_count"] > before["retrain_count"]
                ),
            })
            record.update(self._memory_log_fields())
            self._logs.append(record)
            raise

        elapsed = perf_counter() - started
        after = self._model_status()
        record.update({
            "success": True,
            "training_seconds": elapsed,
            "buffer_samples": after["buffer_samples"],
            "model_version": after["model_version"],
            "retrain_count": after["retrain_count"],
            "retrained": (
                after["retrain_count"] > before["retrain_count"]
            ),
        })
        record.update(self._memory_log_fields())
        self._logs.append(record)

    def memory_result(self):
        """Report current reachable array storage, not RSS or peak memory.

        buffer_array_bytes is a subset of memory_bytes, not an additional
        amount. Log Python-object storage is excluded; only its event
        count is reported.

        The measurement traverses current retained state. Its work and
        temporary traversal storage grow with the reachable object graph.
        """
        return {"memory_bytes": array_storage_bytes((self._model, self._evaluation)),
            "buffer_array_bytes": array_storage_bytes(self._model._buffer_chunks),
            "memory_method": "reachable_ndarray_nbytes",
            "log_events": len(self._logs),
        }

    def _memory_log_fields(self):
        """Keep optional memory measurement failures out of model semantics.

        A failed measurement produces explicit missing values. It must not
        turn completed training into a reported training failure or mask
        the original model exception.
        """
        try:
            result = self.memory_result()
        except Exception as error:
            return {
                "memory_bytes": None,
                "buffer_array_bytes": None,
                "memory_method": "reachable_ndarray_nbytes",
                "memory_status": "unavailable",
                "memory_error_type": type(error).__name__,
                "memory_log_events_before": len(self._logs),
            }

        return {
            "memory_bytes": result["memory_bytes"],
            "buffer_array_bytes": result["buffer_array_bytes"],
            "memory_method": result["memory_method"],
            "memory_status": "ok",
            "memory_error_type": None,
            "memory_log_events_before": result["log_events"],
        }