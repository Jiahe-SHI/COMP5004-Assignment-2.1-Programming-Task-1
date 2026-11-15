"""Buffered preprocessing and paired decision tree snapshots.

Pipeline owns the raw training buffer, rebuilds preprocessing for each
training window, and publishes preprocessing and tree candidates together.
External transformers use an explicit clone_unfitted protocol.
"""

import inspect
from .preprocessing import Imputer, OneHotEncoder, StandardScaler
from .tree import DecisionTreeClassifier
import numpy as np
from ._validation import validate_numeric_X, validate_y
from ._categorical import validate_categorical_X

def _fresh_component(component):
    """Create an unfitted component from configuration only.

    Built-ins are reconstructed explicitly. External transformers must
    implement clone_unfitted(), returning an independent instance of the
    same type without learned state or mutation of the source.

    External implementations are responsible for preserving configuration
    and avoiding shared mutable state. Identity checks alone cannot prove
    those properties.
    """
    component_type = type(component)

    if component_type is StandardScaler:
        return StandardScaler()

    if component_type is Imputer:
        return Imputer(strategy=component.strategy, fill_value=component.fill_value,)

    if component_type is OneHotEncoder:
        return OneHotEncoder(handle_unknown=component.handle_unknown,)

    if component_type is DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=component.max_depth,
            min_samples_split=component.min_samples_split,
            max_features=component.max_features,
            buffer_size=component.buffer_size,
            criterion=component.criterion,
            random_state=component.random_state,
        )

    clone_method = getattr(component, "clone_unfitted", None)
    if not callable(clone_method):
        raise TypeError(
            "Unsupported component type for fresh construction: "
            f"{component_type.__name__}; external transformers must "
            "provide clone_unfitted()."
        )

    _check_transformer_methods(component)
    candidate = clone_method()

    if candidate is component:
        raise ValueError("clone_unfitted() must return a new instance.")
    if type(candidate) is not component_type:
        raise TypeError("clone_unfitted() must return the same component type.")

    _check_transformer_methods(candidate)
    return candidate

def _call_partial_fit(transformer, X, y):
    """Call partial_fit once using its declared signature.

    Explicit y parameters and **kwargs receive labels.
    Otherwise, use the X-only interface. A differently named required
    argument is not assumed to represent labels.

    Only signature binding errors are handled here. Exceptions raised
    inside partial_fit propagate unchanged.
    """
    method = getattr(transformer, "partial_fit", None)
    if not callable(method):
        raise TypeError("Transformer must provide callable partial_fit.")

    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "Transformer partial_fit must have an inspectable signature."
        ) from error

    parameters = signature.parameters
    label_parameter = parameters.get("y")
    args = (X,)
    kwargs = {}

    if label_parameter is not None and label_parameter.kind in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    ):
        if label_parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            args = (X, y)
        else:
            kwargs = {"y": y}
    elif any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    ):
        kwargs = {"y": y}

    try:
        signature.bind(*args, **kwargs)
    except TypeError as error:
        raise TypeError(
            "Unsupported partial_fit signature; expected X with "
            "an optional or required y argument."
        ) from error

    # Keep execution outside the binding-error handler.
    return method(*args, **kwargs)

def _check_transformer_methods(component):
    """Check the required incremental transformer interface."""
    for name in ("partial_fit", "transform"):
        if not callable(getattr(component, name, None)):
            raise TypeError(f"Transformer must provide callable {name}.")


def _transform_checked(transformer, X):
    """Transform a private input copy and validate the output shape.

    The transformer must preserve row order and must not learn during
    transformation. Those semantic requirements cannot be inferred
    from output shape alone.
    """
    result = np.asarray(transformer.transform(X.copy()))

    if result.ndim != 2:
        raise ValueError("Transformer output must be two-dimensional.")
    if result.shape[0] != X.shape[0]:
        raise ValueError("Transformer must preserve the number of samples.")
    if result.shape[1] == 0:
        raise ValueError("Transformer output must contain at least one feature.")

    return result.copy()

class Pipeline:
    """Buffered preprocessing and decision tree classification.

    steps is a nonempty sequence of (name, component) pairs.
    Intermediate components support built-in preprocessors and external
    transformers implementing partial_fit, transform and clone_unfitted.

    Configuration is captured at construction. Supplied components are
    neither trained nor retained as the active prediction model.

    A leading OneHotEncoder enables categorical raw input. Other chains
    require numeric raw input. Each step receives all output columns from
    the preceding step; column-wise routing is not implemented.

    Preprocessing is rebuilt from each entire training buffer. Statistics
    and categories describe that buffer, not all historical observations.
    The active preprocessing chain and tree are published together.
    """

    def __init__(self, steps):
        pairs = list(steps)
        if not pairs:
            raise ValueError("steps must contain at least one component.")

        names = set()
        identities = set()
        templates = []

        for index, pair in enumerate(pairs):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise TypeError("Each step must be a (name, component) pair.")

            name, component = pair
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Step names must be nonempty strings.")
            if name in names:
                raise ValueError("Step names must be unique.")
            if id(component) in identities:
                raise ValueError("A component cannot occupy multiple steps.")

            if index == len(pairs) - 1:
                if type(component) is not DecisionTreeClassifier:
                    raise TypeError("The final step must be a DecisionTreeClassifier.")
            else:
                _check_transformer_methods(component)

            names.add(name)
            identities.add(id(component))
            templates.append((name, _fresh_component(component)))

        self._templates = tuple(templates)
        self._categorical_input = (type(self._templates[0][1]) is OneHotEncoder)
        self._stream_feature_kinds = None
        self._active = None
        self._buffer_chunks = []
        self._n_samples_buffered = 0
        self._stream_n_features = None
        self._stream_label_kind = None
        self._n_retrains = 0
        self._model_version = 0

    @property
    def buffer_size(self):
        """Training threshold captured from the final tree configuration."""
        return self._templates[-1][1].buffer_size

    @property
    def is_fitted_(self):
        return self._active is not None

    @property
    def n_samples_buffered_(self):
        return self._n_samples_buffered

    @property
    def n_retrains_(self):
        """Number of successful Pipeline buffered training events."""
        return self._n_retrains

    @property
    def model_version_(self):
        return self._model_version

    @property
    def classes_(self):
        """Independent class table aligned with predict_proba columns."""
        return self._require_active()[-1][1].classes_.copy()

    @property
    def n_samples_fit_(self):
        return self._require_active()[-1][1].n_samples_fit_

    @property
    def n_features_in_(self):
        """Number of raw input features in the successfully fitted model."""
        self._require_active()
        return self._stream_n_features

    @staticmethod
    def _label_kind(labels):
        return "string" if isinstance(labels[0], str) else "integer"

    def _require_active(self):
        if self._active is None:
            raise RuntimeError("Pipeline must be fitted before prediction.")
        return self._active

    def _checked_raw_X(self, X, *, reset=False, allow_empty=False):
        """Return owned/validated input and candidate feature kinds.

        Categorical values use object storage to avoid coercing numeric
        categories into strings or rounding integers during concatenation.
        Validation never publishes schema changes.
        """
        expected = None if reset else self._stream_n_features

        if not self._categorical_input:
            checked = validate_numeric_X(
                X,
                allow_nan=True,
                allow_empty=allow_empty,
                n_features=expected,
            )
            return checked, None

        columns = validate_categorical_X(
            X, allow_empty=allow_empty, n_features=expected
        )
        previous = None if reset else self._stream_feature_kinds
        kinds = []

        for index, (_, _, current_kind) in enumerate(columns):
            old_kind = None if previous is None else previous[index]
            if (
                old_kind is not None
                and current_kind is not None
                and old_kind != current_kind
            ):
                raise TypeError(
                    f"Feature {index} type must match the current stream."
                )
            kinds.append(old_kind if old_kind is not None else current_kind)

        checked = np.column_stack([values for values, _, _ in columns])
        return checked, tuple(kinds)

    def _checked_training_data(self, X, y, *, reset=False):
        checked_X, feature_kinds = self._checked_raw_X(X, reset=reset)
        checked_y = validate_y(y, n_samples=checked_X.shape[0])
        label_kind = self._label_kind(checked_y)

        if (
            not reset
            and self._stream_label_kind is not None
            and label_kind != self._stream_label_kind
        ):
            raise TypeError("Label type must match the current stream.")

        return checked_X, checked_y, label_kind, feature_kinds

    def _build_candidate(self, X, y):
        """Build an independent complete chain before publication."""
        candidate = tuple((name, _fresh_component(template))
                          for name, template in self._templates
                          )
        transformed = X.copy()

        for _, transformer in candidate[:-1]:
            _call_partial_fit(transformer, transformed.copy(), y.copy())
            transformed = _transform_checked(transformer, transformed)

        candidate[-1][1].fit(transformed, y)
        return candidate

    def _publish(self, candidate, n_features, label_kind, feature_kinds, *, buffered):
        """Publish a complete pair after all training has succeeded."""
        self._active = candidate
        self._stream_n_features = n_features
        self._stream_label_kind = label_kind
        self._stream_feature_kinds = feature_kinds
        self._buffer_chunks = []
        self._n_samples_buffered = 0
        self._model_version += 1
        if buffered:
            self._n_retrains += 1

    def fit(self, X, y):
        """Train immediately and clear pending data only after success.

        A successful direct fit may establish a new input schema.
        Lifetime counters remain cumulative; only model_version_ advances.
        """
        checked_X, checked_y, kind, feature_kinds = (
            self._checked_training_data(X, y, reset=True)
        )
        candidate = self._build_candidate(checked_X, checked_y)
        self._publish(candidate, checked_X.shape[1], kind, feature_kinds, buffered=False,)
        return self

    def partial_fit(self, X, y):
        """Copy one chunk, then train once if the threshold is reached.

        Input validation failure rejects the chunk. Training failure after
        acceptance retains it and the previous model. Use retry_pending_fit
        after resolving the cause; do not submit the same chunk again.
        """
        checked_X, checked_y, kind, feature_kinds = (self._checked_training_data(X, y))
        owned_X = checked_X.copy()
        owned_y = checked_y.copy()
        owned_X.setflags(write=False)
        owned_y.setflags(write=False)

        self._buffer_chunks.append((owned_X, owned_y))
        self._n_samples_buffered += owned_X.shape[0]
        self._stream_n_features = owned_X.shape[1]
        self._stream_label_kind = kind
        self._stream_feature_kinds = feature_kinds
        return self.retry_pending_fit()

    def _buffer_arrays(self):
        """Concatenate once per training attempt, protecting integer values."""
        X_parts = [X for X, _ in self._buffer_chunks]
        y_parts = [y for _, y in self._buffer_chunks]
        X_dtype = np.result_type(*(X.dtype for X in X_parts))

        if X_dtype.kind == "f":
            limit = 2 ** (np.finfo(X_dtype).nmant + 1)
            for X in X_parts:
                if X.dtype.kind in "iu":
                    if int(X.min()) < -limit or int(X.max()) > limit:
                        raise ValueError(
                            "Buffered feature promotion may lose integer "
                            "precision; use a consistent exact numeric dtype."
                        )

        y_dtype = np.result_type(*(y.dtype for y in y_parts))
        if self._stream_label_kind == "integer" and y_dtype.kind not in "iu":
            y_dtype = np.dtype(object)

        return (
            np.concatenate(X_parts, axis=0, dtype=X_dtype),
            np.concatenate(y_parts, axis=0, dtype=y_dtype),
        )

    def retry_pending_fit(self):
        """Retry eligible data without appending or forcing a tail flush."""
        if self._n_samples_buffered < self.buffer_size:
            return self

        X, y = self._buffer_arrays()
        candidate = self._build_candidate(X, y)
        self._publish(
            candidate,
            X.shape[1],
            self._stream_label_kind,
            self._stream_feature_kinds,
            buffered=True,
        )
        return self

    def _prediction_input(self, X):
        active = self._require_active()
        transformed, _ = self._checked_raw_X(X, allow_empty=True)
        for _, transformer in active[:-1]:
            transformed = _transform_checked(transformer, transformed)
        return active[-1][1], transformed

    def predict(self, X):
        """Predict using the last successfully published complete chain."""
        tree, transformed = self._prediction_input(X)
        return tree.predict(transformed)

    def predict_proba(self, X):
        """Return probabilities in the active classes_ order."""
        tree, transformed = self._prediction_input(X)
        return tree.predict_proba(transformed)