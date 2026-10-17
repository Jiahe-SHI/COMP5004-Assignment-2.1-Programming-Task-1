import numpy as np


def _has_method(obj, name):
    return callable(getattr(obj, name, None))


def _validate_named_steps(steps, *, name):
    if isinstance(steps, (str, bytes)) or not isinstance(steps, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple of (name, object) pairs.")

    if len(steps) == 0:
        raise ValueError(f"{name} must contain at least one step.")

    names = set()
    objects = set()

    for index, step in enumerate(steps):
        if not isinstance(step, tuple):
            raise TypeError(f"{name}[{index}] must be a (name, object) tuple.")

        if len(step) != 2:
            raise ValueError(
                f"{name}[{index}] must contain exactly a name and an object; "
                f"got {len(step)} items."
            )

        step_name, obj = step

        if not isinstance(step_name, str):
            raise TypeError(f"The name of {name}[{index}] must be a string.")

        if step_name == "":
            raise ValueError(f"The name of {name}[{index}] must not be empty.")

        if step_name in names:
            raise ValueError(f"Step names must be unique; {step_name!r} is repeated.")

        # Sharing one object would let a later fit overwrite earlier state.
        if id(obj) in objects:
            raise ValueError(
                f"Step {step_name!r} reuses an object from an earlier step; "
                "use a separate object for each step."
            )

        names.add(step_name)
        objects.add(id(obj))

    return list(steps)


def _fit_transform_step(step, X, y):
    if _has_method(step, "fit_transform"):
        return step.fit_transform(X, y)

    step.fit(X, y)
    return step.transform(X)


class Pipeline:
    
    def __init__(self, steps):
        self.steps = _validate_named_steps(steps, name="steps")
        self.named_steps = dict(self.steps)
        self._is_fitted = False

        for step_name, obj in self.steps[:-1]:
            if not (_has_method(obj, "fit") and _has_method(obj, "transform")):
                raise TypeError(
                    f"Intermediate step {step_name!r} must provide "
                    "fit and transform methods."
                )

        last_name, last = self.steps[-1]

        if not _has_method(last, "fit"):
            raise TypeError(f"Final step {last_name!r} must provide a fit method.")

    @property
    def _final_step(self):
        return self.steps[-1][1]

    def _require(self, method):
        """Check that the final step supports method and the pipeline is fitted."""
        last_name, last = self.steps[-1]

        if not _has_method(last, method):
            raise TypeError(
                f"Pipeline does not support {method} because final step "
                f"{last_name!r} has no {method} method."
            )

        if not self._is_fitted:
            raise RuntimeError(f"Pipeline must be fitted before {method}.")

    def _fit_intermediate(self, X, y):
        # A failed fit leaves the pipeline unfitted.
        self._is_fitted = False

        for _, step in self.steps[:-1]:
            X = _fit_transform_step(step, X, y)

        return X

    def _transform_intermediate(self, X):
        for _, step in self.steps[:-1]:
            X = step.transform(X)

        return X

    def fit(self, X, y=None):
        Xt = self._fit_intermediate(X, y)
        self._final_step.fit(Xt, y)
        self._is_fitted = True

        return self

    def transform(self, X):
        self._require("transform")
        return self._final_step.transform(self._transform_intermediate(X))

    def predict(self, X):
        self._require("predict")
        return self._final_step.predict(self._transform_intermediate(X))

    def fit_transform(self, X, y=None):
        last_name, last = self.steps[-1]

        if not _has_method(last, "transform"):
            raise TypeError(
                "Pipeline does not support fit_transform because final step "
                f"{last_name!r} has no transform method."
            )

        Xt = self._fit_intermediate(X, y)
        result = _fit_transform_step(last, Xt, y)
        self._is_fitted = True

        return result


class FeatureUnion:

    def __init__(self, transformers):
        self.transformers = _validate_named_steps(transformers, name="transformers")
        self.named_transformers = dict(self.transformers)
        self._is_fitted = False

        for branch_name, obj in self.transformers:
            if not (_has_method(obj, "fit") and _has_method(obj, "transform")):
                raise TypeError(
                    f"Branch {branch_name!r} must provide fit and transform methods."
                )

    def _n_samples(self, X):
        try:
            return len(X)
        except TypeError as exc:
            raise TypeError("X must be a 2D array-like of samples.") from exc

    def _combine(self, outputs, n_samples):
        arrays = []

        for (branch_name, _), output in zip(self.transformers, outputs):
            array = np.asarray(output)

            if array.ndim != 2:
                raise ValueError(
                    f"Branch {branch_name!r} must return a 2D array; "
                    f"got {array.ndim}D."
                )

            if array.shape[0] != n_samples:
                raise ValueError(
                    f"Branch {branch_name!r} returned {array.shape[0]} rows; "
                    f"expected {n_samples}."
                )

            arrays.append(array)

        return np.hstack(arrays)

    def fit(self, X, y=None):
        self._is_fitted = False

        for _, branch in self.transformers:
            branch.fit(X, y)

        self._is_fitted = True

        return self

    def transform(self, X):
        if not self._is_fitted:
            raise RuntimeError("FeatureUnion must be fitted before transform.")

        n_samples = self._n_samples(X)
        outputs = [branch.transform(X) for _, branch in self.transformers]

        return self._combine(outputs, n_samples)

    def fit_transform(self, X, y=None):
        self._is_fitted = False

        n_samples = self._n_samples(X)
        outputs = [_fit_transform_step(branch, X, y) for _, branch in self.transformers]
        result = self._combine(outputs, n_samples)

        self._is_fitted = True

        return result

