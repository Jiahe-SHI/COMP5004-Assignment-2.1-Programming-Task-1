"""Tests for Pipeline and FeatureUnion composition."""

import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from numcompute.pipeline import Pipeline, FeatureUnion
from numcompute.preprocessing import SimpleImputer, StandardScaler, MinMaxScaler, OneHotEncoder


class DummyEstimator:
    """Predict the training mean of y; record the X seen by fit and predict."""

    def fit(self, X, y=None):
        self.fit_X_ = np.asarray(X, dtype=np.float64).copy()
        self.mean_ = 0.0 if y is None else float(np.mean(y))
        return self

    def predict(self, X):
        if not hasattr(self, "mean_"):
            raise RuntimeError("DummyEstimator must be fitted before predict.")
        self.predict_X_ = np.asarray(X, dtype=np.float64).copy()
        return np.full(len(X), self.mean_)


class CountingTransformer:
    """Identity transformer that counts fit and transform calls."""

    def __init__(self):
        self.fit_calls = 0
        self.transform_calls = 0
        self.y_seen = []

    def fit(self, X, y=None):
        self.fit_calls += 1
        self.y_seen.append(y)
        return self

    def transform(self, X):
        self.transform_calls += 1
        return np.array(X, dtype=np.float64)


class ColumnSelectorStandIn:
    """Test-only stand-in for preprocessing.ColumnSelector (not yet written)."""

    def __init__(self, columns, dtype=None):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = np.asarray(X)[:, self.columns]
        return result if self.dtype is None else result.astype(self.dtype)


class FixedOutput:
    """Transformer that ignores X and returns a fixed output."""

    def __init__(self, output):
        self.output = output

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self.output


class NoTransform:
    def fit(self, X, y=None):
        return self


class NoFit:
    def transform(self, X):
        return X


X_TRAIN = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 60.0], [6.0, 30.0]])
X_TEST = np.array([[0.0, 5.0], [4.0, 100.0]])
Y_TRAIN = np.array([1.0, 2.0, 3.0, 6.0])


class TestPipelineResults(unittest.TestCase):

    def test_matches_steps_run_by_hand(self):
        pipeline = Pipeline([("standard", StandardScaler()), ("minmax", MinMaxScaler())])
        pipeline.fit(X_TRAIN)

        standard = StandardScaler().fit(X_TRAIN)
        minmax = MinMaxScaler().fit(standard.transform(X_TRAIN))
        expected = minmax.transform(standard.transform(X_TEST))

        assert_allclose(pipeline.transform(X_TEST), expected)

    def test_fit_transform_matches_fit_then_transform(self):
        steps = lambda: [("imputer", SimpleImputer()), ("scaler", StandardScaler())]
        X = [[1.0, np.nan], [3.0, 4.0], [5.0, 8.0]]

        assert_allclose(Pipeline(steps()).fit_transform(X), Pipeline(steps()).fit(X).transform(X))

    def test_test_data_uses_training_statistics(self):
        pipeline = Pipeline([("scaler", StandardScaler())]).fit(X_TRAIN)
        scaler = pipeline.named_steps["scaler"]
        mean = scaler.mean_.copy()

        result = pipeline.transform(X_TEST)

        assert_array_equal(scaler.mean_, mean)
        assert_allclose(result, (X_TEST - X_TRAIN.mean(axis=0)) / X_TRAIN.std(axis=0))

    def test_predict_with_dummy_estimator(self):
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", DummyEstimator())])
        pipeline.fit(X_TRAIN, Y_TRAIN)

        assert_array_equal(pipeline.predict(X_TEST), [3.0, 3.0])

        # The estimator receives scaled data, not the raw input.
        scaler = StandardScaler().fit(X_TRAIN)
        model = pipeline.named_steps["model"]
        assert_allclose(model.fit_X_, scaler.transform(X_TRAIN))
        assert_allclose(model.predict_X_, scaler.transform(X_TEST))

    def test_single_estimator_step(self):
        pipeline = Pipeline([("model", DummyEstimator())]).fit(X_TRAIN, Y_TRAIN)
        assert_array_equal(pipeline.predict(X_TEST), [3.0, 3.0])

    def test_nan_through_imputer_then_scaler(self):
        X = np.array([[1.0, np.nan], [np.nan, 2.0], [3.0, 6.0]])
        result = Pipeline([("imputer", SimpleImputer(fill_value=0.0)), ("scaler", StandardScaler())]).fit_transform(X)

        filled = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 6.0]])
        self.assertTrue(np.all(np.isfinite(result)))
        assert_allclose(result, (filled - filled.mean(axis=0)) / filled.std(axis=0))
        self.assertTrue(np.isnan(X[0, 1]))

    def test_y_passed_to_every_step(self):
        first, second = CountingTransformer(), CountingTransformer()
        Pipeline([("first", first), ("second", second)]).fit(X_TRAIN, Y_TRAIN)

        self.assertIs(first.y_seen[0], Y_TRAIN)
        self.assertIs(second.y_seen[0], Y_TRAIN)


class TestPipelineState(unittest.TestCase):

    def test_fit_returns_self(self):
        pipeline = Pipeline([("scaler", StandardScaler())])
        self.assertIs(pipeline.fit(X_TRAIN), pipeline)

    def test_steps_are_not_cloned(self):
        scaler, model = StandardScaler(), DummyEstimator()
        pipeline = Pipeline([("scaler", scaler), ("model", model)]).fit(X_TRAIN, Y_TRAIN)

        self.assertIs(pipeline.named_steps["scaler"], scaler)
        self.assertIs(pipeline.steps[1][1], model)
        assert_allclose(scaler.mean_, X_TRAIN.mean(axis=0))

    def test_steps_list_is_copied(self):
        steps = [("scaler", StandardScaler())]
        pipeline = Pipeline(steps)
        steps.append(("extra", StandardScaler()))

        self.assertEqual(len(pipeline.steps), 1)

    def test_transform_and_predict_do_not_refit(self):
        counter = CountingTransformer()
        pipeline = Pipeline([("counter", counter), ("model", DummyEstimator())]).fit(X_TRAIN, Y_TRAIN)
        self.assertEqual((counter.fit_calls, counter.transform_calls), (1, 1))

        pipeline.predict(X_TEST)
        pipeline.predict(X_TEST)
        self.assertEqual((counter.fit_calls, counter.transform_calls), (1, 3))

        last = CountingTransformer()
        pipeline = Pipeline([("scaler", StandardScaler()), ("last", last)]).fit(X_TRAIN)
        pipeline.transform(X_TEST)
        self.assertEqual((last.fit_calls, last.transform_calls), (1, 1))

    def test_repeated_fit_replaces_state(self):
        pipeline = Pipeline([("scaler", StandardScaler())])
        pipeline.fit(X_TRAIN)
        pipeline.fit(X_TEST)

        assert_allclose(pipeline.named_steps["scaler"].mean_, X_TEST.mean(axis=0))

    def test_failed_fit_leaves_pipeline_unfitted(self):
        pipeline = Pipeline([("scaler", StandardScaler())]).fit(X_TRAIN)

        with self.assertRaises(ValueError):
            pipeline.fit(np.empty((0, 2)))

        with self.assertRaises(RuntimeError):
            pipeline.transform(X_TEST)


class TestPipelineErrors(unittest.TestCase):

    def test_before_fit(self):
        with self.assertRaises(RuntimeError):
            Pipeline([("scaler", StandardScaler())]).transform(X_TEST)

        with self.assertRaises(RuntimeError):
            Pipeline([("scaler", StandardScaler()), ("model", DummyEstimator())]).predict(X_TEST)

    def test_final_step_without_predict(self):
        pipeline = Pipeline([("scaler", StandardScaler())])

        # Unsupported operations raise TypeError whether or not fitted.
        with self.assertRaises(TypeError):
            pipeline.predict(X_TEST)

        pipeline.fit(X_TRAIN)
        with self.assertRaises(TypeError):
            pipeline.predict(X_TEST)

    def test_final_step_without_transform(self):
        model = DummyEstimator()
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])

        with self.assertRaises(TypeError):
            pipeline.fit_transform(X_TRAIN, Y_TRAIN)

        # The unsupported fit_transform is rejected before fitting anything.
        self.assertFalse(hasattr(model, "mean_"))

        pipeline.fit(X_TRAIN, Y_TRAIN)
        with self.assertRaises(TypeError):
            pipeline.transform(X_TEST)

    def test_empty_steps(self):
        for steps in ([], ()):
            with self.subTest(steps=steps), self.assertRaises(ValueError):
                Pipeline(steps)

    def test_steps_not_a_sequence(self):
        for steps in (None, "scaler", ("scaler", StandardScaler()), {"scaler": StandardScaler()}):
            with self.subTest(steps=steps), self.assertRaises(TypeError):
                Pipeline(steps)

    def test_bad_step_tuples(self):
        for step in (["scaler", StandardScaler()], StandardScaler(), "scaler"):
            with self.subTest(step=step), self.assertRaises(TypeError):
                Pipeline([step])

        for step in (("scaler",), ("scaler", StandardScaler(), "extra")):
            with self.subTest(step=step), self.assertRaises(ValueError):
                Pipeline([step])

    def test_bad_names(self):
        for name in (None, 1, b"scaler"):
            with self.subTest(name=name), self.assertRaises(TypeError):
                Pipeline([(name, StandardScaler())])

        with self.assertRaises(ValueError):
            Pipeline([("", StandardScaler())])

    def test_duplicate_names(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            Pipeline([("scaler", StandardScaler()), ("scaler", MinMaxScaler())])

    def test_same_object_in_two_steps(self):
        scaler = StandardScaler()

        with self.assertRaisesRegex(ValueError, "reuses"):
            Pipeline([("first", scaler), ("second", scaler)])

    def test_intermediate_step_missing_methods(self):
        for step in (DummyEstimator(), NoTransform(), NoFit(), object()):
            with self.subTest(step=type(step).__name__), self.assertRaises(TypeError):
                Pipeline([("bad", step), ("scaler", StandardScaler())])

    def test_final_step_without_fit(self):
        with self.assertRaises(TypeError):
            Pipeline([("scaler", StandardScaler()), ("last", NoFit())])

    def test_errors_from_steps_propagate(self):
        with self.assertRaises(ValueError):
            Pipeline([("scaler", StandardScaler())]).fit([[np.inf, 1.0]])


class TestFeatureUnion(unittest.TestCase):

    def test_column_order_follows_branches(self):
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        union = FeatureUnion([
            ("last", ColumnSelectorStandIn([2])),
            ("first_two", ColumnSelectorStandIn([0, 1])),
        ])

        assert_array_equal(union.fit_transform(X), [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])

    def test_matches_branches_run_by_hand(self):
        union = FeatureUnion([("standard", StandardScaler()), ("minmax", MinMaxScaler())]).fit(X_TRAIN)
        expected = np.hstack([
            StandardScaler().fit(X_TRAIN).transform(X_TEST),
            MinMaxScaler().fit(X_TRAIN).transform(X_TEST),
        ])

        result = union.transform(X_TEST)

        self.assertEqual(result.shape, (2, 4))
        assert_allclose(result, expected)

    def test_single_branch(self):
        union = FeatureUnion([("scaler", StandardScaler())])
        assert_allclose(union.fit_transform(X_TRAIN), StandardScaler().fit_transform(X_TRAIN))

    def test_fit_transform_matches_fit_then_transform(self):
        branches = lambda: [("standard", StandardScaler()), ("minmax", MinMaxScaler())]
        assert_allclose(FeatureUnion(branches()).fit_transform(X_TRAIN), FeatureUnion(branches()).fit(X_TRAIN).transform(X_TRAIN))

    def test_fit_returns_self_and_does_not_clone(self):
        scaler = StandardScaler()
        union = FeatureUnion([("scaler", scaler)])

        self.assertIs(union.fit(X_TRAIN), union)
        self.assertIs(union.named_transformers["scaler"], scaler)
        assert_allclose(scaler.mean_, X_TRAIN.mean(axis=0))

    def test_transform_does_not_refit(self):
        counter = CountingTransformer()
        union = FeatureUnion([("counter", counter)]).fit(X_TRAIN)
        union.transform(X_TEST)
        union.transform(X_TEST)

        self.assertEqual((counter.fit_calls, counter.transform_calls), (1, 2))

    def test_before_fit(self):
        with self.assertRaises(RuntimeError):
            FeatureUnion([("scaler", StandardScaler())]).transform(X_TEST)

    def test_row_count_mismatch(self):
        union = FeatureUnion([("scaler", StandardScaler()), ("short", FixedOutput(np.zeros((3, 1))))])

        with self.assertRaisesRegex(ValueError, "rows"):
            union.fit_transform(X_TRAIN)

        with self.assertRaisesRegex(ValueError, "rows"):
            union.fit(X_TRAIN).transform(X_TRAIN)

    def test_failed_fit_transform_leaves_union_unfitted(self):
        # fit succeeds because it does not check outputs; fit_transform does.
        union = FeatureUnion([("short", FixedOutput(np.zeros((3, 1))))]).fit(X_TRAIN)

        with self.assertRaises(ValueError):
            union.fit_transform(X_TRAIN)

        with self.assertRaises(RuntimeError):
            union.transform(X_TRAIN)

    def test_non_2d_branch_output(self):
        for output in (np.zeros(4), np.zeros((4, 1, 1)), 0.0):
            with self.subTest(shape=np.shape(output)), self.assertRaisesRegex(ValueError, "2D"):
                FeatureUnion([("bad", FixedOutput(output))]).fit_transform(X_TRAIN)

    def test_invalid_branches(self):
        with self.assertRaises(ValueError):
            FeatureUnion([])

        with self.assertRaises(ValueError):
            FeatureUnion([("a", StandardScaler()), ("a", MinMaxScaler())])

        with self.assertRaises(TypeError):
            FeatureUnion([["a", StandardScaler()]])

        for branch in (DummyEstimator(), NoFit(), NoTransform()):
            with self.subTest(branch=type(branch).__name__), self.assertRaises(TypeError):
                FeatureUnion([("bad", branch)])


class TestComposition(unittest.TestCase):

    def test_mixed_data_union_inside_pipeline(self):
        X = np.array([[1.0, "red"], [2.0, "blue"], [3.0, "red"]], dtype=object)
        X_new = np.array([[4.0, "blue"]], dtype=object)

        pipeline = Pipeline([
            ("features", FeatureUnion([
                ("numeric", Pipeline([
                    ("select", ColumnSelectorStandIn([0], dtype=np.float64)),
                    ("scaler", StandardScaler()),
                ])),
                ("category", Pipeline([
                    ("select", ColumnSelectorStandIn([1])),
                    ("encoder", OneHotEncoder()),
                ])),
            ])),
            ("model", DummyEstimator()),
        ])

        pipeline.fit(X, [0.0, 1.0, 2.0])
        model = pipeline.named_steps["model"]
        scaled = (np.array([1.0, 2.0, 3.0]) - 2.0) / np.std([1.0, 2.0, 3.0])

        assert_allclose(model.fit_X_, np.column_stack([scaled, [1, 0, 1], [0, 1, 0]]))
        assert_array_equal(pipeline.predict(X_new), [1.0])
        assert_allclose(model.predict_X_, [[2.0 / np.std([1.0, 2.0, 3.0]), 0.0, 1.0]])

    def test_union_then_scaler(self):
        pipeline = Pipeline([
            ("union", FeatureUnion([
                ("a", ColumnSelectorStandIn([0])),
                ("b", ColumnSelectorStandIn([1, 0])),
            ])),
            ("scaler", MinMaxScaler()),
        ])

        result = pipeline.fit_transform(X_TRAIN)

        self.assertEqual(result.shape, (4, 3))
        assert_allclose(result[:, 0], result[:, 2])
        assert_allclose(result.min(axis=0), 0.0)
        assert_allclose(result.max(axis=0), 1.0)


if __name__ == "__main__":
    unittest.main()
