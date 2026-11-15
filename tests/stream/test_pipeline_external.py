"""External transformer construction and Pipeline interoperability."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import DecisionTreeClassifier
from numcompute_stream.pipeline import Pipeline, _fresh_component


class ExternalCenter:
    """Minimal X-only transformer with explicit fresh construction."""

    def __init__(self, offset=0.0):
        self.offset = offset

    def clone_unfitted(self):
        return type(self)(offset=self.offset)

    def partial_fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.mean_ + self.offset


class KeywordCenter(ExternalCenter):
    def partial_fit(self, X, *, y):
        self.labels_ = y.copy()
        return super().partial_fit(X)


def make_pipeline(transformer, buffer_size=2):
    return Pipeline([
        ("external", transformer),
        ("tree", DecisionTreeClassifier(
            buffer_size=buffer_size, random_state=7
        )),
    ])


class TestPipelineExternal(unittest.TestCase):
    def test_x_only_component_runs_without_inheriting_history(self):
        source = ExternalCenter(offset=3.0)
        source.partial_fit(np.array([[100.0], [200.0]]))
        pipeline = make_pipeline(source)

        pipeline.partial_fit([[0.0]], [0])
        pipeline.partial_fit([[2.0]], [1])

        active = pipeline._active[0][1]
        self.assertIsNot(active, source)
        self.assertEqual(active.offset, 3.0)
        np.testing.assert_allclose(active.mean_, [1.0])
        np.testing.assert_allclose(source.mean_, [150.0])
        np.testing.assert_array_equal(pipeline.predict([[0.0], [2.0]]), [0, 1])

        pipeline.partial_fit([[10.0], [14.0]], [2, 3])
        np.testing.assert_allclose(pipeline._active[0][1].mean_, [12.0])
        np.testing.assert_allclose(active.mean_, [1.0])

    def test_keyword_only_labels_reach_external_component(self):
        pipeline = make_pipeline(KeywordCenter())
        pipeline.fit([[0.0], [2.0]], ["left", "right"])

        np.testing.assert_array_equal(pipeline._active[0][1].labels_, ["left", "right"])
        np.testing.assert_array_equal(pipeline.predict([[0.0], [2.0]]), ["left", "right"])

    def test_missing_clone_or_incremental_interface_is_rejected(self):
        class NoClone:
            def partial_fit(self, X):
                return self

            def transform(self, X):
                return X

        class NoPartialFit:
            def clone_unfitted(self):
                return type(self)()

            def transform(self, X):
                return X

        class NoTransform:
            def clone_unfitted(self):
                return type(self)()

            def partial_fit(self, X):
                return self

        for component in (NoClone(), NoPartialFit(), NoTransform()):
            with self.subTest(component=type(component).__name__):
                with self.assertRaises(TypeError):
                    make_pipeline(component)

    def test_invalid_clone_results_are_rejected(self):
        class ReturnsSelf(ExternalCenter):
            def clone_unfitted(self):
                return self

        class ReturnsOtherType(ExternalCenter):
            def clone_unfitted(self):
                return ExternalCenter()

        with self.assertRaisesRegex(ValueError, "new instance"):
            _fresh_component(ReturnsSelf())

        with self.assertRaisesRegex(TypeError, "same component type"):
            _fresh_component(ReturnsOtherType())

    def test_internal_error_preserves_pair_and_buffer_for_retry(self):
        pipeline = make_pipeline(ExternalCenter())
        pipeline.partial_fit([[0.0], [2.0]], [0, 1])
        old_pair = pipeline._active
        original_error = TypeError("external learning failed")

        with patch.object(ExternalCenter, "partial_fit", side_effect=original_error
        ) as mocked:
            with self.assertRaises(TypeError) as caught:
                pipeline.partial_fit([[10.0], [12.0]], [2, 3])

        self.assertIs(caught.exception, original_error)
        self.assertEqual(mocked.call_count, 1)
        self.assertIs(pipeline._active, old_pair)
        self.assertEqual(pipeline.n_samples_buffered_, 2)
        self.assertEqual(pipeline.model_version_, 1)
        np.testing.assert_array_equal(pipeline.predict([[0.0], [2.0]]), [0, 1])

        pipeline.retry_pending_fit()
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.n_samples_fit_, 2)
        self.assertEqual(pipeline.model_version_, 2)
        np.testing.assert_array_equal(pipeline.predict([[10.0], [12.0]]), [2, 3])

    def test_invalid_transform_shape_keeps_pending_data(self):
        class DropsRow(ExternalCenter):
            def transform(self, X):
                return X[:-1]

        class Flattens(ExternalCenter):
            def transform(self, X):
                return X.ravel()

        class RemovesFeatures(ExternalCenter):
            def transform(self, X):
                return X[:, :0]

        for component_type in (DropsRow, Flattens, RemovesFeatures):
            with self.subTest(component=component_type.__name__):
                pipeline = make_pipeline(component_type())

                with self.assertRaises(ValueError):
                    pipeline.partial_fit([[0.0], [2.0]], [0, 1])

                self.assertFalse(pipeline.is_fitted_)
                self.assertEqual(pipeline.n_samples_buffered_, 2)
                self.assertEqual(pipeline.model_version_, 0)

    def test_component_input_mutation_does_not_modify_caller_data(self):
        class MutatingTransformer(ExternalCenter):
            def partial_fit(self, X, y=None):
                self.mean_ = np.mean(X, axis=0)
                X[:] = -999.0
                y[:] = 9
                return self

            def transform(self, X):
                X -= self.mean_
                return X

        X = np.array([[0.0], [2.0]])
        y = np.array([0, 1])
        pipeline = make_pipeline(MutatingTransformer())
        pipeline.fit(X, y)
        predictions = pipeline.predict(X)

        np.testing.assert_array_equal(X, [[0.0], [2.0]])
        np.testing.assert_array_equal(y, [0, 1])
        np.testing.assert_array_equal(pipeline.classes_, [0, 1])
        np.testing.assert_array_equal(predictions, [0, 1])

    def test_clone_internal_error_is_propagated_unchanged(self):
        original_error = TypeError("external clone failed")

        class BrokenClone(ExternalCenter):
            def clone_unfitted(self):
                raise original_error

        with self.assertRaises(TypeError) as caught:
            make_pipeline(BrokenClone())

        self.assertIs(caught.exception, original_error)


if __name__ == "__main__":
    unittest.main()