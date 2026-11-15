"""Public API and boundary tests for the buffered Pipeline."""

import unittest

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    OneHotEncoder,
    Pipeline,
    StandardScaler,
)
from numcompute_stream.pipeline import Pipeline as ModulePipeline


class TestPipelineBoundaries(unittest.TestCase):
    def test_package_export_and_raw_feature_count(self):
        self.assertIs(Pipeline, ModulePipeline)
        pipeline = Pipeline([
            ("encoder", OneHotEncoder()),
            ("tree", DecisionTreeClassifier()),
        ])

        with self.assertRaises(RuntimeError):
            _ = pipeline.n_features_in_

        pipeline.fit(
            [["a", "x"], ["b", "y"], ["c", "x"]],
            [0, 1, 2],
        )

        self.assertEqual(pipeline.n_features_in_, 2)
        self.assertEqual(pipeline._active[-1][1].n_features_in_, 5)

    def test_tree_only_pipeline_supports_empty_prediction(self):
        pipeline = Pipeline([("tree", DecisionTreeClassifier(buffer_size=2)),])
        pipeline.partial_fit([[0.0], [1.0]], [0, 1])

        self.assertEqual(pipeline.predict(np.empty((0, 1))).shape, (0,))
        self.assertEqual(pipeline.predict_proba(np.empty((0, 1))).shape, (0, 2))

        with self.assertRaises(ValueError):
            pipeline.predict(np.empty((0, 2)))

        with self.assertRaises(ValueError):
            pipeline.predict([[np.nan]])

        self.assertEqual(pipeline.model_version_, 1)

    def test_public_arrays_and_state_properties_are_protected(self):
        pipeline = Pipeline([("tree", DecisionTreeClassifier()),
                             ]).fit([[0.0], [1.0]], [0, 1])

        classes = pipeline.classes_
        classes[:] = 99
        probabilities = pipeline.predict_proba([[0.0]])
        probabilities[:] = -1.0

        np.testing.assert_array_equal(pipeline.classes_, [0, 1])
        np.testing.assert_array_equal(pipeline.predict_proba([[0.0]]), [[1.0, 0.0]])

        for name in (
            "buffer_size",
            "is_fitted_",
            "n_features_in_",
            "n_samples_fit_",
            "n_samples_buffered_",
            "n_retrains_",
            "model_version_",
            "classes_",
        ):
            with self.subTest(property=name):
                with self.assertRaises(AttributeError):
                    setattr(pipeline, name, None)

    def test_constructor_rejects_invalid_step_layouts(self):
        shared = StandardScaler()
        invalid_steps = [
            [],
            [("tree",)],
            [("", DecisionTreeClassifier())],
            [
                ("same", StandardScaler()),
                ("same", DecisionTreeClassifier()),
            ],
            [
                ("first", shared),
                ("second", shared),
                ("tree", DecisionTreeClassifier()),
            ],
            [("last", StandardScaler())],
            [("bad", object()), ("tree", DecisionTreeClassifier())],
        ]

        for steps in invalid_steps:
            with self.subTest(steps=steps):
                with self.assertRaises((TypeError, ValueError)):
                    Pipeline(steps)

    def test_constructor_captures_independent_builtin_configuration(self):
        source_encoder = OneHotEncoder(handle_unknown="ignore")
        source_tree = DecisionTreeClassifier(buffer_size=2)
        steps = [
            ("encoder", source_encoder),
            ("tree", source_tree),
        ]
        pipeline = Pipeline(steps)

        source_encoder.handle_unknown = "error"
        source_tree.buffer_size = 100
        source_tree.max_depth = 0
        steps.clear()

        pipeline.partial_fit([["a"], ["b"]], [0, 1])

        self.assertEqual(pipeline.buffer_size, 2)
        self.assertTrue(pipeline.is_fitted_)
        np.testing.assert_array_equal(pipeline.predict([["a"], ["b"]]), [0, 1])
        self.assertEqual(pipeline.predict([["unknown"]]).shape, (1,))
        self.assertFalse(source_tree.is_fitted_)
        self.assertFalse(hasattr(source_encoder, "categories_"))

    def test_mixed_integer_label_dtypes_preserve_exact_values(self):
        pipeline = Pipeline([("tree", DecisionTreeClassifier(buffer_size=2)),])
        large = 2 ** 63 + 1

        pipeline.partial_fit([[0.0]], np.array([-1], dtype=np.int64))
        pipeline.partial_fit([[1.0]], np.array([large], dtype=np.uint64))

        self.assertEqual(pipeline.classes_.tolist(), [-1, large])
        self.assertEqual(pipeline.predict([[0.0], [1.0]]).tolist(), [-1, large])

    def test_unsafe_feature_promotion_keeps_buffer_and_old_model(self):
        pipeline = Pipeline([("tree", DecisionTreeClassifier(buffer_size=2)),])
        pipeline.fit(np.array([[0]], dtype=np.int64), [0])
        pipeline.partial_fit(np.array([[2 ** 60 + 1]], dtype=np.int64), [1])

        with self.assertRaisesRegex(ValueError, "precision"):
            pipeline.partial_fit(np.array([[0.5]]), [0])

        self.assertEqual(pipeline.n_samples_buffered_, 2)
        self.assertEqual(pipeline.model_version_, 1)
        self.assertEqual(pipeline.n_retrains_, 0)
        np.testing.assert_array_equal(pipeline.predict([[0]]), [0])

        # Retrying does not fix incompatible retained dtypes or add rows.
        with self.assertRaisesRegex(ValueError, "precision"):
            pipeline.retry_pending_fit()
        self.assertEqual(pipeline.n_samples_buffered_, 2)

        # Explicit successful fit replaces learning data and clears the buffer.
        pipeline.fit([[0.0], [1.0]], [0, 1])
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.model_version_, 2)

    def test_failed_direct_refit_preserves_active_schema_and_pending_data(self):
        pipeline = Pipeline([("tree", DecisionTreeClassifier(buffer_size=3)),])
        pipeline.fit([[0.0], [1.0]], [0, 1])
        pipeline.partial_fit([[2.0]], [1])
        before = pipeline.predict_proba([[0.0], [1.0]])

        # Raw NaN passes Pipeline validation, but the tree rejects it.
        with self.assertRaises(ValueError):
            pipeline.fit([[np.nan, 1.0]], ["replacement"])

        self.assertEqual(pipeline.n_features_in_, 1)
        self.assertEqual(pipeline.n_samples_buffered_, 1)
        self.assertEqual(pipeline.model_version_, 1)
        np.testing.assert_array_equal(pipeline.predict_proba([[0.0], [1.0]]), before)

        pipeline.partial_fit([[3.0], [4.0]], [1, 1])
        self.assertEqual(pipeline.n_samples_fit_, 3)
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.n_retrains_, 1)
        np.testing.assert_array_equal(pipeline.classes_, [1])


if __name__ == "__main__":
    unittest.main()