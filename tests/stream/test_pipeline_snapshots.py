"""Tests for paired preprocessing/model snapshots."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    OneHotEncoder,
    StandardScaler,
)
from numcompute_stream.pipeline import Pipeline


def categorical_pipeline(buffer_size=4, handle_unknown="ignore"):
    return Pipeline([
        ("encoder", OneHotEncoder(handle_unknown=handle_unknown)),
        ("tree", DecisionTreeClassifier(buffer_size=buffer_size, random_state=7)),
    ])


class TestPipelineSnapshots(unittest.TestCase):
    def test_new_categories_wait_for_paired_publication(self):
        pipeline = categorical_pipeline()
        pipeline.partial_fit([["a"], ["b"], ["a"], ["b"]], [0, 1, 0, 1])
        old_encoder = pipeline._active[0][1]
        old_tree = pipeline._active[-1][1]
        before = pipeline.predict_proba([["a"], ["b"], ["c"]])

        pipeline.partial_fit([["c"], ["d"]], [2, 3])

        self.assertIs(pipeline._active[0][1], old_encoder)
        self.assertIs(pipeline._active[-1][1], old_tree)
        np.testing.assert_array_equal(pipeline.predict_proba([["a"], ["b"], ["c"]]), before)
        self.assertEqual(old_tree.n_features_in_, 2)

        pipeline.partial_fit([["a"], ["b"]], [0, 1])

        self.assertIsNot(pipeline._active[0][1], old_encoder)
        self.assertIsNot(pipeline._active[-1][1], old_tree)
        self.assertEqual(pipeline._active[-1][1].n_features_in_, 4)
        np.testing.assert_array_equal(pipeline.predict([["a"], ["b"], ["c"], ["d"]]), [0, 1, 2, 3])
        # Previously published objects were not modified.
        self.assertEqual(old_encoder.transform([["a"]]).shape, (1, 2))
        np.testing.assert_array_equal(
            old_tree.predict(old_encoder.transform([["a"], ["b"]])),
            [0, 1],
        )

    def test_failed_new_layout_preserves_old_pair_until_retry(self):
        pipeline = categorical_pipeline(buffer_size=3)
        pipeline.partial_fit([["a"], ["b"], ["a"]], [0, 1, 0])
        old_pair = pipeline._active
        before = pipeline.predict([["a"], ["b"]])

        with patch.object(
            DecisionTreeClassifier,
            "fit",
            side_effect=RuntimeError("simulated"),
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                pipeline.partial_fit([["a"], ["b"], ["c"]], [0, 1, 2])

        self.assertIs(pipeline._active, old_pair)
        self.assertEqual(pipeline.n_samples_buffered_, 3)
        self.assertEqual(pipeline.model_version_, 1)
        np.testing.assert_array_equal(pipeline.predict([["a"], ["b"]]), before)

        pipeline.retry_pending_fit()
        self.assertEqual(pipeline.n_samples_fit_, 3)
        self.assertEqual(pipeline.n_samples_buffered_, 0)
        self.assertEqual(pipeline.model_version_, 2)
        np.testing.assert_array_equal(pipeline.predict([["a"], ["b"], ["c"]]), [0, 1, 2])

    def test_categories_can_shrink_in_next_training_window(self):
        pipeline = categorical_pipeline()
        pipeline.partial_fit([["a"], ["b"], ["c"], ["d"]], [0, 1, 2, 3])
        self.assertEqual(pipeline._active[-1][1].n_features_in_, 4)

        pipeline.partial_fit([["z"], ["z"], ["z"], ["z"]], [9, 9, 9, 9])

        self.assertEqual(pipeline._active[-1][1].n_features_in_, 1)
        np.testing.assert_array_equal(pipeline.classes_, [9])
        np.testing.assert_array_equal(pipeline.predict([["z"]]), [9])

    def test_downstream_scaler_is_rebuilt_for_new_encoded_width(self):
        pipeline = Pipeline([
            ("encoder", OneHotEncoder()),
            ("scaler", StandardScaler()),
            ("tree", DecisionTreeClassifier(buffer_size=3)),
        ])
        pipeline.partial_fit([["a"], ["a"], ["a"]], [0, 0, 0])
        old_scaler = pipeline._active[1][1]

        pipeline.partial_fit([["a"], ["b"], ["c"]], [0, 1, 2])
        new_scaler = pipeline._active[1][1]

        np.testing.assert_allclose(old_scaler.mean_, [1.0])
        np.testing.assert_allclose(new_scaler.mean_, [1 / 3] * 3)
        np.testing.assert_array_equal(new_scaler.n_samples_seen_, [3] * 3)
        np.testing.assert_array_equal(pipeline.predict([["a"], ["b"], ["c"]]), [0, 1, 2])

    def test_numeric_statistics_use_only_current_buffer(self):
        pipeline = Pipeline([
            ("imputer", Imputer()),
            ("scaler", StandardScaler()),
            ("tree", DecisionTreeClassifier(buffer_size=2)),
        ])
        pipeline.partial_fit([[0.0], [2.0]], [0, 1])
        old_imputer = pipeline._active[0][1]
        old_scaler = pipeline._active[1][1]

        pipeline.partial_fit([[100.0], [np.nan]], [2, 2])

        np.testing.assert_allclose(pipeline._active[0][1].statistics_, [100.0])
        np.testing.assert_allclose(pipeline._active[1][1].mean_, [100.0])
        np.testing.assert_allclose(old_imputer.statistics_, [1.0])
        np.testing.assert_allclose(old_scaler.mean_, [1.0])

    def test_raw_categories_preserve_types_and_input_ownership(self):
        pipeline = categorical_pipeline(buffer_size=2)
        large = 2 ** 60
        X = np.array([["first", large]], dtype=object)
        pipeline.partial_fit(X, [0])
        X[:] = [["changed", -1]]
        pipeline.partial_fit([["second", large + 1]], [1])

        encoder = pipeline._active[0][1]
        self.assertEqual(encoder.categories_[1].tolist(), [large, large + 1])
        np.testing.assert_array_equal(
            pipeline.predict([["first", large], ["second", large + 1]]),
            [0, 1],
        )

    def test_feature_kind_validation_is_transactional(self):
        pipeline = categorical_pipeline(buffer_size=3)
        pipeline.partial_fit([[None, "a"]], [0])

        # Feature 0 would acquire a numeric kind, but feature 1 is invalid.
        with self.assertRaises(TypeError):
            pipeline.partial_fit([[1, 2]], [1])
        self.assertEqual(pipeline.n_samples_buffered_, 1)

        # Failed validation must not have fixed feature 0 as numeric.
        pipeline.partial_fit([["x", "b"]], [1])
        pipeline.partial_fit([["y", "c"]], [2])
        np.testing.assert_array_equal(
            pipeline.predict([[None, "a"], ["x", "b"], ["y", "c"]]),
            [0, 1, 2],
        )

    def test_unknown_error_does_not_learn_pending_category(self):
        pipeline = categorical_pipeline(buffer_size=2, handle_unknown="error")
        pipeline.partial_fit([["a"], ["b"]], [0, 1])
        pipeline.partial_fit([["c"]], [2])

        with self.assertRaises(ValueError):
            pipeline.predict([["c"]])
        self.assertEqual(pipeline.n_samples_buffered_, 1)
        self.assertEqual(pipeline.model_version_, 1)

        pipeline.partial_fit([["c"]], [2])
        np.testing.assert_array_equal(pipeline.predict([["c"]]), [2])
        self.assertEqual(pipeline.predict(np.empty((0, 1), dtype=object)).shape, (0,))


if __name__ == "__main__":
    unittest.main()