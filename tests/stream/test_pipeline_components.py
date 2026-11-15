"""Tests for constructing fresh Pipeline candidates."""

import unittest

import numpy as np

from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    OneHotEncoder,
    StandardScaler,
)
from numcompute_stream.pipeline import _fresh_component


class TestFreshPipelineComponents(unittest.TestCase):
    def test_scaler_does_not_inherit_statistics(self):
        original = StandardScaler().fit([[0.0], [2.0]])
        candidate = _fresh_component(original)

        self.assertIsNot(candidate, original)
        self.assertFalse(hasattr(candidate, "n_features_in_"))
        with self.assertRaises(RuntimeError):
            candidate.transform([[1.0]])

        candidate.partial_fit([[10.0], [14.0]])

        np.testing.assert_allclose(candidate.mean_, [12.0])
        np.testing.assert_allclose(original.mean_, [1.0])
        np.testing.assert_array_equal(candidate.n_samples_seen_, [2])

    def test_imputer_retains_configuration_not_history(self):
        for strategy in ("mean", "constant"):
            with self.subTest(strategy=strategy):
                original = Imputer(
                    strategy=strategy, fill_value=-7.0
                ).fit([[2.0], [4.0]])
                candidate = _fresh_component(original)

                self.assertIsNot(candidate, original)
                self.assertEqual(candidate.strategy, strategy)
                self.assertEqual(candidate.fill_value, -7.0)
                self.assertFalse(hasattr(candidate, "statistics_"))

                candidate.partial_fit([[np.nan]])
                np.testing.assert_allclose(candidate.transform([[np.nan]]), [[-7.0]])
                np.testing.assert_array_equal(original.n_samples_seen_, [2])

    def test_encoder_does_not_inherit_categories(self):
        original = OneHotEncoder(handle_unknown="error").fit([["old"]])
        candidate = _fresh_component(original)

        self.assertEqual(candidate.handle_unknown, "error")
        self.assertFalse(hasattr(candidate, "categories_"))

        candidate.partial_fit([["new"]])

        np.testing.assert_array_equal(candidate.transform([["new"]]), [[1.0]])
        np.testing.assert_array_equal(original.transform([["old"]]), [[1.0]])
        with self.assertRaises(ValueError):
            candidate.transform([["old"]])
        with self.assertRaises(ValueError):
            original.transform([["new"]])

    def test_tree_retains_parameters_not_model_or_buffer(self):
        original = DecisionTreeClassifier(
            max_depth=3,
            min_samples_split=3,
            max_features="sqrt",
            buffer_size=5,
            criterion="entropy",
            random_state=17,
        )
        original.fit([[0.0], [1.0]], [0, 1])
        original.partial_fit([[2.0]], [1])
        predictions_before = original.predict([[0.0], [1.0]])

        candidate = _fresh_component(original)

        for name in (
            "max_depth",
            "min_samples_split",
            "max_features",
            "buffer_size",
            "criterion",
            "random_state",
        ):
            self.assertEqual(getattr(candidate, name), getattr(original, name))

        self.assertFalse(candidate.is_fitted_)
        self.assertEqual(candidate.n_samples_buffered_, 0)
        self.assertEqual(candidate.n_retrains_, 0)
        self.assertEqual(candidate.model_version_, 0)

        candidate.fit([[10.0], [11.0]], [9, 9])

        self.assertEqual(original.n_samples_buffered_, 1)
        self.assertEqual(original.model_version_, 1)
        np.testing.assert_array_equal(original.predict([[0.0], [1.0]]), predictions_before)
        np.testing.assert_array_equal(candidate.classes_, [9])

    def test_unsupported_types_and_subclasses_are_rejected(self):
        class CustomScaler(StandardScaler):
            pass

        for component in (object(), CustomScaler()):
            with self.subTest(component=type(component).__name__):
                with self.assertRaisesRegex(TypeError, "Unsupported component type"):
                    _fresh_component(component)


if __name__ == "__main__":
    unittest.main()