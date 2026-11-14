"""Cross-component checks for streaming preprocessing and statistics."""

import copy
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import (
    DecisionTreeClassifier,
    Imputer,
    OneHotEncoder,
    StandardScaler,
    StreamingStats,
)


class PreprocessingIntegrationTests(unittest.TestCase):
    def test_shared_observed_statistics_across_components(self):
        X = np.array([
            [1.0, np.nan, np.nan],
            [3.0, 10.0, np.nan],
            [8.0, 14.0, np.nan],
            [np.nan, 18.0, np.nan],
        ])

        scaler = StandardScaler()
        imputer = Imputer(fill_value=-5.0)
        stats = StreamingStats()

        for chunk in np.split(X, [1, 3]):
            scaler.partial_fit(chunk)
            imputer.partial_fit(chunk)
            stats.update_stats(chunk)

        summary = stats.result()

        assert_array_equal(scaler.n_samples_seen_, summary["count"])
        assert_array_equal(imputer.n_samples_seen_, summary["count"])
        assert_allclose(scaler.mean_, summary["mean"])
        assert_allclose(scaler.var_, summary["var"])
        assert_allclose(
            imputer.statistics_,
            np.where(summary["count"] > 0, summary["mean"], -5.0),
        )
        assert_array_equal(summary["count"], [3, 3, 0])
        assert_allclose(summary["mean"], [4.0, 14.0, np.nan])

    def test_numeric_preprocessing_produces_finite_tree_input(self):
        X = np.array([
            [0.0, np.nan],
            [1.0, np.nan],
            [4.0, np.nan],
            [5.0, np.nan],
        ])
        y = np.array([0, 0, 1, 1])

        imputer = Imputer()
        filled = imputer.fit_transform(X)
        scaler = StandardScaler()
        transformed = scaler.fit_transform(filled)

        self.assertTrue(np.isfinite(transformed).all())
        assert_array_equal(filled[:, 1], [0, 0, 0, 0])
        assert_array_equal(transformed[:, 1], [0, 0, 0, 0])

        tree = DecisionTreeClassifier(max_depth=2)
        tree.fit(transformed, y)

        assert_array_equal(tree.predict(transformed), y)

    def test_mixed_feature_blocks_can_train_a_tree(self):
        numeric = np.array([
            [1.0], [np.nan], [1.0], [np.nan]
        ])
        categorical = [["red"], ["blue"], ["red"], ["blue"]]
        y = np.array([1, 0, 1, 0])

        imputer = Imputer()
        scaler = StandardScaler()
        encoder = OneHotEncoder()

        numeric_output = scaler.fit_transform(imputer.fit_transform(numeric))
        category_output = encoder.fit_transform(categorical)
        combined = np.column_stack((numeric_output, category_output))

        self.assertEqual(combined.shape, (4, 3))
        self.assertTrue(np.isfinite(combined).all())

        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(combined, y)

        assert_array_equal(tree.predict(combined), y)

    def test_frozen_numeric_chain_keeps_old_tree_predictions(self):
        X = np.array([[0.0], [1.0], [4.0], [5.0]])
        y = np.array([0, 0, 1, 1])

        imputer = Imputer().fit(X)
        scaler = StandardScaler().fit(imputer.transform(X))
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(scaler.transform(imputer.transform(X)), y)

        frozen_imputer, frozen_scaler, frozen_tree = copy.deepcopy(
            (imputer, scaler, tree)
        )

        probe = np.array([[0.0], [5.0], [np.nan]])
        expected = frozen_tree.predict(
            frozen_scaler.transform(frozen_imputer.transform(probe))
        )

        # Update the live preprocessing objects independently.
        imputer.partial_fit([[100.0], [200.0]])
        scaler.partial_fit([[100.0], [200.0]])

        actual = frozen_tree.predict(
            frozen_scaler.transform(frozen_imputer.transform(probe))
        )

        assert_array_equal(actual, expected)
        assert_allclose(frozen_imputer.statistics_, [2.5])
        assert_allclose(frozen_scaler.mean_, [2.5])
        self.assertFalse(np.allclose(scaler.mean_, frozen_scaler.mean_))

    def test_category_expansion_requires_matching_model_layout(self):
        X = [["red"], ["blue"], ["red"], ["blue"]]
        y = np.array([1, 0, 1, 0])

        encoder = OneHotEncoder().fit(X)
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(encoder.transform(X), y)

        frozen_encoder, frozen_tree = copy.deepcopy((encoder, tree))
        encoder.partial_fit([["green"]])

        self.assertEqual(frozen_encoder.n_features_out_, 2)
        self.assertEqual(encoder.n_features_out_, 3)

        # The old tree rejects a changed input feature count.
        with self.assertRaises(ValueError):
            tree.predict(encoder.transform([["red"]]))

        assert_array_equal(frozen_tree.predict(frozen_encoder.transform(X)), y,)

        # A newly fitted pair supports the expanded layout.
        new_X = [["red"], ["blue"], ["green"]]
        new_y = np.array([1, 0, 2])
        new_encoder = OneHotEncoder().fit(new_X)
        new_tree = DecisionTreeClassifier(max_depth=3)
        new_tree.fit(new_encoder.transform(new_X), new_y)

        assert_array_equal(new_tree.predict(new_encoder.transform(new_X)), new_y,)

    def test_querying_all_components_does_not_update_learning(self):
        X = np.array([
            [1.0, np.nan],
            [3.0, 4.0],
        ])
        imputer = Imputer().fit(X)
        scaler = StandardScaler().fit(X)
        encoder = OneHotEncoder().fit([["red"], ["blue"]])
        stats = StreamingStats(bin_edges=[0, 2, 5]).update_stats(X)

        saved_counts = {
            "imputer": imputer.n_samples_seen_.copy(),
            "scaler": scaler.n_samples_seen_.copy(),
            "stats": stats.result()["count"],
        }
        saved_categories = [values.copy() for values in encoder.categories_]

        imputer.transform([[np.nan, 100.0]])
        scaler.transform([[100.0, np.nan]])
        encoder.transform([["green"]])
        stats.result()
        stats.quantile([0, 0.5, 1])
        stats.histogram()

        assert_array_equal(imputer.n_samples_seen_, saved_counts["imputer"])
        assert_array_equal(scaler.n_samples_seen_, saved_counts["scaler"])
        assert_array_equal(stats.result()["count"], saved_counts["stats"])
        self.assertEqual(stats.result()["n_rows"], 2)

        for actual, expected in zip(encoder.categories_, saved_categories):
            assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()