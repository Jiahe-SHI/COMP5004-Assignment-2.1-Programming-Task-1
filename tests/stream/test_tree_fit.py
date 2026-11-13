"""Tests for recursive batch tree construction and fitted-state handling."""

import unittest
from unittest.mock import patch

import numpy as np

from numcompute_stream.tree import DecisionTreeClassifier, _best_split


def tree_signature(node):
    """Snapshot tree structure and values without retaining mutable arrays."""
    return (
        tuple(node.counts.tolist()), node.prediction, node.n_samples,
        node.depth, node.impurity, node.feature, node.threshold, node.gain,
        None if node.left is None else tree_signature(node.left),
        None if node.right is None else tree_signature(node.right),
    )


class TestTreeFit(unittest.TestCase):
    def test_fit_returns_self_and_publishes_complete_tree(self):
        model = DecisionTreeClassifier(buffer_size=100)
        self.assertFalse(model.is_fitted_)
        self.assertIs(
            model.fit([[0], [1], [2], [3]], ["a", "a", "b", "b"]),
            model,
        )
        self.assertTrue(model.is_fitted_)
        np.testing.assert_array_equal(model.classes_, ["a", "b"])
        self.assertEqual(model.n_classes_, 2)
        self.assertEqual(model.n_features_in_, 1)
        self.assertEqual(model.n_samples_fit_, 4)
        self.assertFalse(model.tree_.is_leaf)
        np.testing.assert_array_equal(model.tree_.left.counts, [2, 0])
        np.testing.assert_array_equal(model.tree_.right.counts, [0, 2])

    def test_single_sample_and_pure_nodes_are_leaves(self):
        for X, y in (([[1]], [7]), ([[0], [1]], [7, 7])):
            with self.subTest(X=X):
                model = DecisionTreeClassifier().fit(X, y)
                self.assertTrue(model.tree_.is_leaf)
                self.assertEqual(model.tree_.depth, 0)
                self.assertEqual(model.classes_[model.tree_.prediction], 7)

    def test_depth_zero_uses_sorted_class_order_for_vote_ties(self):
        model = DecisionTreeClassifier(max_depth=0).fit(
            [[0], [1]], ["z", "a"]
        )
        self.assertTrue(model.tree_.is_leaf)
        self.assertEqual(model.classes_[model.tree_.prediction], "a")
        np.testing.assert_array_equal(model.tree_.counts, [1, 1])

    def test_depth_one_stops_both_children(self):
        model = DecisionTreeClassifier(max_depth=1).fit(
            np.arange(6)[:, None], [0, 0, 1, 1, 2, 2]
        )
        root = model.tree_
        self.assertFalse(root.is_leaf)
        for child in (root.left, root.right):
            self.assertTrue(child.is_leaf)
            self.assertEqual(child.depth, 1)

    def test_min_samples_split_boundary(self):
        X, y = [[0], [1], [2], [3]], [0, 0, 1, 1]
        blocked = DecisionTreeClassifier(min_samples_split=5).fit(X, y)
        allowed = DecisionTreeClassifier(min_samples_split=4).fit(X, y)
        self.assertTrue(blocked.tree_.is_leaf)
        self.assertFalse(allowed.tree_.is_leaf)

    def test_constant_features_and_zero_gain_stop(self):
        cases = (
            ([[1], [1]], [0, 1]),
            ([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0]),
        )
        for X, y in cases:
            with self.subTest(X=X):
                model = DecisionTreeClassifier().fit(X, y)
                self.assertTrue(model.tree_.is_leaf)

    def test_multilevel_multiclass_counts_keep_global_class_order(self):
        model = DecisionTreeClassifier().fit(
            np.arange(6)[:, None], [10, 10, 20, 20, 30, 30]
        )
        root = model.tree_
        self.assertEqual(root.depth, 0)
        self.assertEqual(root.left.depth, 1)
        np.testing.assert_array_equal(root.left.counts, [2, 0, 0])
        np.testing.assert_array_equal(root.right.left.counts, [0, 2, 0])
        np.testing.assert_array_equal(root.right.right.counts, [0, 0, 2])
        self.assertEqual(root.right.right.depth, 2)

    def test_entropy_is_used_by_nodes_and_splits(self):
        model = DecisionTreeClassifier(criterion="entropy").fit(
            [[0], [1], [2], [3]], [0, 0, 1, 1]
        )
        self.assertAlmostEqual(model.tree_.impurity, 1.0)
        self.assertAlmostEqual(model.tree_.gain, 1.0)
        self.assertEqual(model.tree_.left.impurity, 0.0)

    def test_successful_refit_replaces_old_classes_and_schema(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        old_tree = model.tree_
        old_signature = tree_signature(old_tree)
        model.fit([[5, 6], [7, 8]], ["new", "new"])
        self.assertIsNot(model.tree_, old_tree)
        self.assertEqual(model.n_features_in_, 2)
        np.testing.assert_array_equal(model.classes_, ["new"])
        self.assertEqual(tree_signature(old_tree), old_signature)

    def test_invalid_fit_preserves_previous_learned_state(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        old_state = model.__dict__.copy()
        with self.assertRaises(ValueError):
            model.fit([[np.nan]], [0])
        for name, value in old_state.items():
            self.assertIs(model.__dict__[name], value)

    def test_failed_child_construction_preserves_previous_model(self):
        model = DecisionTreeClassifier().fit([[0], [1]], ["a", "b"])
        old_state = model.__dict__.copy()
        calls = 0

        def fail_after_root(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated child construction failure")
            return _best_split(*args, **kwargs)

        with patch(
            "numcompute_stream.tree._best_split",
            side_effect=fail_after_root,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated child"):
                model.fit(np.arange(6)[:, None], [0, 0, 1, 1, 2, 2])
        self.assertEqual(calls, 2)
        for name, value in old_state.items():
            self.assertIs(model.__dict__[name], value)

    def test_failed_initial_fit_remains_unfitted(self):
        model = DecisionTreeClassifier()
        with patch(
            "numcompute_stream.tree._best_split",
            side_effect=RuntimeError,
        ):
            with self.assertRaises(RuntimeError):
                model.fit([[0], [1]], [0, 1])
        self.assertFalse(model.is_fitted_)
        self.assertFalse(hasattr(model, "classes_"))

    def test_candidate_feature_counts_and_selection(self):
        X = np.repeat(np.arange(4)[:, None], 8, axis=1)
        options = ((None, 8), (3, 3), ("sqrt", 2), ("log2", 3))
        for option, expected in options:
            with self.subTest(option=option):
                with patch(
                    "numcompute_stream.tree._best_split",
                    wraps=_best_split,
                ) as spy:
                    model = DecisionTreeClassifier(
                        max_features=option, random_state=17
                    ).fit(X, [0, 0, 1, 1])
                self.assertEqual(model.max_features_, expected)
                features = spy.call_args.kwargs["feature_indices"]
                self.assertEqual(len(features), expected)
                self.assertEqual(len(np.unique(features)), expected)
                self.assertIn(model.tree_.feature, features)

        for option in ("sqrt", "log2"):
            model = DecisionTreeClassifier(max_features=option).fit(
                [[0], [1]], [0, 1]
            )
            self.assertEqual(model.max_features_, 1)

    def test_integer_seed_reproduces_tree_and_refit(self):
        rng = np.random.default_rng(7)
        X = rng.integers(0, 5, size=(30, 5))
        y = (X[:, 0] + X[:, 1]) % 3
        first = DecisionTreeClassifier(
            max_features=2, random_state=42
        ).fit(X, y)
        second = DecisionTreeClassifier(
            max_features=2, random_state=42
        ).fit(X, y)
        expected = tree_signature(first.tree_)
        self.assertEqual(tree_signature(second.tree_), expected)
        first.fit(X, y)
        self.assertEqual(tree_signature(first.tree_), expected)

    def test_does_not_modify_global_random_state(self):
        before = np.random.get_state()
        DecisionTreeClassifier(max_features=1, random_state=42).fit(
            [[0, 0], [1, 1]], [0, 1]
        )
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_training_input_changes_do_not_change_fitted_tree(self):
        X = np.array([[0], [1], [2], [3]])
        y = np.array([10, 10, 20, 20])
        model = DecisionTreeClassifier().fit(X, y)
        expected = tree_signature(model.tree_)
        X[:] = 99
        y[:] = 99
        self.assertEqual(tree_signature(model.tree_), expected)
        np.testing.assert_array_equal(model.classes_, [10, 20])

    def test_large_integer_labels_and_features(self):
        X = np.array(
            [2**64 - 2, 2**64 - 1], dtype=np.uint64
        )[:, None]
        y = [2**80, 2**80 + 1]
        model = DecisionTreeClassifier().fit(X, y)
        self.assertEqual(model.classes_.tolist(), y)
        np.testing.assert_array_equal(model.tree_.left.counts, [1, 0])
        np.testing.assert_array_equal(model.tree_.right.counts, [0, 1])

    def test_invalid_parameters_and_fit_revalidation(self):
        for params in (
            {"max_depth": -1},
            {"min_samples_split": 1},
            {"max_features": "auto"},
            {"buffer_size": 0},
            {"criterion": "unknown"},
            {"random_state": -1},
        ):
            with self.subTest(params=params):
                with self.assertRaises(ValueError):
                    DecisionTreeClassifier(**params)

        with self.assertRaises(ValueError):
            DecisionTreeClassifier(max_features=2).fit([[0], [1]], [0, 1])

        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        old_tree = model.tree_
        model.min_samples_split = 1
        with self.assertRaises(ValueError):
            model.fit([[0], [1]], [0, 1])
        self.assertIs(model.tree_, old_tree)


if __name__ == "__main__":
    unittest.main()