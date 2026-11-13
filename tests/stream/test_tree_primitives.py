"""Hand-calculated tests for label encoding and tree impurity."""

import unittest

import numpy as np

from numcompute_stream.tree import (_encode_labels, _impurity_from_counts,)


class TestLabelEncoding(unittest.TestCase):
    def test_integer_labels_are_sorted_and_reversible(self):
        y = np.array([20, -5, 20, 7])
        classes, encoded = _encode_labels(y)

        np.testing.assert_array_equal(classes, [-5, 7, 20])
        np.testing.assert_array_equal(encoded, [2, 0, 2, 1])
        np.testing.assert_array_equal(classes[encoded], y)
        self.assertTrue(np.issubdtype(encoded.dtype, np.integer))

    def test_string_labels_are_sorted_and_reversible(self):
        y = ["dog", "cat", "dog", "bird"]
        classes, encoded = _encode_labels(y)

        np.testing.assert_array_equal(classes, ["bird", "cat", "dog"])
        np.testing.assert_array_equal(encoded, [2, 1, 2, 0])
        np.testing.assert_array_equal(classes[encoded], y)

    def test_preserves_large_python_integers(self):
        y = [2**80, 2**53 + 1, 2**80, -1]
        classes, encoded = _encode_labels(y)

        self.assertEqual(classes.tolist(), [-1, 2**53 + 1, 2**80])
        self.assertEqual(classes[encoded].tolist(), y)

    def test_preserves_unsigned_integer_labels(self):
        y = np.array([2**64 - 1, 2**63, 2**64 - 1], dtype=np.uint64)
        classes, encoded = _encode_labels(y)

        self.assertEqual(classes.dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(encoded, [1, 0, 1])
        np.testing.assert_array_equal(classes[encoded], y)

    def test_single_class_has_zero_codes(self):
        classes, encoded = _encode_labels(["only", "only"])
        np.testing.assert_array_equal(classes, ["only"])
        np.testing.assert_array_equal(encoded, [0, 0])

    def test_rejects_invalid_labels(self):
        for y in ([1, "cat"], [True, False], [1.0, 2.0], [1, None]):
            with self.subTest(y=y):
                with self.assertRaises(TypeError):
                    _encode_labels(y)

        with self.assertRaises(ValueError):
            _encode_labels([])


class TestImpurity(unittest.TestCase):
    def test_pure_node_has_zero_impurity(self):
        for criterion in ("gini", "entropy"):
            with self.subTest(criterion=criterion):
                self.assertEqual(
                    _impurity_from_counts([0, 5, 0], criterion), 0.0
                )

    def test_balanced_binary_node(self):
        self.assertAlmostEqual(
            _impurity_from_counts([2, 2], "gini"), 0.5
        )
        self.assertAlmostEqual(
            _impurity_from_counts([2, 2], "entropy"), 1.0
        )

    def test_unbalanced_binary_node(self):
        self.assertAlmostEqual(
            _impurity_from_counts([3, 1], "gini"), 0.375
        )
        self.assertAlmostEqual(
            _impurity_from_counts([3, 1], "entropy"),
            0.8112781244591328,
        )

    def test_multiclass_hand_calculation(self):
        # Probabilities: 1/2, 1/4, 1/4.
        self.assertAlmostEqual(
            _impurity_from_counts([2, 1, 1], "gini"), 0.625
        )
        self.assertAlmostEqual(
            _impurity_from_counts([2, 1, 1], "entropy"), 1.5
        )

    def test_multiple_nodes_are_evaluated_together(self):
        counts = np.array([[4, 0], [2, 2], [3, 1], [0, 0]])

        np.testing.assert_allclose(
            _impurity_from_counts(counts, "gini"),
            [0.0, 0.5, 0.375, 0.0],
        )
        np.testing.assert_allclose(
            _impurity_from_counts(counts, "entropy"),
            [0.0, 1.0, 0.8112781244591328, 0.0],
        )

    def test_zero_counts_do_not_produce_numeric_warnings(self):
        with np.errstate(all="raise"):
            for criterion in ("gini", "entropy"):
                self.assertEqual(
                    _impurity_from_counts([0, 0], criterion), 0.0
                )
                self.assertEqual(
                    _impurity_from_counts([5, 0], criterion), 0.0
                )

    def test_large_finite_counts_do_not_overflow(self):
        with np.errstate(all="raise"):
            self.assertAlmostEqual(
                _impurity_from_counts([1e308, 1e308], "gini"), 0.5
            )
            self.assertAlmostEqual(
                _impurity_from_counts([1e308, 1e308], "entropy"), 1.0
            )

    def test_empty_node_batch_returns_empty_vector(self):
        result = _impurity_from_counts(np.empty((0, 3)))
        self.assertEqual(result.shape, (0,))

    def test_input_is_not_modified(self):
        counts = np.array([[3, 1], [2, 2]], dtype=np.int64)
        original = counts.copy()
        counts.setflags(write=False)

        _impurity_from_counts(counts, "entropy")

        np.testing.assert_array_equal(counts, original)
        self.assertFalse(counts.flags.writeable)

    def test_rejects_invalid_counts(self):
        invalid_values = (
            [-1, 2],
            [np.nan, 1],
            [np.inf, 1],
        )
        for counts in invalid_values:
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    _impurity_from_counts(counts)

        for counts in ([True, False], ["1", "2"], [1 + 2j, 1]):
            with self.subTest(counts=counts):
                with self.assertRaises(TypeError):
                    _impurity_from_counts(counts)

    def test_rejects_invalid_shapes(self):
        for counts in (1, [], np.empty((2, 0)), np.zeros((1, 2, 3))):
            with self.subTest(counts=counts):
                with self.assertRaises(ValueError):
                    _impurity_from_counts(counts)

    def test_rejects_invalid_criterion(self):
        with self.assertRaises(ValueError):
            _impurity_from_counts([1, 1], "unknown")
        with self.assertRaises(TypeError):
            _impurity_from_counts([1, 1], None)


if __name__ == "__main__":
    unittest.main()