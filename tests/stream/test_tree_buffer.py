"""Tests for pending tree buffers and input ownership."""

import unittest

import numpy as np

from numcompute_stream import DecisionTreeClassifier


class TestTreeBuffer(unittest.TestCase):
    def test_initial_state_and_readonly_count(self):
        model = DecisionTreeClassifier()

        self.assertEqual(model.n_samples_buffered_, 0)
        self.assertFalse(model.is_fitted_)

        with self.assertRaises(AttributeError):
            model.n_samples_buffered_ = 10

    def test_appends_unequal_chunks_without_training(self):
        model = DecisionTreeClassifier(buffer_size=500)

        for size, expected in ((200, 200), (150, 350), (100, 450)):
            X = np.arange(size * 2).reshape(size, 2)
            y = np.arange(size) % 2
            self.assertIs(model._append_buffer(X, y), model)
            self.assertEqual(model.n_samples_buffered_, expected)
            self.assertFalse(model.is_fitted_)

        self.assertEqual(len(model._buffer_chunks), 3)
        with self.assertRaises(RuntimeError):
            model.predict([[0, 1]])

    def test_copies_readonly_noncontiguous_inputs(self):
        source_X = np.arange(24).reshape(6, 4)
        source_y = np.arange(12)
        X = source_X[:, ::2]
        y = source_y[::2]
        X.setflags(write=False)
        y.setflags(write=False)
        expected_X = X.copy()
        expected_y = y.copy()

        model = DecisionTreeClassifier()
        model._append_buffer(X, y)
        stored_X, stored_y = model._buffer_chunks[0]

        self.assertFalse(np.shares_memory(stored_X, source_X))
        self.assertFalse(np.shares_memory(stored_y, source_y))

        source_X[:] = -1
        source_y[:] = -1
        np.testing.assert_array_equal(stored_X, expected_X)
        np.testing.assert_array_equal(stored_y, expected_y)

    def test_invalid_append_preserves_pending_data(self):
        model = DecisionTreeClassifier()
        model._append_buffer([[0], [1]], [0, 1])
        original_chunk = model._buffer_chunks[0]

        invalid_pairs = [
            ([[2], [3]], [0]),
            ([[np.nan]], [0]),
            ([[np.inf]], [0]),
            ([[2, 3]], [0]),
            (np.empty((0, 1)), np.array([], dtype=int)),
            ([[2]], ["new"]),
        ]
        for X, y in invalid_pairs:
            with self.subTest(X=X, y=y):
                with self.assertRaises((TypeError, ValueError)):
                    model._append_buffer(X, y)
                self.assertEqual(model.n_samples_buffered_, 2)
                self.assertEqual(len(model._buffer_chunks), 1)
                self.assertIs(model._buffer_chunks[0], original_chunk)

    def test_invalid_configuration_does_not_append(self):
        model = DecisionTreeClassifier()
        model._append_buffer([[0]], [0])
        model.buffer_size = 0

        with self.assertRaises(ValueError):
            model._append_buffer([[1]], [1])

        self.assertEqual(model.n_samples_buffered_, 1)
        self.assertEqual(len(model._buffer_chunks), 1)

    def test_append_preserves_active_model_and_predictions(self):
        model = DecisionTreeClassifier().fit([[0], [1]], [0, 1])
        original_tree = model.tree_
        original_classes = model.classes_.copy()
        expected = model.predict([[0], [1]])

        model._append_buffer([[2], [3]], [2, 2])

        self.assertEqual(model.n_samples_buffered_, 2)
        self.assertIs(model.tree_, original_tree)
        np.testing.assert_array_equal(model.classes_, original_classes)
        np.testing.assert_array_equal(model.predict([[0], [1]]), expected)

        with self.assertRaises(ValueError):
            model._append_buffer([[1, 2]], [0])
        with self.assertRaises(TypeError):
            model._append_buffer([[1]], ["zero"])
        self.assertEqual(model.n_samples_buffered_, 2)

    def test_preserves_large_unsigned_labels_and_new_strings(self):
        labels = np.array([2**63 + 1, 2**64 - 1], dtype=np.uint64)
        model = DecisionTreeClassifier()
        model._append_buffer([[0], [1]], labels)

        stored_labels = model._buffer_chunks[0][1]
        self.assertEqual(stored_labels.dtype, labels.dtype)
        np.testing.assert_array_equal(stored_labels, labels)

        string_model = DecisionTreeClassifier()
        string_model._append_buffer([[0]], ["a"])
        string_model._append_buffer([[1]], ["longer_new_class"])
        self.assertEqual(string_model.n_samples_buffered_, 2)
        self.assertEqual(
            string_model._buffer_chunks[1][1][0], "longer_new_class"
        )


if __name__ == "__main__":
    unittest.main()