"""Incremental categorical encoding: mapping, boundaries and lifecycle."""

import copy
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from numcompute_stream import OneHotEncoder


class OneHotEncoderTests(unittest.TestCase):
    def assert_learned_equal(self, actual, expected):
        self.assertEqual(actual.n_features_in_, expected.n_features_in_)
        self.assertEqual(actual.n_features_out_, expected.n_features_out_)
        self.assertEqual(actual._feature_kinds, expected._feature_kinds)
        self.assertEqual(actual._missing_indices, expected._missing_indices)
        for current, saved in zip(actual.categories_, expected.categories_):
            assert_array_equal(current, saved)

    def test_hand_calculated_multiple_features(self):
        encoder = OneHotEncoder()
        self.assertIs(
            encoder.fit([
                ["red", "small"],
                ["blue", "large"],
            ]),
            encoder,
        )

        assert_array_equal(encoder.categories_[0], ["blue", "red"])
        assert_array_equal(encoder.categories_[1], ["large", "small"])
        assert_array_equal(
            encoder.transform([
                ["red", "small"],
                ["blue", "large"],
            ]),
            [[0, 1, 0, 1], [1, 0, 1, 0]],
        )

    def test_new_categories_append_without_reordering(self):
        encoder = OneHotEncoder().fit([["red"]])

        self.assertIs(
            encoder.partial_fit([["green"], ["blue"], ["green"], ["red"]]),
            encoder,
        )

        assert_array_equal(encoder.categories_[0], ["red", "blue", "green"])
        assert_array_equal(
            encoder.transform([["blue"], ["red"], ["green"]]),
            [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        )

    def test_earlier_expansion_shifts_later_feature_block(self):
        encoder = OneHotEncoder().fit([
            ["blue", "small"],
            ["red", "small"],
        ])

        assert_array_equal(
            encoder.transform([["red", "small"]]),
            [[0, 1, 1]],
        )

        encoder.partial_fit([["green", "small"]])

        self.assertEqual(encoder.n_features_out_, 4)
        assert_array_equal(
            encoder.transform([["red", "small"]]),
            [[0, 1, 0, 1]],
        )

    def test_repeated_categories_do_not_expand_state(self):
        encoder = OneHotEncoder().fit([["blue"], ["red"]])
        saved = copy.deepcopy(encoder)

        encoder.partial_fit([["red"], ["red"], ["blue"]])

        self.assert_learned_equal(encoder, saved)

    def test_unknown_ignore_zeros_only_affected_block(self):
        encoder = OneHotEncoder().fit([
            ["blue", "small"],
            ["red", "large"],
        ])
        saved = copy.deepcopy(encoder)

        assert_array_equal(
            encoder.transform([
                ["green", "small"],
                ["red", "medium"],
                ["green", "medium"],
            ]),
            [
                [0, 0, 0, 1],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
            ],
        )
        self.assert_learned_equal(encoder, saved)

    def test_unknown_error_including_unlearned_missing(self):
        encoder = OneHotEncoder(handle_unknown="error").fit([["red"]])
        saved = copy.deepcopy(encoder)

        for X in ([["blue"]], [[None]], [[np.nan]]):
            with self.subTest(X=X):
                with self.assertRaises(ValueError):
                    encoder.transform(X)
                self.assert_learned_equal(encoder, saved)

    def test_missing_is_distinct_from_ordinary_strings(self):
        encoder = OneHotEncoder().fit([["None"], ["nan"], [None], [np.nan]])

        assert_array_equal(
            encoder.categories_[0],np.array(["None", "nan", None], dtype=object),
        )
        assert_array_equal(
            encoder.transform([["None"], ["nan"], [None], [np.nan]]),
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
        )

    def test_missing_category_keeps_index_after_expansion(self):
        encoder = OneHotEncoder().fit([["red"], [None]])
        encoder.partial_fit([["blue"], [np.nan]])

        assert_array_equal(
            encoder.categories_[0],
            np.array(["red", None, "blue"], dtype=object),
        )
        assert_array_equal(
            encoder.transform([[None], ["blue"], ["red"]]),
            [[0, 1, 0], [0, 0, 1], [1, 0, 0]],
        )

    def test_missing_only_column_can_acquire_category_type(self):
        encoder = OneHotEncoder().fit([[None], [np.nan]])

        assert_array_equal(encoder.transform([["blue"], [None]]), [[0], [1]],)

        encoder.partial_fit([["blue"]])

        assert_array_equal(encoder.categories_[0], np.array([None, "blue"], dtype=object),)
        assert_array_equal(encoder.transform([["blue"], [None]]), [[0, 1], [1, 0]],)

    def test_numeric_equivalence_and_large_integers(self):
        large = 2**63 + 1
        encoder = OneHotEncoder().fit([[1], [1.0], [large], [large + 1]])

        assert_array_equal(encoder.categories_[0], np.array([1, large, large + 1], dtype=object),)
        assert_array_equal(
            encoder.transform([[large + 1], [1.0], [large]]),
            [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
        )

    def test_typed_numeric_and_string_arrays(self):
        for X in (
            np.array([[2], [1]], dtype=np.int64),
            np.array([[2], [1]], dtype=np.uint64),
            np.array([[2], [1]], dtype=np.float32),
            np.array([["b"], ["a"]]),
        ):
            with self.subTest(dtype=X.dtype):
                encoder = OneHotEncoder().fit(X)
                assert_array_equal(encoder.transform(X), [[0, 1], [1, 0]],)

    def test_fit_resets_categories_and_feature_schema(self):
        encoder = OneHotEncoder().fit([
            ["red", "small"],
            ["blue", "large"],
        ])

        self.assertIs(encoder.fit([[2], [1]]), encoder)

        self.assertEqual(encoder.n_features_in_, 1)
        self.assertEqual(encoder.n_features_out_, 2)
        assert_array_equal(encoder.categories_[0], [1, 2])
        assert_array_equal(encoder.transform([[2]]), [[0, 1]])

    def test_invalid_parameter_values(self):
        with self.assertRaises(TypeError):
            OneHotEncoder(handle_unknown=1)
        with self.assertRaises(ValueError):
            OneHotEncoder(handle_unknown="skip")

        encoder = OneHotEncoder().fit([["red"]])
        saved = copy.deepcopy(encoder)
        encoder.handle_unknown = "skip"

        for method in (
            encoder.fit,
            encoder.partial_fit,
            encoder.transform,
        ):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method([["red"]])
                self.assert_learned_equal(encoder, saved)

    def test_invalid_learning_inputs_preserve_state(self):
        encoder = OneHotEncoder().fit([["red", "small"]])
        saved = copy.deepcopy(encoder)

        bad_inputs = [
            [],
            ["red", "small"],
            np.empty((0, 2), dtype=object),
            np.empty((2, 0), dtype=object),
            [[True, "small"]],
            [[np.inf, "small"]],
            [[1j, "small"]],
            [["red", "small"], [1, "small"]],
        ]

        for method in (encoder.fit, encoder.partial_fit):
            for X in bad_inputs:
                with self.subTest(method=method.__name__, X=repr(X)):
                    with self.assertRaises((TypeError, ValueError)):
                        method(X)
                    self.assert_learned_equal(encoder, saved)

    def test_late_feature_type_error_does_not_publish_updates(self):
        encoder = OneHotEncoder().fit([["red", 1]])
        saved = copy.deepcopy(encoder)

        # Feature 0 could expand, but feature 1 has an incompatible type.
        with self.assertRaises(TypeError):
            encoder.partial_fit([["blue", "wrong"]])

        self.assert_learned_equal(encoder, saved)
        assert_array_equal(encoder.transform([["red", 1]]), [[1, 1]])

    def test_feature_count_and_kind_must_match(self):
        encoder = OneHotEncoder().fit([["red", 1]])
        saved = copy.deepcopy(encoder)

        for method in (encoder.partial_fit, encoder.transform):
            with self.subTest(method=method.__name__):
                with self.assertRaises(ValueError):
                    method([["red"]])
                with self.assertRaises(TypeError):
                    method([[2, 1]])
                self.assert_learned_equal(encoder, saved)

    def test_unfitted_and_empty_transform(self):
        with self.assertRaises(RuntimeError):
            OneHotEncoder().transform([["red"]])

        encoder = OneHotEncoder().fit([["red", 1]])
        result = encoder.transform(np.empty((0, 2), dtype=object))

        self.assertEqual(result.shape, (0, 2))
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_input_and_output_ownership(self):
        X = np.array([["red", 1], ["blue", 2]], dtype=object)
        original = X.copy()
        encoder = OneHotEncoder().fit(X)
        saved = copy.deepcopy(encoder)

        result = encoder.transform(X)

        assert_array_equal(X, original)
        self.assertFalse(np.shares_memory(result, X))

        X[:] = None
        result[:] = 99

        self.assert_learned_equal(encoder, saved)
        assert_array_equal(encoder.transform(original),
                           [[0, 1, 1, 0], [1, 0, 0, 1]],
        )

    def test_snapshot_preserves_old_layout(self):
        encoder = OneHotEncoder().fit([
            ["red", "small"],
            ["blue", "small"],
        ])
        frozen = copy.deepcopy(encoder)

        encoder.partial_fit([["green", "large"]])

        self.assertEqual(frozen.n_features_out_, 3)
        self.assertEqual(encoder.n_features_out_, 5)
        assert_array_equal(
            frozen.transform([["red", "small"]]),
            [[0, 1, 1]],
        )
        self.assertFalse(np.shares_memory(frozen.categories_[0], encoder.categories_[0]))

    def test_read_only_noncontiguous_input(self):
        base = np.array([
            ["red", "unused", "small"],
            ["blue", "unused", "large"],
            ["green", "unused", "small"],
            ["red", "unused", "large"],
        ], dtype=object)
        X = base[::2, ::2]
        X.flags.writeable = False

        encoder = OneHotEncoder().fit(X)

        assert_array_equal(encoder.transform(X), [[0, 1, 1], [1, 0, 1]],)

    def test_fit_transform_accepts_optional_y(self):
        result = OneHotEncoder().fit_transform([["red"], ["blue"]], [0, 1])

        assert_array_equal(result, [[0, 1], [1, 0]])
        self.assertEqual(result.dtype, np.dtype("float64"))

    def test_chunked_and_batch_outputs_match_after_alignment(self):
        X = np.array([
            ["red", "small"],
            ["blue", "large"],
            ["green", "small"],
            ["blue", "medium"],
            ["yellow", "large"],
        ], dtype=object)

        streaming = OneHotEncoder()
        for chunk in np.split(X, [1, 3]):
            streaming.partial_fit(chunk)

        batch = OneHotEncoder().fit(X)
        streamed_output = streaming.transform(X)
        batch_output = batch.transform(X)

        aligned = np.empty_like(batch_output)
        stream_offset = 0
        batch_offset = 0

        # Independent test-only alignment by category meaning.
        for stream_categories, batch_categories in zip(
                streaming.categories_, batch.categories_
        ):
            for index, category in enumerate(batch_categories):
                source = list(stream_categories).index(category)
                aligned[:, batch_offset + index] = (
                    streamed_output[:, stream_offset + source]
                )

            stream_offset += len(stream_categories)
            batch_offset += len(batch_categories)

        assert_array_equal(aligned, batch_output)


if __name__ == "__main__":
    unittest.main()