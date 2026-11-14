"""Streaming statistics: batch references, boundaries and state ownership."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from numcompute_stream import StreamingStats


class StreamingStatsTests(unittest.TestCase):
    def assert_state_equal(self, actual, expected):
        self.assertEqual(actual.n_features_in_, expected.n_features_in_)
        self.assertEqual(actual.n_rows_seen_, expected.n_rows_seen_)

        actual_result = actual.result()
        expected_result = expected.result()
        self.assertEqual(actual_result["n_rows"], expected_result["n_rows"])
        for key in ("count", "mean", "var"):
            assert_array_equal(actual_result[key], expected_result[key])

        assert_array_equal(
            actual.quantile([0, 0.25, 0.5, 0.75, 1]),
            expected.quantile([0, 0.25, 0.5, 0.75, 1]),
        )

        if expected._bin_edges is None:
            self.assertIsNone(actual._bin_edges)
            self.assertIsNone(actual._hist_counts)
        else:
            for key in ("counts", "bin_edges"):
                assert_array_equal(actual.histogram()[key], expected.histogram()[key],)

    def test_hand_calculated_moments(self):
        stats = StreamingStats()

        self.assertIs(stats.update_stats([[1.0], [3.0]]), stats)
        stats.update_stats([[8.0]])

        result = stats.result()
        self.assertEqual(result["n_rows"], 3)
        assert_array_equal(result["count"], [3])
        assert_allclose(result["mean"], [4.0])
        assert_allclose(result["var"], [26.0 / 3.0])

    def test_unequal_chunks_match_numpy_references(self):
        X = np.random.default_rng(23).normal(size=(107, 4))
        X[::3, 0] = np.nan
        X[::7, 2] = np.nan
        edges = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])

        stats = StreamingStats(bin_edges=edges)
        for chunk in np.split(X, [1, 8, 31, 104]):
            stats.update_stats(chunk)

        result = stats.result()
        self.assertEqual(result["n_rows"], len(X))
        assert_array_equal(result["count"], np.sum(~np.isnan(X), axis=0))
        assert_allclose(result["mean"], np.nanmean(X, axis=0), atol=1e-14)
        assert_allclose(result["var"], np.nanvar(X, axis=0), rtol=1e-13)
        assert_allclose(
            stats.quantile([0, 0.25, 0.5, 0.75, 1]),
            np.nanquantile(X, [0, 0.25, 0.5, 0.75, 1], axis=0),
            atol=1e-14,
        )

        expected_histogram = np.stack([
            np.histogram(X[~np.isnan(X[:, feature]), feature],bins=edges,)[0]
            for feature in range(X.shape[1])
        ])
        assert_array_equal(stats.histogram()["counts"], expected_histogram)

    def test_missing_column_recovers(self):
        stats = StreamingStats(bin_edges=[0, 5, 10])
        stats.update_stats([
            [1.0, np.nan],
            [3.0, np.nan],
        ])

        result = stats.result()
        assert_array_equal(result["count"], [2, 0])
        assert_allclose(result["mean"], [2.0, np.nan])
        assert_allclose(result["var"], [1.0, np.nan])
        assert_allclose(stats.quantile(0.5), [2.0, np.nan])
        assert_array_equal(stats.histogram()["counts"], [[2, 0], [0, 0]])

        stats.update_stats([[np.nan, 10.0]])

        assert_array_equal(stats.result()["count"], [2, 1])
        assert_allclose(stats.result()["mean"], [2.0, 10.0])
        assert_allclose(stats.quantile(0.5), [2.0, 10.0])
        assert_array_equal(stats.histogram()["counts"], [[2, 0], [0, 1]])

    def test_all_missing_chunk_counts_rows_only(self):
        stats = StreamingStats(bin_edges=[0, 5])
        stats.update_stats(np.full((3, 2), np.nan))

        result = stats.result()
        self.assertEqual(result["n_rows"], 3)
        assert_array_equal(result["count"], [0, 0])
        assert_allclose(result["mean"], [np.nan, np.nan])
        assert_allclose(result["var"], [np.nan, np.nan])
        assert_allclose(
            stats.quantile([0, 0.5, 1]), np.full((3, 2), np.nan),
        )
        assert_array_equal(stats.histogram()["counts"], [[0], [0]])

    def test_quantile_linear_interpolation_and_shape(self):
        stats = StreamingStats().update_stats([
            [1.0, 10.0],
            [3.0, np.nan],
            [8.0, np.nan],
        ])

        assert_allclose(
            stats.quantile([0, 0.25, 0.5, 0.75, 1]),
            [
                [1.0, 10.0],
                [2.0, 10.0],
                [3.0, 10.0],
                [5.5, 10.0],
                [8.0, 10.0],
            ],
        )
        self.assertEqual(stats.quantile(0.5).shape, (2,))
        self.assertEqual(stats.quantile([0.5]).shape, (1, 2))

    def test_quantile_preserves_requested_order_and_duplicates(self):
        stats = StreamingStats().update_stats([[1.0], [3.0], [8.0]])

        assert_allclose(
            stats.quantile([1, 0, 0.5, 0.5]),
            [[8.0], [1.0], [3.0], [3.0]],
        )

    def test_invalid_quantiles_do_not_change_state(self):
        stats = StreamingStats().update_stats([[1.0], [3.0]])
        saved = copy.deepcopy(stats)

        for q in (
            -0.1, 1.1, np.nan, np.inf, [], [[0.5]], "0.5", True, 1j,
        ):
            with self.subTest(q=repr(q)):
                with self.assertRaises((TypeError, ValueError)):
                    stats.quantile(q)
                self.assert_state_equal(stats, saved)

    def test_histogram_endpoints_and_out_of_range_values(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([
            [-1.0], [0.0], [1.0], [2.0],
            [4.0], [5.0], [np.nan],
        ])

        assert_array_equal(stats.histogram()["counts"], [[2, 2]])
        assert_array_equal(stats.result()["count"], [6])
        assert_allclose(stats.result()["mean"], [11.0 / 6.0])
        assert_allclose(stats.quantile([0, 1]), [[-1.0], [5.0]])

    def test_histogram_has_independent_feature_counts(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([
            [0.0, 2.0],
            [1.0, 3.0],
        ])
        stats.update_stats([[4.0, 4.0]])

        assert_array_equal(stats.histogram()["counts"], [[2, 1], [0, 3]])

    def test_invalid_bin_edges(self):
        cases = [
            [0],
            [],
            [[0, 1]],
            [0, 0, 1],
            [2, 1],
            [0, np.inf],
            [0, np.nan],
            ["0", "1"],
            [False, True],
            [0j, 1j],
            np.array([2**63, 2**63 + 1], dtype=np.uint64),
        ]

        for edges in cases:
            with self.subTest(edges=repr(edges)):
                with self.assertRaises((TypeError, ValueError)):
                    StreamingStats(bin_edges=edges)

    def test_bin_edges_are_copied(self):
        edges = np.array([0.0, 2.0, 4.0])
        stats = StreamingStats(bin_edges=edges)
        edges[:] = -99

        stats.update_stats([[4.0]])

        assert_array_equal(stats.histogram()["bin_edges"], [0.0, 2.0, 4.0])
        assert_array_equal(stats.histogram()["counts"], [[0, 1]])

    def test_queries_before_update_and_histogram_disabled(self):
        stats = StreamingStats(bin_edges=[0, 1])

        for query in (
            stats.result, lambda: stats.quantile(0.5), stats.histogram,
        ):
            with self.assertRaises(RuntimeError):
                query()

        plain = StreamingStats().update_stats([[1.0]])
        with self.assertRaises(ValueError):
            plain.histogram()

    def test_reset_clears_history_and_retains_edges(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([[1.0, 3.0], [2.0, 4.0]])

        self.assertIs(stats.reset(), stats)
        self.assertEqual(stats.n_rows_seen_, 0)
        self.assertIsNone(stats.n_features_in_)

        for query in (
            stats.result, lambda: stats.quantile(0.5), stats.histogram,
        ):
            with self.assertRaises(RuntimeError):
                query()

        stats.update_stats([[4.0]])

        assert_array_equal(stats.result()["count"], [1])
        assert_allclose(stats.quantile([0, 0.5, 1]), [[4], [4], [4]])
        assert_array_equal(stats.histogram()["counts"], [[0, 1]])
        assert_array_equal(stats.histogram()["bin_edges"], [0.0, 2.0, 4.0])

    def test_invalid_updates_preserve_all_state(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([[1.0, 2.0], [3.0, np.nan]])
        saved = copy.deepcopy(stats)

        bad_inputs = [
            [],
            [1, 2],
            np.empty((0, 2)),
            np.empty((2, 0)),
            [[1.0]],
            [[np.inf, 1.0]],
            [[True, False]],
            [["1", "2"]],
            np.array([[1, 2]], dtype=object),
            [[1j, 2j]],
        ]

        for X in bad_inputs:
            with self.subTest(X=repr(X)):
                with self.assertRaises((TypeError, ValueError)):
                    stats.update_stats(X)
                self.assert_state_equal(stats, saved)

    def test_numeric_overflow_preserves_all_state(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([[1.0], [3.0]])
        saved = copy.deepcopy(stats)

        with self.assertRaises(FloatingPointError):
            stats.update_stats([[-1e308], [1e308]])

        self.assert_state_equal(stats, saved)

    def test_histogram_failure_preserves_all_state(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([[1.0, 2.0]])
        saved = copy.deepcopy(stats)

        real_histogram = np.histogram
        first_result = real_histogram([3.0], bins=[0, 2, 4])

        # Fail on the second feature, after the first candidate was computed.
        with patch(
            "numcompute_stream.stats.np.histogram",
            side_effect=[
                first_result,
                RuntimeError("simulated histogram failure"),
            ],
        ):
            with self.assertRaisesRegex(
                RuntimeError, "simulated histogram failure"
            ):
                stats.update_stats([[3.0, 4.0]])

        self.assert_state_equal(stats, saved)

    def test_input_mutation_does_not_change_retained_history(self):
        X = np.array([[1.0, np.nan], [3.0, 4.0]])
        original = X.copy()
        stats = StreamingStats(bin_edges=[0, 2, 4])

        stats.update_stats(X)
        saved = copy.deepcopy(stats)
        assert_array_equal(X, original)

        X[:] = -999

        self.assert_state_equal(stats, saved)

    def test_returned_arrays_are_independent(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([[1.0, 2.0], [3.0, 4.0]])
        saved = copy.deepcopy(stats)

        result = stats.result()
        for key in ("count", "mean", "var"):
            result[key][:] = -99

        quantiles = stats.quantile([0, 0.5, 1])
        quantiles[:] = -99

        histogram = stats.histogram()
        histogram["counts"][:] = -99
        histogram["bin_edges"][:] = -99

        self.assert_state_equal(stats, saved)

    def test_deepcopy_snapshot_remains_independent(self):
        stats = StreamingStats(bin_edges=[0, 2, 4])
        stats.update_stats([[1.0], [3.0]])
        frozen = copy.deepcopy(stats)

        stats.update_stats([[4.0]])

        assert_array_equal(frozen.result()["count"], [2])
        assert_allclose(frozen.quantile(0.5), [2.0])
        assert_array_equal(frozen.histogram()["counts"], [[1, 1]])
        self.assertFalse(
            np.shares_memory(stats._history[0][0], frozen._history[0][0])
        )

    def test_read_only_noncontiguous_input(self):
        X = np.arange(48, dtype=float).reshape(8, 6)[::2, ::2]
        X[0, 1] = np.nan
        X.flags.writeable = False

        stats = StreamingStats(bin_edges=[0, 12, 24, 48])
        stats.update_stats(X)

        assert_allclose(stats.result()["mean"], np.nanmean(X, axis=0))
        assert_allclose(stats.result()["var"], np.nanvar(X, axis=0))
        assert_allclose(stats.quantile(0.5), np.nanmedian(X, axis=0))


if __name__ == "__main__":
    unittest.main()