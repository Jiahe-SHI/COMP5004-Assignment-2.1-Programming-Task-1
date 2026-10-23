import unittest
from unittest.mock import patch
from benchmark.run_benchmarks import (
    check_equivalent, run_case, prepare_inputs,
)
import numpy as np

class TestCheckEquivalent(unittest.TestCase):
    def test_close_results(self):
        # Small floating-point differences are acceptable.
        check_equivalent(1.0, 1.000000000001)
        check_equivalent([1.0, 2.0], [1.0, 2.000000000001])

    def test_topk_results(self):
        expected = ([9.0, 9.0], [1, 2])
        actual = ([9.0, 9.0], [1, 2])
        check_equivalent(expected, actual, exact=True)

        # Equal values do not make different indices acceptable.
        wrong = ([9.0, 9.0], [2, 1])
        with self.assertRaises(AssertionError):
            check_equivalent(expected, wrong, exact=True)

    def test_wrong_shape(self):
        with self.assertRaises(AssertionError):
            check_equivalent([1, 2], [[1, 2]])

    def test_wrong_values(self):
        with self.assertRaises(AssertionError):
            check_equivalent([1, 2], [1, 3])

        # Missing results must not pass the correctness check.
        with self.assertRaises(AssertionError):
            check_equivalent(float("nan"), float("nan"))

    def test_invalid_flag(self):
        with self.assertRaises(TypeError):
            check_equivalent([1], [1], exact="yes")

class TestRunCase(unittest.TestCase):
    def test_timing_and_speedup(self):
        # Fixed times check the calculation, not actual performance.
        with patch(
            "numcompute.benchmarking.time.perf_counter",
            side_effect=[0, 2, 10, 14, 20, 21, 30, 31],
        ):
            result = run_case(
                "sum_test", sum, sum,
                args=([1, 2, 3],),
                warmup=0, repeats=2,
            )

        self.assertEqual(result["case_id"], "sum_test")
        self.assertTrue(result["equivalent"])
        self.assertEqual(result["baseline"]["times_seconds"], [2, 4])
        self.assertEqual(result["candidate"]["times_seconds"], [1, 1])
        self.assertEqual(result["speedup"], 3.0)

    def test_mismatch_stops_timing(self):
        # sum([1, 2]) is 3, but len([1, 2]) is 2.
        with patch("benchmark.run_benchmarks.benchmark") as timer:
            with self.assertRaises(AssertionError):
                run_case(
                    "wrong_result", sum, len,
                    args=([1, 2],),
                    warmup=0, repeats=1,
                )

            timer.assert_not_called()

    def test_zero_time(self):
        with patch(
            "numcompute.benchmarking.time.perf_counter",
            side_effect=[0, 2, 10, 10],
        ):
            result = run_case(
                "zero_time", sum, sum,
                args=([1, 2],),
                warmup=0, repeats=1,
            )

        self.assertIsNone(result["speedup"])

"""
check the shape and type of data, 
Reproducibility, 
no impact on global random state, 
parameter validation
"""
class TestPrepareInputs(unittest.TestCase):
    def test_shapes_and_values(self):
        data = prepare_inputs(n=20, seed=42)

        self.assertEqual(data["values"].shape, (20,))
        self.assertEqual(data["features"].shape, (20, 10))
        self.assertEqual(data["y_true"].shape, (20,))
        self.assertEqual(data["y_pred"].shape, (20,))

        for array in data.values():
            self.assertEqual(array.dtype, np.dtype("float64"))
            self.assertTrue(np.all(np.isfinite(array)))

        self.assertTrue(np.all(data["features"][:, 0] == 7.0))

    def test_reproducible_data(self):
        # The same seed must reproduce every input array.
        first = prepare_inputs(n=20, seed=42)
        second = prepare_inputs(n=20, seed=42)

        for name in first:
            np.testing.assert_array_equal(first[name], second[name])

    def test_global_random_state_unchanged(self):
        before = np.random.get_state()
        prepare_inputs(n=20, seed=42)
        after = np.random.get_state()

        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            prepare_inputs(n=9)

        with self.assertRaises(TypeError):
            prepare_inputs(n=True)

        with self.assertRaises(ValueError):
            prepare_inputs(seed=-1)

        with self.assertRaises(TypeError):
            prepare_inputs(seed=1.5)

if __name__ == "__main__":
    unittest.main()