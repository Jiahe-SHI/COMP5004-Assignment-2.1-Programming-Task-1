import unittest
from unittest.mock import patch
from benchmark.run_benchmarks import check_equivalent, run_case

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

if __name__ == "__main__":
    unittest.main()