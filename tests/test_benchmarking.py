import unittest
from unittest.mock import Mock, call, patch

from numcompute.benchmarking import benchmark

"""
first, we record:
Timing results, 
parameter checks
exception propagation.
Use the mock clock, the durations—3, 1, and 2 seconds—are fixed values ​​used to verify the timing logic; 
they do not represent actual performance results. 
It can also be confirmed that the function underwent a total of 2 warm-up runs followed by 3 formal measurement runs, with the parameters passed in correctly.
"""
class TestBenchmark(unittest.TestCase):
    def test_timing_and_calls(self):
        func = Mock()

        with patch(
            "numcompute.benchmarking.time.perf_counter",
            side_effect=[0, 3, 10, 11, 20, 22],
        ):
            result = benchmark(
                func, 5, offset=2, warmup=2, repeats=3
            )

        self.assertEqual(result["times_seconds"], [3, 1, 2])
        self.assertEqual(result["median_seconds"], 2)
        self.assertEqual(result["min_seconds"], 1)
        self.assertEqual(
            func.call_args_list, [call(5, offset=2)] * 5
        )

    def test_invalid_parameters(self):
        cases = [
            ({"warmup": -1}, ValueError),
            ({"repeats": 0}, ValueError),
            ({"warmup": True}, TypeError),
            ({"repeats": 1.5}, TypeError),
        ]
        for options, error in cases:
            with self.subTest(options=options):
                with self.assertRaises(error):
                    benchmark(lambda: None, **options)

        with self.assertRaises(TypeError):
            benchmark(123)

    def test_function_error_propagates(self):
        func = Mock(side_effect=RuntimeError("failed"))

        with self.assertRaisesRegex(RuntimeError, "failed"):
            benchmark(func, warmup=0)


if __name__ == "__main__":
    unittest.main()