import unittest
from unittest.mock import patch
from benchmark.run_benchmarks import (
    check_equivalent, run_case, prepare_inputs, run_experiments,
    save_raw_timings, save_summary,
)
import numpy as np
import csv
import tempfile
from pathlib import Path

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

class TestRunExperiments(unittest.TestCase):
    def test_all_cases_run(self):
        results = run_experiments(
            n=20, seed=42, warmup=0, repeats=1
        )

        names = []
        for result in results:
            names.append(result["case_id"])
            self.assertTrue(result["equivalent"])
            self.assertEqual(result["dtype"], "float64")
            self.assertEqual(
                len(result["baseline"]["times_seconds"]), 1
            )
            self.assertEqual(
                len(result["candidate"]["times_seconds"]), 1
            )

        self.assertEqual(names, [
            "mean_20",
            "standardization_20",
            "mse_20",
            "topk_fixed_20",
            "topk_fraction_20",
        ])

    def test_case_settings(self):
        results = run_experiments(
            n=20, seed=42, warmup=0, repeats=1
        )

        self.assertEqual(results[0]["shape"], [20])
        self.assertEqual(results[1]["shape"], [20, 10])
        self.assertIsNone(results[2]["k"])

        # Top-k uses both fixed and proportional selections.
        self.assertEqual(results[3]["k"], 10)
        self.assertEqual(results[4]["k"], 2)
        self.assertEqual(
            results[3]["comparison"], "full_sort_vs_argpartition"
        )
        self.assertEqual(
            results[0]["comparison"], "loop_vs_vectorized"
        )

    def test_invalid_parameters(self):
        with self.assertRaises(ValueError):
            run_experiments(n=9)

        with self.assertRaises(ValueError):
            run_experiments(seed=-1)

        with self.assertRaises(TypeError):
            run_experiments(warmup=True)

        with self.assertRaises(ValueError):
            run_experiments(repeats=0)

class TestSaveRawTimings(unittest.TestCase):
    def test_csv_content(self):
        # Fixed samples check file contents, not actual performance.
        results = [
            {
                "case_id": "mean_20",
                "baseline": {"times_seconds": [0.2, 0.3]},
                "candidate": {"times_seconds": [0.01, 0.02]},
            }
        ]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results" / "raw.csv"
            saved_path = save_raw_timings(results, path, "test_run")

            self.assertEqual(saved_path, path)

            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))

            self.assertEqual(rows, [
                [
                    "run_id", "case_id", "implementation",
                    "repeat_index", "seconds_per_call",
                ],
                ["test_run", "mean_20", "baseline", "1", "0.2"],
                ["test_run", "mean_20", "baseline", "2", "0.3"],
                ["test_run", "mean_20", "candidate", "1", "0.01"],
                ["test_run", "mean_20", "candidate", "2", "0.02"],
            ])

    def test_existing_file_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.csv"
            save_raw_timings([], path, "first_run")
            original = path.read_text(encoding="utf-8")

            with self.assertRaises(FileExistsError):
                save_raw_timings([], path, "second_run")

            self.assertEqual(
                path.read_text(encoding="utf-8"), original
            )

    def test_invalid_run_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "raw.csv"

            with self.assertRaises(TypeError):
                save_raw_timings([], path, 123)

            with self.assertRaises(ValueError):
                save_raw_timings([], path, "   ")

            self.assertFalse(path.exists())

class TestSaveSummary(unittest.TestCase):
    def test_csv_content(self):
        # Fixed numbers check export contents, not performance.
        results = [
            {
                "case_id": "mean_20",
                "task": "mean",
                "shape": [20],
                "dtype": "float64",
                "k": None,
                "comparison": "loop_vs_vectorized",
                "baseline_name": "loop_mean",
                "candidate_name": "mean",
                "baseline": {
                    "median_seconds": 2.0,
                    "min_seconds": 1.8,
                },
                "candidate": {
                    "median_seconds": 0.5,
                    "min_seconds": 0.4,
                },
                "speedup": 4.0,
                "equivalent": True,
            }
        ]

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results" / "summary.csv"
            saved_path = save_summary(results, path, "test_run")

            self.assertEqual(saved_path, path)

            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))

            self.assertEqual(rows, [
                [
                    "run_id", "case_id", "task", "shape", "dtype", "k",
                    "comparison", "baseline_name", "candidate_name",
                    "baseline_median_seconds", "candidate_median_seconds",
                    "baseline_min_seconds", "candidate_min_seconds",
                    "speedup", "equivalent",
                ],
                [
                    "test_run", "mean_20", "mean", "[20]", "float64", "",
                    "loop_vs_vectorized", "loop_mean", "mean",
                    "2.0", "0.5", "1.8", "0.4", "4.0", "True",
                ],
            ])

    def test_existing_file_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.csv"
            save_summary([], path, "first_run")
            original = path.read_text(encoding="utf-8")

            with self.assertRaises(FileExistsError):
                save_summary([], path, "second_run")

            self.assertEqual(
                path.read_text(encoding="utf-8"), original
            )

    def test_invalid_run_id(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summary.csv"

            with self.assertRaises(TypeError):
                save_summary([], path, 123)

            with self.assertRaises(ValueError):
                save_summary([], path, "   ")

            self.assertFalse(path.exists())

if __name__ == "__main__":
    unittest.main()