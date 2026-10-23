import unittest

from benchmark.run_benchmarks import check_equivalent


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


if __name__ == "__main__":
    unittest.main()