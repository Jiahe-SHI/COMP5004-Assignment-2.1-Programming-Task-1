"""Tests for incremental CSV parsing and schema validation."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from numcompute_stream.io import iter_csv_chunks


class TestCSVChunks(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.path = Path(self.temp_dir.name) / "input.csv"

    def write_csv(self, text):
        with self.path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(text)

    def read_chunks(self, **kwargs):
        options = {"target_column": "target", "label_dtype": "int"}
        options.update(kwargs)
        return list(iter_csv_chunks(self.path, **options))

    def test_chunk_sizes_tail_and_values(self):
        self.write_csv(
            "a,b,target\n"
            "1,10,0\n2,20,1\n3,30,0\n4,40,1\n5,50,1\n"
        )
        chunks = self.read_chunks(chunk_size=2)
        self.assertEqual([len(y) for _, y in chunks], [2, 2, 1])
        np.testing.assert_array_equal(
            np.concatenate([X for X, _ in chunks]),
            [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]],
        )
        np.testing.assert_array_equal(
            np.concatenate([y for _, y in chunks]), [0, 1, 0, 1, 1]
        )
        for X, y in chunks:
            self.assertEqual(X.dtype, np.dtype("float64"))
            self.assertEqual(y.ndim, 1)

    def test_exact_multiple_has_no_empty_tail(self):
        self.write_csv("a,target\n1,0\n2,1\n3,0\n4,1\n")
        chunks = self.read_chunks(chunk_size=2)
        self.assertEqual([len(y) for _, y in chunks], [2, 2])

    def test_feature_selection_preserves_requested_order(self):
        self.write_csv("a,target,b,city\n1,0,10,Adelaide\n2,1,20,Perth\n")
        X, y = self.read_chunks(feature_columns=["b", "a"])[0]
        np.testing.assert_array_equal(X, [[10, 1], [20, 2]])
        np.testing.assert_array_equal(y, [0, 1])

    def test_default_features_exclude_target_in_middle(self):
        self.write_csv("a,target,b\n1,0,10\n")
        X, y = self.read_chunks()[0]
        np.testing.assert_array_equal(X, [[1, 10]])
        np.testing.assert_array_equal(y, [0])

    def test_bom_quoted_fields_and_string_labels(self):
        self.write_csv(
            '\ufeffa,target\r\n'
            '"1.5","class, one"\r\n'
            '"2.5","class ""two"""\r\n'
        )
        X, y = self.read_chunks(label_dtype="str")[0]
        np.testing.assert_array_equal(X, [[1.5], [2.5]])
        self.assertEqual(y.tolist(), ["class, one", 'class "two"'])

    def test_missing_markers_and_whitespace(self):
        self.write_csv(
            "a,b,target\n"
            " , NA , 0 \n"
            "NaN, 2.5 ,1\n"
        )
        X, y = self.read_chunks()[0]
        np.testing.assert_array_equal(
            np.isnan(X), [[True, True], [True, False]]
        )
        self.assertEqual(X[1, 1], 2.5)
        np.testing.assert_array_equal(y, [0, 1])

    def test_custom_missing_markers_are_case_sensitive(self):
        self.write_csv("a,target\n ?,0\n")
        X, _ = self.read_chunks(missing_values=[" ? "])[0]
        self.assertTrue(np.isnan(X[0, 0]))

        self.write_csv("a,target\nna,0\n")
        with self.assertRaisesRegex(ValueError, "line 2.*column 'a'"):
            self.read_chunks(missing_values=["NA"])

    def test_large_integer_labels_preserve_precision(self):
        large = 2**64 + 1
        self.write_csv(f"a,target\n1,{large}\n2,-{large}\n")
        _, y = self.read_chunks()[0]
        self.assertEqual(y.dtype, np.dtype("object"))
        self.assertEqual(y.tolist(), [large, -large])

    def test_header_only_and_empty_file(self):
        self.write_csv("a,target\n")
        self.assertEqual(self.read_chunks(), [])

        self.write_csv("")
        with self.assertRaisesRegex(ValueError, "empty"):
            self.read_chunks()

    def test_invalid_headers(self):
        cases = [
            ("a,a,target\n", "duplicate"),
            ("a,,target\n", "non-empty"),
            ("a,b\n", "Target column"),
            ("target\n", "At least one feature"),
        ]
        for text, message in cases:
            with self.subTest(text=text):
                self.write_csv(text)
                with self.assertRaisesRegex(ValueError, message):
                    self.read_chunks()

    def test_invalid_feature_selection(self):
        self.write_csv("a,b,target\n1,2,0\n")
        cases = [
            ([], ValueError),
            (["a", "a"], ValueError),
            (["target"], ValueError),
            (["unknown"], ValueError),
            ([""], ValueError),
            ("a", TypeError),
            ([1], TypeError),
        ]
        for columns, error in cases:
            with self.subTest(columns=columns):
                with self.assertRaises(error):
                    self.read_chunks(feature_columns=columns)

    def test_invalid_options(self):
        self.write_csv("a,target\n1,0\n")
        cases = [
            ({"chunk_size": 0}, ValueError),
            ({"chunk_size": -1}, ValueError),
            ({"chunk_size": True}, TypeError),
            ({"chunk_size": np.bool_(True)}, TypeError),
            ({"chunk_size": 1.5}, TypeError),
            ({"target_column": ""}, ValueError),
            ({"target_column": None}, ValueError),
            ({"label_dtype": "float"}, ValueError),
            ({"missing_values": "NA"}, TypeError),
            ({"missing_values": [None]}, TypeError),
        ]
        for options, error in cases:
            with self.subTest(options=options):
                with self.assertRaises(error):
                    self.read_chunks(**options)

        chunks = self.read_chunks(chunk_size=np.int64(1))
        self.assertEqual(len(chunks), 1)

    def test_invalid_features_include_location(self):
        for value in ("hello", "inf", "-inf", "1e999", "nan"):
            with self.subTest(value=value):
                self.write_csv(f"a,target\n{value},0\n")
                with self.assertRaisesRegex(
                    ValueError, "line 2.*column 'a'"
                ):
                    self.read_chunks()

    def test_missing_labels_rejected_for_both_types(self):
        for dtype in ("str", "int"):
            for label in ("", " ", "NA", "NaN"):
                with self.subTest(dtype=dtype, label=label):
                    self.write_csv(f"a,target\n1,{label}\n")
                    with self.assertRaisesRegex(
                        ValueError, "line 2.*missing labels"
                    ):
                        self.read_chunks(label_dtype=dtype)

    def test_integer_label_syntax(self):
        self.write_csv("a,target\n1,+12\n2,-3\n")
        _, y = self.read_chunks()[0]
        self.assertEqual(y.tolist(), [12, -3])

        for label in ("1.0", "1e2", "True", "1_000"):
            with self.subTest(label=label):
                self.write_csv(f"a,target\n1,{label}\n")
                with self.assertRaisesRegex(
                    ValueError, "line 2.*integer label"
                ):
                    self.read_chunks()

    def test_wrong_field_count_and_blank_row(self):
        for record in ("1", "1,0,extra", ""):
            with self.subTest(record=record):
                self.write_csv(f"a,target\n{record}\n")
                with self.assertRaisesRegex(
                    ValueError, "line 2.*expected 2 fields"
                ):
                    self.read_chunks()

    def test_malformed_quoting_reports_csv_error(self):
        self.write_csv('a,target\n"1,0\n')
        with self.assertRaisesRegex(ValueError, "CSV parsing error"):
            self.read_chunks()

    def test_later_error_does_not_invalidate_yielded_chunk(self):
        self.write_csv("a,target\n1,0\n2,1\n3,0\nbad,1\n")
        stream = iter_csv_chunks(
            self.path,
            target_column="target",
            label_dtype="int",
            chunk_size=2,
        )
        try:
            X, y = next(stream)
            np.testing.assert_array_equal(X, [[1], [2]])
            np.testing.assert_array_equal(y, [0, 1])

            with self.assertRaisesRegex(ValueError, "line 5"):
                next(stream)

            np.testing.assert_array_equal(X, [[1], [2]])
            np.testing.assert_array_equal(y, [0, 1])
            with self.assertRaises(StopIteration):
                next(stream)
        finally:
            stream.close()

    def test_chunks_have_independent_arrays(self):
        self.write_csv("a,target\n1,0\n2,1\n3,0\n4,1\n")
        first, second = self.read_chunks(chunk_size=2)
        self.assertFalse(np.shares_memory(first[0], second[0]))
        self.assertFalse(np.shares_memory(first[1], second[1]))

        first[0][:] = 99
        first[1][:] = 99
        np.testing.assert_array_equal(second[0], [[3], [4]])
        np.testing.assert_array_equal(second[1], [0, 1])


if __name__ == "__main__":
    unittest.main()