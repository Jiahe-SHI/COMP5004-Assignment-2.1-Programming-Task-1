"""Contract tests using isolated temporary CSV fixtures."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose
from numcompute.io import load_csv, iter_csv_chunks


class TestCSV(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'input.csv'

    def write(self, text):
        self.path.write_text(text, encoding='utf-8')
        return self.path

    def test_comma_header_dtype(self):
        out = load_csv(self.write('a,b\n1,2\n3,4\n'), skip_header=1, dtype=np.float32)
        assert_array_equal(out, [[1, 2], [3, 4]])
        self.assertEqual(out.dtype, np.dtype('float32'))

    def test_tab(self):
        assert_array_equal(load_csv(self.write('1\t2\n3\t4\n'), delimiter='\t'), [[1, 2], [3, 4]])

    def test_shapes(self):
        for text, shape in [('1,2\n', (1, 2)), ('1\n2\n', (2, 1)), ('1\n', (1, 1))]:
            with self.subTest(shape=shape):
                self.assertEqual(load_csv(self.write(text)).shape, shape)

    def test_physical_header_blank(self):
        assert_array_equal(load_csv(self.write('\na,b\n\n1,2\n \n'), skip_header=2), [[1, 2]])

    def test_selected_order(self):
        assert_array_equal(load_csv(self.write('1,NA,3\n4,,6\n'), usecols=[2, 0], missing='skip'), [[3, 1], [6, 4]])

    def test_float_missing(self):
        assert_allclose(load_csv(self.write(' 1 , NA \n,4\n')), [[1, np.nan], [np.nan, 4]], equal_nan=True)

    def test_custom_fill(self):
        assert_array_equal(load_csv(self.write('1, ? \n'), missing_values=['?'], fill_value=-1), [[1, -1]])

    def test_skip(self):
        assert_array_equal(load_csv(self.write('1,2\n3,\n4,5\n'), missing='skip'), [[1, 2], [4, 5]])

    def test_integer_missing(self):
        self.write('1,\n')
        with self.assertRaises(ValueError):
            load_csv(self.path, dtype=int)
        assert_array_equal(load_csv(self.path, dtype=int, fill_value=-1), [[1, -1]])

    def test_integer_precision(self):
        for dtype, text, expected in [(np.int64, '9007199254740993', 9007199254740993),
                                      (np.uint64, '18446744073709551615', 18446744073709551615),
                                      (np.int8, '-128', -128), (np.int16, '1e2', 100)]:
            with self.subTest(dtype=dtype):
                self.assertEqual(int(load_csv(self.write(text), dtype=dtype)[0, 0]), expected)

    def test_invalid_integer(self):
        for text in ['1.5', '128', '-129', 'nan', 'inf', 'word']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load_csv(self.write(text), dtype=np.int8, missing_values=())
        for fill in [1.5, 128, np.nan]:
            with self.subTest(fill=fill), self.assertRaises(ValueError):
                load_csv(self.write('1,\n'), dtype=np.int8, fill_value=fill)

    #"阿德莱德" is the Chinese word for "Adelaide", and "城市" is the Chinese word for "city". 
    #It is used to test whether UTF-8 can be correctly read and saved with test data
    def test_utf8_strings(self):
        assert_array_equal(load_csv(self.write('阿德莱德,\nSydney,NA\n'), dtype=str), [['阿德莱德', ''], ['Sydney', '']])
        self.assertEqual(load_csv(self.write('城市\n'), dtype='S6')[0, 0], '城市'.encode('utf-8'))

    def test_truncation(self):
        for dtype in ['U2', 'S2']:
            with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                load_csv(self.write('long\n'), dtype=dtype)
        with self.assertRaises(ValueError):
            load_csv(self.write('NA\n'), dtype='U2', fill_value='long')

    def test_hash_data(self):
        assert_array_equal(load_csv(self.write('#tag,x\n'), dtype=str), [['#tag', 'x']])

    def test_empty(self):
        for text, opts in [('', {}), ('\n \n', {}), ('a,b\n', {'skip_header': 1}), ('NA,\n', {'missing': 'skip'})]:
            with self.subTest(text=text):
                self.write(text)
                with self.assertRaises(ValueError):
                    load_csv(self.path, **opts)
                with self.assertRaises(ValueError):
                    list(iter_csv_chunks(self.path, **opts))

    def test_bad_width(self):
        with self.assertRaises(ValueError):
            load_csv(self.write('1,2\n3,4,5\n'), usecols=[0])

    def test_quotes(self):
        with self.assertRaises(ValueError):
            load_csv(self.write('1,"two"\n'), usecols=[0])

    def test_columns(self):
        self.write('1,2\n')
        for cols in [[], [0, 0], [-1], [2]]:
            with self.subTest(cols=cols), self.assertRaises(ValueError):
                load_csv(self.path, usecols=cols)
        for cols in [[True], [0.5], '0', 0]:
            with self.subTest(cols=cols), self.assertRaises(TypeError):
                load_csv(self.path, usecols=cols)

    def test_options(self):
        self.write('1,2\n')
        for opts in [{'delimiter': '::'}, {'delimiter': '\n'}, {'missing': 'omit'}, {'skip_header': -1}]:
            with self.subTest(opts=opts), self.assertRaises(ValueError):
                load_csv(self.path, **opts)
        for opts in [{'delimiter': 1}, {'skip_header': True}, {'dtype': object}, {'dtype': complex},
                     {'dtype': bool}, {'dtype': None}, {'missing_values': 'NA'}]:
            with self.subTest(opts=opts), self.assertRaises(TypeError):
                load_csv(self.path, **opts)

    def test_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            load_csv(self.path)

    def test_invalid_numeric(self):
        with self.assertRaises(ValueError):
            load_csv(self.write('1,hello\n'))

    def test_chunks_equivalence(self):
        self.write('a,b\n1,2\n3,NA\n4,5\n6,7\n8,9\n')
        for missing in ['skip', 'fill']:
            with self.subTest(missing=missing):
                chunks = list(iter_csv_chunks(self.path, chunk_size=2, skip_header=1, missing=missing))
                self.assertEqual([len(c) for c in chunks], [2, 2] if missing == 'skip' else [2, 2, 1])
                assert_allclose(np.concatenate(chunks), load_csv(self.path, skip_header=1, missing=missing), equal_nan=True)

    def test_chunks_strings(self):
        self.write('a\nlonger\n城市\n')
        assert_array_equal(np.concatenate(list(iter_csv_chunks(self.path, chunk_size=1, dtype=str))), load_csv(self.path, dtype=str))

    def test_chunk_size(self):
        for value, error in [(0, ValueError), (-1, ValueError), (True, TypeError), (1.5, TypeError)]:
            with self.subTest(value=value), self.assertRaises(error):
                list(iter_csv_chunks(self.path, chunk_size=value))

    def test_incremental_error(self):
        self.write('1,2\n3,4\n5,6,7\n')
        chunks = iter_csv_chunks(self.path, chunk_size=2)
        self.addCleanup(chunks.close)
        assert_array_equal(next(chunks), [[1, 2], [3, 4]])
        with self.assertRaises(ValueError):
            next(chunks)

    def test_independent_chunks(self):
        chunks = list(iter_csv_chunks(self.write('1\n2\n'), chunk_size=1))
        chunks[0][0, 0] = 99
        assert_array_equal(chunks[1], [[2]])
        assert_array_equal(load_csv(self.path), [[1], [2]])


if __name__ == '__main__':
    unittest.main()
