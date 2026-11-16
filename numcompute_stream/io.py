"""Read numeric features and classification labels from CSV chunks."""

import csv
import re
from numbers import Integral

import numpy as np


def _string_sequence(values, name):
    """Validate an iterable of strings without treating one string as a list."""
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of strings")
    try:
        result = list(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an iterable of strings") from exc
    if any(not isinstance(value, str) for value in result):
        raise TypeError(f"{name} must contain only strings")
    return result


def iter_csv_chunks(
    path,
    *,
    target_column,
    chunk_size=100,
    feature_columns=None,
    label_dtype="str",
    missing_values=("", "NA", "NaN"),
    encoding="utf-8-sig",
):
    """Yield independent (X, y) arrays from a CSV file with a header.

    X has shape (n_rows, n_features) and dtype float64. Selected
    features must be numeric; configured missing markers become NaN.
    Other non-finite values are rejected.

    y has shape (n_rows,). String labels use a NumPy string dtype.
    Integer labels use object dtype to preserve Python integer precision.

    Header names are matched exactly. Data fields and missing markers
    are stripped of surrounding whitespace; matching is case-sensitive.
    Integer labels must match [+-]?[0-9]+. Labels cannot be missing.

    Rows are read incrementally. A final short chunk is yielded.
    A header-only file yields nothing; an empty file is invalid.
    Blank data rows and rows with the wrong field count are invalid.

    Validation and file opening occur when iteration begins. If a later
    row is invalid, previously yielded chunks remain valid; the current
    incomplete chunk is not yielded. Exhaust or close the generator to
    release its file handle.
    """
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("target_column must be a non-empty column name")
    if isinstance(chunk_size, (bool, np.bool_)) or not isinstance(
        chunk_size, Integral):
        raise TypeError("chunk_size must be a positive integer")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)

    if not isinstance(label_dtype, str) or label_dtype not in ("str", "int"):
        raise ValueError("label_dtype must be 'str' or 'int'")

    markers = {
        value.strip()
        for value in _string_sequence(missing_values, "missing_values")
    }

    selected = None
    if feature_columns is not None:
        selected = _string_sequence(feature_columns, "feature_columns")
        if not selected or any(not name.strip() for name in selected):
            raise ValueError("feature_columns must contain non-empty names")
        if len(set(selected)) != len(selected):
            raise ValueError("feature_columns must not contain duplicates")
        if target_column in selected:
            raise ValueError("target_column must not be included in features")

    def make_chunk(rows, labels):
        X = np.array(rows, dtype=np.float64)
        dtype = object if label_dtype == "int" else str
        y = np.array(labels, dtype=dtype)
        return X, y

    with open(path, "r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader, None)
            if header is None:
                raise ValueError("CSV file is empty: expected a header")
            if not header or any(not name.strip() for name in header):
                raise ValueError("CSV header must contain non-empty names")
            if len(set(header)) != len(header):
                raise ValueError("CSV header contains duplicate column names")
            if target_column not in header:
                raise ValueError(f"Target column {target_column!r} is not in the header")

            if selected is None:
                selected = [name for name in header if name != target_column]
            if not selected:
                raise ValueError("At least one feature column is required")

            unknown = [name for name in selected if name not in header]
            if unknown:
                raise ValueError(f"Unknown feature columns: {unknown!r}")

            feature_indices = [header.index(name) for name in selected]
            target_index = header.index(target_column)
            rows = []
            labels = []

            for record in reader:
                line = reader.line_num
                if len(record) != len(header):
                    raise ValueError(
                        f"CSV line {line}: expected {len(header)} fields, "
                        f"got {len(record)}"
                    )

                features = []
                for name, index in zip(selected, feature_indices):
                    text = record[index].strip()
                    if text in markers:
                        features.append(np.nan)
                        continue
                    try:
                        value = float(text)
                    except ValueError as exc:
                        raise ValueError(
                            f"CSV line {line}, column {name!r}: "
                            f"expected a numeric value, got {text!r}"
                        ) from exc
                    if not np.isfinite(value):
                        raise ValueError(
                            f"CSV line {line}, column {name!r}: "
                            "expected a finite value or a configured "
                            "missing marker"
                        )
                    features.append(value)

                text = record[target_index].strip()
                if not text or text in markers:
                    raise ValueError(
                        f"CSV line {line}, column {target_column!r}: "
                        "missing labels are not allowed"
                    )

                if label_dtype == "int":
                    if re.fullmatch(r"[+-]?[0-9]+", text) is None:
                        raise ValueError(
                            f"CSV line {line}, column {target_column!r}: "
                            f"expected an integer label, got {text!r}"
                        )
                    try:
                        label = int(text)
                    except ValueError as exc:
                        raise ValueError(
                            f"CSV line {line}, column {target_column!r}: "
                            "integer label could not be parsed"
                        ) from exc
                else:
                    label = text

                rows.append(features)
                labels.append(label)

                if len(rows) == chunk_size:
                    chunk = make_chunk(rows, labels)
                    rows = []
                    labels = []
                    yield chunk

            if rows:
                yield make_chunk(rows, labels)

        except csv.Error as exc:
            raise ValueError(
                f"CSV parsing error near line {reader.line_num}: {exc}"
            ) from exc