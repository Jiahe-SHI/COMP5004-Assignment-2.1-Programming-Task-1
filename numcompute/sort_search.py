import numpy as np

from .utils import _as_float_array, _validate_integer


def _as_numeric_array(values, *, name="values", ndim=1):
    try:
        array = np.asarray(values)
    except ValueError as exc:
        raise ValueError(f"{name} must have a regular rectangular shape.") from exc
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric values, not boolean, "
                        "complex, string or object data.")
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D; got {array.ndim}D.")
    if isinstance(values, (list, tuple)):
        _as_float_array(values, name=name, ndim=ndim)
    return array


def _validate_bool(value, *, name):
    if not isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a boolean.")


def _reject_nan(array, *, name="values"):
    if np.any(np.isnan(array)):
        raise ValueError(f"{name} must not contain NaN.")


def _stable_order(values, descending=False):
    if descending:
        return values.size - 1 - np.argsort(values[::-1], kind="stable")[::-1]
    return np.argsort(values, kind="stable")


def stable_sort(values, descending=False, return_indices=False):
    array = _as_numeric_array(values)
    _validate_bool(descending, name="descending")
    _validate_bool(return_indices, name="return_indices")
    _reject_nan(array)
    if not descending and not return_indices:
        return np.sort(array, kind="stable")
    indices = _stable_order(array, descending)
    result = array[indices]
    return (result, indices) if return_indices else result


def multi_key_sort(X, columns, return_indices=False):
    array = _as_numeric_array(X, name="X", ndim=2)
    _validate_bool(return_indices, name="return_indices")
    _reject_nan(array, name="X")
    keys = np.asarray(columns)
    if keys.ndim != 1 or keys.size == 0:
        raise ValueError("columns must be a non-empty 1D sequence.")
    if keys.dtype.kind not in "iu":
        raise TypeError("columns must contain integers, not booleans.")
    if isinstance(columns, (list, tuple)):
        for column in columns:
            _validate_integer(column, name="column", minimum=0)
    if np.any(keys < 0) or np.any(keys >= array.shape[1]):
        raise ValueError("columns must contain valid non-negative column indices.")
    if np.unique(keys).size != keys.size:
        raise ValueError("columns must not contain repeated indices.")
    indices = np.lexsort(array[:, keys[::-1]].T)
    result = array[indices]
    return (result, indices) if return_indices else result


def topk(values, k, largest=True, return_indices=True):
    array = _as_numeric_array(values)
    k = _validate_integer(k, name="k", minimum=0)
    _validate_bool(largest, name="largest")
    _validate_bool(return_indices, name="return_indices")
    _reject_nan(array)
    if k > array.size:
        raise ValueError("k must not exceed the number of values.")
    if k == 0:
        indices = np.empty(0, dtype=np.intp)
    else:
        position = array.size - k if largest else k - 1
        partition = np.argpartition(array, position)
        cutoff = array[partition[position]]
        better = np.flatnonzero(array > cutoff if largest else array < cutoff)
        tied = np.flatnonzero(array == cutoff)[:k - better.size]
        candidates = np.concatenate((better, tied))
        indices = candidates[_stable_order(array[candidates], largest)]
    result = array[indices]
    return (result, indices) if return_indices else result


def quickselect(values, k):
    array = _as_numeric_array(values)
    k = _validate_integer(k, name="k", minimum=0)
    _reject_nan(array)
    if k >= array.size:
        raise ValueError("k must satisfy 0 <= k < the number of values.")
    work = array.copy()
    while work.size > 1:
        first, middle, last = work[0], work[work.size // 2], work[-1]
        pivot = max(min(first, middle), min(max(first, middle), last))
        smaller = work < pivot
        larger = work > pivot
        left_count = np.count_nonzero(smaller)
        equal_end = work.size - np.count_nonzero(larger)
        if k < left_count:
            work = work[smaller]
        elif k < equal_end:
            return pivot
        else:
            k -= equal_end
            work = work[larger]
    return work[0]


def binary_search(values, target):
    array = _as_numeric_array(values)
    needle = _as_numeric_array(target, name="target", ndim=0).item()
    if np.isnan(needle):
        raise ValueError("target must not be NaN.")
    left, right = 0, array.size
    while left < right:
        middle = (left + right) // 2
        if array[middle].item() < needle:
            left = middle + 1
        else:
            right = middle
    return left, bool(left < array.size and array[left].item() == needle)
