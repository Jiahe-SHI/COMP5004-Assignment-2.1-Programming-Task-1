"""Timing utilities for performance comparisons."""

import time

from statistics import median

from .utils import _validate_integer

import platform
import numpy as np


def benchmark(func, *args, warmup=1, repeats=5, **kwargs):
    """Time repeated calls; return samples, median and minimum in seconds.

    warmup >= 0 and repeats >= 1 must be integers, excluding booleans.
    Warmup is untimed. Errors from func propagate unchanged.
    """

    if not callable(func): # check parameters, ensure it is a callable object
        raise TypeError("func must be callable.")
    """
    the times of warm up could be 0 but the formal test at least 1 times
    """
    warmup = _validate_integer(warmup, name="warmup", minimum=0)
    repeats = _validate_integer(repeats, name="repeats", minimum=1)

    for _ in range(warmup):
        func(*args, **kwargs)

    times = []
    for _ in range(repeats): # record start time > excute functions > compute time gap > preserve the result
        start = time.perf_counter()
        func(*args, **kwargs)
        times.append(time.perf_counter() - start)

    return {
        "times_seconds": times,
        "median_seconds": median(times),
        "min_seconds": min(times),
    }

def environment_info():
    """Return runtime versions, OS and CPU information."""
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.platform(),
        "cpu": platform.processor() or "unknown",
    }