from numcompute.utils import _as_float_array


def loop_mean(values):
    """
    Compute mean using a Python loop on finite 1D data.
    """
    data = _as_float_array(values, ndim=1, finite=True)
    
    if data.size == 0: # check input and create an independent float
        raise ValueError("values must not be empty.")

    total = 0.0
    for value in data: # commulative summation
        total += float(value)

    return total / data.size # gain mean by deviation of quantity