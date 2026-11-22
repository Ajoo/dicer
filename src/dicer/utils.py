import numpy as np

def exclusive_cumsum(x):
    shape = x.shape
    shape = (*shape[:-1], shape[-1] + 1)
    y = np.zeros(shape, dtype=x.dtype)
    y[..., 1:] = np.cumsum(x, -1)
    return y