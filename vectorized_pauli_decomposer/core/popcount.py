import numpy as np

from ..custom_popcount.numpy_popcount import numpy_int64popcount, numpy_int64popcount_


M1 = 0x5555555555555555
M2 = 0x3333333333333333
M4 = 0x0f0f0f0f0f0f0f0f
H01 = 0x0101010101010101


def slow_popcount(x):
    x -= np.right_shift(x, 1) & M1
    x = np.bitwise_and(M2, x) + np.bitwise_and(M2, np.right_shift(x, 2))
    x = np.bitwise_and(M4, x + np.right_shift(x, 4))

    return np.right_shift(x * H01, 56)

    # return np.bitwise_count(x).astype(np.int64)


def popcount(x):
    assert (x.dtype == np.int64) or (x.dtype == np.uint64)
    return numpy_int64popcount(x)


def popcount_(x):
    assert (x.dtype == np.int64) or (x.dtype == np.uint64)
    return numpy_int64popcount_(x)
