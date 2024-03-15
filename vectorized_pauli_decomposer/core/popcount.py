import numpy as np

M1 = 0x5555555555555555
M2 = 0x3333333333333333
M4 = 0x0f0f0f0f0f0f0f0f
H01 = 0x0101010101010101


def popcount(x):
    x -= np.right_shift(x, 1) & M1
    x = np.bitwise_and(M2, x) + np.bitwise_and(M2, np.right_shift(x, 2))
    x = np.bitwise_and(M4, x + np.right_shift(x, 4))

    return np.right_shift(x * H01, 56)

    # return np.bitwise_count(x).astype(np.int64)
