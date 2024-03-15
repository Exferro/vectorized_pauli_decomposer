import numpy as np
from . import cffi_popcount


def numpy_int64popcount(a: np.ndarray) -> np.ndarray:
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    assert (a.dtype == np.int64)

    result = np.zeros_like(a)
    elem_num = a.size

    _elem_num = cffi_popcount.ffi.cast('int', elem_num)
    _a = cffi_popcount.ffi.cast('long long int*', a.ctypes.data)
    _result = cffi_popcount.ffi.cast('long long int*', result.ctypes.data)

    cffi_popcount.lib.cffi_int64popcount( _elem_num, _a, _result)

    return result

def numpy_int64popcount_(a: np.ndarray) -> np.ndarray:
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)
    assert (a.dtype == np.int64)

    elem_num = a.size

    _elem_num = cffi_popcount.ffi.cast('int', elem_num)
    _a = cffi_popcount.ffi.cast('long long int*', a.ctypes.data)

    cffi_popcount.lib.cffi_int64popcount_( _elem_num, _a)

    return a
