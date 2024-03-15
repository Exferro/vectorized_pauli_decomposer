import os
import cffi
import inspect


ffi = cffi.FFI()
debug = False
use_openmp = True
directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
output_dir = os.path.join(directory, '_cffi_outputs')

with open('%s/cffi_int64popcount.h' % directory) as my_header:
    ffi.cdef(my_header.read())

with open('%s/cffi_int64popcount.c' % directory) as my_source:
    if debug:
        #print('Building the debug build...')
        ffi.set_source(
            '_cffi_popcount',
            my_source.read(),
            extra_compile_args=[ '-pedantic', '-Wall', '-g', '-O0'],
        )
    # -ffast-math assumes there are no nans or infs!
    # -O3 includes -ffast-math!
    # https://stackoverflow.com/questions/22931147/stdisinf-does-not-work-with-ffast-math-how-to-check-for-infinity
    else:
        if use_openmp:
            #print('Building for performance with OpenMP...')
            ffi.set_source(
                '_cffi_popcount',
                my_source.read(),
                extra_compile_args=['-fopenmp', '-D use_openmp', '-O3','-march=native'],
                extra_link_args=['-fopenmp'],
            )
        else:
            #print('Building for performance without OpenMP...')
            ffi.set_source('_cffi_popcount',
                my_source.read(),
                extra_compile_args=['-O3','-march=native'],
)

ffi.compile(tmpdir=output_dir)
#ffi.compile(verbose=True)

import sys
sys.path.append(output_dir)

from _cffi_popcount import *