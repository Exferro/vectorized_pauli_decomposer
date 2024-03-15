#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include <inttypes.h>

void cffi_int64popcount(
    const int elem_num,
    const long long int* a,
    long long int* result
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < elem_num; elem_idx++) {

        result[elem_idx] = __builtin_popcountll(a[elem_idx]);
    }
}

void cffi_int64popcount_(
    const int elem_num,
    long long int* a
) {
    #pragma omp parallel for
    for (int elem_idx = 0; elem_idx < elem_num; elem_idx++) {

        a[elem_idx] = __builtin_popcountll(a[elem_idx]);
    }
}