/**
 * Bitwise hamming distance.
 *
 * @param n number of elements
 * @param a first tensor
 * @param b second tensor
 * @param dist distance tensor
 */
void cffi_int64popcount(
    const int n,
    const long long int* a,
    long long int* dist
);

void cffi_int64popcount_(
    const int n,
    long long int* a
);