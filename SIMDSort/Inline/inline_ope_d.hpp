#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256d _mm256_isnan_pd(__m256d x) {
    __m256d y = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);

    return y;
}

__forceinline __m256d _mm256_not_pd(__m256d x) {
    const __m256d setbits = _mm256_castsi256_pd(_mm256_set1_epi32(~0u));

    __m256d y = _mm256_xor_pd(x, setbits);

    return y;
}