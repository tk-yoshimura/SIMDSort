#pragma once
#pragma unmanaged

#include "../utils.h"

__forceinline __m256 _mm256_abs_ps(const __m256 x) {
    const __m256 bitmask = _mm256_set1_ps(_m32(0x7FFFFFFFu).f);

    const __m256 ret = _mm256_and_ps(x, bitmask);

    return ret;
}

__forceinline __m256 _mm256_isnan_ps(__m256 x) {
    __m256 y = _mm256_cmp_ps(x, x, _CMP_NEQ_UQ);

    return y;
}

__forceinline __m256 _mm256_not_ps(__m256 x) {
    const __m256 setbits = _mm256_castsi256_ps(_mm256_set1_epi32(~0u));

    __m256 y = _mm256_xor_ps(x, setbits);

    return y;
}