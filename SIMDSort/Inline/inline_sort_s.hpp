#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"

__forceinline __m128 _mm_evensort_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmp_ps(x, y, _CMP_GT_OQ), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

__forceinline __m128 _mm_oddsort_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmp_ps(x, y, _CMP_GT_OQ), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

__forceinline __m256 _mm256_evensort_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmp_ps(x, y, _CMP_GT_OQ), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

__forceinline __m256 _mm256_oddsort_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmp_ps(x, y, _CMP_GT_OQ), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

__forceinline __m128 _mm_sort_ps(__m128 x) {
    x = _mm_oddsort_ps(x);
    x = _mm_evensort_ps(x);
    x = _mm_oddsort_ps(x);

    return x;
}

__forceinline __m256 _mm256_sort_ps(__m256 x) {
    x = _mm256_oddsort_ps(x);
    x = _mm256_evensort_ps(x);
    x = _mm256_oddsort_ps(x);
    x = _mm256_evensort_ps(x);
    x = _mm256_oddsort_ps(x);
    x = _mm256_evensort_ps(x);
    x = _mm256_oddsort_ps(x);

    return x;
}