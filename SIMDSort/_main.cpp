﻿#include <stdio.h>
#include <intrin.h>
#include "simdsort.h"
#include "Inline/inline_misc.hpp"

// compare x > y (ignore nan)
__forceinline __m128 _mm_needswap_ps(__m128 x, __m128 y) {
    __m128 gt = _mm_cmp_ps(x, y, _CMP_GT_OQ);

    return gt;
}

// compare x > y (ignore nan)
__forceinline __m256 _mm256_needswap_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);

    return gt;
}

// sort batches4 x elems2 
__forceinline __m256 _mm256_sort4x2_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// sort batches2 x elems3
__forceinline __m256 _mm256_sort2x3_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 2, 4, 3, 5, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 3, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 3, 3, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 4, 4, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches2 x elems4
__forceinline __m256 _mm256_sort2x4_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, ~0u, ~0u, 0, 0, ~0u));

    __m256 y, c;

    y = _mm256_permute_ps(x, _MM_PERM_ABCD);
    c = _mm256_xor_ps(xormask, _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_ABCD);
    c = _mm256_xor_ps(xormask, _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches1 x elems5
__forceinline __m256 _mm256_sort1x5_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 3, 2, 4, 5, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 4, 3, 5, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 2, 4, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 3, 5, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches1 x elems6
__forceinline __m256 _mm256_sort1x6_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, ~0u, 0, 0));
    const __m256i perm0 = _mm256_setr_epi32(5, 2, 1, 4, 3, 0, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(5, 1, 1, 3, 3, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0));
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches1 x elems7
__forceinline __m256 _mm256_sort1x7_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 4, 3, 6, 5, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 3, 5, 5, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort
__forceinline __m128 _mm_sort_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y, c;

    y = _mm_permute_ps(x, _MM_PERM_ABCD);
    c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_needswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm_blendv_ps(x, y, c);

    y = _mm_permute_ps(x, _MM_PERM_CDAB);
    c = _mm_permute_ps(_mm_needswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm_blendv_ps(x, y, c);

    y = _mm_permute_ps(x, _MM_PERM_ABCD);
    c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_needswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm_blendv_ps(x, y, c);

    return x;
}

// sort
__forceinline __m256 _mm256_sort_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

int main() {
    __m256 x1 = _mm256_setr_ps(1, 2, 3, 4, 6, 5, 7, 8);
    __m256 x2 = _mm256_setr_ps(2, 3, 1, 6, 5, 4, 0, 7);
    __m256 x3 = _mm256_setr_ps(3, 1, 2, 5, 4, 6, 7, 8);

    __m256 y1 = _mm256_sort1x6_ps(x1);
    __m256 y2 = _mm256_sort1x6_ps(x2);
    __m256 y3 = _mm256_sort1x6_ps(x3);

    printf("end");
    return getchar();
}