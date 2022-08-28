#include "sortwithkey.h"
#include "../Inline/inline_cmp_s.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_epi32.hpp"
#include "../Inline/inline_blend_epi.hpp"

#pragma region needs swap

// needs swap (sort order definition)
__forceinline static __m256 _mm256_needsswap_ps(__m256 x, __m256 y) {
    return _mm256_cmplt_maxnan_ps(x, y);
}

#pragma endregion needs swap

#pragma region needs sort

// needs sort
__forceinline static bool _mm256_needssort_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_needsswap_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion needs sort

#pragma region horizontal sort

// sort batches4 x elems2 
__forceinline static __m256kv _mm256_sort4x2_ps(__m256kv x) {
    __m256 yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    __m256i yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    __m256 zk = _mm256_blendv_ps(x.k, yk, c);
    __m256i zv = _mm256_blendv_epi32(x.v, yv, c);

    return __m256kv(zk, zv);
}

// sort batches2 x elems3
__forceinline static __m256kv _mm256_sort2x3_ps(__m256kv x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 2, 4, 3, 5, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 3, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 3, 3, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 4, 4, 6, 7);

    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

// sort batches2 x elems4
__forceinline static __m256kv _mm256_sort2x4_ps(__m256kv x) {
    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_ABCD);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_ABCD);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_ABBA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

// sort batches1 x elems5
__forceinline static __m256kv _mm256_sort1x5_ps(__m256kv x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 3, 2, 4, 5, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 4, 3, 5, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 2, 4, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 3, 5, 6, 7);

    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

// sort batches1 x elems6
__forceinline static __m256kv _mm256_sort1x6_ps(__m256kv x) {
    const __m256i perm0 = _mm256_setr_epi32(5, 2, 1, 4, 3, 0, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 1, 1, 3, 3, 0, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);

    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

// sort batches1 x elems7
__forceinline static __m256kv _mm256_sort1x7_ps(__m256kv x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 4, 3, 6, 5, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 3, 5, 5, 7);

    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm1);
    yv = _mm256_permutevar8x32_epi32(x.v, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp1);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm0);
    yv = _mm256_permutevar8x32_epi32(x.v, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp0);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

// sort elems8
__forceinline static __m256kv _mm256_sort_ps(__m256kv x) {
    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i permcmp = _mm256_setr_epi32(0, 1, 2, 3, 3, 2, 1, 0);

    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_ABCD);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_ABCD);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_ABBA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permutevar8x32_ps(x.k, perm);
    yv = _mm256_permutevar8x32_epi32(x.v, perm);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_BADC);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_BADC);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_BABA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

// sort elems8 (ho, lo sorted)
__forceinline static __m256kv _mm256_halfsort_ps(__m256kv x) {
    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i permcmp = _mm256_setr_epi32(0, 1, 2, 3, 3, 2, 1, 0);

    __m256 yk, c;
    __m256i yv;

    yk = _mm256_permutevar8x32_ps(x.k, perm);
    yv = _mm256_permutevar8x32_epi32(x.v, perm);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x.k, yk), permcmp);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_BADC);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_BADC);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_BABA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    yk = _mm256_permute_ps(x.k, _MM_PERM_CDAB);
    yv = _mm256_shuffle_epi32(x.v, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_ps(x.k, yk, c);
    x.v = _mm256_blendv_epi32(x.v, yv, c);

    return x;
}

#pragma endregion horizontal sort

#pragma region cmp and swap

// compare and swap
__forceinline static void _mm256_cmpswap_ps(__m256kv a, __m256kv b, __m256kv& x, __m256kv& y) {
    __m256 swaps = _mm256_needsswap_ps(a.k, b.k), notswaps = _mm256_not_ps(swaps);

    x.k = _mm256_blendv_ps(a.k, b.k, swaps);
    y.k = _mm256_blendv_ps(a.k, b.k, notswaps);
    x.v = _mm256_blendv_epi32(a.v, b.v, swaps);
    y.v = _mm256_blendv_epi32(a.v, b.v, notswaps);
}

// compare and swap
__forceinline static uint _mm256_cmpswap_indexed_ps(__m256kv a, __m256kv b, __m256kv& x, __m256kv& y) {
    __m256 swaps = _mm256_needsswap_ps(a.k, b.k), notswaps = _mm256_not_ps(swaps);

    uint index = _mm256_movemask_ps(swaps);

    x.k = _mm256_blendv_ps(a.k, b.k, swaps);
    y.k = _mm256_blendv_ps(a.k, b.k, notswaps);
    x.v = _mm256_blendv_epi32(a.v, b.v, swaps);
    y.v = _mm256_blendv_epi32(a.v, b.v, notswaps);

    return index;
}

// compare and swap with permutate
__forceinline static void _mm256_cmpswap_withperm_ps(__m256kv a, __m256kv b, __m256kv& x, __m256kv& y) {

    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_permute_ps(x.k, _MM_PERM_CBAD);
    a.v = _mm256_shuffle_epi32(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_permute_ps(x.k, _MM_PERM_CBAD);
    a.v = _mm256_shuffle_epi32(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_permute_ps(x.k, _MM_PERM_CBAD);
    a.v = _mm256_shuffle_epi32(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x.k), _MM_PERM_BADC));
    a.v = _mm256_castpd_si256(_mm256_permute4x64_pd(_mm256_castsi256_pd(x.v), _MM_PERM_BADC));
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_permute_ps(x.k, _MM_PERM_CBAD);
    a.v = _mm256_shuffle_epi32(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_permute_ps(x.k, _MM_PERM_CBAD);
    a.v = _mm256_shuffle_epi32(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a.k = _mm256_permute_ps(x.k, _MM_PERM_CBAD);
    a.v = _mm256_shuffle_epi32(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);
}

#pragma endregion cmp and swap

#pragma region combsort

// combsort h=9...15
static int combsort_h9to15_s(const uint n, const uint h, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_FLOAT_STRIDE || h >= AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_ps(h & AVX2_FLOAT_REMAIN_MASK);

    uint e = n - h * 2;

    __m256kv a0, a1, b0, b1;
    __m256kv x0, x1, y0, y1;

    if (e > 0) {
        _mm256_maskload_x2_ps(k_ptr, a0.k, a1.k, mask);
        _mm256_maskload_x2_epi32(v_ptr, a0.v, a1.v, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x2_ps(k_ptr + i + h, b0.k, b1.k, mask);
            _mm256_maskload_x2_epi32(v_ptr + i + h, b0.v, b1.v, mask);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);

            _mm256_maskstore_x2_ps(k_ptr + i, x0.k, x1.k, mask);
            _mm256_maskstore_x2_epi32(v_ptr + i, x0.v, x1.v, mask);

            a0 = y0;
            a1 = y1;
        }
        _mm256_maskstore_x2_ps(k_ptr + i, a0.k, a1.k, mask);
        _mm256_maskstore_x2_epi32(v_ptr + i, a0.v, a1.v, mask);
    }
    {
        _mm256_maskload_x2_ps(k_ptr + e, a0.k, a1.k, mask);
        _mm256_maskload_x2_epi32(v_ptr + e, a0.v, a1.v, mask);
        _mm256_maskload_x2_ps(k_ptr + e + h, b0.k, b1.k, mask);
        _mm256_maskload_x2_epi32(v_ptr + e + h, b0.v, b1.v, mask);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        _mm256_maskstore_x2_ps(k_ptr + e, x0.k, x1.k, mask);
        _mm256_maskstore_x2_epi32(v_ptr + e, x0.v, x1.v, mask);
        _mm256_maskstore_x2_ps(k_ptr + e + h, y0.k, y1.k, mask);
        _mm256_maskstore_x2_epi32(v_ptr + e + h, y0.v, y1.v, mask);
    }

    return SUCCESS;
}

// combsort h=16
static int combsort_h16_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 4;

    __m256kv a0, a1, b0, b1;
    __m256kv x0, x1, y0, y1;

    if (e > 0) {
        _mm256_loadu_x2_ps(k_ptr, a0.k, a1.k);
        _mm256_loadu_x2_epi32(v_ptr, a0.v, a1.v);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(k_ptr + i + AVX2_FLOAT_STRIDE * 2, b0.k, b1.k);
            _mm256_loadu_x2_epi32(v_ptr + i + AVX2_EPI32_STRIDE * 2, b0.v, b1.v);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);

            _mm256_storeu_x2_ps(k_ptr + i, x0.k, x1.k);
            _mm256_storeu_x2_epi32(v_ptr + i, x0.v, x1.v);

            a0 = y0;
            a1 = y1;
        }
        _mm256_storeu_x2_ps(k_ptr + i, a0.k, a1.k);
        _mm256_storeu_x2_epi32(v_ptr + i, a0.v, a1.v);
    }
    {
        _mm256_loadu_x2_ps(k_ptr + e, a0.k, a1.k);
        _mm256_loadu_x2_epi32(v_ptr + e, a0.v, a1.v);
        _mm256_loadu_x2_ps(k_ptr + e + AVX2_FLOAT_STRIDE * 2, b0.k, b1.k);
        _mm256_loadu_x2_epi32(v_ptr + e + AVX2_EPI32_STRIDE * 2, b0.v, b1.v);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        _mm256_storeu_x2_ps(k_ptr + e, x0.k, x1.k);
        _mm256_storeu_x2_epi32(v_ptr + e, x0.v, x1.v);
        _mm256_storeu_x2_ps(k_ptr + e + AVX2_FLOAT_STRIDE * 2, y0.k, y1.k);
        _mm256_storeu_x2_epi32(v_ptr + e + AVX2_FLOAT_STRIDE * 2, y0.v, y1.v);
    }

    return SUCCESS;
}

// combsort h=17...23
static int combsort_h17to23_s(const uint n, const uint h, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_FLOAT_STRIDE * 2 || h >= AVX2_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_ps(h & AVX2_FLOAT_REMAIN_MASK);

    uint e = n - h * 2;

    __m256kv a0, a1, a2, b0, b1, b2;
    __m256kv x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_maskload_x3_ps(k_ptr, a0.k, a1.k, a2.k, mask);
        _mm256_maskload_x3_epi32(v_ptr, a0.v, a1.v, a2.v, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x3_ps(k_ptr + i + h, b0.k, b1.k, b2.k, mask);
            _mm256_maskload_x3_epi32(v_ptr + i + h, b0.v, b1.v, b2.v, mask);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);

            _mm256_maskstore_x3_ps(k_ptr + i, x0.k, x1.k, x2.k, mask);
            _mm256_maskstore_x3_epi32(v_ptr + i, x0.v, x1.v, x2.v, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_maskstore_x3_ps(k_ptr + i, a0.k, a1.k, a2.k, mask);
        _mm256_maskstore_x3_epi32(v_ptr + i, a0.v, a1.v, a2.v, mask);
    }
    {
        _mm256_maskload_x3_ps(k_ptr + e, a0.k, a1.k, a2.k, mask);
        _mm256_maskload_x3_epi32(v_ptr + e, a0.v, a1.v, a2.v, mask);
        _mm256_maskload_x3_ps(k_ptr + e + h, b0.k, b1.k, b2.k, mask);
        _mm256_maskload_x3_epi32(v_ptr + e + h, b0.v, b1.v, b2.v, mask);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);

        _mm256_maskstore_x3_ps(k_ptr + e, x0.k, x1.k, x2.k, mask);
        _mm256_maskstore_x3_epi32(v_ptr + e, x0.v, x1.v, x2.v, mask);
        _mm256_maskstore_x3_ps(k_ptr + e + h, y0.k, y1.k, y2.k, mask);
        _mm256_maskstore_x3_epi32(v_ptr + e + h, y0.v, y1.v, y2.v, mask);
    }

    return SUCCESS;
}

// combsort h=24
static int combsort_h24_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 6) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 6;

    __m256kv a0, a1, a2, b0, b1, b2;
    __m256kv x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_loadu_x3_ps(k_ptr, a0.k, a1.k, a2.k);
        _mm256_loadu_x3_epi32(v_ptr, a0.v, a1.v, a2.v);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(k_ptr + i + AVX2_FLOAT_STRIDE * 3, b0.k, b1.k, b2.k);
            _mm256_loadu_x3_epi32(v_ptr + i + AVX2_EPI32_STRIDE * 3, b0.v, b1.v, b2.v);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);

            _mm256_storeu_x3_ps(k_ptr + i, x0.k, x1.k, x2.k);
            _mm256_storeu_x3_epi32(v_ptr + i, x0.v, x1.v, x2.v);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_storeu_x3_ps(k_ptr + i, a0.k, a1.k, a2.k);
        _mm256_storeu_x3_epi32(v_ptr + i, a0.v, a1.v, a2.v);
    }
    {
        _mm256_loadu_x3_ps(k_ptr + e, a0.k, a1.k, a2.k);
        _mm256_loadu_x3_epi32(v_ptr + e, a0.v, a1.v, a2.v);
        _mm256_loadu_x3_ps(k_ptr + e + AVX2_FLOAT_STRIDE * 3, b0.k, b1.k, b2.k);
        _mm256_loadu_x3_epi32(v_ptr + e + AVX2_EPI32_STRIDE * 3, b0.v, b1.v, b2.v);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);

        _mm256_storeu_x3_ps(k_ptr + e, x0.k, x1.k, x2.k);
        _mm256_storeu_x3_epi32(v_ptr + e, x0.v, x1.v, x2.v);
        _mm256_storeu_x3_ps(k_ptr + e + AVX2_FLOAT_STRIDE * 3, y0.k, y1.k, y2.k);
        _mm256_storeu_x3_epi32(v_ptr + e + AVX2_EPI32_STRIDE * 3, y0.v, y1.v, y2.v);
    }

    return SUCCESS;
}

// combsort h=25...31
static int combsort_h25to31_s(const uint n, const uint h, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_FLOAT_STRIDE * 3 || h >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_ps(h & AVX2_FLOAT_REMAIN_MASK);

    uint e = n - h * 2;

    __m256kv a0, a1, a2, a3, b0, b1, b2, b3;
    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_maskload_x4_ps(k_ptr, a0.k, a1.k, a2.k, a3.k, mask);
        _mm256_maskload_x4_epi32(v_ptr, a0.v, a1.v, a2.v, a3.v, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x4_ps(k_ptr + i + h, b0.k, b1.k, b2.k, b3.k, mask);
            _mm256_maskload_x4_epi32(v_ptr + i + h, b0.v, b1.v, b2.v, b3.v, mask);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);
            _mm256_cmpswap_ps(a3, b3, x3, y3);

            _mm256_maskstore_x4_ps(k_ptr + i, x0.k, x1.k, x2.k, x3.k, mask);
            _mm256_maskstore_x4_epi32(v_ptr + i, x0.v, x1.v, x2.v, x3.v, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_maskstore_x4_ps(k_ptr + i, a0.k, a1.k, a2.k, a3.k, mask);
        _mm256_maskstore_x4_epi32(v_ptr + i, a0.v, a1.v, a2.v, a3.v, mask);
    }
    {
        _mm256_maskload_x4_ps(k_ptr + e, a0.k, a1.k, a2.k, a3.k, mask);
        _mm256_maskload_x4_epi32(v_ptr + e, a0.v, a1.v, a2.v, a3.v, mask);
        _mm256_maskload_x4_ps(k_ptr + e + h, b0.k, b1.k, b2.k, b3.k, mask);
        _mm256_maskload_x4_epi32(v_ptr + e + h, b0.v, b1.v, b2.v, b3.v, mask);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_maskstore_x4_ps(k_ptr + e, x0.k, x1.k, x2.k, x3.k, mask);
        _mm256_maskstore_x4_epi32(v_ptr + e, x0.v, x1.v, x2.v, x3.v, mask);
        _mm256_maskstore_x4_ps(k_ptr + e + h, y0.k, y1.k, y2.k, y3.k, mask);
        _mm256_maskstore_x4_epi32(v_ptr + e + h, y0.v, y1.v, y2.v, y3.v, mask);
    }

    return SUCCESS;
}

// combsort h=32
static int combsort_h32_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 8) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 8;

    __m256kv a0, a1, a2, a3, b0, b1, b2, b3;
    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_loadu_x4_ps(k_ptr, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi32(v_ptr, a0.v, a1.v, a2.v, a3.v);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(k_ptr + i + AVX2_FLOAT_STRIDE * 4, b0.k, b1.k, b2.k, b3.k);
            _mm256_loadu_x4_epi32(v_ptr + i + AVX2_EPI32_STRIDE * 4, b0.v, b1.v, b2.v, b3.v);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);
            _mm256_cmpswap_ps(a3, b3, x3, y3);

            _mm256_storeu_x4_ps(k_ptr + i, x0.k, x1.k, x2.k, x3.k);
            _mm256_storeu_x4_epi32(v_ptr + i, x0.v, x1.v, x2.v, x3.v);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_storeu_x4_ps(k_ptr + i, a0.k, a1.k, a2.k, a3.k);
        _mm256_storeu_x4_epi32(v_ptr + i, a0.v, a1.v, a2.v, a3.v);
    }
    {
        _mm256_loadu_x4_ps(k_ptr + e, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi32(v_ptr + e, a0.v, a1.v, a2.v, a3.v);
        _mm256_loadu_x4_ps(k_ptr + e + AVX2_FLOAT_STRIDE * 4, b0.k, b1.k, b2.k, b3.k);
        _mm256_loadu_x4_epi32(v_ptr + e + AVX2_EPI32_STRIDE * 4, b0.v, b1.v, b2.v, b3.v);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(k_ptr + e, x0.k, x1.k, x2.k, x3.k);
        _mm256_storeu_x4_epi32(v_ptr + e, x0.v, x1.v, x2.v, x3.v);
        _mm256_storeu_x4_ps(k_ptr + e + AVX2_FLOAT_STRIDE * 4, y0.k, y1.k, y2.k, y3.k);
        _mm256_storeu_x4_epi32(v_ptr + e + AVX2_EPI32_STRIDE * 4, y0.v, y1.v, y2.v, y3.v);
    }

    return SUCCESS;
}

// combsort h>32
static int combsort_h33plus_s(const uint n, const uint h, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h + AVX2_FLOAT_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_FLOAT_STRIDE * 4;

    __m256kv a0, a1, a2, a3, b0, b1, b2, b3;
    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    for (uint i = 0; i < e; i += AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(k_ptr + i, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi32(v_ptr + i, a0.v, a1.v, a2.v, a3.v);
        _mm256_loadu_x4_ps(k_ptr + i + h, b0.k, b1.k, b2.k, b3.k);
        _mm256_loadu_x4_epi32(v_ptr + i + h, b0.v, b1.v, b2.v, b3.v);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(k_ptr + i, x0.k, x1.k, x2.k, x3.k);
        _mm256_storeu_x4_epi32(v_ptr + i, x0.v, x1.v, x2.v, x3.v);
        _mm256_storeu_x4_ps(k_ptr + i + h, y0.k, y1.k, y2.k, y3.k);
        _mm256_storeu_x4_epi32(v_ptr + i + h, y0.v, y1.v, y2.v, y3.v);
    }
    {
        _mm256_loadu_x4_ps(k_ptr + e, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi32(v_ptr + e, a0.v, a1.v, a2.v, a3.v);
        _mm256_loadu_x4_ps(k_ptr + e + h, b0.k, b1.k, b2.k, b3.k);
        _mm256_loadu_x4_epi32(v_ptr + e + h, b0.v, b1.v, b2.v, b3.v);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(k_ptr + e, x0.k, x1.k, x2.k, x3.k);
        _mm256_storeu_x4_epi32(v_ptr + e, x0.v, x1.v, x2.v, x3.v);
        _mm256_storeu_x4_ps(k_ptr + e + h, y0.k, y1.k, y2.k, y3.k);
        _mm256_storeu_x4_epi32(v_ptr + e + h, y0.v, y1.v, y2.v, y3.v);
    }

    return SUCCESS;
}

#pragma endregion combsort

#pragma region paracombsort

// paracombsort 2x8
static int paracombsort_p2x8_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    const uint e = n - AVX2_FLOAT_STRIDE * 4;
    const uint c = n % AVX2_FLOAT_STRIDE;

    __m256kv a0, a1, b0, b1;
    __m256kv x0, x1, y0, y1;

    for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
        a1.k = _mm256_loadu_ps(k_ptr + i);
        a1.v = _mm256_loadu_epi32(v_ptr + i);
        b1.k = _mm256_loadu_ps(k_ptr + i + AVX2_FLOAT_STRIDE);
        b1.v = _mm256_loadu_epi32(v_ptr + i + AVX2_EPI32_STRIDE);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        a0 = x1;
        a1 = y1;
        b1.k = _mm256_loadu_ps(k_ptr + i + AVX2_FLOAT_STRIDE * 2);
        b1.v = _mm256_loadu_epi32(v_ptr + i + AVX2_EPI32_STRIDE * 2);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        b0.k = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x1.k), _MM_PERM_ABDC));
        b0.v = _mm256_castpd_si256(_mm256_permute4x64_pd(_mm256_castsi256_pd(x1.v), _MM_PERM_ABDC));
        a1 = y1;
        b1.k = _mm256_loadu_ps(k_ptr + i + AVX2_FLOAT_STRIDE * 3);
        b1.v = _mm256_loadu_epi32(v_ptr + i + AVX2_EPI32_STRIDE * 3);
        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        for (j = i; j + AVX2_FLOAT_STRIDE <= e; j += AVX2_FLOAT_STRIDE) {
            _mm256_storeu_ps(k_ptr + j, x0.k);
            _mm256_storeu_epi32(v_ptr + j, x0.v);
            a0 = y0;
            b0.k = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x1.k), _MM_PERM_ABDC));
            b0.v = _mm256_castpd_si256(_mm256_permute4x64_pd(_mm256_castsi256_pd(x1.v), _MM_PERM_ABDC));
            a1 = y1;
            b1.k = _mm256_loadu_ps(k_ptr + j + AVX2_FLOAT_STRIDE * 4);
            b1.v = _mm256_loadu_epi32(v_ptr + j + AVX2_EPI32_STRIDE * 4);
            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
        }

        _mm256_storeu_ps(k_ptr + j, x0.k);
        _mm256_storeu_epi32(v_ptr + j, x0.v);
        a0 = y0;
        b0.k = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x1.k), _MM_PERM_ABDC));
        b0.v = _mm256_castpd_si256(_mm256_permute4x64_pd(_mm256_castsi256_pd(x1.v), _MM_PERM_ABDC));
        _mm256_cmpswap_ps(a0, b0, x0, y0);
        j += AVX2_FLOAT_STRIDE;

        _mm256_storeu_ps(k_ptr + j, x0.k);
        _mm256_storeu_epi32(v_ptr + j, x0.v);
        a0 = y0;
        b0 = y1;
        _mm256_cmpswap_ps(a0, b0, x0, y0);
        j += AVX2_FLOAT_STRIDE;

        _mm256_storeu_ps(k_ptr + j, x0.k);
        _mm256_storeu_epi32(v_ptr + j, x0.v);
        _mm256_storeu_ps(k_ptr + j + AVX2_FLOAT_STRIDE, y0.k);
        _mm256_storeu_epi32(v_ptr + j + AVX2_EPI32_STRIDE, y0.v);
    }

    return SUCCESS;
}

#pragma endregion paracombsort

#pragma region backtracksort

// backtracksort 8 elems wise
__forceinline static int backtracksort_p8_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 2) {
        return SUCCESS;
    }

    uint i = 0, e = n - AVX2_FLOAT_STRIDE * 2;

    __m256kv a, b, x, y;
    a.k = _mm256_loadu_ps(k_ptr);
    a.v = _mm256_loadu_epi32(v_ptr);
    b.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE);
    b.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE);

    if (e <= 0) {
        _mm256_cmpswap_ps(a, b, x, y);

        _mm256_storeu_ps(k_ptr, x.k);
        _mm256_storeu_epi32(v_ptr, x.v);
        _mm256_storeu_ps(k_ptr + AVX2_FLOAT_STRIDE, y.k);
        _mm256_storeu_epi32(v_ptr + AVX2_EPI32_STRIDE, y.v);

        return SUCCESS;
    }

    while (true) {
        int indexes = _mm256_cmpswap_indexed_ps(a, b, x, y);

        if (indexes > 0) {
            _mm256_storeu_ps(k_ptr + i, x.k);
            _mm256_storeu_epi32(v_ptr + i, x.v);
            _mm256_storeu_ps(k_ptr + i + AVX2_FLOAT_STRIDE, y.k);
            _mm256_storeu_epi32(v_ptr + i + AVX2_EPI32_STRIDE, y.v);

            if (i >= AVX2_FLOAT_STRIDE) {
                i -= AVX2_FLOAT_STRIDE;
                a.k = _mm256_loadu_ps(k_ptr + i);
                a.v = _mm256_loadu_epi32(v_ptr + i);
                b = x;
                continue;
            }
            else if (i > 0) {
                i = 0;
                a.k = _mm256_loadu_ps(k_ptr);
                a.v = _mm256_loadu_epi32(v_ptr);
                b.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE);
                b.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE);
                continue;
            }
            else {
                i = AVX2_FLOAT_STRIDE;
                if (i <= e) {
                    a = y;
                    b.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE * 2);
                    b.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE * 2);
                    continue;
                }
            }
        }
        else if (i < e) {
            i += AVX2_FLOAT_STRIDE;

            if (i <= e) {
                a = b;
                b.k = _mm256_loadu_ps(k_ptr + i + AVX2_FLOAT_STRIDE);
                b.v = _mm256_loadu_epi32(v_ptr + i + AVX2_EPI32_STRIDE);
                continue;
            }
        }
        else {
            break;
        }

        i = e;
        a.k = _mm256_loadu_ps(k_ptr + i);
        a.v = _mm256_loadu_epi32(v_ptr + i);
        b.k = _mm256_loadu_ps(k_ptr + i + AVX2_FLOAT_STRIDE);
        b.v = _mm256_loadu_epi32(v_ptr + i + AVX2_EPI32_STRIDE);
    }

    return SUCCESS;
}

#pragma endregion backtracksort

#pragma region batchsort

// batchsort 8 elems wise
static int batchsort_p8_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (n < AVX2_FLOAT_STRIDE) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE;

    __m256kv x0, x1, x2, x3;
    __m256kv y0, y1, y2, y3;

    float* const ke_ptr = k_ptr + e;
    uint* const ve_ptr = v_ptr + e;

    {
        float* kc_ptr = k_ptr;
        uint* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);
            y3 = _mm256_sort_ps(x3);

            _mm256_storeu_x4_ps(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            kc_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi32(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);

            _mm256_storeu_x3_ps(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi32(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(vc_ptr, x0.v, x1.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);

            _mm256_storeu_x2_ps(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(kc_ptr, x0.k);
            _mm256_loadu_x1_epi32(vc_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(kc_ptr, y0.k);
            _mm256_storeu_x1_epi32(vc_ptr, y0.v);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ke_ptr, x0.k);
            _mm256_loadu_x1_epi32(ve_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ke_ptr, y0.k);
            _mm256_storeu_x1_epi32(ve_ptr, y0.v);
        }
    }
    {
        float* kc_ptr = k_ptr + AVX2_FLOAT_STRIDE / 2;
        uint* vc_ptr = v_ptr + AVX2_EPI32_STRIDE / 2;
        uint r = n - AVX2_FLOAT_STRIDE / 2;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);
            y3 = _mm256_halfsort_ps(x3);

            _mm256_storeu_x4_ps(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            kc_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi32(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);

            _mm256_storeu_x3_ps(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi32(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(vc_ptr, x0.v, x1.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);

            _mm256_storeu_x2_ps(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(kc_ptr, x0.k);
            _mm256_loadu_x1_epi32(vc_ptr, x0.v);

            y0 = _mm256_halfsort_ps(x0);

            _mm256_storeu_x1_ps(kc_ptr, y0.k);
            _mm256_storeu_x1_epi32(vc_ptr, y0.v);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ke_ptr, x0.k);
            _mm256_loadu_x1_epi32(ve_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ke_ptr, y0.k);
            _mm256_storeu_x1_epi32(ve_ptr, y0.v);
        }
    }
    {
        float* kc_ptr = k_ptr;
        uint* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);
            y3 = _mm256_halfsort_ps(x3);

            _mm256_storeu_x4_ps(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            kc_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi32(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);

            _mm256_storeu_x3_ps(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi32(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(vc_ptr, x0.v, x1.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);

            _mm256_storeu_x2_ps(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(kc_ptr, x0.k);
            _mm256_loadu_x1_epi32(vc_ptr, x0.v);

            y0 = _mm256_halfsort_ps(x0);

            _mm256_storeu_x1_ps(kc_ptr, y0.k);
            _mm256_storeu_x1_epi32(vc_ptr, y0.v);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ke_ptr, x0.k);
            _mm256_loadu_x1_epi32(ve_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ke_ptr, y0.k);
            _mm256_storeu_x1_epi32(ve_ptr, y0.v);
        }
    }
    {
        float* kc_ptr = k_ptr + AVX2_FLOAT_STRIDE / 2;
        uint* vc_ptr = v_ptr + AVX2_EPI32_STRIDE / 2;
        uint r = n - AVX2_FLOAT_STRIDE / 2;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);
            y3 = _mm256_halfsort_ps(x3);

            _mm256_storeu_x4_ps(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            kc_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi32(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);

            _mm256_storeu_x3_ps(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi32(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(vc_ptr, x0.v, x1.v);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);

            _mm256_storeu_x2_ps(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(kc_ptr, x0.k);
            _mm256_loadu_x1_epi32(vc_ptr, x0.v);

            y0 = _mm256_halfsort_ps(x0);

            _mm256_storeu_x1_ps(kc_ptr, y0.k);
            _mm256_storeu_x1_epi32(vc_ptr, y0.v);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ke_ptr, x0.k);
            _mm256_loadu_x1_epi32(ve_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ke_ptr, y0.k);
            _mm256_storeu_x1_epi32(ve_ptr, y0.v);
        }
    }

    return SUCCESS;
}

#pragma endregion batchsort

#pragma region scansort

// scansort 8 elems wise
__forceinline static int scansort_p8_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    uint e = n - AVX2_FLOAT_STRIDE;

    uint indexes;
    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    {
        uint i = 0;
        while (true) {
            if (i + AVX2_FLOAT_STRIDE * 4 + 1 <= n) {
                _mm256_loadu_x4_ps(k_ptr + i, x0.k, x1.k, x2.k, x3.k);
                _mm256_loadu_x4_ps(k_ptr + i + 1, y0.k, y1.k, y2.k, y3.k);

                uint i0 = _mm256_movemask_ps(_mm256_needsswap_ps(x0.k, y0.k));
                uint i1 = _mm256_movemask_ps(_mm256_needsswap_ps(x1.k, y1.k));
                uint i2 = _mm256_movemask_ps(_mm256_needsswap_ps(x2.k, y2.k));
                uint i3 = _mm256_movemask_ps(_mm256_needsswap_ps(x3.k, y3.k));

                indexes = (i0) | (i1 << (AVX2_FLOAT_STRIDE)) | (i2 << (AVX2_FLOAT_STRIDE * 2)) | (i3 << (AVX2_FLOAT_STRIDE * 3));

                if (indexes == 0u) {
                    i += AVX2_FLOAT_STRIDE * 4;
                    continue;
                }
            }
            else if (i + AVX2_FLOAT_STRIDE * 2 + 1 <= n) {
                _mm256_loadu_x2_ps(k_ptr + i, x0.k, x1.k);
                _mm256_loadu_x2_ps(k_ptr + i + 1, y0.k, y1.k);

                uint i0 = _mm256_movemask_ps(_mm256_needsswap_ps(x0.k, y0.k));
                uint i1 = _mm256_movemask_ps(_mm256_needsswap_ps(x1.k, y1.k));

                indexes = (i0) | (i1 << (AVX2_FLOAT_STRIDE));

                if (indexes == 0u) {
                    i += AVX2_FLOAT_STRIDE * 2;
                    continue;
                }
            }
            else if (i + AVX2_FLOAT_STRIDE + 1 <= n) {
                _mm256_loadu_x1_ps(k_ptr + i, x0.k);
                _mm256_loadu_x1_ps(k_ptr + i + 1, y0.k);

                indexes = (uint)_mm256_movemask_ps(_mm256_needsswap_ps(x0.k, y0.k));

                if (indexes == 0u) {
                    i += AVX2_FLOAT_STRIDE;
                    continue;
                }
            }
            else {
                i = e;

                x0.k = _mm256_loadu_ps(k_ptr + i);

                if (!_mm256_needssort_ps(x0.k)) {
                    break;
                }

                x0.v = _mm256_loadu_epi32(v_ptr + i);

                y0 = _mm256_sort_ps(x0);
                _mm256_storeu_ps(k_ptr + i, y0.k);
                _mm256_storeu_epi32(v_ptr + i, y0.v);

                indexes = 0xFFu - (uint)_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0.k, y0.k));

                if ((indexes & 1u) == 0u || i == 0) {
                    break;
                }
            }

            uint index = bsf(indexes);

            if (index >= AVX2_FLOAT_STRIDE - 2) {
                uint forward = index - (AVX2_FLOAT_STRIDE - 2);
                i += forward;
            }
            else {
                uint backward = (AVX2_FLOAT_STRIDE - 2) - index;
                i = (i > backward) ? i - backward : 0;
            }

            x0.k = _mm256_loadu_ps(k_ptr + i);
            x0.v = _mm256_loadu_epi32(v_ptr + i);

            while (true) {
                y0 = _mm256_sort_ps(x0);
                _mm256_storeu_ps(k_ptr + i, y0.k);
                _mm256_storeu_epi32(v_ptr + i, y0.v);

                indexes = (uint)_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0.k, y0.k));
                if ((indexes & 1u) == 0u && i > 0u) {
                    uint backward = AVX2_FLOAT_STRIDE - 2;
                    i = (i > backward) ? i - backward : 0;

                    x0.k = _mm256_loadu_ps(k_ptr + i);

                    if (_mm256_needssort_ps(x0.k)) {
                        x0.v = _mm256_loadu_epi32(v_ptr + i);
                        continue;
                    }
                }

                uint forward = AVX2_FLOAT_STRIDE - 1;
                i += forward;
                break;
            }
        }
    }

    return SUCCESS;
}

#pragma endregion scansort

#pragma region permcombsort

// permcombsort 8 elems wise
__forceinline static int permcombsort_p8_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_FLOAT_STRIDE * 2 || n >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_FLOAT_STRIDE;

    __m256kv a, b, x, y;

    for (uint h = c > 2 ? 2 : 1; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a.k = _mm256_loadu_ps(k_ptr + i * AVX2_FLOAT_STRIDE);
            a.v = _mm256_loadu_epi32(v_ptr + i * AVX2_EPI32_STRIDE);
            b.k = _mm256_loadu_ps(k_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b.v = _mm256_loadu_epi32(v_ptr + (i + h) * AVX2_EPI32_STRIDE);

            _mm256_cmpswap_withperm_ps(a, b, x, y);

            _mm256_storeu_ps(k_ptr + i * AVX2_FLOAT_STRIDE, x.k);
            _mm256_storeu_epi32(v_ptr + i * AVX2_EPI32_STRIDE, x.v);
            _mm256_storeu_ps(k_ptr + (i + h) * AVX2_FLOAT_STRIDE, y.k);
            _mm256_storeu_epi32(v_ptr + (i + h) * AVX2_EPI32_STRIDE, y.v);
        }
    }

    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0) {
        b.k = _mm256_loadu_ps(k_ptr + (n - AVX2_FLOAT_STRIDE));
        b.v = _mm256_loadu_epi32(v_ptr + (n - AVX2_EPI32_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a.k = _mm256_loadu_ps(k_ptr + i * AVX2_FLOAT_STRIDE);
            a.v = _mm256_loadu_epi32(v_ptr + i * AVX2_EPI32_STRIDE);

            _mm256_cmpswap_ps(a, b, x, y);
            x = _mm256_sort_ps(x);
            b = y;

            _mm256_storeu_ps(k_ptr + i * AVX2_FLOAT_STRIDE, x.k);
            _mm256_storeu_epi32(v_ptr + i * AVX2_EPI32_STRIDE, x.v);
        }

        a.k = _mm256_loadu_ps(k_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a.v = _mm256_loadu_epi32(v_ptr + (n - AVX2_EPI32_STRIDE * 2));

        _mm256_cmpswap_withperm_ps(a, b, x, y);

        x = _mm256_sort_ps(x);
        y = _mm256_sort_ps(y);

        _mm256_storeu_ps(k_ptr + (n - AVX2_FLOAT_STRIDE * 2), x.k);
        _mm256_storeu_epi32(v_ptr + (n - AVX2_EPI32_STRIDE * 2), x.v);
        _mm256_storeu_ps(k_ptr + (n - AVX2_FLOAT_STRIDE), y.k);
        _mm256_storeu_epi32(v_ptr + (n - AVX2_EPI32_STRIDE), y.v);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a.k = _mm256_loadu_ps(k_ptr + i * AVX2_FLOAT_STRIDE);
            a.v = _mm256_loadu_epi32(v_ptr + i * AVX2_EPI32_STRIDE);
            a = _mm256_sort_ps(a);
            _mm256_storeu_ps(k_ptr + i * AVX2_FLOAT_STRIDE, a.k);
            _mm256_storeu_epi32(v_ptr + i * AVX2_EPI32_STRIDE, a.v);
        }
    }

    return SUCCESS;
}

// permcombsort 8 elems wise 4 batches
__forceinline static int permcombsort_p4x8_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_FLOAT_STRIDE * 2 || n >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_FLOAT_STRIDE;

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + n;
    float* k2_ptr = k_ptr + n * 2;
    float* k3_ptr = k_ptr + n * 3;
    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + n;
    uint* v2_ptr = v_ptr + n * 2;
    uint* v3_ptr = v_ptr + n * 3;

    __m256kv a0, b0, x0, y0, a1, b1, x1, y1, a2, b2, x2, y2, a3, b3, x3, y3;

    for (uint h = c > 2 ? 2 : 1; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a0.k = _mm256_loadu_ps(k0_ptr + i * AVX2_FLOAT_STRIDE);
            a1.k = _mm256_loadu_ps(k1_ptr + i * AVX2_FLOAT_STRIDE);
            a2.k = _mm256_loadu_ps(k2_ptr + i * AVX2_FLOAT_STRIDE);
            a3.k = _mm256_loadu_ps(k3_ptr + i * AVX2_FLOAT_STRIDE);
            b0.k = _mm256_loadu_ps(k0_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b1.k = _mm256_loadu_ps(k1_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b2.k = _mm256_loadu_ps(k2_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b3.k = _mm256_loadu_ps(k3_ptr + (i + h) * AVX2_FLOAT_STRIDE);

            a0.v = _mm256_loadu_epi32(v0_ptr + i * AVX2_EPI32_STRIDE);
            a1.v = _mm256_loadu_epi32(v1_ptr + i * AVX2_EPI32_STRIDE);
            a2.v = _mm256_loadu_epi32(v2_ptr + i * AVX2_EPI32_STRIDE);
            a3.v = _mm256_loadu_epi32(v3_ptr + i * AVX2_EPI32_STRIDE);
            b0.v = _mm256_loadu_epi32(v0_ptr + (i + h) * AVX2_EPI32_STRIDE);
            b1.v = _mm256_loadu_epi32(v1_ptr + (i + h) * AVX2_EPI32_STRIDE);
            b2.v = _mm256_loadu_epi32(v2_ptr + (i + h) * AVX2_EPI32_STRIDE);
            b3.v = _mm256_loadu_epi32(v3_ptr + (i + h) * AVX2_EPI32_STRIDE);

            _mm256_cmpswap_withperm_ps(a0, b0, x0, y0);
            _mm256_cmpswap_withperm_ps(a1, b1, x1, y1);
            _mm256_cmpswap_withperm_ps(a2, b2, x2, y2);
            _mm256_cmpswap_withperm_ps(a3, b3, x3, y3);

            _mm256_storeu_ps(k0_ptr + i * AVX2_FLOAT_STRIDE, x0.k);
            _mm256_storeu_ps(k1_ptr + i * AVX2_FLOAT_STRIDE, x1.k);
            _mm256_storeu_ps(k2_ptr + i * AVX2_FLOAT_STRIDE, x2.k);
            _mm256_storeu_ps(k3_ptr + i * AVX2_FLOAT_STRIDE, x3.k);
            _mm256_storeu_ps(k0_ptr + (i + h) * AVX2_FLOAT_STRIDE, y0.k);
            _mm256_storeu_ps(k1_ptr + (i + h) * AVX2_FLOAT_STRIDE, y1.k);
            _mm256_storeu_ps(k2_ptr + (i + h) * AVX2_FLOAT_STRIDE, y2.k);
            _mm256_storeu_ps(k3_ptr + (i + h) * AVX2_FLOAT_STRIDE, y3.k);

            _mm256_storeu_epi32(v0_ptr + i * AVX2_EPI32_STRIDE, x0.v);
            _mm256_storeu_epi32(v1_ptr + i * AVX2_EPI32_STRIDE, x1.v);
            _mm256_storeu_epi32(v2_ptr + i * AVX2_EPI32_STRIDE, x2.v);
            _mm256_storeu_epi32(v3_ptr + i * AVX2_EPI32_STRIDE, x3.v);
            _mm256_storeu_epi32(v0_ptr + (i + h) * AVX2_EPI32_STRIDE, y0.v);
            _mm256_storeu_epi32(v1_ptr + (i + h) * AVX2_EPI32_STRIDE, y1.v);
            _mm256_storeu_epi32(v2_ptr + (i + h) * AVX2_EPI32_STRIDE, y2.v);
            _mm256_storeu_epi32(v3_ptr + (i + h) * AVX2_EPI32_STRIDE, y3.v);
        }
    }

    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0) {
        b0.k = _mm256_loadu_ps(k0_ptr + (n - AVX2_FLOAT_STRIDE));
        b1.k = _mm256_loadu_ps(k1_ptr + (n - AVX2_FLOAT_STRIDE));
        b2.k = _mm256_loadu_ps(k2_ptr + (n - AVX2_FLOAT_STRIDE));
        b3.k = _mm256_loadu_ps(k3_ptr + (n - AVX2_FLOAT_STRIDE));

        b0.v = _mm256_loadu_epi32(v0_ptr + (n - AVX2_EPI32_STRIDE));
        b1.v = _mm256_loadu_epi32(v1_ptr + (n - AVX2_EPI32_STRIDE));
        b2.v = _mm256_loadu_epi32(v2_ptr + (n - AVX2_EPI32_STRIDE));
        b3.v = _mm256_loadu_epi32(v3_ptr + (n - AVX2_EPI32_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a0.k = _mm256_loadu_ps(k0_ptr + i * AVX2_FLOAT_STRIDE);
            a1.k = _mm256_loadu_ps(k1_ptr + i * AVX2_FLOAT_STRIDE);
            a2.k = _mm256_loadu_ps(k2_ptr + i * AVX2_FLOAT_STRIDE);
            a3.k = _mm256_loadu_ps(k3_ptr + i * AVX2_FLOAT_STRIDE);

            a0.v = _mm256_loadu_epi32(v0_ptr + i * AVX2_EPI32_STRIDE);
            a1.v = _mm256_loadu_epi32(v1_ptr + i * AVX2_EPI32_STRIDE);
            a2.v = _mm256_loadu_epi32(v2_ptr + i * AVX2_EPI32_STRIDE);
            a3.v = _mm256_loadu_epi32(v3_ptr + i * AVX2_EPI32_STRIDE);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);
            _mm256_cmpswap_ps(a3, b3, x3, y3);

            x0 = _mm256_sort_ps(x0);
            x1 = _mm256_sort_ps(x1);
            x2 = _mm256_sort_ps(x2);
            x3 = _mm256_sort_ps(x3);
            b0 = y0;
            b1 = y1;
            b2 = y2;
            b3 = y3;

            _mm256_storeu_ps(k0_ptr + i * AVX2_FLOAT_STRIDE, x0.k);
            _mm256_storeu_ps(k1_ptr + i * AVX2_FLOAT_STRIDE, x1.k);
            _mm256_storeu_ps(k2_ptr + i * AVX2_FLOAT_STRIDE, x2.k);
            _mm256_storeu_ps(k3_ptr + i * AVX2_FLOAT_STRIDE, x3.k);

            _mm256_storeu_epi32(v0_ptr + i * AVX2_EPI32_STRIDE, x0.v);
            _mm256_storeu_epi32(v1_ptr + i * AVX2_EPI32_STRIDE, x1.v);
            _mm256_storeu_epi32(v2_ptr + i * AVX2_EPI32_STRIDE, x2.v);
            _mm256_storeu_epi32(v3_ptr + i * AVX2_EPI32_STRIDE, x3.v);
        }

        a0.k = _mm256_loadu_ps(k0_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a1.k = _mm256_loadu_ps(k1_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a2.k = _mm256_loadu_ps(k2_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a3.k = _mm256_loadu_ps(k3_ptr + (n - AVX2_FLOAT_STRIDE * 2));

        a0.v = _mm256_loadu_epi32(v0_ptr + (n - AVX2_EPI32_STRIDE * 2));
        a1.v = _mm256_loadu_epi32(v1_ptr + (n - AVX2_EPI32_STRIDE * 2));
        a2.v = _mm256_loadu_epi32(v2_ptr + (n - AVX2_EPI32_STRIDE * 2));
        a3.v = _mm256_loadu_epi32(v3_ptr + (n - AVX2_EPI32_STRIDE * 2));

        _mm256_cmpswap_withperm_ps(a0, b0, x0, y0);
        _mm256_cmpswap_withperm_ps(a1, b1, x1, y1);
        _mm256_cmpswap_withperm_ps(a2, b2, x2, y2);
        _mm256_cmpswap_withperm_ps(a3, b3, x3, y3);

        x0 = _mm256_sort_ps(x0);
        x1 = _mm256_sort_ps(x1);
        x2 = _mm256_sort_ps(x2);
        x3 = _mm256_sort_ps(x3);
        y0 = _mm256_sort_ps(y0);
        y1 = _mm256_sort_ps(y1);
        y2 = _mm256_sort_ps(y2);
        y3 = _mm256_sort_ps(y3);

        _mm256_storeu_ps(k0_ptr + (n - AVX2_FLOAT_STRIDE * 2), x0.k);
        _mm256_storeu_ps(k1_ptr + (n - AVX2_FLOAT_STRIDE * 2), x1.k);
        _mm256_storeu_ps(k2_ptr + (n - AVX2_FLOAT_STRIDE * 2), x2.k);
        _mm256_storeu_ps(k3_ptr + (n - AVX2_FLOAT_STRIDE * 2), x3.k);
        _mm256_storeu_ps(k0_ptr + (n - AVX2_FLOAT_STRIDE), y0.k);
        _mm256_storeu_ps(k1_ptr + (n - AVX2_FLOAT_STRIDE), y1.k);
        _mm256_storeu_ps(k2_ptr + (n - AVX2_FLOAT_STRIDE), y2.k);
        _mm256_storeu_ps(k3_ptr + (n - AVX2_FLOAT_STRIDE), y3.k);

        _mm256_storeu_epi32(v0_ptr + (n - AVX2_EPI32_STRIDE * 2), x0.v);
        _mm256_storeu_epi32(v1_ptr + (n - AVX2_EPI32_STRIDE * 2), x1.v);
        _mm256_storeu_epi32(v2_ptr + (n - AVX2_EPI32_STRIDE * 2), x2.v);
        _mm256_storeu_epi32(v3_ptr + (n - AVX2_EPI32_STRIDE * 2), x3.v);
        _mm256_storeu_epi32(v0_ptr + (n - AVX2_EPI32_STRIDE), y0.v);
        _mm256_storeu_epi32(v1_ptr + (n - AVX2_EPI32_STRIDE), y1.v);
        _mm256_storeu_epi32(v2_ptr + (n - AVX2_EPI32_STRIDE), y2.v);
        _mm256_storeu_epi32(v3_ptr + (n - AVX2_EPI32_STRIDE), y3.v);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a0.k = _mm256_loadu_ps(k0_ptr + i * AVX2_FLOAT_STRIDE);
            a1.k = _mm256_loadu_ps(k1_ptr + i * AVX2_FLOAT_STRIDE);
            a2.k = _mm256_loadu_ps(k2_ptr + i * AVX2_FLOAT_STRIDE);
            a3.k = _mm256_loadu_ps(k3_ptr + i * AVX2_FLOAT_STRIDE);
            a0.v = _mm256_loadu_epi32(v0_ptr + i * AVX2_EPI32_STRIDE);
            a1.v = _mm256_loadu_epi32(v1_ptr + i * AVX2_EPI32_STRIDE);
            a2.v = _mm256_loadu_epi32(v2_ptr + i * AVX2_EPI32_STRIDE);
            a3.v = _mm256_loadu_epi32(v3_ptr + i * AVX2_EPI32_STRIDE);

            a0 = _mm256_sort_ps(a0);
            a1 = _mm256_sort_ps(a1);
            a2 = _mm256_sort_ps(a2);
            a3 = _mm256_sort_ps(a3);

            _mm256_storeu_ps(k0_ptr + i * AVX2_FLOAT_STRIDE, a0.k);
            _mm256_storeu_ps(k1_ptr + i * AVX2_FLOAT_STRIDE, a1.k);
            _mm256_storeu_ps(k2_ptr + i * AVX2_FLOAT_STRIDE, a2.k);
            _mm256_storeu_ps(k3_ptr + i * AVX2_FLOAT_STRIDE, a3.k);
            _mm256_storeu_epi32(v0_ptr + i * AVX2_EPI32_STRIDE, a0.v);
            _mm256_storeu_epi32(v1_ptr + i * AVX2_EPI32_STRIDE, a1.v);
            _mm256_storeu_epi32(v2_ptr + i * AVX2_EPI32_STRIDE, a2.v);
            _mm256_storeu_epi32(v3_ptr + i * AVX2_EPI32_STRIDE, a3.v);
        }
    }

    return SUCCESS;
}

#pragma endregion permcombsort

#pragma region shortsort

// shortsort elems9
__forceinline static int shortsort_n9_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 1);
    x.v = _mm256_loadu_epi32(v_ptr + 1);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 1, y.k);
    _mm256_storeu_epi32(v_ptr + 1, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort4x2_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems9
__forceinline static int shortsort_n4x9_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 9;
    float* k2_ptr = k_ptr + 18;
    float* k3_ptr = k_ptr + 27;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 9;
    uint* v2_ptr = v_ptr + 18;
    uint* v3_ptr = v_ptr + 27;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 1);
    x1.k = _mm256_loadu_ps(k1_ptr + 1);
    x2.k = _mm256_loadu_ps(k2_ptr + 1);
    x3.k = _mm256_loadu_ps(k3_ptr + 1);
    x0.v = _mm256_loadu_epi32(v0_ptr + 1);
    x1.v = _mm256_loadu_epi32(v1_ptr + 1);
    x2.v = _mm256_loadu_epi32(v2_ptr + 1);
    x3.v = _mm256_loadu_epi32(v3_ptr + 1);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 1, y0.k);
    _mm256_storeu_ps(k1_ptr + 1, y1.k);
    _mm256_storeu_ps(k2_ptr + 1, y2.k);
    _mm256_storeu_ps(k3_ptr + 1, y3.k);
    _mm256_storeu_epi32(v0_ptr + 1, y0.v);
    _mm256_storeu_epi32(v1_ptr + 1, y1.v);
    _mm256_storeu_epi32(v2_ptr + 1, y2.v);
    _mm256_storeu_epi32(v3_ptr + 1, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort4x2_ps(x0);
    y1 = _mm256_sort4x2_ps(x1);
    y2 = _mm256_sort4x2_ps(x2);
    y3 = _mm256_sort4x2_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    return SUCCESS;
}

// shortsort elems10
__forceinline static int shortsort_n10_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 2);
    x.v = _mm256_loadu_epi32(v_ptr + 2);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 2, y.k);
    _mm256_storeu_epi32(v_ptr + 2, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort2x4_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems10
__forceinline static int shortsort_n4x10_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 10;
    float* k2_ptr = k_ptr + 20;
    float* k3_ptr = k_ptr + 30;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 10;
    uint* v2_ptr = v_ptr + 20;
    uint* v3_ptr = v_ptr + 30;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 2);
    x1.k = _mm256_loadu_ps(k1_ptr + 2);
    x2.k = _mm256_loadu_ps(k2_ptr + 2);
    x3.k = _mm256_loadu_ps(k3_ptr + 2);
    x0.v = _mm256_loadu_epi32(v0_ptr + 2);
    x1.v = _mm256_loadu_epi32(v1_ptr + 2);
    x2.v = _mm256_loadu_epi32(v2_ptr + 2);
    x3.v = _mm256_loadu_epi32(v3_ptr + 2);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 2, y0.k);
    _mm256_storeu_ps(k1_ptr + 2, y1.k);
    _mm256_storeu_ps(k2_ptr + 2, y2.k);
    _mm256_storeu_ps(k3_ptr + 2, y3.k);
    _mm256_storeu_epi32(v0_ptr + 2, y0.v);
    _mm256_storeu_epi32(v1_ptr + 2, y1.v);
    _mm256_storeu_epi32(v2_ptr + 2, y2.v);
    _mm256_storeu_epi32(v3_ptr + 2, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort2x4_ps(x0);
    y1 = _mm256_sort2x4_ps(x1);
    y2 = _mm256_sort2x4_ps(x2);
    y3 = _mm256_sort2x4_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    return SUCCESS;
}

// shortsort elems11
__forceinline static int shortsort_n11_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 3);
    x.v = _mm256_loadu_epi32(v_ptr + 3);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 3, y.k);
    _mm256_storeu_epi32(v_ptr + 3, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort1x6_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems11
__forceinline static int shortsort_n4x11_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 11;
    float* k2_ptr = k_ptr + 22;
    float* k3_ptr = k_ptr + 33;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 11;
    uint* v2_ptr = v_ptr + 22;
    uint* v3_ptr = v_ptr + 33;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 3);
    x1.k = _mm256_loadu_ps(k1_ptr + 3);
    x2.k = _mm256_loadu_ps(k2_ptr + 3);
    x3.k = _mm256_loadu_ps(k3_ptr + 3);
    x0.v = _mm256_loadu_epi32(v0_ptr + 3);
    x1.v = _mm256_loadu_epi32(v1_ptr + 3);
    x2.v = _mm256_loadu_epi32(v2_ptr + 3);
    x3.v = _mm256_loadu_epi32(v3_ptr + 3);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 3, y0.k);
    _mm256_storeu_ps(k1_ptr + 3, y1.k);
    _mm256_storeu_ps(k2_ptr + 3, y2.k);
    _mm256_storeu_ps(k3_ptr + 3, y3.k);
    _mm256_storeu_epi32(v0_ptr + 3, y0.v);
    _mm256_storeu_epi32(v1_ptr + 3, y1.v);
    _mm256_storeu_epi32(v2_ptr + 3, y2.v);
    _mm256_storeu_epi32(v3_ptr + 3, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort1x6_ps(x0);
    y1 = _mm256_sort1x6_ps(x1);
    y2 = _mm256_sort1x6_ps(x2);
    y3 = _mm256_sort1x6_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    return SUCCESS;
}

// shortsort elems12
__forceinline static int shortsort_n12_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 4);
    x.v = _mm256_loadu_epi32(v_ptr + 4);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 4, y.k);
    _mm256_storeu_epi32(v_ptr + 4, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems12
__forceinline static int shortsort_n4x12_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 12;
    float* k2_ptr = k_ptr + 24;
    float* k3_ptr = k_ptr + 36;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 12;
    uint* v2_ptr = v_ptr + 24;
    uint* v3_ptr = v_ptr + 36;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 4);
    x1.k = _mm256_loadu_ps(k1_ptr + 4);
    x2.k = _mm256_loadu_ps(k2_ptr + 4);
    x3.k = _mm256_loadu_ps(k3_ptr + 4);
    x0.v = _mm256_loadu_epi32(v0_ptr + 4);
    x1.v = _mm256_loadu_epi32(v1_ptr + 4);
    x2.v = _mm256_loadu_epi32(v2_ptr + 4);
    x3.v = _mm256_loadu_epi32(v3_ptr + 4);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 4, y0.k);
    _mm256_storeu_ps(k1_ptr + 4, y1.k);
    _mm256_storeu_ps(k2_ptr + 4, y2.k);
    _mm256_storeu_ps(k3_ptr + 4, y3.k);
    _mm256_storeu_epi32(v0_ptr + 4, y0.v);
    _mm256_storeu_epi32(v1_ptr + 4, y1.v);
    _mm256_storeu_epi32(v2_ptr + 4, y2.v);
    _mm256_storeu_epi32(v3_ptr + 4, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    return SUCCESS;
}

// shortsort elems13
__forceinline static int shortsort_n13_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 5);
    x.v = _mm256_loadu_epi32(v_ptr + 5);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 5, y.k);
    _mm256_storeu_epi32(v_ptr + 5, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 2);
    x.v = _mm256_loadu_epi32(v_ptr + 2);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 2, y.k);
    _mm256_storeu_epi32(v_ptr + 2, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 5);
    x.v = _mm256_loadu_epi32(v_ptr + 5);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 5, y.k);
    _mm256_storeu_epi32(v_ptr + 5, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems13
__forceinline static int shortsort_n4x13_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 13;
    float* k2_ptr = k_ptr + 26;
    float* k3_ptr = k_ptr + 39;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 13;
    uint* v2_ptr = v_ptr + 26;
    uint* v3_ptr = v_ptr + 39;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 5);
    x1.k = _mm256_loadu_ps(k1_ptr + 5);
    x2.k = _mm256_loadu_ps(k2_ptr + 5);
    x3.k = _mm256_loadu_ps(k3_ptr + 5);
    x0.v = _mm256_loadu_epi32(v0_ptr + 5);
    x1.v = _mm256_loadu_epi32(v1_ptr + 5);
    x2.v = _mm256_loadu_epi32(v2_ptr + 5);
    x3.v = _mm256_loadu_epi32(v3_ptr + 5);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 5, y0.k);
    _mm256_storeu_ps(k1_ptr + 5, y1.k);
    _mm256_storeu_ps(k2_ptr + 5, y2.k);
    _mm256_storeu_ps(k3_ptr + 5, y3.k);
    _mm256_storeu_epi32(v0_ptr + 5, y0.v);
    _mm256_storeu_epi32(v1_ptr + 5, y1.v);
    _mm256_storeu_epi32(v2_ptr + 5, y2.v);
    _mm256_storeu_epi32(v3_ptr + 5, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 2);
    x1.k = _mm256_loadu_ps(k1_ptr + 2);
    x2.k = _mm256_loadu_ps(k2_ptr + 2);
    x3.k = _mm256_loadu_ps(k3_ptr + 2);
    x0.v = _mm256_loadu_epi32(v0_ptr + 2);
    x1.v = _mm256_loadu_epi32(v1_ptr + 2);
    x2.v = _mm256_loadu_epi32(v2_ptr + 2);
    x3.v = _mm256_loadu_epi32(v3_ptr + 2);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 2, y0.k);
    _mm256_storeu_ps(k1_ptr + 2, y1.k);
    _mm256_storeu_ps(k2_ptr + 2, y2.k);
    _mm256_storeu_ps(k3_ptr + 2, y3.k);
    _mm256_storeu_epi32(v0_ptr + 2, y0.v);
    _mm256_storeu_epi32(v1_ptr + 2, y1.v);
    _mm256_storeu_epi32(v2_ptr + 2, y2.v);
    _mm256_storeu_epi32(v3_ptr + 2, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 5);
    x1.k = _mm256_loadu_ps(k1_ptr + 5);
    x2.k = _mm256_loadu_ps(k2_ptr + 5);
    x3.k = _mm256_loadu_ps(k3_ptr + 5);
    x0.v = _mm256_loadu_epi32(v0_ptr + 5);
    x1.v = _mm256_loadu_epi32(v1_ptr + 5);
    x2.v = _mm256_loadu_epi32(v2_ptr + 5);
    x3.v = _mm256_loadu_epi32(v3_ptr + 5);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 5, y0.k);
    _mm256_storeu_ps(k1_ptr + 5, y1.k);
    _mm256_storeu_ps(k2_ptr + 5, y2.k);
    _mm256_storeu_ps(k3_ptr + 5, y3.k);
    _mm256_storeu_epi32(v0_ptr + 5, y0.v);
    _mm256_storeu_epi32(v1_ptr + 5, y1.v);
    _mm256_storeu_epi32(v2_ptr + 5, y2.v);
    _mm256_storeu_epi32(v3_ptr + 5, y3.v);

    return SUCCESS;
}

// shortsort elems14
__forceinline static int shortsort_n14_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 6);
    x.v = _mm256_loadu_epi32(v_ptr + 6);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 6, y.k);
    _mm256_storeu_epi32(v_ptr + 6, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 3);
    x.v = _mm256_loadu_epi32(v_ptr + 3);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 3, y.k);
    _mm256_storeu_epi32(v_ptr + 3, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 6);
    x.v = _mm256_loadu_epi32(v_ptr + 6);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 6, y.k);
    _mm256_storeu_epi32(v_ptr + 6, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 3);
    x.v = _mm256_loadu_epi32(v_ptr + 3);
    y = _mm256_sort4x2_ps(x);
    _mm256_storeu_ps(k_ptr + 3, y.k);
    _mm256_storeu_epi32(v_ptr + 3, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems14
__forceinline static int shortsort_n4x14_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 14;
    float* k2_ptr = k_ptr + 28;
    float* k3_ptr = k_ptr + 42;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 14;
    uint* v2_ptr = v_ptr + 28;
    uint* v3_ptr = v_ptr + 42;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 6);
    x1.k = _mm256_loadu_ps(k1_ptr + 6);
    x2.k = _mm256_loadu_ps(k2_ptr + 6);
    x3.k = _mm256_loadu_ps(k3_ptr + 6);
    x0.v = _mm256_loadu_epi32(v0_ptr + 6);
    x1.v = _mm256_loadu_epi32(v1_ptr + 6);
    x2.v = _mm256_loadu_epi32(v2_ptr + 6);
    x3.v = _mm256_loadu_epi32(v3_ptr + 6);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 6, y0.k);
    _mm256_storeu_ps(k1_ptr + 6, y1.k);
    _mm256_storeu_ps(k2_ptr + 6, y2.k);
    _mm256_storeu_ps(k3_ptr + 6, y3.k);
    _mm256_storeu_epi32(v0_ptr + 6, y0.v);
    _mm256_storeu_epi32(v1_ptr + 6, y1.v);
    _mm256_storeu_epi32(v2_ptr + 6, y2.v);
    _mm256_storeu_epi32(v3_ptr + 6, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 3);
    x1.k = _mm256_loadu_ps(k1_ptr + 3);
    x2.k = _mm256_loadu_ps(k2_ptr + 3);
    x3.k = _mm256_loadu_ps(k3_ptr + 3);
    x0.v = _mm256_loadu_epi32(v0_ptr + 3);
    x1.v = _mm256_loadu_epi32(v1_ptr + 3);
    x2.v = _mm256_loadu_epi32(v2_ptr + 3);
    x3.v = _mm256_loadu_epi32(v3_ptr + 3);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 3, y0.k);
    _mm256_storeu_ps(k1_ptr + 3, y1.k);
    _mm256_storeu_ps(k2_ptr + 3, y2.k);
    _mm256_storeu_ps(k3_ptr + 3, y3.k);
    _mm256_storeu_epi32(v0_ptr + 3, y0.v);
    _mm256_storeu_epi32(v1_ptr + 3, y1.v);
    _mm256_storeu_epi32(v2_ptr + 3, y2.v);
    _mm256_storeu_epi32(v3_ptr + 3, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 6);
    x1.k = _mm256_loadu_ps(k1_ptr + 6);
    x2.k = _mm256_loadu_ps(k2_ptr + 6);
    x3.k = _mm256_loadu_ps(k3_ptr + 6);
    x0.v = _mm256_loadu_epi32(v0_ptr + 6);
    x1.v = _mm256_loadu_epi32(v1_ptr + 6);
    x2.v = _mm256_loadu_epi32(v2_ptr + 6);
    x3.v = _mm256_loadu_epi32(v3_ptr + 6);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 6, y0.k);
    _mm256_storeu_ps(k1_ptr + 6, y1.k);
    _mm256_storeu_ps(k2_ptr + 6, y2.k);
    _mm256_storeu_ps(k3_ptr + 6, y3.k);
    _mm256_storeu_epi32(v0_ptr + 6, y0.v);
    _mm256_storeu_epi32(v1_ptr + 6, y1.v);
    _mm256_storeu_epi32(v2_ptr + 6, y2.v);
    _mm256_storeu_epi32(v3_ptr + 6, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 3);
    x1.k = _mm256_loadu_ps(k1_ptr + 3);
    x2.k = _mm256_loadu_ps(k2_ptr + 3);
    x3.k = _mm256_loadu_ps(k3_ptr + 3);
    x0.v = _mm256_loadu_epi32(v0_ptr + 3);
    x1.v = _mm256_loadu_epi32(v1_ptr + 3);
    x2.v = _mm256_loadu_epi32(v2_ptr + 3);
    x3.v = _mm256_loadu_epi32(v3_ptr + 3);
    y0 = _mm256_sort4x2_ps(x0);
    y1 = _mm256_sort4x2_ps(x1);
    y2 = _mm256_sort4x2_ps(x2);
    y3 = _mm256_sort4x2_ps(x3);
    _mm256_storeu_ps(k0_ptr + 3, y0.k);
    _mm256_storeu_ps(k1_ptr + 3, y1.k);
    _mm256_storeu_ps(k2_ptr + 3, y2.k);
    _mm256_storeu_ps(k3_ptr + 3, y3.k);
    _mm256_storeu_epi32(v0_ptr + 3, y0.v);
    _mm256_storeu_epi32(v1_ptr + 3, y1.v);
    _mm256_storeu_epi32(v2_ptr + 3, y2.v);
    _mm256_storeu_epi32(v3_ptr + 3, y3.v);

    return SUCCESS;
}

// shortsort elems15
__forceinline static int shortsort_n15_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 7);
    x.v = _mm256_loadu_epi32(v_ptr + 7);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 7, y.k);
    _mm256_storeu_epi32(v_ptr + 7, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 3);
    x.v = _mm256_loadu_epi32(v_ptr + 3);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 3, y.k);
    _mm256_storeu_epi32(v_ptr + 3, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 7);
    x.v = _mm256_loadu_epi32(v_ptr + 7);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 7, y.k);
    _mm256_storeu_epi32(v_ptr + 7, y.v);

    x.k = _mm256_loadu_ps(k_ptr + 4);
    x.v = _mm256_loadu_epi32(v_ptr + 4);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + 4, y.k);
    _mm256_storeu_epi32(v_ptr + 4, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems15
__forceinline static int shortsort_n4x15_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + 15;
    float* k2_ptr = k_ptr + 30;
    float* k3_ptr = k_ptr + 45;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + 15;
    uint* v2_ptr = v_ptr + 30;
    uint* v3_ptr = v_ptr + 45;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 7);
    x1.k = _mm256_loadu_ps(k1_ptr + 7);
    x2.k = _mm256_loadu_ps(k2_ptr + 7);
    x3.k = _mm256_loadu_ps(k3_ptr + 7);
    x0.v = _mm256_loadu_epi32(v0_ptr + 7);
    x1.v = _mm256_loadu_epi32(v1_ptr + 7);
    x2.v = _mm256_loadu_epi32(v2_ptr + 7);
    x3.v = _mm256_loadu_epi32(v3_ptr + 7);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 7, y0.k);
    _mm256_storeu_ps(k1_ptr + 7, y1.k);
    _mm256_storeu_ps(k2_ptr + 7, y2.k);
    _mm256_storeu_ps(k3_ptr + 7, y3.k);
    _mm256_storeu_epi32(v0_ptr + 7, y0.v);
    _mm256_storeu_epi32(v1_ptr + 7, y1.v);
    _mm256_storeu_epi32(v2_ptr + 7, y2.v);
    _mm256_storeu_epi32(v3_ptr + 7, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 3);
    x1.k = _mm256_loadu_ps(k1_ptr + 3);
    x2.k = _mm256_loadu_ps(k2_ptr + 3);
    x3.k = _mm256_loadu_ps(k3_ptr + 3);
    x0.v = _mm256_loadu_epi32(v0_ptr + 3);
    x1.v = _mm256_loadu_epi32(v1_ptr + 3);
    x2.v = _mm256_loadu_epi32(v2_ptr + 3);
    x3.v = _mm256_loadu_epi32(v3_ptr + 3);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 3, y0.k);
    _mm256_storeu_ps(k1_ptr + 3, y1.k);
    _mm256_storeu_ps(k2_ptr + 3, y2.k);
    _mm256_storeu_ps(k3_ptr + 3, y3.k);
    _mm256_storeu_epi32(v0_ptr + 3, y0.v);
    _mm256_storeu_epi32(v1_ptr + 3, y1.v);
    _mm256_storeu_epi32(v2_ptr + 3, y2.v);
    _mm256_storeu_epi32(v3_ptr + 3, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 7);
    x1.k = _mm256_loadu_ps(k1_ptr + 7);
    x2.k = _mm256_loadu_ps(k2_ptr + 7);
    x3.k = _mm256_loadu_ps(k3_ptr + 7);
    x0.v = _mm256_loadu_epi32(v0_ptr + 7);
    x1.v = _mm256_loadu_epi32(v1_ptr + 7);
    x2.v = _mm256_loadu_epi32(v2_ptr + 7);
    x3.v = _mm256_loadu_epi32(v3_ptr + 7);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 7, y0.k);
    _mm256_storeu_ps(k1_ptr + 7, y1.k);
    _mm256_storeu_ps(k2_ptr + 7, y2.k);
    _mm256_storeu_ps(k3_ptr + 7, y3.k);
    _mm256_storeu_epi32(v0_ptr + 7, y0.v);
    _mm256_storeu_epi32(v1_ptr + 7, y1.v);
    _mm256_storeu_epi32(v2_ptr + 7, y2.v);
    _mm256_storeu_epi32(v3_ptr + 7, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + 4);
    x1.k = _mm256_loadu_ps(k1_ptr + 4);
    x2.k = _mm256_loadu_ps(k2_ptr + 4);
    x3.k = _mm256_loadu_ps(k3_ptr + 4);
    x0.v = _mm256_loadu_epi32(v0_ptr + 4);
    x1.v = _mm256_loadu_epi32(v1_ptr + 4);
    x2.v = _mm256_loadu_epi32(v2_ptr + 4);
    x3.v = _mm256_loadu_epi32(v3_ptr + 4);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + 4, y0.k);
    _mm256_storeu_ps(k1_ptr + 4, y1.k);
    _mm256_storeu_ps(k2_ptr + 4, y2.k);
    _mm256_storeu_ps(k3_ptr + 4, y3.k);
    _mm256_storeu_epi32(v0_ptr + 4, y0.v);
    _mm256_storeu_epi32(v1_ptr + 4, y1.v);
    _mm256_storeu_epi32(v2_ptr + 4, y2.v);
    _mm256_storeu_epi32(v3_ptr + 4, y3.v);

    return SUCCESS;
}

// shortsort elems16
__forceinline static int shortsort_n16_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256kv x, y;

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE);
    x.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(k_ptr + AVX2_FLOAT_STRIDE, y.k);
    _mm256_storeu_epi32(v_ptr + AVX2_EPI32_STRIDE, y.v);

    x.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE / 2);
    x.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE / 2);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(k_ptr + AVX2_FLOAT_STRIDE / 2, y.k);
    _mm256_storeu_epi32(v_ptr + AVX2_EPI32_STRIDE / 2, y.v);

    x.k = _mm256_loadu_ps(k_ptr);
    x.v = _mm256_loadu_epi32(v_ptr);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(k_ptr, y.k);
    _mm256_storeu_epi32(v_ptr, y.v);

    x.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE);
    x.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(k_ptr + AVX2_FLOAT_STRIDE, y.k);
    _mm256_storeu_epi32(v_ptr + AVX2_EPI32_STRIDE, y.v);

    x.k = _mm256_loadu_ps(k_ptr + AVX2_FLOAT_STRIDE / 2);
    x.v = _mm256_loadu_epi32(v_ptr + AVX2_EPI32_STRIDE / 2);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(k_ptr + AVX2_FLOAT_STRIDE / 2, y.k);
    _mm256_storeu_epi32(v_ptr + AVX2_EPI32_STRIDE / 2, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems16
__forceinline static int shortsort_n4x16_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* k0_ptr = k_ptr;
    float* k1_ptr = k_ptr + AVX2_FLOAT_STRIDE * 2;
    float* k2_ptr = k_ptr + AVX2_FLOAT_STRIDE * 4;
    float* k3_ptr = k_ptr + AVX2_FLOAT_STRIDE * 6;

    uint* v0_ptr = v_ptr;
    uint* v1_ptr = v_ptr + AVX2_EPI32_STRIDE * 2;
    uint* v2_ptr = v_ptr + AVX2_EPI32_STRIDE * 4;
    uint* v3_ptr = v_ptr + AVX2_EPI32_STRIDE * 6;

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + AVX2_FLOAT_STRIDE);
    x1.k = _mm256_loadu_ps(k1_ptr + AVX2_FLOAT_STRIDE);
    x2.k = _mm256_loadu_ps(k2_ptr + AVX2_FLOAT_STRIDE);
    x3.k = _mm256_loadu_ps(k3_ptr + AVX2_FLOAT_STRIDE);
    x0.v = _mm256_loadu_epi32(v0_ptr + AVX2_EPI32_STRIDE);
    x1.v = _mm256_loadu_epi32(v1_ptr + AVX2_EPI32_STRIDE);
    x2.v = _mm256_loadu_epi32(v2_ptr + AVX2_EPI32_STRIDE);
    x3.v = _mm256_loadu_epi32(v3_ptr + AVX2_EPI32_STRIDE);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(k0_ptr + AVX2_FLOAT_STRIDE, y0.k);
    _mm256_storeu_ps(k1_ptr + AVX2_FLOAT_STRIDE, y1.k);
    _mm256_storeu_ps(k2_ptr + AVX2_FLOAT_STRIDE, y2.k);
    _mm256_storeu_ps(k3_ptr + AVX2_FLOAT_STRIDE, y3.k);
    _mm256_storeu_epi32(v0_ptr + AVX2_EPI32_STRIDE, y0.v);
    _mm256_storeu_epi32(v1_ptr + AVX2_EPI32_STRIDE, y1.v);
    _mm256_storeu_epi32(v2_ptr + AVX2_EPI32_STRIDE, y2.v);
    _mm256_storeu_epi32(v3_ptr + AVX2_EPI32_STRIDE, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + AVX2_FLOAT_STRIDE / 2);
    x1.k = _mm256_loadu_ps(k1_ptr + AVX2_FLOAT_STRIDE / 2);
    x2.k = _mm256_loadu_ps(k2_ptr + AVX2_FLOAT_STRIDE / 2);
    x3.k = _mm256_loadu_ps(k3_ptr + AVX2_FLOAT_STRIDE / 2);
    x0.v = _mm256_loadu_epi32(v0_ptr + AVX2_EPI32_STRIDE / 2);
    x1.v = _mm256_loadu_epi32(v1_ptr + AVX2_EPI32_STRIDE / 2);
    x2.v = _mm256_loadu_epi32(v2_ptr + AVX2_EPI32_STRIDE / 2);
    x3.v = _mm256_loadu_epi32(v3_ptr + AVX2_EPI32_STRIDE / 2);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(k0_ptr + AVX2_FLOAT_STRIDE / 2, y0.k);
    _mm256_storeu_ps(k1_ptr + AVX2_FLOAT_STRIDE / 2, y1.k);
    _mm256_storeu_ps(k2_ptr + AVX2_FLOAT_STRIDE / 2, y2.k);
    _mm256_storeu_ps(k3_ptr + AVX2_FLOAT_STRIDE / 2, y3.k);
    _mm256_storeu_epi32(v0_ptr + AVX2_EPI32_STRIDE / 2, y0.v);
    _mm256_storeu_epi32(v1_ptr + AVX2_EPI32_STRIDE / 2, y1.v);
    _mm256_storeu_epi32(v2_ptr + AVX2_EPI32_STRIDE / 2, y2.v);
    _mm256_storeu_epi32(v3_ptr + AVX2_EPI32_STRIDE / 2, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr);
    x1.k = _mm256_loadu_ps(k1_ptr);
    x2.k = _mm256_loadu_ps(k2_ptr);
    x3.k = _mm256_loadu_ps(k3_ptr);
    x0.v = _mm256_loadu_epi32(v0_ptr);
    x1.v = _mm256_loadu_epi32(v1_ptr);
    x2.v = _mm256_loadu_epi32(v2_ptr);
    x3.v = _mm256_loadu_epi32(v3_ptr);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(k0_ptr, y0.k);
    _mm256_storeu_ps(k1_ptr, y1.k);
    _mm256_storeu_ps(k2_ptr, y2.k);
    _mm256_storeu_ps(k3_ptr, y3.k);
    _mm256_storeu_epi32(v0_ptr, y0.v);
    _mm256_storeu_epi32(v1_ptr, y1.v);
    _mm256_storeu_epi32(v2_ptr, y2.v);
    _mm256_storeu_epi32(v3_ptr, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + AVX2_FLOAT_STRIDE);
    x1.k = _mm256_loadu_ps(k1_ptr + AVX2_FLOAT_STRIDE);
    x2.k = _mm256_loadu_ps(k2_ptr + AVX2_FLOAT_STRIDE);
    x3.k = _mm256_loadu_ps(k3_ptr + AVX2_FLOAT_STRIDE);
    x0.v = _mm256_loadu_epi32(v0_ptr + AVX2_EPI32_STRIDE);
    x1.v = _mm256_loadu_epi32(v1_ptr + AVX2_EPI32_STRIDE);
    x2.v = _mm256_loadu_epi32(v2_ptr + AVX2_EPI32_STRIDE);
    x3.v = _mm256_loadu_epi32(v3_ptr + AVX2_EPI32_STRIDE);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(k0_ptr + AVX2_FLOAT_STRIDE, y0.k);
    _mm256_storeu_ps(k1_ptr + AVX2_FLOAT_STRIDE, y1.k);
    _mm256_storeu_ps(k2_ptr + AVX2_FLOAT_STRIDE, y2.k);
    _mm256_storeu_ps(k3_ptr + AVX2_FLOAT_STRIDE, y3.k);
    _mm256_storeu_epi32(v0_ptr + AVX2_EPI32_STRIDE, y0.v);
    _mm256_storeu_epi32(v1_ptr + AVX2_EPI32_STRIDE, y1.v);
    _mm256_storeu_epi32(v2_ptr + AVX2_EPI32_STRIDE, y2.v);
    _mm256_storeu_epi32(v3_ptr + AVX2_EPI32_STRIDE, y3.v);

    x0.k = _mm256_loadu_ps(k0_ptr + AVX2_FLOAT_STRIDE / 2);
    x1.k = _mm256_loadu_ps(k1_ptr + AVX2_FLOAT_STRIDE / 2);
    x2.k = _mm256_loadu_ps(k2_ptr + AVX2_FLOAT_STRIDE / 2);
    x3.k = _mm256_loadu_ps(k3_ptr + AVX2_FLOAT_STRIDE / 2);
    x0.v = _mm256_loadu_epi32(v0_ptr + AVX2_EPI32_STRIDE / 2);
    x1.v = _mm256_loadu_epi32(v1_ptr + AVX2_EPI32_STRIDE / 2);
    x2.v = _mm256_loadu_epi32(v2_ptr + AVX2_EPI32_STRIDE / 2);
    x3.v = _mm256_loadu_epi32(v3_ptr + AVX2_EPI32_STRIDE / 2);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(k0_ptr + AVX2_FLOAT_STRIDE / 2, y0.k);
    _mm256_storeu_ps(k1_ptr + AVX2_FLOAT_STRIDE / 2, y1.k);
    _mm256_storeu_ps(k2_ptr + AVX2_FLOAT_STRIDE / 2, y2.k);
    _mm256_storeu_ps(k3_ptr + AVX2_FLOAT_STRIDE / 2, y3.k);
    _mm256_storeu_epi32(v0_ptr + AVX2_EPI32_STRIDE / 2, y0.v);
    _mm256_storeu_epi32(v1_ptr + AVX2_EPI32_STRIDE / 2, y1.v);
    _mm256_storeu_epi32(v2_ptr + AVX2_EPI32_STRIDE / 2, y2.v);
    _mm256_storeu_epi32(v3_ptr + AVX2_EPI32_STRIDE / 2, y3.v);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

// longsort elems 32...63
__forceinline static int longsort_n32to63_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE * 4 || n >= AVX2_FLOAT_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    paracombsort_p2x8_s(n, v_ptr, k_ptr);
    backtracksort_p8_s(n, v_ptr, k_ptr);

    batchsort_p8_s(n, v_ptr, k_ptr);
    scansort_p8_s(n, v_ptr, k_ptr);

    return SUCCESS;
}

// longsort elems 64+
__forceinline static int longsort_n64plus_s(const uint n, uint* __restrict v_ptr, float* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    uint h;

    for (h = (uint)(n * 10uLL / 13uLL); h > 33; h = (uint)(h * 10uLL / 13uLL)) {
        combsort_h33plus_s(n, h, v_ptr, k_ptr);
    }
    if (h >= 32) {
        combsort_h32_s(n, v_ptr, k_ptr);
        h = h * 10 / 13;
    }
    for (; h > 25; h = h * 10 / 13) {
        combsort_h25to31_s(n, h, v_ptr, k_ptr);
    }
    if (h >= 24) {
        combsort_h24_s(n, v_ptr, k_ptr);
        h = h * 10 / 13;
    }
    for (; h > 17; h = h * 10 / 13) {
        combsort_h17to23_s(n, h, v_ptr, k_ptr);
    }
    if (h >= 16) {
        combsort_h16_s(n, v_ptr, k_ptr);
        h = h * 10 / 13;
    }
    for (; h > 9; h = h * 10 / 13) {
        combsort_h9to15_s(n, h, v_ptr, k_ptr);
    }

    paracombsort_p2x8_s(n, v_ptr, k_ptr);

    batchsort_p8_s(n, v_ptr, k_ptr);
    scansort_p8_s(n, v_ptr, k_ptr);

    return SUCCESS;
}

#pragma endregion longsort

#pragma region sort

int sortwithkeydsc_maxnan_s2_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 2) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)k_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= AVX2_FLOAT_STRIDE * 4 / 2) {
            _mm256_loadu_x4_ps(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);
            y2 = _mm256_sort4x2_ps(x2);
            y3 = _mm256_sort4x2_ps(x3);

            _mm256_storeu_x4_ps(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_FLOAT_STRIDE * 4;
            v_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4 / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x2_ps(k_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);

            _mm256_storeu_x2_ps(k_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_FLOAT_STRIDE * 2;
            v_ptr += AVX2_EPI32_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_loadu_x1_ps(k_ptr, x0.k);
            _mm256_loadu_x1_epi32(v_ptr, x0.v);

            y0 = _mm256_sort4x2_ps(x0);

            _mm256_storeu_x1_ps(k_ptr, y0.k);
            _mm256_storeu_x1_epi32(v_ptr, y0.v);

            k_ptr += AVX2_FLOAT_STRIDE;
            v_ptr += AVX2_EPI32_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
    }
    else {
        while (r >= AVX2_FLOAT_STRIDE * 4 / 2) {
            _mm256_load_x4_ps(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_load_x4_epi32(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);
            y2 = _mm256_sort4x2_ps(x2);
            y3 = _mm256_sort4x2_ps(x3);

            _mm256_stream_x4_ps(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_stream_x4_epi32(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_FLOAT_STRIDE * 4;
            v_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4 / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x2_ps(k_ptr, x0.k, x1.k);
            _mm256_load_x2_epi32(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);

            _mm256_stream_x2_ps(k_ptr, y0.k, y1.k);
            _mm256_stream_x2_epi32(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_FLOAT_STRIDE * 2;
            v_ptr += AVX2_EPI32_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_load_x1_ps(k_ptr, x0.k);
            _mm256_load_x1_epi32(v_ptr, x0.v);

            y0 = _mm256_sort4x2_ps(x0);

            _mm256_stream_x1_ps(k_ptr, y0.k);
            _mm256_stream_x1_epi32(v_ptr, y0.v);

            k_ptr += AVX2_FLOAT_STRIDE;
            v_ptr += AVX2_EPI32_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps((r * 2) & AVX2_FLOAT_REMAIN_MASK);

        _mm256_maskload_x1_ps(k_ptr, x0.k, mask);
        _mm256_maskload_x1_epi32(v_ptr, x0.v, mask);

        y0 = _mm256_sort4x2_ps(x0);

        _mm256_maskstore_x1_ps(k_ptr, y0.k, mask);
        _mm256_maskstore_x1_epi32(v_ptr, y0.v, mask);
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s3_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 3) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 9) {
        x0.k = _mm256_loadu_ps(k_ptr);
        x1.k = _mm256_loadu_ps(k_ptr + 6);
        x2.k = _mm256_loadu_ps(k_ptr + 12);
        x3.k = _mm256_loadu_ps(k_ptr + 18);
        x0.v = _mm256_loadu_epi32(v_ptr);
        x1.v = _mm256_loadu_epi32(v_ptr + 6);
        x2.v = _mm256_loadu_epi32(v_ptr + 12);
        x3.v = _mm256_loadu_epi32(v_ptr + 18);

        y0 = _mm256_sort2x3_ps(x0);
        y1 = _mm256_sort2x3_ps(x1);
        y2 = _mm256_sort2x3_ps(x2);
        y3 = _mm256_sort2x3_ps(x3);

        _mm256_storeu_ps(k_ptr, y0.k);
        _mm256_storeu_ps(k_ptr + 6, y1.k);
        _mm256_storeu_ps(k_ptr + 12, y2.k);
        _mm256_storeu_ps(k_ptr + 18, y3.k);
        _mm256_storeu_epi32(v_ptr, y0.v);
        _mm256_storeu_epi32(v_ptr + 6, y1.v);
        _mm256_storeu_epi32(v_ptr + 12, y2.v);
        _mm256_storeu_epi32(v_ptr + 18, y3.v);

        k_ptr += AVX2_FLOAT_STRIDE * 3;
        v_ptr += AVX2_EPI32_STRIDE * 3;
        r -= 8;
    }
    while (r >= 2) {
        const __m256i mask = _mm256_setmask_ps(6);

        x0.k = _mm256_maskload_ps(k_ptr, mask);
        x0.v = _mm256_maskload_epi32(v_ptr, mask);

        y0 = _mm256_sort2x3_ps(x0);

        _mm256_maskstore_ps(k_ptr, mask, y0.k);
        _mm256_maskstore_epi32(v_ptr, mask, y0.v);

        k_ptr += AVX2_FLOAT_STRIDE * 3 / 4;
        v_ptr += AVX2_EPI32_STRIDE * 3 / 4;
        r -= 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(3);

        x0.k = _mm256_maskload_ps(k_ptr, mask);
        x0.v = _mm256_maskload_epi32(v_ptr, mask);

        y0 = _mm256_sort2x3_ps(x0);

        _mm256_maskstore_ps(k_ptr, mask, y0.k);
        _mm256_maskstore_epi32(v_ptr, mask, y0.v);
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s4_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 4) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)k_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x4_ps(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);
            y2 = _mm256_sort2x4_ps(x2);
            y3 = _mm256_sort2x4_ps(x3);

            _mm256_storeu_x4_ps(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_FLOAT_STRIDE * 4;
            v_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_loadu_x2_ps(k_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);

            _mm256_storeu_x2_ps(k_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_FLOAT_STRIDE * 2;
            v_ptr += AVX2_EPI32_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE / 4) {
            _mm256_loadu_x1_ps(k_ptr, x0.k);
            _mm256_loadu_x1_epi32(v_ptr, x0.v);

            y0 = _mm256_sort2x4_ps(x0);

            _mm256_storeu_x1_ps(k_ptr, y0.k);
            _mm256_storeu_x1_epi32(v_ptr, y0.v);

            k_ptr += AVX2_FLOAT_STRIDE;
            v_ptr += AVX2_EPI32_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 4;
        }
    }
    else {
        while (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x4_ps(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_load_x4_epi32(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);
            y2 = _mm256_sort2x4_ps(x2);
            y3 = _mm256_sort2x4_ps(x3);

            _mm256_stream_x4_ps(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_stream_x4_epi32(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_FLOAT_STRIDE * 4;
            v_ptr += AVX2_EPI32_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_load_x2_ps(k_ptr, x0.k, x1.k);
            _mm256_load_x2_epi32(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);

            _mm256_stream_x2_ps(k_ptr, y0.k, y1.k);
            _mm256_stream_x2_epi32(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_FLOAT_STRIDE * 2;
            v_ptr += AVX2_EPI32_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE / 4) {
            _mm256_load_x1_ps(k_ptr, x0.k);
            _mm256_load_x1_epi32(v_ptr, x0.v);

            y0 = _mm256_sort2x4_ps(x0);

            _mm256_stream_x1_ps(k_ptr, y0.k);
            _mm256_stream_x1_epi32(v_ptr, y0.v);

            k_ptr += AVX2_FLOAT_STRIDE;
            v_ptr += AVX2_EPI32_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 4;
        }
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(4);

        _mm256_maskload_x1_ps(k_ptr, x0.k, mask);
        _mm256_maskload_x1_epi32(v_ptr, x0.v, mask);

        y0 = _mm256_sort2x4_ps(x0);

        _mm256_maskstore_x1_ps(k_ptr, y0.k, mask);
        _mm256_maskstore_x1_epi32(v_ptr, y0.v, mask);
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s5_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 5) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0.k = _mm256_loadu_ps(k_ptr);
        x1.k = _mm256_loadu_ps(k_ptr + 5);
        x2.k = _mm256_loadu_ps(k_ptr + 10);
        x3.k = _mm256_loadu_ps(k_ptr + 15);
        x0.v = _mm256_loadu_epi32(v_ptr);
        x1.v = _mm256_loadu_epi32(v_ptr + 5);
        x2.v = _mm256_loadu_epi32(v_ptr + 10);
        x3.v = _mm256_loadu_epi32(v_ptr + 15);

        y0 = _mm256_sort1x5_ps(x0);
        y1 = _mm256_sort1x5_ps(x1);
        y2 = _mm256_sort1x5_ps(x2);
        y3 = _mm256_sort1x5_ps(x3);

        _mm256_storeu_ps(k_ptr, y0.k);
        _mm256_storeu_ps(k_ptr + 5, y1.k);
        _mm256_storeu_ps(k_ptr + 10, y2.k);
        _mm256_storeu_ps(k_ptr + 15, y3.k);
        _mm256_storeu_epi32(v_ptr, y0.v);
        _mm256_storeu_epi32(v_ptr + 5, y1.v);
        _mm256_storeu_epi32(v_ptr + 10, y2.v);
        _mm256_storeu_epi32(v_ptr + 15, y3.v);

        k_ptr += 20;
        v_ptr += 20;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(5);

        x0.k = _mm256_maskload_ps(k_ptr, mask);
        x0.v = _mm256_maskload_epi32(v_ptr, mask);

        y0 = _mm256_sort1x5_ps(x0);

        _mm256_maskstore_ps(k_ptr, mask, y0.k);
        _mm256_maskstore_epi32(v_ptr, mask, y0.v);

        k_ptr += 5;
        v_ptr += 5;
        r -= 1;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s6_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 6) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0.k = _mm256_loadu_ps(k_ptr);
        x1.k = _mm256_loadu_ps(k_ptr + 6);
        x2.k = _mm256_loadu_ps(k_ptr + 12);
        x3.k = _mm256_loadu_ps(k_ptr + 18);
        x0.v = _mm256_loadu_epi32(v_ptr);
        x1.v = _mm256_loadu_epi32(v_ptr + 6);
        x2.v = _mm256_loadu_epi32(v_ptr + 12);
        x3.v = _mm256_loadu_epi32(v_ptr + 18);

        y0 = _mm256_sort1x6_ps(x0);
        y1 = _mm256_sort1x6_ps(x1);
        y2 = _mm256_sort1x6_ps(x2);
        y3 = _mm256_sort1x6_ps(x3);

        _mm256_storeu_ps(k_ptr, y0.k);
        _mm256_storeu_ps(k_ptr + 6, y1.k);
        _mm256_storeu_ps(k_ptr + 12, y2.k);
        _mm256_storeu_ps(k_ptr + 18, y3.k);
        _mm256_storeu_epi32(v_ptr, y0.v);
        _mm256_storeu_epi32(v_ptr + 6, y1.v);
        _mm256_storeu_epi32(v_ptr + 12, y2.v);
        _mm256_storeu_epi32(v_ptr + 18, y3.v);

        k_ptr += 24;
        v_ptr += 24;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(6);

        x0.k = _mm256_maskload_ps(k_ptr, mask);
        x0.v = _mm256_maskload_epi32(v_ptr, mask);

        y0 = _mm256_sort1x6_ps(x0);

        _mm256_maskstore_ps(k_ptr, mask, y0.k);
        _mm256_maskstore_epi32(v_ptr, mask, y0.v);

        k_ptr += 6;
        v_ptr += 6;
        r -= 1;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s7_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 7) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0.k = _mm256_loadu_ps(k_ptr);
        x1.k = _mm256_loadu_ps(k_ptr + 7);
        x2.k = _mm256_loadu_ps(k_ptr + 14);
        x3.k = _mm256_loadu_ps(k_ptr + 21);
        x0.v = _mm256_loadu_epi32(v_ptr);
        x1.v = _mm256_loadu_epi32(v_ptr + 7);
        x2.v = _mm256_loadu_epi32(v_ptr + 14);
        x3.v = _mm256_loadu_epi32(v_ptr + 21);

        y0 = _mm256_sort1x7_ps(x0);
        y1 = _mm256_sort1x7_ps(x1);
        y2 = _mm256_sort1x7_ps(x2);
        y3 = _mm256_sort1x7_ps(x3);

        _mm256_storeu_ps(k_ptr, y0.k);
        _mm256_storeu_ps(k_ptr + 7, y1.k);
        _mm256_storeu_ps(k_ptr + 14, y2.k);
        _mm256_storeu_ps(k_ptr + 21, y3.k);
        _mm256_storeu_epi32(v_ptr, y0.v);
        _mm256_storeu_epi32(v_ptr + 7, y1.v);
        _mm256_storeu_epi32(v_ptr + 14, y2.v);
        _mm256_storeu_epi32(v_ptr + 21, y3.v);

        k_ptr += 28;
        v_ptr += 28;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(7);

        x0.k = _mm256_maskload_ps(k_ptr, mask);
        x0.v = _mm256_maskload_epi32(v_ptr, mask);

        y0 = _mm256_sort1x7_ps(x0);

        _mm256_maskstore_ps(k_ptr, mask, y0.k);
        _mm256_maskstore_epi32(v_ptr, mask, y0.v);

        k_ptr += 7;
        v_ptr += 7;
        r -= 1;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s8_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 8) {
        return FAILURE_BADPARAM;
    }

    __m256kv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)k_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= 4) {
            _mm256_loadu_x4_ps(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi32(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);
            y3 = _mm256_sort_ps(x3);

            _mm256_storeu_x4_ps(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi32(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_FLOAT_STRIDE * 4;
            v_ptr += AVX2_EPI32_STRIDE * 4;
            r -= 4;
        }
        if (r >= 2) {
            _mm256_loadu_x2_ps(k_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi32(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);

            _mm256_storeu_x2_ps(k_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi32(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_FLOAT_STRIDE * 2;
            v_ptr += AVX2_EPI32_STRIDE * 2;
            r -= 2;
        }
        if (r > 0) {
            _mm256_loadu_x1_ps(k_ptr, x0.k);
            _mm256_loadu_x1_epi32(v_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(k_ptr, y0.k);
            _mm256_storeu_x1_epi32(v_ptr, y0.v);
        }
    }
    else {
        while (r >= 4) {
            _mm256_load_x4_ps(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_load_x4_epi32(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);
            y3 = _mm256_sort_ps(x3);

            _mm256_stream_x4_ps(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_stream_x4_epi32(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_FLOAT_STRIDE * 4;
            v_ptr += AVX2_EPI32_STRIDE * 4;
            r -= 4;
        }
        if (r >= 2) {
            _mm256_load_x2_ps(k_ptr, x0.k, x1.k);
            _mm256_load_x2_epi32(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);

            _mm256_stream_x2_ps(k_ptr, y0.k, y1.k);
            _mm256_stream_x2_epi32(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_FLOAT_STRIDE * 2;
            v_ptr += AVX2_EPI32_STRIDE * 2;
            r -= 2;
        }
        if (r > 0) {
            _mm256_load_x1_ps(k_ptr, x0.k);
            _mm256_load_x1_epi32(v_ptr, x0.v);

            y0 = _mm256_sort_ps(x0);

            _mm256_stream_x1_ps(k_ptr, y0.k);
            _mm256_stream_x1_epi32(v_ptr, y0.v);
        }
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s9_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 9) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x9_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n9_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s10_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 10) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x10_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n10_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s11_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 11) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x11_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n11_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s12_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 12) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x12_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n12_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s13_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 13) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x13_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n13_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s14_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 14) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x14_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n14_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s15_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != 15) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x15_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n15_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s16_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s != AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x16_s(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n16_s(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s17to31_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s <= AVX2_FLOAT_STRIDE * 2 || s >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        permcombsort_p4x8_s(s, v_ptr, k_ptr);
        scansort_p8_s(s, v_ptr, k_ptr);
        scansort_p8_s(s, v_ptr + s, k_ptr + s);
        scansort_p8_s(s, v_ptr + s * 2, k_ptr + s * 2);
        scansort_p8_s(s, v_ptr + s * 3, k_ptr + s * 3);

        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        permcombsort_p8_s(s, v_ptr, k_ptr);
        scansort_p8_s(s, v_ptr, k_ptr);

        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s32to63_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s < AVX2_FLOAT_STRIDE * 4 || s >= AVX2_FLOAT_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < n; i++) {
        longsort_n32to63_s(s, v_ptr, k_ptr);

        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_maxnan_s64plus_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s < AVX2_FLOAT_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < n; i++) {
        longsort_n64plus_s(s, v_ptr, k_ptr);

        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

#pragma endregion sort