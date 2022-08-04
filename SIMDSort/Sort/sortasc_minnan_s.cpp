#include "sort.h"
#include "../Inline/inline_cmp_s.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma region needs swap

// needs swap (sort order definition)
__forceinline static __m256 _mm256_needsswap_ps(__m256 x, __m256 y) {
    return _mm256_cmpgt_minnan_ps(x, y);
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
__forceinline static __m256 _mm256_sort4x2_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// sort batches2 x elems3
__forceinline static __m256 _mm256_sort2x3_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 2, 4, 3, 5, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 3, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 3, 3, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 4, 4, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches2 x elems4
__forceinline static __m256 _mm256_sort2x4_ps(__m256 x) {
    __m256 y, c;

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_ABCD);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_ABBA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches1 x elems5
__forceinline static __m256 _mm256_sort1x5_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 3, 2, 4, 5, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 4, 3, 5, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 2, 4, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 3, 5, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches1 x elems6
__forceinline static __m256 _mm256_sort1x6_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(5, 2, 1, 4, 3, 0, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 1, 1, 3, 3, 0, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort batches1 x elems7
__forceinline static __m256 _mm256_sort1x7_ps(__m256 x) {
    const __m256i perm0 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(0, 2, 1, 4, 3, 6, 5, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 1, 1, 3, 3, 5, 5, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort elems8
__forceinline static __m256 _mm256_sort_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i permcmp = _mm256_setr_epi32(0, 1, 2, 3, 3, 2, 1, 0);

    __m256 y, c;

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_ABCD);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_ABBA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_BADC);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_BABA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

// sort elems8 (ho, lo sorted)
__forceinline static __m256 _mm256_halfsort_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
    const __m256i permcmp = _mm256_setr_epi32(0, 1, 2, 3, 3, 2, 1, 0);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_BADC);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_BABA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

#pragma endregion horizontal sort

#pragma region cmp and swap

// compare and swap
__forceinline static void _mm256_cmpswap_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 swaps = _mm256_needsswap_ps(a, b);

    x = _mm256_blendv_ps(a, b, swaps);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(swaps));
}

// compare and swap
__forceinline static uint _mm256_cmpswap_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 swaps = _mm256_needsswap_ps(a, b);

    uint index = _mm256_movemask_ps(swaps);

    x = _mm256_blendv_ps(a, b, swaps);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(swaps));

    return index;
}

// compare and swap with permutate
__forceinline static void _mm256_cmpswap_withperm_ps(__m256 a, __m256 b, __m256& x, __m256& y) {

    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_permute_ps(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_permute_ps(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_permute_ps(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x), _MM_PERM_BADC));
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_permute_ps(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_permute_ps(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);

    a = _mm256_permute_ps(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_ps(a, b, x, y);
}

#pragma endregion cmp and swap

#pragma region combsort

// combsort h=9...15
static int combsort_h9to15_s(const uint n, const uint h, float* v_ptr) {
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

    __m256 a0, a1, b0, b1;
    __m256 x0, x1, y0, y1;

    if (e > 0) {
        _mm256_maskload_x2_ps(v_ptr, a0, a1, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x2_ps(v_ptr + i + h, b0, b1, mask);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);

            _mm256_maskstore_x2_ps(v_ptr + i, x0, x1, mask);

            a0 = y0;
            a1 = y1;
        }
        _mm256_maskstore_x2_ps(v_ptr + i, a0, a1, mask);
    }
    {
        _mm256_maskload_x2_ps(v_ptr + e, a0, a1, mask);
        _mm256_maskload_x2_ps(v_ptr + e + h, b0, b1, mask);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        _mm256_maskstore_x2_ps(v_ptr + e, x0, x1, mask);
        _mm256_maskstore_x2_ps(v_ptr + e + h, y0, y1, mask);
    }

    return SUCCESS;
}

// combsort h=16
static int combsort_h16_s(const uint n, float* v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 4;

    __m256 a0, a1, b0, b1;
    __m256 x0, x1, y0, y1;

    if (e > 0) {
        _mm256_loadu_x2_ps(v_ptr, a0, a1);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(v_ptr + i + AVX2_FLOAT_STRIDE * 2, b0, b1);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);

            _mm256_storeu_x2_ps(v_ptr + i, x0, x1);

            a0 = y0;
            a1 = y1;
        }
        _mm256_storeu_x2_ps(v_ptr + i, a0, a1);
    }
    {
        _mm256_loadu_x2_ps(v_ptr + e, a0, a1);
        _mm256_loadu_x2_ps(v_ptr + e + AVX2_FLOAT_STRIDE * 2, b0, b1);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        _mm256_storeu_x2_ps(v_ptr + e, x0, x1);
        _mm256_storeu_x2_ps(v_ptr + e + AVX2_FLOAT_STRIDE * 2, y0, y1);
    }

    return SUCCESS;
}

// combsort h=17...23
static int combsort_h17to23_s(const uint n, const uint h, float* v_ptr) {
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

    __m256 a0, a1, a2, b0, b1, b2;
    __m256 x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_maskload_x3_ps(v_ptr, a0, a1, a2, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x3_ps(v_ptr + i + h, b0, b1, b2, mask);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);

            _mm256_maskstore_x3_ps(v_ptr + i, x0, x1, x2, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_maskstore_x3_ps(v_ptr + i, a0, a1, a2, mask);
    }
    {
        _mm256_maskload_x3_ps(v_ptr + e, a0, a1, a2, mask);
        _mm256_maskload_x3_ps(v_ptr + e + h, b0, b1, b2, mask);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);

        _mm256_maskstore_x3_ps(v_ptr + e, x0, x1, x2, mask);
        _mm256_maskstore_x3_ps(v_ptr + e + h, y0, y1, y2, mask);
    }

    return SUCCESS;
}

// combsort h=24
static int combsort_h24_s(const uint n, float* v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 6) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 6;

    __m256 a0, a1, a2, b0, b1, b2;
    __m256 x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_loadu_x3_ps(v_ptr, a0, a1, a2);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(v_ptr + i + AVX2_FLOAT_STRIDE * 3, b0, b1, b2);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);

            _mm256_storeu_x3_ps(v_ptr + i, x0, x1, x2);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_storeu_x3_ps(v_ptr + i, a0, a1, a2);
    }
    {
        _mm256_loadu_x3_ps(v_ptr + e, a0, a1, a2);
        _mm256_loadu_x3_ps(v_ptr + e + AVX2_FLOAT_STRIDE * 3, b0, b1, b2);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);

        _mm256_storeu_x3_ps(v_ptr + e, x0, x1, x2);
        _mm256_storeu_x3_ps(v_ptr + e + AVX2_FLOAT_STRIDE * 3, y0, y1, y2);
    }

    return SUCCESS;
}

// combsort h=25...31
static int combsort_h25to31_s(const uint n, const uint h, float* v_ptr) {
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

    __m256 a0, a1, a2, a3, b0, b1, b2, b3;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_maskload_x4_ps(v_ptr, a0, a1, a2, a3, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x4_ps(v_ptr + i + h, b0, b1, b2, b3, mask);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);
            _mm256_cmpswap_ps(a3, b3, x3, y3);

            _mm256_maskstore_x4_ps(v_ptr + i, x0, x1, x2, x3, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_maskstore_x4_ps(v_ptr + i, a0, a1, a2, a3, mask);
    }
    {
        _mm256_maskload_x4_ps(v_ptr + e, a0, a1, a2, a3, mask);
        _mm256_maskload_x4_ps(v_ptr + e + h, b0, b1, b2, b3, mask);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_maskstore_x4_ps(v_ptr + e, x0, x1, x2, x3, mask);
        _mm256_maskstore_x4_ps(v_ptr + e + h, y0, y1, y2, y3, mask);
    }

    return SUCCESS;
}

// combsort h=32
static int combsort_h32_s(const uint n, float* v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 8) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 8;

    __m256 a0, a1, a2, a3, b0, b1, b2, b3;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_loadu_x4_ps(v_ptr, a0, a1, a2, a3);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(v_ptr + i + AVX2_FLOAT_STRIDE * 4, b0, b1, b2, b3);

            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
            _mm256_cmpswap_ps(a2, b2, x2, y2);
            _mm256_cmpswap_ps(a3, b3, x3, y3);

            _mm256_storeu_x4_ps(v_ptr + i, x0, x1, x2, x3);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_storeu_x4_ps(v_ptr + i, a0, a1, a2, a3);
    }
    {
        _mm256_loadu_x4_ps(v_ptr + e, a0, a1, a2, a3);
        _mm256_loadu_x4_ps(v_ptr + e + AVX2_FLOAT_STRIDE * 4, b0, b1, b2, b3);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(v_ptr + e, x0, x1, x2, x3);
        _mm256_storeu_x4_ps(v_ptr + e + AVX2_FLOAT_STRIDE * 4, y0, y1, y2, y3);
    }

    return SUCCESS;
}

// combsort h>32
static int combsort_h33plus_s(const uint n, const uint h, float* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h + AVX2_FLOAT_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_FLOAT_STRIDE * 4;

    __m256 a0, a1, a2, a3, b0, b1, b2, b3;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    for (uint i = 0; i < e; i += AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(v_ptr + i, a0, a1, a2, a3);
        _mm256_loadu_x4_ps(v_ptr + i + h, b0, b1, b2, b3);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(v_ptr + i, x0, x1, x2, x3);
        _mm256_storeu_x4_ps(v_ptr + i + h, y0, y1, y2, y3);
    }
    {
        _mm256_loadu_x4_ps(v_ptr + e, a0, a1, a2, a3);
        _mm256_loadu_x4_ps(v_ptr + e + h, b0, b1, b2, b3);

        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);
        _mm256_cmpswap_ps(a2, b2, x2, y2);
        _mm256_cmpswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(v_ptr + e, x0, x1, x2, x3);
        _mm256_storeu_x4_ps(v_ptr + e + h, y0, y1, y2, y3);
    }

    return SUCCESS;
}

#pragma endregion combsort

#pragma region paracombsort

// paracombsort 2x8
static int paracombsort_p2x8_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    const uint e = n - AVX2_FLOAT_STRIDE * 4;
    const uint c = n % AVX2_FLOAT_STRIDE;

    __m256 a0, a1, b0, b1;
    __m256 x0, x1, y0, y1;

    for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
        a1 = _mm256_loadu_ps(v_ptr + i);
        b1 = _mm256_loadu_ps(v_ptr + i + AVX2_FLOAT_STRIDE);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        a0 = x1;
        a1 = y1;
        b1 = _mm256_loadu_ps(v_ptr + i + AVX2_FLOAT_STRIDE * 2);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        b0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x1), _MM_PERM_ABDC));
        a1 = y1;
        b1 = _mm256_loadu_ps(v_ptr + i + AVX2_FLOAT_STRIDE * 3);
        _mm256_cmpswap_ps(a0, b0, x0, y0);
        _mm256_cmpswap_ps(a1, b1, x1, y1);

        for (j = i; j + AVX2_FLOAT_STRIDE <= e; j += AVX2_FLOAT_STRIDE) {
            _mm256_storeu_ps(v_ptr + j, x0);
            a0 = y0;
            b0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x1), _MM_PERM_ABDC));
            a1 = y1;
            b1 = _mm256_loadu_ps(v_ptr + j + AVX2_FLOAT_STRIDE * 4);
            _mm256_cmpswap_ps(a0, b0, x0, y0);
            _mm256_cmpswap_ps(a1, b1, x1, y1);
        }

        _mm256_storeu_ps(v_ptr + j, x0);
        a0 = y0;
        b0 = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(x1), _MM_PERM_ABDC));
        _mm256_cmpswap_ps(a0, b0, x0, y0);
        j += AVX2_FLOAT_STRIDE;

        _mm256_storeu_ps(v_ptr + j, x0);
        a0 = y0;
        b0 = y1;
        _mm256_cmpswap_ps(a0, b0, x0, y0);
        j += AVX2_FLOAT_STRIDE;

        _mm256_storeu_ps(v_ptr + j, x0);
        _mm256_storeu_ps(v_ptr + j + AVX2_FLOAT_STRIDE, y0);
    }

    return SUCCESS;
}

#pragma endregion paracombsort

#pragma region backtracksort

// backtracksort 8 elems wise
__forceinline static int backtracksort_p8_s(const uint n, float* v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 2) {
        return SUCCESS;
    }

    uint i = 0, e = n - AVX2_FLOAT_STRIDE * 2;

    __m256 a, b, x, y;
    a = _mm256_loadu_ps(v_ptr);
    b = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE);

    if (e <= 0) {
        _mm256_cmpswap_ps(a, b, x, y);

        _mm256_storeu_ps(v_ptr, x);
        _mm256_storeu_ps(v_ptr + AVX2_FLOAT_STRIDE, y);

        return SUCCESS;
    }

    while (true) {
        int indexes = _mm256_cmpswap_indexed_ps(a, b, x, y);

        if (indexes > 0) {
            _mm256_storeu_ps(v_ptr + i, x);
            _mm256_storeu_ps(v_ptr + i + AVX2_FLOAT_STRIDE, y);

            if (i >= AVX2_FLOAT_STRIDE) {
                i -= AVX2_FLOAT_STRIDE;
                a = _mm256_loadu_ps(v_ptr + i);
                b = x;
                continue;
            }
            else if (i > 0) {
                i = 0;
                a = _mm256_loadu_ps(v_ptr);
                b = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE);
                continue;
            }
            else {
                i = AVX2_FLOAT_STRIDE;
                if (i <= e) {
                    a = y;
                    b = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE * 2);
                    continue;
                }
            }
        }
        else if (i < e) {
            i += AVX2_FLOAT_STRIDE;

            if (i <= e) {
                a = b;
                b = _mm256_loadu_ps(v_ptr + i + AVX2_FLOAT_STRIDE);
                continue;
            }
        }
        else {
            break;
        }

        i = e;
        a = _mm256_loadu_ps(v_ptr + i);
        b = _mm256_loadu_ps(v_ptr + i + AVX2_FLOAT_STRIDE);
    }

    return SUCCESS;
}

#pragma endregion backtracksort

#pragma region batchsort

// batchsort 8 elems wise
static int batchsort_p8_s(const uint n, float* v_ptr) {
    if (n < AVX2_FLOAT_STRIDE) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE;

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    float* const ve_ptr = v_ptr + e;

    {
        float* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);
            y3 = _mm256_sort_ps(x3);

            _mm256_storeu_x4_ps(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(vc_ptr, x0, x1, x2);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);

            _mm256_storeu_x3_ps(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(vc_ptr, x0, x1);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);

            _mm256_storeu_x2_ps(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(vc_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ve_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ve_ptr, y0);
        }
    }
    {
        float* vc_ptr = v_ptr + AVX2_FLOAT_STRIDE / 2;
        uint r = n - AVX2_FLOAT_STRIDE / 2;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);
            y3 = _mm256_halfsort_ps(x3);

            _mm256_storeu_x4_ps(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(vc_ptr, x0, x1, x2);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);

            _mm256_storeu_x3_ps(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(vc_ptr, x0, x1);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);

            _mm256_storeu_x2_ps(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(vc_ptr, x0);

            y0 = _mm256_halfsort_ps(x0);

            _mm256_storeu_x1_ps(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ve_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ve_ptr, y0);
        }
    }
    {
        float* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);
            y3 = _mm256_halfsort_ps(x3);

            _mm256_storeu_x4_ps(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(vc_ptr, x0, x1, x2);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);

            _mm256_storeu_x3_ps(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(vc_ptr, x0, x1);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);

            _mm256_storeu_x2_ps(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(vc_ptr, x0);

            y0 = _mm256_halfsort_ps(x0);

            _mm256_storeu_x1_ps(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ve_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ve_ptr, y0);
        }
    }
    {
        float* vc_ptr = v_ptr + AVX2_FLOAT_STRIDE / 2;
        uint r = n - AVX2_FLOAT_STRIDE / 2;

        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);
            y3 = _mm256_halfsort_ps(x3);

            _mm256_storeu_x4_ps(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 3) {
            _mm256_loadu_x3_ps(vc_ptr, x0, x1, x2);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);
            y2 = _mm256_halfsort_ps(x2);

            _mm256_storeu_x3_ps(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(vc_ptr, x0, x1);

            y0 = _mm256_halfsort_ps(x0);
            y1 = _mm256_halfsort_ps(x1);

            _mm256_storeu_x2_ps(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(vc_ptr, x0);

            y0 = _mm256_halfsort_ps(x0);

            _mm256_storeu_x1_ps(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_ps(ve_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_storeu_x1_ps(ve_ptr, y0);
        }
    }

    return SUCCESS;
}

#pragma endregion batchsort

#pragma region scansort

// scansort 8 elems wise
__forceinline static int scansort_p8_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    uint e = n - AVX2_FLOAT_STRIDE;

    uint indexes;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    {
        uint i = 0;
        while (true) {
            if (i + AVX2_FLOAT_STRIDE * 4 + 1 <= n) {
                _mm256_loadu_x4_ps(v_ptr + i, x0, x1, x2, x3);
                _mm256_loadu_x4_ps(v_ptr + i + 1, y0, y1, y2, y3);

                uint i0 = _mm256_movemask_ps(_mm256_needsswap_ps(x0, y0));
                uint i1 = _mm256_movemask_ps(_mm256_needsswap_ps(x1, y1));
                uint i2 = _mm256_movemask_ps(_mm256_needsswap_ps(x2, y2));
                uint i3 = _mm256_movemask_ps(_mm256_needsswap_ps(x3, y3));

                indexes = (i0) | (i1 << (AVX2_FLOAT_STRIDE)) | (i2 << (AVX2_FLOAT_STRIDE * 2)) | (i3 << (AVX2_FLOAT_STRIDE * 3));

                if (indexes == 0u) {
                    i += AVX2_FLOAT_STRIDE * 4;
                    continue;
                }
            }
            else if (i + AVX2_FLOAT_STRIDE * 2 + 1 <= n) {
                _mm256_loadu_x2_ps(v_ptr + i, x0, x1);
                _mm256_loadu_x2_ps(v_ptr + i + 1, y0, y1);

                uint i0 = _mm256_movemask_ps(_mm256_needsswap_ps(x0, y0));
                uint i1 = _mm256_movemask_ps(_mm256_needsswap_ps(x1, y1));

                indexes = (i0) | (i1 << (AVX2_FLOAT_STRIDE));

                if (indexes == 0u) {
                    i += AVX2_FLOAT_STRIDE * 2;
                    continue;
                }
            }
            else if (i + AVX2_FLOAT_STRIDE + 1 <= n) {
                _mm256_loadu_x1_ps(v_ptr + i, x0);
                _mm256_loadu_x1_ps(v_ptr + i + 1, y0);

                indexes = (uint)_mm256_movemask_ps(_mm256_needsswap_ps(x0, y0));

                if (indexes == 0u) {
                    i += AVX2_FLOAT_STRIDE;
                    continue;
                }
            }
            else {
                i = e;

                x0 = _mm256_loadu_ps(v_ptr + i);

                if (!_mm256_needssort_ps(x0)) {
                    break;
                }

                y0 = _mm256_sort_ps(x0);
                _mm256_storeu_ps(v_ptr + i, y0);

                indexes = 0xFFu - (uint)_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, y0));

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

            x0 = _mm256_loadu_ps(v_ptr + i);

            while (true) {
                y0 = _mm256_sort_ps(x0);
                _mm256_storeu_ps(v_ptr + i, y0);

                indexes = (uint)_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x0, y0));
                if ((indexes & 1u) == 0u && i > 0u) {
                    uint backward = AVX2_FLOAT_STRIDE - 2;
                    i = (i > backward) ? i - backward : 0;

                    x0 = _mm256_loadu_ps(v_ptr + i);

                    if (_mm256_needssort_ps(x0)) {
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
__forceinline static int permcombsort_p8_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_FLOAT_STRIDE * 2 || n >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_FLOAT_STRIDE;

    __m256 a, b, x, y;

    for (uint h = c > 2 ? 2 : 1; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a = _mm256_loadu_ps(v_ptr + i * AVX2_FLOAT_STRIDE);
            b = _mm256_loadu_ps(v_ptr + (i + h) * AVX2_FLOAT_STRIDE);

            _mm256_cmpswap_withperm_ps(a, b, x, y);

            _mm256_storeu_ps(v_ptr + i * AVX2_FLOAT_STRIDE, x);
            _mm256_storeu_ps(v_ptr + (i + h) * AVX2_FLOAT_STRIDE, y);
        }
    }

    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0) {
        b = _mm256_loadu_ps(v_ptr + (n - AVX2_FLOAT_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a = _mm256_loadu_ps(v_ptr + i * AVX2_FLOAT_STRIDE);

            _mm256_cmpswap_ps(a, b, x, y);
            x = _mm256_sort_ps(x);
            b = y;

            _mm256_storeu_ps(v_ptr + i * AVX2_FLOAT_STRIDE, x);
        }

        a = _mm256_loadu_ps(v_ptr + (n - AVX2_FLOAT_STRIDE * 2));

        _mm256_cmpswap_withperm_ps(a, b, x, y);

        x = _mm256_sort_ps(x);
        y = _mm256_sort_ps(y);

        _mm256_storeu_ps(v_ptr + (n - AVX2_FLOAT_STRIDE * 2), x);
        _mm256_storeu_ps(v_ptr + (n - AVX2_FLOAT_STRIDE), y);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a = _mm256_loadu_ps(v_ptr + i * AVX2_FLOAT_STRIDE);
            a = _mm256_sort_ps(a);
            _mm256_storeu_ps(v_ptr + i * AVX2_FLOAT_STRIDE, a);
        }
    }

    return SUCCESS;
}

// permcombsort 8 elems wise 4 batches
__forceinline static int permcombsort_p4x8_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_FLOAT_STRIDE * 2 || n >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_FLOAT_STRIDE;

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + n;
    float* v2_ptr = v_ptr + n * 2;
    float* v3_ptr = v_ptr + n * 3;

    __m256 a0, b0, x0, y0, a1, b1, x1, y1, a2, b2, x2, y2, a3, b3, x3, y3;

    for (uint h = c > 2 ? 2 : 1; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a0 = _mm256_loadu_ps(v0_ptr + i * AVX2_FLOAT_STRIDE);
            a1 = _mm256_loadu_ps(v1_ptr + i * AVX2_FLOAT_STRIDE);
            a2 = _mm256_loadu_ps(v2_ptr + i * AVX2_FLOAT_STRIDE);
            a3 = _mm256_loadu_ps(v3_ptr + i * AVX2_FLOAT_STRIDE);
            b0 = _mm256_loadu_ps(v0_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b1 = _mm256_loadu_ps(v1_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b2 = _mm256_loadu_ps(v2_ptr + (i + h) * AVX2_FLOAT_STRIDE);
            b3 = _mm256_loadu_ps(v3_ptr + (i + h) * AVX2_FLOAT_STRIDE);

            _mm256_cmpswap_withperm_ps(a0, b0, x0, y0);
            _mm256_cmpswap_withperm_ps(a1, b1, x1, y1);
            _mm256_cmpswap_withperm_ps(a2, b2, x2, y2);
            _mm256_cmpswap_withperm_ps(a3, b3, x3, y3);

            _mm256_storeu_ps(v0_ptr + i * AVX2_FLOAT_STRIDE, x0);
            _mm256_storeu_ps(v1_ptr + i * AVX2_FLOAT_STRIDE, x1);
            _mm256_storeu_ps(v2_ptr + i * AVX2_FLOAT_STRIDE, x2);
            _mm256_storeu_ps(v3_ptr + i * AVX2_FLOAT_STRIDE, x3);
            _mm256_storeu_ps(v0_ptr + (i + h) * AVX2_FLOAT_STRIDE, y0);
            _mm256_storeu_ps(v1_ptr + (i + h) * AVX2_FLOAT_STRIDE, y1);
            _mm256_storeu_ps(v2_ptr + (i + h) * AVX2_FLOAT_STRIDE, y2);
            _mm256_storeu_ps(v3_ptr + (i + h) * AVX2_FLOAT_STRIDE, y3);
        }
    }

    if ((n & AVX2_FLOAT_REMAIN_MASK) != 0) {
        b0 = _mm256_loadu_ps(v0_ptr + (n - AVX2_FLOAT_STRIDE));
        b1 = _mm256_loadu_ps(v1_ptr + (n - AVX2_FLOAT_STRIDE));
        b2 = _mm256_loadu_ps(v2_ptr + (n - AVX2_FLOAT_STRIDE));
        b3 = _mm256_loadu_ps(v3_ptr + (n - AVX2_FLOAT_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a0 = _mm256_loadu_ps(v0_ptr + i * AVX2_FLOAT_STRIDE);
            a1 = _mm256_loadu_ps(v1_ptr + i * AVX2_FLOAT_STRIDE);
            a2 = _mm256_loadu_ps(v2_ptr + i * AVX2_FLOAT_STRIDE);
            a3 = _mm256_loadu_ps(v3_ptr + i * AVX2_FLOAT_STRIDE);

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

            _mm256_storeu_ps(v0_ptr + i * AVX2_FLOAT_STRIDE, x0);
            _mm256_storeu_ps(v1_ptr + i * AVX2_FLOAT_STRIDE, x1);
            _mm256_storeu_ps(v2_ptr + i * AVX2_FLOAT_STRIDE, x2);
            _mm256_storeu_ps(v3_ptr + i * AVX2_FLOAT_STRIDE, x3);
        }

        a0 = _mm256_loadu_ps(v0_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a1 = _mm256_loadu_ps(v1_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a2 = _mm256_loadu_ps(v2_ptr + (n - AVX2_FLOAT_STRIDE * 2));
        a3 = _mm256_loadu_ps(v3_ptr + (n - AVX2_FLOAT_STRIDE * 2));

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

        _mm256_storeu_ps(v0_ptr + (n - AVX2_FLOAT_STRIDE * 2), x0);
        _mm256_storeu_ps(v1_ptr + (n - AVX2_FLOAT_STRIDE * 2), x1);
        _mm256_storeu_ps(v2_ptr + (n - AVX2_FLOAT_STRIDE * 2), x2);
        _mm256_storeu_ps(v3_ptr + (n - AVX2_FLOAT_STRIDE * 2), x3);
        _mm256_storeu_ps(v0_ptr + (n - AVX2_FLOAT_STRIDE), y0);
        _mm256_storeu_ps(v1_ptr + (n - AVX2_FLOAT_STRIDE), y1);
        _mm256_storeu_ps(v2_ptr + (n - AVX2_FLOAT_STRIDE), y2);
        _mm256_storeu_ps(v3_ptr + (n - AVX2_FLOAT_STRIDE), y3);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a0 = _mm256_loadu_ps(v0_ptr + i * AVX2_FLOAT_STRIDE);
            a1 = _mm256_loadu_ps(v1_ptr + i * AVX2_FLOAT_STRIDE);
            a2 = _mm256_loadu_ps(v2_ptr + i * AVX2_FLOAT_STRIDE);
            a3 = _mm256_loadu_ps(v3_ptr + i * AVX2_FLOAT_STRIDE);
            a0 = _mm256_sort_ps(a0);
            a1 = _mm256_sort_ps(a1);
            a2 = _mm256_sort_ps(a2);
            a3 = _mm256_sort_ps(a3);
            _mm256_storeu_ps(v0_ptr + i * AVX2_FLOAT_STRIDE, a0);
            _mm256_storeu_ps(v1_ptr + i * AVX2_FLOAT_STRIDE, a1);
            _mm256_storeu_ps(v2_ptr + i * AVX2_FLOAT_STRIDE, a2);
            _mm256_storeu_ps(v3_ptr + i * AVX2_FLOAT_STRIDE, a3);
        }
    }

    return SUCCESS;
}

#pragma endregion permcombsort

#pragma region shortsort

// shortsort elems9
__forceinline static int shortsort_n9_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 1);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 1, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort4x2_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    return SUCCESS;
}

// shortsort batches4 x elems9
__forceinline static int shortsort_n4x9_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 9;
    float* v2_ptr = v_ptr + 18;
    float* v3_ptr = v_ptr + 27;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 1);
    x1 = _mm256_loadu_ps(v1_ptr + 1);
    x2 = _mm256_loadu_ps(v2_ptr + 1);
    x3 = _mm256_loadu_ps(v3_ptr + 1);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 1, y0);
    _mm256_storeu_ps(v1_ptr + 1, y1);
    _mm256_storeu_ps(v2_ptr + 1, y2);
    _mm256_storeu_ps(v3_ptr + 1, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort4x2_ps(x0);
    y1 = _mm256_sort4x2_ps(x1);
    y2 = _mm256_sort4x2_ps(x2);
    y3 = _mm256_sort4x2_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    return SUCCESS;
}

// shortsort elems10
__forceinline static int shortsort_n10_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 2);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 2, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort2x4_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    return SUCCESS;
}

// shortsort batches4 x elems10
__forceinline static int shortsort_n4x10_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 10;
    float* v2_ptr = v_ptr + 20;
    float* v3_ptr = v_ptr + 30;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 2);
    x1 = _mm256_loadu_ps(v1_ptr + 2);
    x2 = _mm256_loadu_ps(v2_ptr + 2);
    x3 = _mm256_loadu_ps(v3_ptr + 2);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 2, y0);
    _mm256_storeu_ps(v1_ptr + 2, y1);
    _mm256_storeu_ps(v2_ptr + 2, y2);
    _mm256_storeu_ps(v3_ptr + 2, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort2x4_ps(x0);
    y1 = _mm256_sort2x4_ps(x1);
    y2 = _mm256_sort2x4_ps(x2);
    y3 = _mm256_sort2x4_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    return SUCCESS;
}

// shortsort elems11
__forceinline static int shortsort_n11_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 3);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 3, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort1x6_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    return SUCCESS;
}

// shortsort batches4 x elems11
__forceinline static int shortsort_n4x11_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 11;
    float* v2_ptr = v_ptr + 22;
    float* v3_ptr = v_ptr + 33;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 3);
    x1 = _mm256_loadu_ps(v1_ptr + 3);
    x2 = _mm256_loadu_ps(v2_ptr + 3);
    x3 = _mm256_loadu_ps(v3_ptr + 3);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 3, y0);
    _mm256_storeu_ps(v1_ptr + 3, y1);
    _mm256_storeu_ps(v2_ptr + 3, y2);
    _mm256_storeu_ps(v3_ptr + 3, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort1x6_ps(x0);
    y1 = _mm256_sort1x6_ps(x1);
    y2 = _mm256_sort1x6_ps(x2);
    y3 = _mm256_sort1x6_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    return SUCCESS;
}

// shortsort elems12
__forceinline static int shortsort_n12_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 4);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 4, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    return SUCCESS;
}

// shortsort batches4 x elems12
__forceinline static int shortsort_n4x12_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 12;
    float* v2_ptr = v_ptr + 24;
    float* v3_ptr = v_ptr + 36;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 4);
    x1 = _mm256_loadu_ps(v1_ptr + 4);
    x2 = _mm256_loadu_ps(v2_ptr + 4);
    x3 = _mm256_loadu_ps(v3_ptr + 4);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 4, y0);
    _mm256_storeu_ps(v1_ptr + 4, y1);
    _mm256_storeu_ps(v2_ptr + 4, y2);
    _mm256_storeu_ps(v3_ptr + 4, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    return SUCCESS;
}

// shortsort elems13
__forceinline static int shortsort_n13_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 5);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 5, y);

    x = _mm256_loadu_ps(v_ptr + 2);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 2, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 5);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 5, y);

    return SUCCESS;
}

// shortsort batches4 x elems13
__forceinline static int shortsort_n4x13_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 13;
    float* v2_ptr = v_ptr + 26;
    float* v3_ptr = v_ptr + 39;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 5);
    x1 = _mm256_loadu_ps(v1_ptr + 5);
    x2 = _mm256_loadu_ps(v2_ptr + 5);
    x3 = _mm256_loadu_ps(v3_ptr + 5);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 5, y0);
    _mm256_storeu_ps(v1_ptr + 5, y1);
    _mm256_storeu_ps(v2_ptr + 5, y2);
    _mm256_storeu_ps(v3_ptr + 5, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 2);
    x1 = _mm256_loadu_ps(v1_ptr + 2);
    x2 = _mm256_loadu_ps(v2_ptr + 2);
    x3 = _mm256_loadu_ps(v3_ptr + 2);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 2, y0);
    _mm256_storeu_ps(v1_ptr + 2, y1);
    _mm256_storeu_ps(v2_ptr + 2, y2);
    _mm256_storeu_ps(v3_ptr + 2, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 5);
    x1 = _mm256_loadu_ps(v1_ptr + 5);
    x2 = _mm256_loadu_ps(v2_ptr + 5);
    x3 = _mm256_loadu_ps(v3_ptr + 5);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 5, y0);
    _mm256_storeu_ps(v1_ptr + 5, y1);
    _mm256_storeu_ps(v2_ptr + 5, y2);
    _mm256_storeu_ps(v3_ptr + 5, y3);

    return SUCCESS;
}

// shortsort elems14
__forceinline static int shortsort_n14_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 6);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 6, y);

    x = _mm256_loadu_ps(v_ptr + 3);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 3, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 6);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 6, y);

    x = _mm256_loadu_ps(v_ptr + 3);
    y = _mm256_sort4x2_ps(x);
    _mm256_storeu_ps(v_ptr + 3, y);

    return SUCCESS;
}

// shortsort batches4 x elems14
__forceinline static int shortsort_n4x14_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 14;
    float* v2_ptr = v_ptr + 28;
    float* v3_ptr = v_ptr + 42;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 6);
    x1 = _mm256_loadu_ps(v1_ptr + 6);
    x2 = _mm256_loadu_ps(v2_ptr + 6);
    x3 = _mm256_loadu_ps(v3_ptr + 6);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 6, y0);
    _mm256_storeu_ps(v1_ptr + 6, y1);
    _mm256_storeu_ps(v2_ptr + 6, y2);
    _mm256_storeu_ps(v3_ptr + 6, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 3);
    x1 = _mm256_loadu_ps(v1_ptr + 3);
    x2 = _mm256_loadu_ps(v2_ptr + 3);
    x3 = _mm256_loadu_ps(v3_ptr + 3);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 3, y0);
    _mm256_storeu_ps(v1_ptr + 3, y1);
    _mm256_storeu_ps(v2_ptr + 3, y2);
    _mm256_storeu_ps(v3_ptr + 3, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 6);
    x1 = _mm256_loadu_ps(v1_ptr + 6);
    x2 = _mm256_loadu_ps(v2_ptr + 6);
    x3 = _mm256_loadu_ps(v3_ptr + 6);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 6, y0);
    _mm256_storeu_ps(v1_ptr + 6, y1);
    _mm256_storeu_ps(v2_ptr + 6, y2);
    _mm256_storeu_ps(v3_ptr + 6, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 3);
    x1 = _mm256_loadu_ps(v1_ptr + 3);
    x2 = _mm256_loadu_ps(v2_ptr + 3);
    x3 = _mm256_loadu_ps(v3_ptr + 3);
    y0 = _mm256_sort4x2_ps(x0);
    y1 = _mm256_sort4x2_ps(x1);
    y2 = _mm256_sort4x2_ps(x2);
    y3 = _mm256_sort4x2_ps(x3);
    _mm256_storeu_ps(v0_ptr + 3, y0);
    _mm256_storeu_ps(v1_ptr + 3, y1);
    _mm256_storeu_ps(v2_ptr + 3, y2);
    _mm256_storeu_ps(v3_ptr + 3, y3);

    return SUCCESS;
}

// shortsort elems15
__forceinline static int shortsort_n15_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 7);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 7, y);

    x = _mm256_loadu_ps(v_ptr + 3);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 3, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + 7);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 7, y);

    x = _mm256_loadu_ps(v_ptr + 4);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + 4, y);

    return SUCCESS;
}

// shortsort batches4 x elems15
__forceinline static int shortsort_n4x15_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + 15;
    float* v2_ptr = v_ptr + 30;
    float* v3_ptr = v_ptr + 45;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 7);
    x1 = _mm256_loadu_ps(v1_ptr + 7);
    x2 = _mm256_loadu_ps(v2_ptr + 7);
    x3 = _mm256_loadu_ps(v3_ptr + 7);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 7, y0);
    _mm256_storeu_ps(v1_ptr + 7, y1);
    _mm256_storeu_ps(v2_ptr + 7, y2);
    _mm256_storeu_ps(v3_ptr + 7, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 3);
    x1 = _mm256_loadu_ps(v1_ptr + 3);
    x2 = _mm256_loadu_ps(v2_ptr + 3);
    x3 = _mm256_loadu_ps(v3_ptr + 3);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 3, y0);
    _mm256_storeu_ps(v1_ptr + 3, y1);
    _mm256_storeu_ps(v2_ptr + 3, y2);
    _mm256_storeu_ps(v3_ptr + 3, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 7);
    x1 = _mm256_loadu_ps(v1_ptr + 7);
    x2 = _mm256_loadu_ps(v2_ptr + 7);
    x3 = _mm256_loadu_ps(v3_ptr + 7);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 7, y0);
    _mm256_storeu_ps(v1_ptr + 7, y1);
    _mm256_storeu_ps(v2_ptr + 7, y2);
    _mm256_storeu_ps(v3_ptr + 7, y3);

    x0 = _mm256_loadu_ps(v0_ptr + 4);
    x1 = _mm256_loadu_ps(v1_ptr + 4);
    x2 = _mm256_loadu_ps(v2_ptr + 4);
    x3 = _mm256_loadu_ps(v3_ptr + 4);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + 4, y0);
    _mm256_storeu_ps(v1_ptr + 4, y1);
    _mm256_storeu_ps(v2_ptr + 4, y2);
    _mm256_storeu_ps(v3_ptr + 4, y3);

    return SUCCESS;
}

// shortsort elems16
__forceinline static int shortsort_n16_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x, y;

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE);
    y = _mm256_sort_ps(x);
    _mm256_storeu_ps(v_ptr + AVX2_FLOAT_STRIDE, y);

    x = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE / 2);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(v_ptr + AVX2_FLOAT_STRIDE / 2, y);

    x = _mm256_loadu_ps(v_ptr);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(v_ptr, y);

    x = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(v_ptr + AVX2_FLOAT_STRIDE, y);

    x = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE / 2);
    y = _mm256_halfsort_ps(x);
    _mm256_storeu_ps(v_ptr + AVX2_FLOAT_STRIDE / 2, y);

    return SUCCESS;
}

// shortsort batches4 x elems16
__forceinline static int shortsort_n4x16_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n != AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    float* v0_ptr = v_ptr;
    float* v1_ptr = v_ptr + AVX2_FLOAT_STRIDE * 2;
    float* v2_ptr = v_ptr + AVX2_FLOAT_STRIDE * 4;
    float* v3_ptr = v_ptr + AVX2_FLOAT_STRIDE * 6;

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + AVX2_FLOAT_STRIDE);
    x1 = _mm256_loadu_ps(v1_ptr + AVX2_FLOAT_STRIDE);
    x2 = _mm256_loadu_ps(v2_ptr + AVX2_FLOAT_STRIDE);
    x3 = _mm256_loadu_ps(v3_ptr + AVX2_FLOAT_STRIDE);
    y0 = _mm256_sort_ps(x0);
    y1 = _mm256_sort_ps(x1);
    y2 = _mm256_sort_ps(x2);
    y3 = _mm256_sort_ps(x3);
    _mm256_storeu_ps(v0_ptr + AVX2_FLOAT_STRIDE, y0);
    _mm256_storeu_ps(v1_ptr + AVX2_FLOAT_STRIDE, y1);
    _mm256_storeu_ps(v2_ptr + AVX2_FLOAT_STRIDE, y2);
    _mm256_storeu_ps(v3_ptr + AVX2_FLOAT_STRIDE, y3);

    x0 = _mm256_loadu_ps(v0_ptr + AVX2_FLOAT_STRIDE / 2);
    x1 = _mm256_loadu_ps(v1_ptr + AVX2_FLOAT_STRIDE / 2);
    x2 = _mm256_loadu_ps(v2_ptr + AVX2_FLOAT_STRIDE / 2);
    x3 = _mm256_loadu_ps(v3_ptr + AVX2_FLOAT_STRIDE / 2);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(v0_ptr + AVX2_FLOAT_STRIDE / 2, y0);
    _mm256_storeu_ps(v1_ptr + AVX2_FLOAT_STRIDE / 2, y1);
    _mm256_storeu_ps(v2_ptr + AVX2_FLOAT_STRIDE / 2, y2);
    _mm256_storeu_ps(v3_ptr + AVX2_FLOAT_STRIDE / 2, y3);

    x0 = _mm256_loadu_ps(v0_ptr);
    x1 = _mm256_loadu_ps(v1_ptr);
    x2 = _mm256_loadu_ps(v2_ptr);
    x3 = _mm256_loadu_ps(v3_ptr);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(v0_ptr, y0);
    _mm256_storeu_ps(v1_ptr, y1);
    _mm256_storeu_ps(v2_ptr, y2);
    _mm256_storeu_ps(v3_ptr, y3);

    x0 = _mm256_loadu_ps(v0_ptr + AVX2_FLOAT_STRIDE);
    x1 = _mm256_loadu_ps(v1_ptr + AVX2_FLOAT_STRIDE);
    x2 = _mm256_loadu_ps(v2_ptr + AVX2_FLOAT_STRIDE);
    x3 = _mm256_loadu_ps(v3_ptr + AVX2_FLOAT_STRIDE);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(v0_ptr + AVX2_FLOAT_STRIDE, y0);
    _mm256_storeu_ps(v1_ptr + AVX2_FLOAT_STRIDE, y1);
    _mm256_storeu_ps(v2_ptr + AVX2_FLOAT_STRIDE, y2);
    _mm256_storeu_ps(v3_ptr + AVX2_FLOAT_STRIDE, y3);

    x0 = _mm256_loadu_ps(v0_ptr + AVX2_FLOAT_STRIDE / 2);
    x1 = _mm256_loadu_ps(v1_ptr + AVX2_FLOAT_STRIDE / 2);
    x2 = _mm256_loadu_ps(v2_ptr + AVX2_FLOAT_STRIDE / 2);
    x3 = _mm256_loadu_ps(v3_ptr + AVX2_FLOAT_STRIDE / 2);
    y0 = _mm256_halfsort_ps(x0);
    y1 = _mm256_halfsort_ps(x1);
    y2 = _mm256_halfsort_ps(x2);
    y3 = _mm256_halfsort_ps(x3);
    _mm256_storeu_ps(v0_ptr + AVX2_FLOAT_STRIDE / 2, y0);
    _mm256_storeu_ps(v1_ptr + AVX2_FLOAT_STRIDE / 2, y1);
    _mm256_storeu_ps(v2_ptr + AVX2_FLOAT_STRIDE / 2, y2);
    _mm256_storeu_ps(v3_ptr + AVX2_FLOAT_STRIDE / 2, y3);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

// longsort elems 32...63
__forceinline static int longsort_n32to63_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE * 4 || n >= AVX2_FLOAT_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    paracombsort_p2x8_s(n, v_ptr);
    backtracksort_p8_s(n, v_ptr);

    batchsort_p8_s(n, v_ptr);
    scansort_p8_s(n, v_ptr);

    return SUCCESS;
}

// longsort elems 64+
__forceinline static int longsort_n64plus_s(const uint n, float* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    uint h;

    for (h = (uint)(n * 10uLL / 13uLL); h > 33; h = (uint)(h * 10uLL / 13uLL)) {
        combsort_h33plus_s(n, h, v_ptr);
    }
    if (h >= 32) {
        combsort_h32_s(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 25; h = h * 10 / 13) {
        combsort_h25to31_s(n, h, v_ptr);
    }
    if (h >= 24) {
        combsort_h24_s(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 17; h = h * 10 / 13) {
        combsort_h17to23_s(n, h, v_ptr);
    }
    if (h >= 16) {
        combsort_h16_s(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 9; h = h * 10 / 13) {
        combsort_h9to15_s(n, h, v_ptr);
    }

    paracombsort_p2x8_s(n, v_ptr);

    batchsort_p8_s(n, v_ptr);
    scansort_p8_s(n, v_ptr);

    return SUCCESS;
}

#pragma endregion longsort

#pragma region sort

int sortasc_minnan_s2_s(const uint n, const uint s, float* v_ptr) {
    if (s != 2) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= AVX2_FLOAT_STRIDE * 4 / 2) {
            _mm256_loadu_x4_ps(v_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);
            y2 = _mm256_sort4x2_ps(x2);
            y3 = _mm256_sort4x2_ps(x3);

            _mm256_storeu_x4_ps(v_ptr, y0, y1, y2, y3);

            v_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4 / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x2_ps(v_ptr, x0, x1);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);

            _mm256_storeu_x2_ps(v_ptr, y0, y1);

            v_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_loadu_x1_ps(v_ptr, x0);

            y0 = _mm256_sort4x2_ps(x0);

            _mm256_storeu_x1_ps(v_ptr, y0);

            v_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
    }
    else {
        while (r >= AVX2_FLOAT_STRIDE * 4 / 2) {
            _mm256_load_x4_ps(v_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);
            y2 = _mm256_sort4x2_ps(x2);
            y3 = _mm256_sort4x2_ps(x3);

            _mm256_stream_x4_ps(v_ptr, y0, y1, y2, y3);

            v_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4 / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x2_ps(v_ptr, x0, x1);

            y0 = _mm256_sort4x2_ps(x0);
            y1 = _mm256_sort4x2_ps(x1);

            _mm256_stream_x2_ps(v_ptr, y0, y1);

            v_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_load_x1_ps(v_ptr, x0);

            y0 = _mm256_sort4x2_ps(x0);

            _mm256_stream_x1_ps(v_ptr, y0);

            v_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps((r * 2) & AVX2_FLOAT_REMAIN_MASK);

        _mm256_maskload_x1_ps(v_ptr, x0, mask);

        y0 = _mm256_sort4x2_ps(x0);

        _mm256_maskstore_x1_ps(v_ptr, y0, mask);
    }

    return SUCCESS;
}

int sortasc_minnan_s3_s(const uint n, const uint s, float* v_ptr) {
    if (s != 3) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 9) {
        x0 = _mm256_loadu_ps(v_ptr);
        x1 = _mm256_loadu_ps(v_ptr + 6);
        x2 = _mm256_loadu_ps(v_ptr + 12);
        x3 = _mm256_loadu_ps(v_ptr + 18);

        y0 = _mm256_sort2x3_ps(x0);
        y1 = _mm256_sort2x3_ps(x1);
        y2 = _mm256_sort2x3_ps(x2);
        y3 = _mm256_sort2x3_ps(x3);

        _mm256_storeu_ps(v_ptr, y0);
        _mm256_storeu_ps(v_ptr + 6, y1);
        _mm256_storeu_ps(v_ptr + 12, y2);
        _mm256_storeu_ps(v_ptr + 18, y3);

        v_ptr += AVX2_FLOAT_STRIDE * 3;
        r -= 8;
    }
    while (r >= 2) {
        const __m256i mask = _mm256_setmask_ps(6);

        x0 = _mm256_maskload_ps(v_ptr, mask);

        y0 = _mm256_sort2x3_ps(x0);

        _mm256_maskstore_ps(v_ptr, mask, y0);

        v_ptr += AVX2_FLOAT_STRIDE * 3 / 4;
        r -= 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(3);

        x0 = _mm256_maskload_ps(v_ptr, mask);

        y0 = _mm256_sort2x3_ps(x0);

        _mm256_maskstore_ps(v_ptr, mask, y0);
    }

    return SUCCESS;
}

int sortasc_minnan_s4_s(const uint n, const uint s, float* v_ptr) {
    if (s != 4) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x4_ps(v_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);
            y2 = _mm256_sort2x4_ps(x2);
            y3 = _mm256_sort2x4_ps(x3);

            _mm256_storeu_x4_ps(v_ptr, y0, y1, y2, y3);

            v_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_loadu_x2_ps(v_ptr, x0, x1);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);

            _mm256_storeu_x2_ps(v_ptr, y0, y1);

            v_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE / 4) {
            _mm256_loadu_x1_ps(v_ptr, x0);

            y0 = _mm256_sort2x4_ps(x0);

            _mm256_storeu_x1_ps(v_ptr, y0);

            v_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 4;
        }
    }
    else {
        while (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x4_ps(v_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);
            y2 = _mm256_sort2x4_ps(x2);
            y3 = _mm256_sort2x4_ps(x3);

            _mm256_stream_x4_ps(v_ptr, y0, y1, y2, y3);

            v_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE;
        }
        if (r >= AVX2_FLOAT_STRIDE / 2) {
            _mm256_load_x2_ps(v_ptr, x0, x1);

            y0 = _mm256_sort2x4_ps(x0);
            y1 = _mm256_sort2x4_ps(x1);

            _mm256_stream_x2_ps(v_ptr, y0, y1);

            v_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE / 2;
        }
        if (r >= AVX2_FLOAT_STRIDE / 4) {
            _mm256_load_x1_ps(v_ptr, x0);

            y0 = _mm256_sort2x4_ps(x0);

            _mm256_stream_x1_ps(v_ptr, y0);

            v_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE / 4;
        }
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(4);

        _mm256_maskload_x1_ps(v_ptr, x0, mask);

        y0 = _mm256_sort2x4_ps(x0);

        _mm256_maskstore_x1_ps(v_ptr, y0, mask);
    }

    return SUCCESS;
}

int sortasc_minnan_s5_s(const uint n, const uint s, float* v_ptr) {
    if (s != 5) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0 = _mm256_loadu_ps(v_ptr);
        x1 = _mm256_loadu_ps(v_ptr + 5);
        x2 = _mm256_loadu_ps(v_ptr + 10);
        x3 = _mm256_loadu_ps(v_ptr + 15);

        y0 = _mm256_sort1x5_ps(x0);
        y1 = _mm256_sort1x5_ps(x1);
        y2 = _mm256_sort1x5_ps(x2);
        y3 = _mm256_sort1x5_ps(x3);

        _mm256_storeu_ps(v_ptr, y0);
        _mm256_storeu_ps(v_ptr + 5, y1);
        _mm256_storeu_ps(v_ptr + 10, y2);
        _mm256_storeu_ps(v_ptr + 15, y3);

        v_ptr += 20;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(5);

        x0 = _mm256_maskload_ps(v_ptr, mask);

        y0 = _mm256_sort1x5_ps(x0);

        _mm256_maskstore_ps(v_ptr, mask, y0);

        v_ptr += 5;
        r -= 1;
    }

    return SUCCESS;
}

int sortasc_minnan_s6_s(const uint n, const uint s, float* v_ptr) {
    if (s != 6) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0 = _mm256_loadu_ps(v_ptr);
        x1 = _mm256_loadu_ps(v_ptr + 6);
        x2 = _mm256_loadu_ps(v_ptr + 12);
        x3 = _mm256_loadu_ps(v_ptr + 18);

        y0 = _mm256_sort1x6_ps(x0);
        y1 = _mm256_sort1x6_ps(x1);
        y2 = _mm256_sort1x6_ps(x2);
        y3 = _mm256_sort1x6_ps(x3);

        _mm256_storeu_ps(v_ptr, y0);
        _mm256_storeu_ps(v_ptr + 6, y1);
        _mm256_storeu_ps(v_ptr + 12, y2);
        _mm256_storeu_ps(v_ptr + 18, y3);

        v_ptr += 24;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(6);

        x0 = _mm256_maskload_ps(v_ptr, mask);

        y0 = _mm256_sort1x6_ps(x0);

        _mm256_maskstore_ps(v_ptr, mask, y0);

        v_ptr += 6;
        r -= 1;
    }

    return SUCCESS;
}

int sortasc_minnan_s7_s(const uint n, const uint s, float* v_ptr) {
    if (s != 7) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0 = _mm256_loadu_ps(v_ptr);
        x1 = _mm256_loadu_ps(v_ptr + 7);
        x2 = _mm256_loadu_ps(v_ptr + 14);
        x3 = _mm256_loadu_ps(v_ptr + 21);

        y0 = _mm256_sort1x7_ps(x0);
        y1 = _mm256_sort1x7_ps(x1);
        y2 = _mm256_sort1x7_ps(x2);
        y3 = _mm256_sort1x7_ps(x3);

        _mm256_storeu_ps(v_ptr, y0);
        _mm256_storeu_ps(v_ptr + 7, y1);
        _mm256_storeu_ps(v_ptr + 14, y2);
        _mm256_storeu_ps(v_ptr + 21, y3);

        v_ptr += 28;
        r -= 4;
    }
    while (r > 0) {
        const __m256i mask = _mm256_setmask_ps(7);

        x0 = _mm256_maskload_ps(v_ptr, mask);

        y0 = _mm256_sort1x7_ps(x0);

        _mm256_maskstore_ps(v_ptr, mask, y0);

        v_ptr += 7;
        r -= 1;
    }

    return SUCCESS;
}

int sortasc_minnan_s8_s(const uint n, const uint s, float* v_ptr) {
    if (s != 8) {
        return FAILURE_BADPARAM;
    }

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= 4) {
            _mm256_loadu_x4_ps(v_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);
            y3 = _mm256_sort_ps(x3);

            _mm256_storeu_x4_ps(v_ptr, y0, y1, y2, y3);

            v_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= 4;
        }
        if (r >= 2) {
            _mm256_loadu_x2_ps(v_ptr, x0, x1);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);

            _mm256_storeu_x2_ps(v_ptr, y0, y1);

            v_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= 2;
        }
        if (r > 0) {
            _mm256_load_x1_ps(v_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_stream_x1_ps(v_ptr, y0);
        }
    }
    else {
        while (r >= 4) {
            _mm256_load_x4_ps(v_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);
            y2 = _mm256_sort_ps(x2);
            y3 = _mm256_sort_ps(x3);

            _mm256_stream_x4_ps(v_ptr, y0, y1, y2, y3);

            v_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= 4;
        }
        if (r >= 2) {
            _mm256_load_x2_ps(v_ptr, x0, x1);

            y0 = _mm256_sort_ps(x0);
            y1 = _mm256_sort_ps(x1);

            _mm256_stream_x2_ps(v_ptr, y0, y1);

            v_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= 2;
        }
        if (r > 0) {
            _mm256_load_x1_ps(v_ptr, x0);

            y0 = _mm256_sort_ps(x0);

            _mm256_stream_x1_ps(v_ptr, y0);
        }
    }

    return SUCCESS;
}

int sortasc_minnan_s9_s(const uint n, const uint s, float* v_ptr) {
    if (s != 9) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x9_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n9_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s10_s(const uint n, const uint s, float* v_ptr) {
    if (s != 10) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x10_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n10_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s11_s(const uint n, const uint s, float* v_ptr) {
    if (s != 11) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x11_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n11_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s12_s(const uint n, const uint s, float* v_ptr) {
    if (s != 12) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x12_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n12_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s13_s(const uint n, const uint s, float* v_ptr) {
    if (s != 13) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x13_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n13_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s14_s(const uint n, const uint s, float* v_ptr) {
    if (s != 14) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x14_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n14_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s15_s(const uint n, const uint s, float* v_ptr) {
    if (s != 15) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x15_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n15_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s16_s(const uint n, const uint s, float* v_ptr) {
    if (s != AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x16_s(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n16_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s17to31_s(const uint n, const uint s, float* v_ptr) {
    if (s <= AVX2_FLOAT_STRIDE * 2 || s >= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        permcombsort_p4x8_s(s, v_ptr);
        scansort_p8_s(s, v_ptr);
        scansort_p8_s(s, v_ptr + s);
        scansort_p8_s(s, v_ptr + s * 2);
        scansort_p8_s(s, v_ptr + s * 3);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        permcombsort_p8_s(s, v_ptr);
        scansort_p8_s(s, v_ptr);

        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s32to63_s(const uint n, const uint s, float* v_ptr) {
    if (s < AVX2_FLOAT_STRIDE * 4 || s >= AVX2_FLOAT_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < n; i++) {
        longsort_n32to63_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_minnan_s64plus_s(const uint n, const uint s, float* v_ptr) {
    if (s < AVX2_FLOAT_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < n; i++) {
        longsort_n64plus_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

#pragma endregion sort