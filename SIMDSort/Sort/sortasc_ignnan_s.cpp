#include "sort.h"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_cmp_s.hpp"
#include "../Inline/inline_loadstore_xn_s.hpp"

#pragma region needs swap

// needs swap
__forceinline __m128 _mm_needsswap_ps(__m128 x, __m128 y) {
    return _mm_cmpgt_ignnan_ps(x, y);
}

// needs swap
__forceinline __m256 _mm256_needsswap_ps(__m256 x, __m256 y) {
    return _mm256_cmpgt_ignnan_ps(x, y);
}

#pragma endregion needs swap

#pragma region needs sort

// needs sort
__forceinline bool _mm_needssort_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_needsswap_ps(x, y)) > 0;

    return needssort;
}

// needs sort
__forceinline bool _mm256_needssort_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_needsswap_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion needs sort

#pragma region horizontal sort

// sort batches4 x elems2 
__forceinline __m256 _mm256_sort4x2_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
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
__forceinline __m256 _mm256_sort2x4_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, ~0u, ~0u, 0, 0, ~0u));

    __m256 y, c;

    y = _mm256_permute_ps(x, _MM_PERM_ABCD);
    c = _mm256_xor_ps(xormask, _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_ABCD);
    c = _mm256_xor_ps(xormask, _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_DBBD));
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
__forceinline __m256 _mm256_sort1x6_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, ~0u, 0, 0));
    const __m256i perm0 = _mm256_setr_epi32(5, 2, 1, 4, 3, 0, 6, 7);
    const __m256i perm1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 6, 7);
    const __m256i permcmp0 = _mm256_setr_epi32(5, 1, 1, 3, 3, 5, 6, 7);
    const __m256i permcmp1 = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm1);
    c = _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp1);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm0);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp0));
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

// sort elems4
__forceinline __m128 _mm_sort_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y, c;

    y = _mm_permute_ps(x, _MM_PERM_ABCD);
    c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_needsswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm_blendv_ps(x, y, c);

    y = _mm_permute_ps(x, _MM_PERM_CDAB);
    c = _mm_permute_ps(_mm_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm_blendv_ps(x, y, c);

    y = _mm_permute_ps(x, _MM_PERM_ABCD);
    c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_needsswap_ps(x, y), _MM_PERM_DBBD));
    x = _mm_blendv_ps(x, y, c);

    return x;
}

// sort elems8
__forceinline __m256 _mm256_sort_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

#pragma endregion horizontal sort

#pragma region cmp and swap

// compare and swap
__forceinline void _mm_cmpswap_ps(__m128 a, __m128 b, __m128& x, __m128& y) {
    __m128 swaps = _mm_needsswap_ps(a, b);

    x = _mm_blendv_ps(a, b, swaps);
    y = _mm_blendv_ps(a, b, _mm_not_ps(swaps));
}

// compare and swap
__forceinline uint _mm_cmpswap_indexed_ps(__m128 a, __m128 b, __m128& x, __m128& y) {
    __m128 swaps = _mm_needsswap_ps(a, b);

    uint index = _mm_movemask_ps(swaps);

    x = _mm_blendv_ps(a, b, swaps);
    y = _mm_blendv_ps(a, b, _mm_not_ps(swaps));

    return index;
}

// compare and swap
__forceinline void _mm256_cmpswap_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 swaps = _mm256_needsswap_ps(a, b);

    x = _mm256_blendv_ps(a, b, swaps);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(swaps));
}

// compare and swap
__forceinline uint _mm256_cmpswap_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 swaps = _mm256_needsswap_ps(a, b);

    uint index = _mm256_movemask_ps(swaps);

    x = _mm256_blendv_ps(a, b, swaps);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(swaps));

    return index;
}

#pragma endregion cmp and swap

#pragma region combsort

int combsort_h8_s(const uint n, outfloats v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 2) {
        return SUCCESS;
    }

    uint e = n - AVX2_FLOAT_STRIDE * 2;

    __m256 a0, b0;
    __m256 x0, y0;

    if (e > 0) {
        _mm256_loadu_x1_ps(v_ptr, a0);

        uint i = 0;
        for (; i < e; i += AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(v_ptr + i + AVX2_FLOAT_STRIDE, b0);

            _mm256_cmpswap_ps(a0, b0, x0, y0);

            _mm256_storeu_x1_ps(v_ptr + i, x0);

            a0 = y0;
        }
        _mm256_storeu_x1_ps(v_ptr + i, a0);
    }
    {
        _mm256_loadu_x1_ps(v_ptr + e, a0);
        _mm256_loadu_x1_ps(v_ptr + e + AVX2_FLOAT_STRIDE, b0);

        _mm256_cmpswap_ps(a0, b0, x0, y0);

        _mm256_storeu_x1_ps(v_ptr + e, x0);
        _mm256_storeu_x1_ps(v_ptr + e + AVX2_FLOAT_STRIDE, y0);
    }

    return SUCCESS;
}

int combsort_h9to15_s(const uint n, const uint h, outfloats v_ptr) {
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

int combsort_h16_s(const uint n, outfloats v_ptr) {
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

int combsort_h17to23_s(const uint n, const uint h, outfloats v_ptr) {
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

int combsort_h24_s(const uint n, outfloats v_ptr) {
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

int combsort_h25to31_s(const uint n, const uint h, outfloats v_ptr) {
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

int combsort_h32_s(const uint n, outfloats v_ptr) {
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

int combsort_h33plus_s(const uint n, const uint h, outfloats v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < AVX2_FLOAT_STRIDE * 4 || n - AVX2_FLOAT_STRIDE * 4 < h) {
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

#pragma region backtracksort

int backtracksort_p8_s(const uint n, outfloats v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 2) {
        return SUCCESS;
    }

    uint i = 0, e = n - AVX2_FLOAT_STRIDE * 2;
    __m256 a = _mm256_loadu_ps(v_ptr), b = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE);
    __m256 x, y;

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

#pragma region scansort

void scansortiter_p8_s(const uint i, const uint n, outfloats v_ptr) {
    if (i > n) {
        return;
    }

    v_ptr += i;

    uint r = n - i;

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(v_ptr, x0, x1, x2, x3);

        y0 = _mm256_sort_ps(x0);
        y1 = _mm256_sort_ps(x1);
        y2 = _mm256_sort_ps(x2);
        y3 = _mm256_sort_ps(x3);

        _mm256_storeu_x4_ps(v_ptr, y0, y1, y2, y3);

        v_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_loadu_x3_ps(v_ptr, x0, x1, x2);

        y0 = _mm256_sort_ps(x0);
        y1 = _mm256_sort_ps(x1);
        y2 = _mm256_sort_ps(x2);

        _mm256_storeu_x3_ps(v_ptr, y0, y1, y2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x2_ps(v_ptr, x0, x1);

        y0 = _mm256_sort_ps(x0);
        y1 = _mm256_sort_ps(x1);

        _mm256_storeu_x2_ps(v_ptr, y0, y1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x1_ps(v_ptr, x0);

        y0 = _mm256_sort_ps(x0);

        _mm256_storeu_x1_ps(v_ptr, y0);
    }
}

int scansort_p8_s(const uint n, outfloats v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    scansortiter_p8_s(0, n, v_ptr);
    scansortiter_p8_s(AVX2_FLOAT_STRIDE / 2, n, v_ptr);

    uint e = n - AVX2_FLOAT_STRIDE;

    uint i = 0;
    while (true) {
        __m256 x = _mm256_loadu_ps(v_ptr + i);

        if (_mm256_needssort_ps(x)) {
            __m256 y = _mm256_sort_ps(x);
            _mm256_storeu_ps(v_ptr + i, y);

            if (i > 0) {
                uint indexes = _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x, y));
                if ((indexes & 1) == 0) {
                    const uint back = AVX2_FLOAT_STRIDE - 2;

                    i = (i > back) ? i - back : 0;
                    continue;
                }
            }
        }

        if (i < e) {
            i += AVX2_FLOAT_STRIDE - 1;
            if (i > e) {
                i = e;
            }
        }
        else {
            break;
        }
    }

    return SUCCESS;
}

#pragma endregion scansort

#pragma region shortsort

__forceinline int shortsort_n9to11_s(const uint n, outfloats v_ptr) {
#ifdef _DEBUG
    if (n <= AVX1_FLOAT_STRIDE * 2 || n >= AVX1_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m128 x0, x1, x2, x3, y0, y1, y2, y3;
    __m128 noswap = _mm_setzero_ps();

    do {
        x0 = _mm_loadu_ps(v_ptr);
        x1 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE);
        y0 = _mm_sort_ps(x0);
        y1 = _mm_sort_ps(x1);
        _mm_storeu_ps(v_ptr, y0);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE, y1);

        x2 = _mm_loadu_ps(v_ptr + n - AVX1_FLOAT_STRIDE * 2);
        x3 = _mm_loadu_ps(v_ptr + n - AVX1_FLOAT_STRIDE);
        y2 = _mm_sort_ps(x2);
        y3 = _mm_sort_ps(x3);
        _mm_storeu_ps(v_ptr + n - AVX1_FLOAT_STRIDE * 2, y2);
        _mm_storeu_ps(v_ptr + n - AVX1_FLOAT_STRIDE, y3);

        noswap = _mm_and_ps(
            _mm_and_ps(_mm_cmpeq_ignnan_ps(x0, y0), _mm_cmpeq_ignnan_ps(x1, y1)),
            _mm_and_ps(_mm_cmpeq_ignnan_ps(x2, y2), _mm_cmpeq_ignnan_ps(x3, y3))
        );
    } while (_mm_movemask_ps(noswap) != 0xFu);

    return SUCCESS;
}

__forceinline int shortsort_n12_s(const uint n, outfloats v_ptr) {
#ifdef _DEBUG
    if (n != AVX1_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m128 x0, x1, x2, x3, x4, y0, y1, y2, y3, y4;
    __m128 noswap = _mm_setzero_ps();

    do {
        x0 = _mm_loadu_ps(v_ptr);
        x1 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE);
        x2 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE * 2);
        y0 = _mm_sort_ps(x0);
        y1 = _mm_sort_ps(x1);
        y2 = _mm_sort_ps(x2);
        _mm_storeu_ps(v_ptr, y0);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE, y1);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE * 2, y2);

        x3 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE / 2);
        x4 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE * 3 / 2);
        y3 = _mm_sort_ps(x3);
        y4 = _mm_sort_ps(x4);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE / 2, y3);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE * 3 / 2, y4);

        noswap = _mm_and_ps(
            _mm_and_ps(_mm_and_ps(_mm_cmpeq_ignnan_ps(x0, y0), _mm_cmpeq_ignnan_ps(x1, y1)), _mm_cmpeq_ignnan_ps(x2, y2)),
            _mm_and_ps(_mm_cmpeq_ignnan_ps(x3, y3), _mm_cmpeq_ignnan_ps(x4, y4))
        );
    } while (_mm_movemask_ps(noswap) != 0xFu);

    return SUCCESS;
}

__forceinline int shortsort_n13to15_s(const uint n, outfloats v_ptr) {
#ifdef _DEBUG
    if (n <= AVX1_FLOAT_STRIDE * 3 || n >= AVX1_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m128 x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5;
    __m128 noswap = _mm_setzero_ps();

    do {
        x0 = _mm_loadu_ps(v_ptr);
        x1 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE);
        x2 = _mm_loadu_ps(v_ptr + AVX1_FLOAT_STRIDE * 2);
        y0 = _mm_sort_ps(x0);
        y1 = _mm_sort_ps(x1);
        y2 = _mm_sort_ps(x2);
        _mm_storeu_ps(v_ptr, y0);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE, y1);
        _mm_storeu_ps(v_ptr + AVX1_FLOAT_STRIDE * 2, y2);

        x3 = _mm_loadu_ps(v_ptr + n - AVX1_FLOAT_STRIDE * 3);
        x4 = _mm_loadu_ps(v_ptr + n - AVX1_FLOAT_STRIDE * 2);
        x5 = _mm_loadu_ps(v_ptr + n - AVX1_FLOAT_STRIDE);
        y3 = _mm_sort_ps(x3);
        y4 = _mm_sort_ps(x4);
        y5 = _mm_sort_ps(x5);
        _mm_storeu_ps(v_ptr + n - AVX1_FLOAT_STRIDE * 3, y3);
        _mm_storeu_ps(v_ptr + n - AVX1_FLOAT_STRIDE * 2, y4);
        _mm_storeu_ps(v_ptr + n - AVX1_FLOAT_STRIDE, y5);

        noswap = _mm_and_ps(
            _mm_and_ps(_mm_and_ps(_mm_cmpeq_ignnan_ps(x0, y0), _mm_cmpeq_ignnan_ps(x1, y1)), _mm_cmpeq_ignnan_ps(x2, y2)),
            _mm_and_ps(_mm_and_ps(_mm_cmpeq_ignnan_ps(x3, y3), _mm_cmpeq_ignnan_ps(x4, y4)), _mm_cmpeq_ignnan_ps(x5, y5))
        );
    } while (_mm_movemask_ps(noswap) != 0xFu);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

__forceinline int longsort_s(const uint n, outfloats v_ptr) {
    uint h;

    for (h = (uint)(n * 10L / 13L); h > 33; h = (uint)(h * 10L / 13L)) {
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
    if (h >= 8) {
        combsort_h8_s(n, v_ptr);
    }

    backtracksort_p8_s(n, v_ptr);
    scansort_p8_s(n, v_ptr);

    return SUCCESS;
}

#pragma endregion longsort

#pragma region sort

int sortasc_ignnan_s2_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 2 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

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
    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps((r * 2) & AVX2_FLOAT_REMAIN_MASK);

        _mm256_maskload_x1_ps(v_ptr, x0, mask);

        y0 = _mm256_sort4x2_ps(x0);

        _mm256_maskstore_x1_ps(v_ptr, y0, mask);
    }

    return SUCCESS;
}

int sortasc_ignnan_s3_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s4_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 4 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

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
    if (r > 0) {
        __m128 x = _mm_load_ps(v_ptr);

        __m128 y = _mm_sort_ps(x);

        _mm_stream_ps(v_ptr, y);
    }

    return SUCCESS;
}

int sortasc_ignnan_s5_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s6_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s7_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s8_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != 8) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

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

    return SUCCESS;
}

int sortasc_ignnan_s9to11_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s <= AVX1_FLOAT_STRIDE * 2 || s >= AVX1_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    for (uint i = 0; i < n; i++) {
        shortsort_n9to11_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_ignnan_s12_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s != AVX1_FLOAT_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    for (uint i = 0; i < n; i++) {
        shortsort_n12_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_ignnan_s13to15_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s <= AVX1_FLOAT_STRIDE * 3 || s >= AVX1_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    for (uint i = 0; i < n; i++) {
        shortsort_n13to15_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_ignnan_s16to32_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s < AVX2_FLOAT_STRIDE * 2 || s > AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    for (uint i = 0; i < n; i++) {
        backtracksort_p8_s(s, v_ptr);
        scansort_p8_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_ignnan_slong_s(const uint n, const uint s, outfloats v_ptr) {
#ifdef _DEBUG
    if (s < AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    for (uint i = 0; i < n; i++) {
        longsort_s(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

#pragma endregion sort