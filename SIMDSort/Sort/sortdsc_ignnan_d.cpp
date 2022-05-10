#include "sort.h"
#include "../Inline/inline_cmp_d.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

#pragma region needs swap

// needs swap (sort order definition)
__forceinline static __m256d _mm256_needsswap_pd(__m256d x, __m256d y) {
    return _mm256_cmplt_ignnan_pd(x, y);
}

#pragma endregion needs swap

#pragma region needs sort

// needs sort
__forceinline static bool _mm256_needssort_pd(__m256d x) {
    __m256d y = _mm256_permute4x64_pd(x, _MM_PERM_DDCB);

    bool needssort = _mm256_movemask_pd(_mm256_needsswap_pd(x, y)) > 0;

    return needssort;
}

#pragma endregion needs sort

#pragma region horizontal sort

// sort batches4 x elems2 
__forceinline static __m256d _mm256_sort2x2_pd(__m256d x) {
    __m256d y = _mm256_permute4x64_pd(x, _MM_PERM_CDAB);
    __m256d c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_CCAA);
    __m256d z = _mm256_blendv_pd(x, y, c);

    return z;
}

// sort batches1 x elems3
__forceinline static __m256d _mm256_sort1x3_pd(__m256d x) {
    __m256d y, c;

    y = _mm256_permute4x64_pd(x, _MM_PERM_DCAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_DCAA);
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_DBCA);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_DBBA);
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_DCAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_DCAA);
    x = _mm256_blendv_pd(x, y, c);

    return x;
}

// sort elems4
__forceinline static __m256d _mm256_sort_pd(__m256d x) {
    __m256d y, c;

    y = _mm256_permute4x64_pd(x, _MM_PERM_CDAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_BADC);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_BABA);
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_ABCD);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_ABBA);
    x = _mm256_blendv_pd(x, y, c);

    return x;
}

// sort elems4 (ho, lo sorted)
__forceinline static __m256d _mm256_halfsort_pd(__m256d x) {
    __m256d y, c;

    y = _mm256_permute4x64_pd(x, _MM_PERM_BADC);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_BABA);
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_ABCD);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_ABBA);
    x = _mm256_blendv_pd(x, y, c);

    return x;
}

// sort elems8
__forceinline static __m256dx2 _mm256x2_sort_pd(__m256dx2 x) {
    __m256d y0, y1, c0, c1, swaps;

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, _mm256_permute4x64_pd(y1, _MM_PERM_ADCB));

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, _mm256_permute4x64_pd(y1, _MM_PERM_ADCB));

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, _mm256_permute4x64_pd(y1, _MM_PERM_ADCB));

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, y1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_ABBA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_ABBA);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_CDAB);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_CCAA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_CDAB);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_CCAA);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_ABBA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_ABBA);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    return x;
}

#pragma endregion horizontal sort

#pragma region cmp and swap

// compare and swap
__forceinline static void _mm256_cmpswap_pd(__m256d a, __m256d b, __m256d& x, __m256d& y) {
    __m256d swaps = _mm256_needsswap_pd(a, b);

    x = _mm256_blendv_pd(a, b, swaps);
    y = _mm256_blendv_pd(a, b, _mm256_not_pd(swaps));
}

// compare and swap
__forceinline static uint _mm256_cmpswap_indexed_pd(__m256d a, __m256d b, __m256d& x, __m256d& y) {
    __m256d swaps = _mm256_needsswap_pd(a, b);

    uint index = _mm256_movemask_pd(swaps);

    x = _mm256_blendv_pd(a, b, swaps);
    y = _mm256_blendv_pd(a, b, _mm256_not_pd(swaps));

    return index;
}

// compare and swap with permutate
__forceinline static void _mm256_cmpswap_withperm_pd(__m256d a, __m256d b, __m256d& x, __m256d& y) {

    _mm256_cmpswap_pd(a, b, x, y);

    a = _mm256_permute4x64_pd(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_pd(a, b, x, y);

    a = _mm256_permute4x64_pd(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_pd(a, b, x, y);

    a = _mm256_permute4x64_pd(x, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_pd(a, b, x, y);
}

#pragma endregion cmp and swap

#pragma region combsort

// combsort h=5...7
static int combsort_h5to7_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE || h >= AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, b0, b1;
    __m256d x0, x1, y0, y1;

    if (e > 0) {
        _mm256_maskload_x2_pd(v_ptr, a0, a1, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x2_pd(v_ptr + i + h, b0, b1, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);

            _mm256_maskstore_x2_pd(v_ptr + i, x0, x1, mask);

            a0 = y0;
            a1 = y1;
        }
        _mm256_maskstore_x2_pd(v_ptr + i, a0, a1, mask);
    }
    {
        _mm256_maskload_x2_pd(v_ptr + e, a0, a1, mask);
        _mm256_maskload_x2_pd(v_ptr + e + h, b0, b1, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        _mm256_maskstore_x2_pd(v_ptr + e, x0, x1, mask);
        _mm256_maskstore_x2_pd(v_ptr + e + h, y0, y1, mask);
    }

    return SUCCESS;
}

// combsort h=8
static int combsort_h8_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 4;

    __m256d a0, a1, b0, b1;
    __m256d x0, x1, y0, y1;

    if (e > 0) {
        _mm256_loadu_x2_pd(v_ptr, a0, a1);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 2, b0, b1);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);

            _mm256_storeu_x2_pd(v_ptr + i, x0, x1);

            a0 = y0;
            a1 = y1;
        }
        _mm256_storeu_x2_pd(v_ptr + i, a0, a1);
    }
    {
        _mm256_loadu_x2_pd(v_ptr + e, a0, a1);
        _mm256_loadu_x2_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 2, b0, b1);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        _mm256_storeu_x2_pd(v_ptr + e, x0, x1);
        _mm256_storeu_x2_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 2, y0, y1);
    }

    return SUCCESS;
}

// combsort h=9...11
static int combsort_h9to11_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 2 || h >= AVX2_DOUBLE_STRIDE * 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, a2, b0, b1, b2;
    __m256d x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_maskload_x3_pd(v_ptr, a0, a1, a2, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x3_pd(v_ptr + i + h, b0, b1, b2, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);

            _mm256_maskstore_x3_pd(v_ptr + i, x0, x1, x2, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_maskstore_x3_pd(v_ptr + i, a0, a1, a2, mask);
    }
    {
        _mm256_maskload_x3_pd(v_ptr + e, a0, a1, a2, mask);
        _mm256_maskload_x3_pd(v_ptr + e + h, b0, b1, b2, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);

        _mm256_maskstore_x3_pd(v_ptr + e, x0, x1, x2, mask);
        _mm256_maskstore_x3_pd(v_ptr + e + h, y0, y1, y2, mask);
    }

    return SUCCESS;
}

// combsort h=12
static int combsort_h12_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 6) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 6;

    __m256d a0, a1, a2, b0, b1, b2;
    __m256d x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_loadu_x3_pd(v_ptr, a0, a1, a2);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 3, b0, b1, b2);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);

            _mm256_storeu_x3_pd(v_ptr + i, x0, x1, x2);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_storeu_x3_pd(v_ptr + i, a0, a1, a2);
    }
    {
        _mm256_loadu_x3_pd(v_ptr + e, a0, a1, a2);
        _mm256_loadu_x3_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 3, b0, b1, b2);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);

        _mm256_storeu_x3_pd(v_ptr + e, x0, x1, x2);
        _mm256_storeu_x3_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 3, y0, y1, y2);
    }

    return SUCCESS;
}

// combsort h=13...15
static int combsort_h13to15_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 3 || h >= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_maskload_x4_pd(v_ptr, a0, a1, a2, a3, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x4_pd(v_ptr + i + h, b0, b1, b2, b3, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);

            _mm256_maskstore_x4_pd(v_ptr + i, x0, x1, x2, x3, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_maskstore_x4_pd(v_ptr + i, a0, a1, a2, a3, mask);
    }
    {
        _mm256_maskload_x4_pd(v_ptr + e, a0, a1, a2, a3, mask);
        _mm256_maskload_x4_pd(v_ptr + e + h, b0, b1, b2, b3, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_maskstore_x4_pd(v_ptr + e, x0, x1, x2, x3, mask);
        _mm256_maskstore_x4_pd(v_ptr + e + h, y0, y1, y2, y3, mask);
    }

    return SUCCESS;
}

// combsort h=16
static int combsort_h16_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 8) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 8;

    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_loadu_x4_pd(v_ptr, a0, a1, a2, a3);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 4, b0, b1, b2, b3);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);

            _mm256_storeu_x4_pd(v_ptr + i, x0, x1, x2, x3);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_storeu_x4_pd(v_ptr + i, a0, a1, a2, a3);
    }
    {
        _mm256_loadu_x4_pd(v_ptr + e, a0, a1, a2, a3);
        _mm256_loadu_x4_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 4, b0, b1, b2, b3);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_storeu_x4_pd(v_ptr + e, x0, x1, x2, x3);
        _mm256_storeu_x4_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 4, y0, y1, y2, y3);
    }

    return SUCCESS;
}

// combsort h>16
static int combsort_h17plus_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h + AVX2_DOUBLE_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_DOUBLE_STRIDE * 4;

    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    for (uint i = 0; i < e; i += AVX2_DOUBLE_STRIDE * 4) {
        _mm256_loadu_x4_pd(v_ptr + i, a0, a1, a2, a3);
        _mm256_loadu_x4_pd(v_ptr + i + h, b0, b1, b2, b3);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_storeu_x4_pd(v_ptr + i, x0, x1, x2, x3);
        _mm256_storeu_x4_pd(v_ptr + i + h, y0, y1, y2, y3);
    }
    {
        _mm256_loadu_x4_pd(v_ptr + e, a0, a1, a2, a3);
        _mm256_loadu_x4_pd(v_ptr + e + h, b0, b1, b2, b3);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_storeu_x4_pd(v_ptr + e, x0, x1, x2, x3);
        _mm256_storeu_x4_pd(v_ptr + e + h, y0, y1, y2, y3);
    }

    return SUCCESS;
}

#pragma endregion combsort

#pragma region paracombsort

// paracombsort 2x4
static int paracombsort_p2x4_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    const uint e = n - AVX2_DOUBLE_STRIDE * 4;
    const uint c = n % AVX2_DOUBLE_STRIDE;

    __m256d a0, a1, b0, b1;
    __m256d x0, x1, y0, y1;

    for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
        a1 = _mm256_loadu_pd(v_ptr + i);
        b1 = _mm256_loadu_pd(v_ptr + i + AVX2_DOUBLE_STRIDE);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        a0 = x1;
        a1 = y1;
        b1 = _mm256_loadu_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 2);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        b0 = _mm256_permute4x64_pd(x1, _MM_PERM_ABDC);
        a1 = y1;
        b1 = _mm256_loadu_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 3);
        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        for (j = i; j + AVX2_DOUBLE_STRIDE <= e; j += AVX2_DOUBLE_STRIDE) {
            _mm256_storeu_pd(v_ptr + j, x0);
            a0 = y0;
            b0 = _mm256_permute4x64_pd(x1, _MM_PERM_ABDC);
            a1 = y1;
            b1 = _mm256_loadu_pd(v_ptr + j + AVX2_DOUBLE_STRIDE * 4);
            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
        }

        _mm256_storeu_pd(v_ptr + j, x0);
        a0 = y0;
        b0 = _mm256_permute4x64_pd(x1, _MM_PERM_ABDC);
        _mm256_cmpswap_pd(a0, b0, x0, y0);
        j += AVX2_DOUBLE_STRIDE;

        _mm256_storeu_pd(v_ptr + j, x0);
        a0 = y0;
        b0 = y1;
        _mm256_cmpswap_pd(a0, b0, x0, y0);
        j += AVX2_DOUBLE_STRIDE;

        _mm256_storeu_pd(v_ptr + j, x0);
        _mm256_storeu_pd(v_ptr + j + AVX2_DOUBLE_STRIDE, y0);
    }

    return SUCCESS;
}

#pragma endregion paracombsort

#pragma region backtracksort

// backtracksort 4 elems wise
__forceinline static int backtracksort_p4_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 2) {
        return SUCCESS;
    }

    uint i = 0, e = n - AVX2_DOUBLE_STRIDE * 2;
    __m256d a = _mm256_loadu_pd(v_ptr), b = _mm256_loadu_pd(v_ptr + AVX2_DOUBLE_STRIDE);
    __m256d x, y;

    if (e <= 0) {
        _mm256_cmpswap_pd(a, b, x, y);

        _mm256_storeu_pd(v_ptr, x);
        _mm256_storeu_pd(v_ptr + AVX2_DOUBLE_STRIDE, y);

        return SUCCESS;
    }

    while (true) {
        int indexes = _mm256_cmpswap_indexed_pd(a, b, x, y);

        if (indexes > 0) {
            _mm256_storeu_pd(v_ptr + i, x);
            _mm256_storeu_pd(v_ptr + i + AVX2_DOUBLE_STRIDE, y);

            if (i >= AVX2_DOUBLE_STRIDE) {
                i -= AVX2_DOUBLE_STRIDE;
                a = _mm256_loadu_pd(v_ptr + i);
                b = x;
                continue;
            }
            else if (i > 0) {
                i = 0;
                a = _mm256_loadu_pd(v_ptr);
                b = _mm256_loadu_pd(v_ptr + AVX2_DOUBLE_STRIDE);
                continue;
            }
            else {
                i = AVX2_DOUBLE_STRIDE;
                if (i <= e) {
                    a = y;
                    b = _mm256_loadu_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2);
                    continue;
                }
            }
        }
        else if (i < e) {
            i += AVX2_DOUBLE_STRIDE;

            if (i <= e) {
                a = b;
                b = _mm256_loadu_pd(v_ptr + i + AVX2_DOUBLE_STRIDE);
                continue;
            }
        }
        else {
            break;
        }

        i = e;
        a = _mm256_loadu_pd(v_ptr + i);
        b = _mm256_loadu_pd(v_ptr + i + AVX2_DOUBLE_STRIDE);
    }

    return SUCCESS;
}

#pragma endregion backtracksort

#pragma region batchsort

// batchsort 4 elems wise
static int batchsort_p4_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE;

    __m256d x0, x1, x2, x3;
    __m256d y0, y1, y2, y3;

    double* const ve_ptr = v_ptr + e;

    {
        double* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);
            y2 = _mm256_sort_pd(x2);
            y3 = _mm256_sort_pd(x3);

            _mm256_storeu_x4_pd(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(vc_ptr, x0, x1, x2);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);
            y2 = _mm256_sort_pd(x2);

            _mm256_storeu_x3_pd(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(vc_ptr, x0, x1);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);

            _mm256_storeu_x2_pd(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(vc_ptr, x0);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ve_ptr, x0);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ve_ptr, y0);
        }
    }
    {
        double* vc_ptr = v_ptr + AVX2_DOUBLE_STRIDE / 2;
        uint r = n - AVX2_DOUBLE_STRIDE / 2;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);
            y3 = _mm256_halfsort_pd(x3);

            _mm256_storeu_x4_pd(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(vc_ptr, x0, x1, x2);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);

            _mm256_storeu_x3_pd(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(vc_ptr, x0, x1);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);

            _mm256_storeu_x2_pd(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(vc_ptr, x0);

            y0 = _mm256_halfsort_pd(x0);

            _mm256_storeu_x1_pd(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ve_ptr, x0);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ve_ptr, y0);
        }
    }
    {
        double* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);
            y3 = _mm256_halfsort_pd(x3);

            _mm256_storeu_x4_pd(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(vc_ptr, x0, x1, x2);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);

            _mm256_storeu_x3_pd(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(vc_ptr, x0, x1);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);

            _mm256_storeu_x2_pd(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(vc_ptr, x0);

            y0 = _mm256_halfsort_pd(x0);

            _mm256_storeu_x1_pd(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ve_ptr, x0);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ve_ptr, y0);
        }
    }
    {
        double* vc_ptr = v_ptr + AVX2_DOUBLE_STRIDE / 2;
        uint r = n - AVX2_DOUBLE_STRIDE / 2;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(vc_ptr, x0, x1, x2, x3);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);
            y3 = _mm256_halfsort_pd(x3);

            _mm256_storeu_x4_pd(vc_ptr, y0, y1, y2, y3);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(vc_ptr, x0, x1, x2);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);

            _mm256_storeu_x3_pd(vc_ptr, y0, y1, y2);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(vc_ptr, x0, x1);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);

            _mm256_storeu_x2_pd(vc_ptr, y0, y1);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(vc_ptr, x0);

            y0 = _mm256_halfsort_pd(x0);

            _mm256_storeu_x1_pd(vc_ptr, y0);
        }
        if ((r & AVX2_FLOAT_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ve_ptr, x0);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ve_ptr, y0);
        }
    }

    return SUCCESS;
}

#pragma endregion batchsort

#pragma region scansort

// scansort 4 elems wise
__forceinline static int scansort_p4_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    uint e = n - AVX2_DOUBLE_STRIDE;

    uint indexes;
    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    {
        uint i = 0;
        while (true) {
            if (i + AVX2_DOUBLE_STRIDE * 4 + 1 <= n) {
                _mm256_loadu_x4_pd(v_ptr + i, x0, x1, x2, x3);
                _mm256_loadu_x4_pd(v_ptr + i + 1, y0, y1, y2, y3);

                uint i0 = _mm256_movemask_pd(_mm256_needsswap_pd(x0, y0));
                uint i1 = _mm256_movemask_pd(_mm256_needsswap_pd(x1, y1));
                uint i2 = _mm256_movemask_pd(_mm256_needsswap_pd(x2, y2));
                uint i3 = _mm256_movemask_pd(_mm256_needsswap_pd(x3, y3));

                indexes = (i0) | (i1 << (AVX2_DOUBLE_STRIDE)) | (i2 << (AVX2_DOUBLE_STRIDE * 2)) | (i3 << (AVX2_DOUBLE_STRIDE * 3));

                if (indexes == 0u) {
                    i += AVX2_DOUBLE_STRIDE * 4;
                    continue;
                }
            }
            else if (i + AVX2_DOUBLE_STRIDE * 2 + 1 <= n) {
                _mm256_loadu_x2_pd(v_ptr + i, x0, x1);
                _mm256_loadu_x2_pd(v_ptr + i + 1, y0, y1);

                uint i0 = _mm256_movemask_pd(_mm256_needsswap_pd(x0, y0));
                uint i1 = _mm256_movemask_pd(_mm256_needsswap_pd(x1, y1));

                indexes = (i0) | (i1 << (AVX2_DOUBLE_STRIDE));

                if (indexes == 0u) {
                    i += AVX2_DOUBLE_STRIDE * 2;
                    continue;
                }
            }
            else if (i + AVX2_DOUBLE_STRIDE + 1 <= n) {
                _mm256_loadu_x1_pd(v_ptr + i, x0);
                _mm256_loadu_x1_pd(v_ptr + i + 1, y0);

                indexes = (uint)_mm256_movemask_pd(_mm256_needsswap_pd(x0, y0));

                if (indexes == 0u) {
                    i += AVX2_DOUBLE_STRIDE;
                    continue;
                }
            }
            else {
                i = e;

                x0 = _mm256_loadu_pd(v_ptr + i);

                if (!_mm256_needssort_pd(x0)) {
                    break;
                }

                y0 = _mm256_sort_pd(x0);
                _mm256_storeu_pd(v_ptr + i, y0);

                indexes = 0xFu - (uint)_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, y0));

                if ((indexes & 1u) == 0u || i == 0) {
                    break;
                }
            }

            uint index = bsf(indexes);

            if (index >= AVX2_DOUBLE_STRIDE - 2) {
                uint forward = index - (AVX2_DOUBLE_STRIDE - 2);
                i += forward;
            }
            else {
                uint backward = (AVX2_DOUBLE_STRIDE - 2) - index;
                i = (i > backward) ? i - backward : 0;
            }

            x0 = _mm256_loadu_pd(v_ptr + i);

            while (true) {
                y0 = _mm256_sort_pd(x0);
                _mm256_storeu_pd(v_ptr + i, y0);

                indexes = (uint)_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0, y0));
                if ((indexes & 1u) == 0u && i > 0u) {
                    uint backward = AVX2_DOUBLE_STRIDE - 2;
                    i = (i > backward) ? i - backward : 0;

                    x0 = _mm256_loadu_pd(v_ptr + i);

                    if (_mm256_needssort_pd(x0)) {
                        continue;
                    }
                }

                uint forward = AVX2_DOUBLE_STRIDE - 1;
                i += forward;
                break;
            }
        }
    }

    return SUCCESS;
}

#pragma endregion scansort

#pragma region permcombsort

// permcombsort 4 elems wise
__forceinline static int permcombsort_p4_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_DOUBLE_STRIDE * 4 || n >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_DOUBLE_STRIDE;

    __m256d a, b, x, y;

    for (uint h = c > 2 ? 2 : 1; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a = _mm256_loadu_pd(v_ptr + i * AVX2_DOUBLE_STRIDE);
            b = _mm256_loadu_pd(v_ptr + (i + h) * AVX2_DOUBLE_STRIDE);

            _mm256_cmpswap_withperm_pd(a, b, x, y);

            _mm256_storeu_pd(v_ptr + i * AVX2_DOUBLE_STRIDE, x);
            _mm256_storeu_pd(v_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y);
        }
    }

    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0) {
        b = _mm256_loadu_pd(v_ptr + (n - AVX2_DOUBLE_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a = _mm256_loadu_pd(v_ptr + i * AVX2_DOUBLE_STRIDE);

            _mm256_cmpswap_pd(a, b, x, y);
            x = _mm256_sort_pd(x);
            b = y;

            _mm256_storeu_pd(v_ptr + i * AVX2_DOUBLE_STRIDE, x);
        }

        a = _mm256_loadu_pd(v_ptr + (n - AVX2_DOUBLE_STRIDE * 2));

        _mm256_cmpswap_withperm_pd(a, b, x, y);

        x = _mm256_sort_pd(x);
        y = _mm256_sort_pd(y);

        _mm256_storeu_pd(v_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x);
        _mm256_storeu_pd(v_ptr + (n - AVX2_DOUBLE_STRIDE), y);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a = _mm256_loadu_pd(v_ptr + i * AVX2_DOUBLE_STRIDE);
            a = _mm256_sort_pd(a);
            _mm256_storeu_pd(v_ptr + i * AVX2_DOUBLE_STRIDE, a);
        }
    }

    return SUCCESS;
}

// permcombsort 4 elems wise 4 batches
__forceinline static int permcombsort_p4x4_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_DOUBLE_STRIDE * 4 || n >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_DOUBLE_STRIDE;

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + n;
    double* v2_ptr = v_ptr + n * 2;
    double* v3_ptr = v_ptr + n * 3;

    __m256d a0, b0, x0, y0, a1, b1, x1, y1, a2, b2, x2, y2, a3, b3, x3, y3;

    for (uint h = c > 4 ? 4 : 2; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a0 = _mm256_loadu_pd(v0_ptr + i * AVX2_DOUBLE_STRIDE);
            a1 = _mm256_loadu_pd(v1_ptr + i * AVX2_DOUBLE_STRIDE);
            a2 = _mm256_loadu_pd(v2_ptr + i * AVX2_DOUBLE_STRIDE);
            a3 = _mm256_loadu_pd(v3_ptr + i * AVX2_DOUBLE_STRIDE);
            b0 = _mm256_loadu_pd(v0_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b1 = _mm256_loadu_pd(v1_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b2 = _mm256_loadu_pd(v2_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b3 = _mm256_loadu_pd(v3_ptr + (i + h) * AVX2_DOUBLE_STRIDE);

            _mm256_cmpswap_withperm_pd(a0, b0, x0, y0);
            _mm256_cmpswap_withperm_pd(a1, b1, x1, y1);
            _mm256_cmpswap_withperm_pd(a2, b2, x2, y2);
            _mm256_cmpswap_withperm_pd(a3, b3, x3, y3);

            _mm256_storeu_pd(v0_ptr + i * AVX2_DOUBLE_STRIDE, x0);
            _mm256_storeu_pd(v1_ptr + i * AVX2_DOUBLE_STRIDE, x1);
            _mm256_storeu_pd(v2_ptr + i * AVX2_DOUBLE_STRIDE, x2);
            _mm256_storeu_pd(v3_ptr + i * AVX2_DOUBLE_STRIDE, x3);
            _mm256_storeu_pd(v0_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y0);
            _mm256_storeu_pd(v1_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y1);
            _mm256_storeu_pd(v2_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y2);
            _mm256_storeu_pd(v3_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y3);
        }
    }

    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0) {
        b0 = _mm256_loadu_pd(v0_ptr + (n - AVX2_DOUBLE_STRIDE));
        b1 = _mm256_loadu_pd(v1_ptr + (n - AVX2_DOUBLE_STRIDE));
        b2 = _mm256_loadu_pd(v2_ptr + (n - AVX2_DOUBLE_STRIDE));
        b3 = _mm256_loadu_pd(v3_ptr + (n - AVX2_DOUBLE_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a0 = _mm256_loadu_pd(v0_ptr + i * AVX2_DOUBLE_STRIDE);
            a1 = _mm256_loadu_pd(v1_ptr + i * AVX2_DOUBLE_STRIDE);
            a2 = _mm256_loadu_pd(v2_ptr + i * AVX2_DOUBLE_STRIDE);
            a3 = _mm256_loadu_pd(v3_ptr + i * AVX2_DOUBLE_STRIDE);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);

            x0 = _mm256_sort_pd(x0);
            x1 = _mm256_sort_pd(x1);
            x2 = _mm256_sort_pd(x2);
            x3 = _mm256_sort_pd(x3);
            b0 = y0;
            b1 = y1;
            b2 = y2;
            b3 = y3;

            _mm256_storeu_pd(v0_ptr + i * AVX2_DOUBLE_STRIDE, x0);
            _mm256_storeu_pd(v1_ptr + i * AVX2_DOUBLE_STRIDE, x1);
            _mm256_storeu_pd(v2_ptr + i * AVX2_DOUBLE_STRIDE, x2);
            _mm256_storeu_pd(v3_ptr + i * AVX2_DOUBLE_STRIDE, x3);
        }

        a0 = _mm256_loadu_pd(v0_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a1 = _mm256_loadu_pd(v1_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a2 = _mm256_loadu_pd(v2_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a3 = _mm256_loadu_pd(v3_ptr + (n - AVX2_DOUBLE_STRIDE * 2));

        _mm256_cmpswap_withperm_pd(a0, b0, x0, y0);
        _mm256_cmpswap_withperm_pd(a1, b1, x1, y1);
        _mm256_cmpswap_withperm_pd(a2, b2, x2, y2);
        _mm256_cmpswap_withperm_pd(a3, b3, x3, y3);

        x0 = _mm256_sort_pd(x0);
        x1 = _mm256_sort_pd(x1);
        x2 = _mm256_sort_pd(x2);
        x3 = _mm256_sort_pd(x3);
        y0 = _mm256_sort_pd(y0);
        y1 = _mm256_sort_pd(y1);
        y2 = _mm256_sort_pd(y2);
        y3 = _mm256_sort_pd(y3);

        _mm256_storeu_pd(v0_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x0);
        _mm256_storeu_pd(v1_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x1);
        _mm256_storeu_pd(v2_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x2);
        _mm256_storeu_pd(v3_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x3);
        _mm256_storeu_pd(v0_ptr + (n - AVX2_DOUBLE_STRIDE), y0);
        _mm256_storeu_pd(v1_ptr + (n - AVX2_DOUBLE_STRIDE), y1);
        _mm256_storeu_pd(v2_ptr + (n - AVX2_DOUBLE_STRIDE), y2);
        _mm256_storeu_pd(v3_ptr + (n - AVX2_DOUBLE_STRIDE), y3);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a0 = _mm256_loadu_pd(v0_ptr + i * AVX2_DOUBLE_STRIDE);
            a1 = _mm256_loadu_pd(v1_ptr + i * AVX2_DOUBLE_STRIDE);
            a2 = _mm256_loadu_pd(v2_ptr + i * AVX2_DOUBLE_STRIDE);
            a3 = _mm256_loadu_pd(v3_ptr + i * AVX2_DOUBLE_STRIDE);
            a0 = _mm256_sort_pd(a0);
            a1 = _mm256_sort_pd(a1);
            a2 = _mm256_sort_pd(a2);
            a3 = _mm256_sort_pd(a3);
            _mm256_storeu_pd(v0_ptr + i * AVX2_DOUBLE_STRIDE, a0);
            _mm256_storeu_pd(v1_ptr + i * AVX2_DOUBLE_STRIDE, a1);
            _mm256_storeu_pd(v2_ptr + i * AVX2_DOUBLE_STRIDE, a2);
            _mm256_storeu_pd(v3_ptr + i * AVX2_DOUBLE_STRIDE, a3);
        }
    }

    return SUCCESS;
}

#pragma endregion permcombsort

#pragma region shortsort

// shortsort elems5
__forceinline static int shortsort_n5_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256d x, y;

    x = _mm256_loadu_pd(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr, y);

    x = _mm256_loadu_pd(v_ptr + 1);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 1, y);

    x = _mm256_loadu_pd(v_ptr);
    y = _mm256_sort2x2_pd(x);
    _mm256_storeu_pd(v_ptr, y);

    return SUCCESS;
}

// shortsort batches4 x elems5
__forceinline static int shortsort_n4x5_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 5;
    double* v2_ptr = v_ptr + 10;
    double* v3_ptr = v_ptr + 15;

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_pd(v0_ptr);
    x1 = _mm256_loadu_pd(v1_ptr);
    x2 = _mm256_loadu_pd(v2_ptr);
    x3 = _mm256_loadu_pd(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr, y0);
    _mm256_storeu_pd(v1_ptr, y1);
    _mm256_storeu_pd(v2_ptr, y2);
    _mm256_storeu_pd(v3_ptr, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 1);
    x1 = _mm256_loadu_pd(v1_ptr + 1);
    x2 = _mm256_loadu_pd(v2_ptr + 1);
    x3 = _mm256_loadu_pd(v3_ptr + 1);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 1, y0);
    _mm256_storeu_pd(v1_ptr + 1, y1);
    _mm256_storeu_pd(v2_ptr + 1, y2);
    _mm256_storeu_pd(v3_ptr + 1, y3);

    x0 = _mm256_loadu_pd(v0_ptr);
    x1 = _mm256_loadu_pd(v1_ptr);
    x2 = _mm256_loadu_pd(v2_ptr);
    x3 = _mm256_loadu_pd(v3_ptr);
    y0 = _mm256_sort2x2_pd(x0);
    y1 = _mm256_sort2x2_pd(x1);
    y2 = _mm256_sort2x2_pd(x2);
    y3 = _mm256_sort2x2_pd(x3);
    _mm256_storeu_pd(v0_ptr, y0);
    _mm256_storeu_pd(v1_ptr, y1);
    _mm256_storeu_pd(v2_ptr, y2);
    _mm256_storeu_pd(v3_ptr, y3);

    return SUCCESS;
}

// shortsort elems6
__forceinline static int shortsort_n6_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256d x, y;

    x = _mm256_loadu_pd(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr, y);

    x = _mm256_loadu_pd(v_ptr + 2);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 2, y);

    x = _mm256_loadu_pd(v_ptr + 1);
    y = _mm256_sort2x2_pd(x);
    _mm256_storeu_pd(v_ptr + 1, y);

    x = _mm256_loadu_pd(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr, y);

    x = _mm256_loadu_pd(v_ptr + 2);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 2, y);

    return SUCCESS;
}

// shortsort batches4 x elems6
__forceinline static int shortsort_n4x6_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 6;
    double* v2_ptr = v_ptr + 12;
    double* v3_ptr = v_ptr + 18;

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_pd(v0_ptr);
    x1 = _mm256_loadu_pd(v1_ptr);
    x2 = _mm256_loadu_pd(v2_ptr);
    x3 = _mm256_loadu_pd(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr, y0);
    _mm256_storeu_pd(v1_ptr, y1);
    _mm256_storeu_pd(v2_ptr, y2);
    _mm256_storeu_pd(v3_ptr, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 2);
    x1 = _mm256_loadu_pd(v1_ptr + 2);
    x2 = _mm256_loadu_pd(v2_ptr + 2);
    x3 = _mm256_loadu_pd(v3_ptr + 2);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 2, y0);
    _mm256_storeu_pd(v1_ptr + 2, y1);
    _mm256_storeu_pd(v2_ptr + 2, y2);
    _mm256_storeu_pd(v3_ptr + 2, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 1);
    x1 = _mm256_loadu_pd(v1_ptr + 1);
    x2 = _mm256_loadu_pd(v2_ptr + 1);
    x3 = _mm256_loadu_pd(v3_ptr + 1);
    y0 = _mm256_sort2x2_pd(x0);
    y1 = _mm256_sort2x2_pd(x1);
    y2 = _mm256_sort2x2_pd(x2);
    y3 = _mm256_sort2x2_pd(x3);
    _mm256_storeu_pd(v0_ptr + 1, y0);
    _mm256_storeu_pd(v1_ptr + 1, y1);
    _mm256_storeu_pd(v2_ptr + 1, y2);
    _mm256_storeu_pd(v3_ptr + 1, y3);

    x0 = _mm256_loadu_pd(v0_ptr);
    x1 = _mm256_loadu_pd(v1_ptr);
    x2 = _mm256_loadu_pd(v2_ptr);
    x3 = _mm256_loadu_pd(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr, y0);
    _mm256_storeu_pd(v1_ptr, y1);
    _mm256_storeu_pd(v2_ptr, y2);
    _mm256_storeu_pd(v3_ptr, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 2);
    x1 = _mm256_loadu_pd(v1_ptr + 2);
    x2 = _mm256_loadu_pd(v2_ptr + 2);
    x3 = _mm256_loadu_pd(v3_ptr + 2);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 2, y0);
    _mm256_storeu_pd(v1_ptr + 2, y1);
    _mm256_storeu_pd(v2_ptr + 2, y2);
    _mm256_storeu_pd(v3_ptr + 2, y3);

    return SUCCESS;
}

// shortsort elems7
__forceinline static int shortsort_n7_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256d x, y;

    x = _mm256_loadu_pd(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr, y);

    x = _mm256_loadu_pd(v_ptr + 3);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 3, y);

    x = _mm256_loadu_pd(v_ptr + 1);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 1, y);

    x = _mm256_loadu_pd(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr, y);

    x = _mm256_loadu_pd(v_ptr + 3);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 3, y);

    x = _mm256_loadu_pd(v_ptr + 2);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(v_ptr + 2, y);

    return SUCCESS;
}

// shortsort batches4 x elems7
__forceinline static int shortsort_n4x7_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 7;
    double* v2_ptr = v_ptr + 14;
    double* v3_ptr = v_ptr + 21;

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    x0 = _mm256_loadu_pd(v0_ptr);
    x1 = _mm256_loadu_pd(v1_ptr);
    x2 = _mm256_loadu_pd(v2_ptr);
    x3 = _mm256_loadu_pd(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr, y0);
    _mm256_storeu_pd(v1_ptr, y1);
    _mm256_storeu_pd(v2_ptr, y2);
    _mm256_storeu_pd(v3_ptr, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 3);
    x1 = _mm256_loadu_pd(v1_ptr + 3);
    x2 = _mm256_loadu_pd(v2_ptr + 3);
    x3 = _mm256_loadu_pd(v3_ptr + 3);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 3, y0);
    _mm256_storeu_pd(v1_ptr + 3, y1);
    _mm256_storeu_pd(v2_ptr + 3, y2);
    _mm256_storeu_pd(v3_ptr + 3, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 1);
    x1 = _mm256_loadu_pd(v1_ptr + 1);
    x2 = _mm256_loadu_pd(v2_ptr + 1);
    x3 = _mm256_loadu_pd(v3_ptr + 1);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 1, y0);
    _mm256_storeu_pd(v1_ptr + 1, y1);
    _mm256_storeu_pd(v2_ptr + 1, y2);
    _mm256_storeu_pd(v3_ptr + 1, y3);

    x0 = _mm256_loadu_pd(v0_ptr);
    x1 = _mm256_loadu_pd(v1_ptr);
    x2 = _mm256_loadu_pd(v2_ptr);
    x3 = _mm256_loadu_pd(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr, y0);
    _mm256_storeu_pd(v1_ptr, y1);
    _mm256_storeu_pd(v2_ptr, y2);
    _mm256_storeu_pd(v3_ptr, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 3);
    x1 = _mm256_loadu_pd(v1_ptr + 3);
    x2 = _mm256_loadu_pd(v2_ptr + 3);
    x3 = _mm256_loadu_pd(v3_ptr + 3);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 3, y0);
    _mm256_storeu_pd(v1_ptr + 3, y1);
    _mm256_storeu_pd(v2_ptr + 3, y2);
    _mm256_storeu_pd(v3_ptr + 3, y3);

    x0 = _mm256_loadu_pd(v0_ptr + 2);
    x1 = _mm256_loadu_pd(v1_ptr + 2);
    x2 = _mm256_loadu_pd(v2_ptr + 2);
    x3 = _mm256_loadu_pd(v3_ptr + 2);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(v0_ptr + 2, y0);
    _mm256_storeu_pd(v1_ptr + 2, y1);
    _mm256_storeu_pd(v2_ptr + 2, y2);
    _mm256_storeu_pd(v3_ptr + 2, y3);

    return SUCCESS;
}

// shortsort elems9
__forceinline static int shortsort_n9_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 1, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 1, y.imm0, y.imm1);

    x.imm0 = _mm256_loadu_pd(v_ptr);
    y.imm0 = _mm256_sort2x2_pd(x.imm0);
    _mm256_storeu_pd(v_ptr, y.imm0);

    return SUCCESS;
}

// shortsort batches4 x elems9
__forceinline static int shortsort_n4x9_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 9;
    double* v2_ptr = v_ptr + 18;
    double* v3_ptr = v_ptr + 27;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 1, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 1, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 1, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 1, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 1, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 1, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 1, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 1, y3.imm0, y3.imm1);

    x0.imm0 = _mm256_loadu_pd(v0_ptr);
    x1.imm0 = _mm256_loadu_pd(v1_ptr);
    x2.imm0 = _mm256_loadu_pd(v2_ptr);
    x3.imm0 = _mm256_loadu_pd(v3_ptr);
    y0.imm0 = _mm256_sort2x2_pd(x0.imm0);
    y1.imm0 = _mm256_sort2x2_pd(x1.imm0);
    y2.imm0 = _mm256_sort2x2_pd(x2.imm0);
    y3.imm0 = _mm256_sort2x2_pd(x3.imm0);
    _mm256_storeu_pd(v0_ptr, y0.imm0);
    _mm256_storeu_pd(v1_ptr, y1.imm0);
    _mm256_storeu_pd(v2_ptr, y2.imm0);
    _mm256_storeu_pd(v3_ptr, y3.imm0);

    return SUCCESS;
}

// shortsort elems10
__forceinline static int shortsort_n10_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 2, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 2, y.imm0, y.imm1);

    x.imm0 = _mm256_loadu_pd(v_ptr);
    y.imm0 = _mm256_sort_pd(x.imm0);
    _mm256_storeu_pd(v_ptr, y.imm0);

    return SUCCESS;
}

// shortsort batches4 x elems10
__forceinline static int shortsort_n4x10_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 10;
    double* v2_ptr = v_ptr + 20;
    double* v3_ptr = v_ptr + 30;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 2, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 2, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 2, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 2, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 2, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 2, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 2, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 2, y3.imm0, y3.imm1);

    x0.imm0 = _mm256_loadu_pd(v0_ptr);
    x1.imm0 = _mm256_loadu_pd(v1_ptr);
    x2.imm0 = _mm256_loadu_pd(v2_ptr);
    x3.imm0 = _mm256_loadu_pd(v3_ptr);
    y0.imm0 = _mm256_sort_pd(x0.imm0);
    y1.imm0 = _mm256_sort_pd(x1.imm0);
    y2.imm0 = _mm256_sort_pd(x2.imm0);
    y3.imm0 = _mm256_sort_pd(x3.imm0);
    _mm256_storeu_pd(v0_ptr, y0.imm0);
    _mm256_storeu_pd(v1_ptr, y1.imm0);
    _mm256_storeu_pd(v2_ptr, y2.imm0);
    _mm256_storeu_pd(v3_ptr, y3.imm0);

    return SUCCESS;
}

// shortsort elems11
__forceinline static int shortsort_n11_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 3, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 3, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    return SUCCESS;
}

// shortsort batches4 x elems11
__forceinline static int shortsort_n4x11_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 11;
    double* v2_ptr = v_ptr + 22;
    double* v3_ptr = v_ptr + 33;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 3, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 3, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 3, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 3, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 3, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 3, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 3, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 3, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    return SUCCESS;
}

// shortsort elems12
__forceinline static int shortsort_n12_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 4, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 4, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    return SUCCESS;
}

// shortsort batches4 x elems12
__forceinline static int shortsort_n4x12_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 12;
    double* v2_ptr = v_ptr + 24;
    double* v3_ptr = v_ptr + 36;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 4, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 4, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 4, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 4, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 4, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 4, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 4, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 4, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    return SUCCESS;
}

// shortsort elems13
__forceinline static int shortsort_n13_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 5, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 5, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 2, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 2, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 5, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 5, y.imm0, y.imm1);

    return SUCCESS;
}

// shortsort batches4 x elems13
__forceinline static int shortsort_n4x13_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 13;
    double* v2_ptr = v_ptr + 26;
    double* v3_ptr = v_ptr + 39;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 5, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 5, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 5, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 5, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 5, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 5, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 5, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 5, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 2, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 2, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 2, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 2, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 2, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 2, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 2, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 2, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 5, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 5, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 5, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 5, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 5, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 5, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 5, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 5, y3.imm0, y3.imm1);

    return SUCCESS;
}

// shortsort elems14
__forceinline static int shortsort_n14_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 6, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 6, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 3, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 3, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 6, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 6, y.imm0, y.imm1);

    x.imm0 = _mm256_loadu_pd(v_ptr + 5);
    y.imm0 = _mm256_sort2x2_pd(x.imm0);
    _mm256_storeu_pd(v_ptr + 5, y.imm0);

    return SUCCESS;
}

// shortsort batches4 x elems14
__forceinline static int shortsort_n4x14_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 14;
    double* v2_ptr = v_ptr + 28;
    double* v3_ptr = v_ptr + 42;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 6, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 6, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 6, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 6, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 6, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 6, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 6, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 6, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 3, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 3, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 3, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 3, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 3, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 3, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 3, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 3, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 6, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 6, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 6, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 6, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 6, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 6, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 6, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 6, y3.imm0, y3.imm1);

    x0.imm0 = _mm256_loadu_pd(v0_ptr + 5);
    x1.imm0 = _mm256_loadu_pd(v1_ptr + 5);
    x2.imm0 = _mm256_loadu_pd(v2_ptr + 5);
    x3.imm0 = _mm256_loadu_pd(v3_ptr + 5);
    y0.imm0 = _mm256_sort2x2_pd(x0.imm0);
    y1.imm0 = _mm256_sort2x2_pd(x1.imm0);
    y2.imm0 = _mm256_sort2x2_pd(x2.imm0);
    y3.imm0 = _mm256_sort2x2_pd(x3.imm0);
    _mm256_storeu_pd(v0_ptr + 5, y0.imm0);
    _mm256_storeu_pd(v1_ptr + 5, y1.imm0);
    _mm256_storeu_pd(v2_ptr + 5, y2.imm0);
    _mm256_storeu_pd(v3_ptr + 5, y3.imm0);

    return SUCCESS;
}

// shortsort elems15
__forceinline static int shortsort_n15_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 7, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 7, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 3, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 3, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 7, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 7, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + 4, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + 4, y.imm0, y.imm1);

    return SUCCESS;
}

// shortsort batches4 x elems15
__forceinline static int shortsort_n4x15_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 15;
    double* v2_ptr = v_ptr + 30;
    double* v3_ptr = v_ptr + 45;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 7, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 7, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 7, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 7, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 7, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 7, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 7, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 7, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 3, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 3, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 3, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 3, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 3, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 3, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 3, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 3, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 7, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 7, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 7, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 7, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 7, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 7, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 7, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 7, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + 4, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + 4, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + 4, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + 4, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + 4, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + 4, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + 4, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + 4, y3.imm0, y3.imm1);

    return SUCCESS;
}

// shortsort elems16
__forceinline static int shortsort_n16_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dx2 x, y;

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, y.imm0, y.imm1);

    _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE, x.imm0, x.imm1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE, y.imm0, y.imm1);

    return SUCCESS;
}

// shortsort batches4 x elems16
__forceinline static int shortsort_n4x16_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n != AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* v0_ptr = v_ptr;
    double* v1_ptr = v_ptr + 16;
    double* v2_ptr = v_ptr + 32;
    double* v3_ptr = v_ptr + 48;

    __m256dx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE * 2, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE * 2, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE * 2, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE * 2, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE * 2, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE * 2, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE * 2, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE * 2, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE * 2, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE * 2, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE * 2, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE * 2, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE * 2, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE * 2, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE * 2, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE * 2, y3.imm0, y3.imm1);

    _mm256_loadu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE, x0.imm0, x0.imm1);
    _mm256_loadu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE, x1.imm0, x1.imm1);
    _mm256_loadu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE, x2.imm0, x2.imm1);
    _mm256_loadu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE, x3.imm0, x3.imm1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(v0_ptr + AVX2_DOUBLE_STRIDE, y0.imm0, y0.imm1);
    _mm256_storeu_x2_pd(v1_ptr + AVX2_DOUBLE_STRIDE, y1.imm0, y1.imm1);
    _mm256_storeu_x2_pd(v2_ptr + AVX2_DOUBLE_STRIDE, y2.imm0, y2.imm1);
    _mm256_storeu_x2_pd(v3_ptr + AVX2_DOUBLE_STRIDE, y3.imm0, y3.imm1);

    return SUCCESS;
}

// shortsort elems 17...31
__forceinline static int shortsort_n17to31_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_DOUBLE_STRIDE * 4 || n >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    combsort_h12_d(n, v_ptr);
    paracombsort_p2x4_d(n, v_ptr);

    batchsort_p4_d(n, v_ptr);
    scansort_p4_d(n, v_ptr);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

// longsort elems 32+
__forceinline static int longsort_n32plus_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    uint h;

    for (h = (uint)(n * 10uLL / 13uLL); h > 33; h = (uint)(h * 10uLL / 13uLL)) {
        combsort_h17plus_d(n, h, v_ptr);
    }
    if (h >= 16) {
        combsort_h16_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 12; h = h * 10 / 13) {
        combsort_h13to15_d(n, h, v_ptr);
    }
    if (h >= 12) {
        combsort_h12_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 8; h = h * 10 / 13) {
        combsort_h9to11_d(n, h, v_ptr);
    }
    if (h >= 8) {
        combsort_h8_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 4; h = h * 10 / 13) {
        combsort_h5to7_d(n, h, v_ptr);
    }

    paracombsort_p2x4_d(n, v_ptr);

    batchsort_p4_d(n, v_ptr);
    scansort_p4_d(n, v_ptr);

    return SUCCESS;
}

#pragma endregion longsort

#pragma region sort

int sortdsc_ignnan_s2_d(const uint n, const uint s, double* v_ptr) {
    if (s != 2 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE * 4 / 2) {
        _mm256_load_x4_pd(v_ptr, x0, x1, x2, x3);

        y0 = _mm256_sort2x2_pd(x0);
        y1 = _mm256_sort2x2_pd(x1);
        y2 = _mm256_sort2x2_pd(x2);
        y3 = _mm256_sort2x2_pd(x3);

        _mm256_stream_x4_pd(v_ptr, y0, y1, y2, y3);

        v_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE * 4 / 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x2_pd(v_ptr, x0, x1);

        y0 = _mm256_sort2x2_pd(x0);
        y1 = _mm256_sort2x2_pd(x1);

        _mm256_stream_x2_pd(v_ptr, y0, y1);

        v_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x1_pd(v_ptr, x0);

        y0 = _mm256_sort2x2_pd(x0);

        _mm256_stream_x1_pd(v_ptr, y0);

        v_ptr += AVX2_DOUBLE_STRIDE;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd((r * 2) & AVX2_DOUBLE_REMAIN_MASK);

        _mm256_maskload_x1_pd(v_ptr, x0, mask);

        y0 = _mm256_sort2x2_pd(x0);

        _mm256_maskstore_x1_pd(v_ptr, y0, mask);
    }

    return SUCCESS;
}

int sortdsc_ignnan_s3_d(const uint n, const uint s, double* v_ptr) {
    if (s != 3) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0 = _mm256_loadu_pd(v_ptr);
        x1 = _mm256_loadu_pd(v_ptr + 3);
        x2 = _mm256_loadu_pd(v_ptr + 6);
        x3 = _mm256_loadu_pd(v_ptr + 9);

        y0 = _mm256_sort1x3_pd(x0);
        y1 = _mm256_sort1x3_pd(x1);
        y2 = _mm256_sort1x3_pd(x2);
        y3 = _mm256_sort1x3_pd(x3);

        _mm256_storeu_pd(v_ptr, y0);
        _mm256_storeu_pd(v_ptr + 3, y1);
        _mm256_storeu_pd(v_ptr + 6, y2);
        _mm256_storeu_pd(v_ptr + 9, y3);

        v_ptr += AVX2_DOUBLE_STRIDE * 3;
        r -= 4;
    }
    while (r >= 1) {
        const __m256i mask = _mm256_setmask_pd(3);

        x0 = _mm256_maskload_pd(v_ptr, mask);

        y0 = _mm256_sort1x3_pd(x0);

        _mm256_maskstore_pd(v_ptr, mask, y0);

        v_ptr += 3;
        r -= 1;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s4_d(const uint n, const uint s, double* v_ptr) {
    if (s != 4 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }

    __m256d x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= AVX2_DOUBLE_STRIDE) {
        _mm256_load_x4_pd(v_ptr, x0, x1, x2, x3);

        y0 = _mm256_sort_pd(x0);
        y1 = _mm256_sort_pd(x1);
        y2 = _mm256_sort_pd(x2);
        y3 = _mm256_sort_pd(x3);

        _mm256_stream_x4_pd(v_ptr, y0, y1, y2, y3);

        v_ptr += AVX2_DOUBLE_STRIDE * 4;
        r -= AVX2_DOUBLE_STRIDE;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 2) {
        _mm256_load_x2_pd(v_ptr, x0, x1);

        y0 = _mm256_sort_pd(x0);
        y1 = _mm256_sort_pd(x1);

        _mm256_stream_x2_pd(v_ptr, y0, y1);

        v_ptr += AVX2_DOUBLE_STRIDE * 2;
        r -= AVX2_DOUBLE_STRIDE / 2;
    }
    if (r >= AVX2_DOUBLE_STRIDE / 4) {
        _mm256_load_x1_pd(v_ptr, x0);

        y0 = _mm256_sort_pd(x0);

        _mm256_stream_x1_pd(v_ptr, y0);
    }

    return SUCCESS;
}

int sortdsc_ignnan_s5_d(const uint n, const uint s, double* v_ptr) {
    if (s != 5) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x5_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n5_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s6_d(const uint n, const uint s, double* v_ptr) {
    if (s != 6) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x6_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n6_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s7_d(const uint n, const uint s, double* v_ptr) {
    if (s != 7) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x7_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n7_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s8_d(const uint n, const uint s, double* v_ptr) {
    if (s != AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }

    __m256d s0, s1, s2, s3, s4, s5, s6, s7;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        for (uint i = 0; i < (n & (~3u)); i += 4u) {
            _mm256_loadu_x4_pd(v_ptr, s0, s1, s2, s3);
            _mm256_loadu_x4_pd(v_ptr + AVX2_DOUBLE_STRIDE * 4, s4, s5, s6, s7);

            __m256dx2 x0 = __m256dx2(s0, s1), x1 = __m256dx2(s2, s3), x2 = __m256dx2(s4, s5), x3 = __m256dx2(s6, s7);

            __m256dx2 y0 = _mm256x2_sort_pd(x0);
            __m256dx2 y1 = _mm256x2_sort_pd(x1);
            __m256dx2 y2 = _mm256x2_sort_pd(x2);
            __m256dx2 y3 = _mm256x2_sort_pd(x3);

            _mm256_storeu_x4_pd(v_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1);
            _mm256_storeu_x4_pd(v_ptr + AVX2_DOUBLE_STRIDE * 4, y2.imm0, y2.imm1, y3.imm0, y3.imm1);

            v_ptr += s * 4;
        }
        for (uint i = (n & (~3u)); i < n; i++) {
            _mm256_loadu_x2_pd(v_ptr, s0, s1);

            __m256dx2 x0 = __m256dx2(s0, s1);

            __m256dx2 y0 = _mm256x2_sort_pd(x0);

            _mm256_storeu_x2_pd(v_ptr, y0.imm0, y0.imm1);

            v_ptr += s;
        }
    }
    else {
        for (uint i = 0; i < (n & (~3u)); i += 4u) {
            _mm256_load_x4_pd(v_ptr, s0, s1, s2, s3);
            _mm256_load_x4_pd(v_ptr + AVX2_DOUBLE_STRIDE * 4, s4, s5, s6, s7);

            __m256dx2 x0 = __m256dx2(s0, s1), x1 = __m256dx2(s2, s3), x2 = __m256dx2(s4, s5), x3 = __m256dx2(s6, s7);

            __m256dx2 y0 = _mm256x2_sort_pd(x0);
            __m256dx2 y1 = _mm256x2_sort_pd(x1);
            __m256dx2 y2 = _mm256x2_sort_pd(x2);
            __m256dx2 y3 = _mm256x2_sort_pd(x3);

            _mm256_store_x4_pd(v_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1);
            _mm256_store_x4_pd(v_ptr + AVX2_DOUBLE_STRIDE * 4, y2.imm0, y2.imm1, y3.imm0, y3.imm1);

            v_ptr += s * 4;
        }
        for (uint i = (n & (~3u)); i < n; i++) {
            _mm256_load_x2_pd(v_ptr, s0, s1);

            __m256dx2 x0 = __m256dx2(s0, s1);

            __m256dx2 y0 = _mm256x2_sort_pd(x0);

            _mm256_store_x2_pd(v_ptr, y0.imm0, y0.imm1);

            v_ptr += s;
        }
    }

    return SUCCESS;
}

int sortdsc_ignnan_s9_d(const uint n, const uint s, double* v_ptr) {
    if (s != 9) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x9_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n9_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s10_d(const uint n, const uint s, double* v_ptr) {
    if (s != 10) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x10_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n10_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s11_d(const uint n, const uint s, double* v_ptr) {
    if (s != 11) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x11_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n11_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s12_d(const uint n, const uint s, double* v_ptr) {
    if (s != 12) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x12_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n12_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s13_d(const uint n, const uint s, double* v_ptr) {
    if (s != 13) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x13_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n13_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s14_d(const uint n, const uint s, double* v_ptr) {
    if (s != 14) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x14_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n14_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s15_d(const uint n, const uint s, double* v_ptr) {
    if (s != 15) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x15_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n15_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s16_d(const uint n, const uint s, double* v_ptr) {
    if (s != 16) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x16_d(s, v_ptr);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n16_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s17to31_d(const uint n, const uint s, double* v_ptr) {
    if (s <= AVX2_DOUBLE_STRIDE * 4 || s >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        permcombsort_p4x4_d(s, v_ptr);
        scansort_p4_d(s, v_ptr);
        scansort_p4_d(s, v_ptr + s);
        scansort_p4_d(s, v_ptr + s * 2);
        scansort_p4_d(s, v_ptr + s * 3);
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        permcombsort_p4_d(s, v_ptr);
        scansort_p4_d(s, v_ptr);

        v_ptr += s;
    }

    return SUCCESS;
}

int sortdsc_ignnan_s32plus_d(const uint n, const uint s, double* v_ptr) {
    if (s < AVX2_DOUBLE_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < n; i++) {
        longsort_n32plus_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

#pragma endregion sort