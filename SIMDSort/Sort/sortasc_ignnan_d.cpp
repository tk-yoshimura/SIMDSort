#include "../simdsort.h"
#include "sort.h"
#include "../Inline/inline_cmp_d.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"

#pragma region needs swap

// needs swap (sort order definition)
__forceinline static __m256d _mm256_needsswap_pd(__m256d x, __m256d y) {
    return _mm256_cmpgt_ignnan_pd(x, y);
}

#pragma endregion needs swap

#pragma region needs sort

// needs sort
__forceinline static bool _mm256_needssort_pd(__m256d x) {
    __m256d y = _mm256_permute4x64_pd(x, _MM_PERM_DDCB);

    bool needssort = _mm256_movemask_pd(_mm256_needsswap_pd(x, y)) > 0;

    return needssort;
}

// needs sort
__forceinline static bool _mm256x2_needssort_pd(__m256dx2 x) {
    __m256d y0 = _mm256_blend_pd(_mm256_permute4x64_pd(x.imm0, _MM_PERM_DDCB), _mm256_permute4x64_pd(x.imm1, _MM_PERM_ADCB), 0b1000);
    __m256d y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_DDCB);

    bool needssort = _mm256_movemask_pd(_mm256_needsswap_pd(x.imm0, y0)) > 0 || _mm256_movemask_pd(_mm256_needsswap_pd(x.imm1, y1));

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
    const __m256d xormask = _mm256_castsi256_pd(_mm256_setr_epi32(~0u, ~0u, 0, 0, 0, 0, ~0u, ~0u));

    __m256d y, c;

    y = _mm256_permute4x64_pd(x, _MM_PERM_ABCD);
    c = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_DBBD));
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_CDAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_pd(x, y, c);

    y = _mm256_permute4x64_pd(x, _MM_PERM_ABCD);
    c = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x, y), _MM_PERM_DBBD));
    x = _mm256_blendv_pd(x, y, c);

    return x;
}

// sort elems8
__forceinline __m256dx2 _mm256x2_sort_pd(__m256dx2 x) {
    const __m256d xormask = _mm256_castsi256_pd(_mm256_setr_epi32(~0u, ~0u, 0, 0, 0, 0, ~0u, ~0u));

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
    c0 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_DBBD));
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_DBBD));
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_CDAB);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_CCAA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_CDAB);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_CCAA);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    c0 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_DBBD));
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_DBBD));
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

#pragma endregion cmp and swap

#pragma region combsort

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

// combsort h=17...19
static int combsort_h17to19_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 4 || h >= AVX2_DOUBLE_STRIDE * 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, a2, a3, a4, b0, b1, b2, b3, b4;
    __m256d x0, x1, x2, x3, x4, y0, y1, y2, y3, y4;

    if (e > 0) {
        _mm256_maskload_x5_pd(v_ptr, a0, a1, a2, a3, a4, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x5_pd(v_ptr + i + h, b0, b1, b2, b3, b4, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);

            _mm256_maskstore_x5_pd(v_ptr + i, x0, x1, x2, x3, x4, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
        }
        _mm256_maskstore_x5_pd(v_ptr + i, a0, a1, a2, a3, a4, mask);
    }
    {
        _mm256_maskload_x5_pd(v_ptr + e, a0, a1, a2, a3, a4, mask);
        _mm256_maskload_x5_pd(v_ptr + e + h, b0, b1, b2, b3, b4, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);

        _mm256_maskstore_x5_pd(v_ptr + e, x0, x1, x2, x3, x4, mask);
        _mm256_maskstore_x5_pd(v_ptr + e + h, y0, y1, y2, y3, y4, mask);
    }

    return SUCCESS;
}

// combsort h=20
static int combsort_h20_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 10) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 10;

    __m256d a0, a1, a2, a3, a4, b0, b1, b2, b3, b4;
    __m256d x0, x1, x2, x3, x4, y0, y1, y2, y3, y4;

    if (e > 0) {
        _mm256_loadu_x5_pd(v_ptr, a0, a1, a2, a3, a4);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 5) {
            _mm256_loadu_x5_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 5, b0, b1, b2, b3, b4);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);

            _mm256_storeu_x5_pd(v_ptr + i, x0, x1, x2, x3, x4);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
        }
        _mm256_storeu_x5_pd(v_ptr + i, a0, a1, a2, a3, a4);
    }
    {
        _mm256_loadu_x5_pd(v_ptr + e, a0, a1, a2, a3, a4);
        _mm256_loadu_x5_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 5, b0, b1, b2, b3, b4);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);

        _mm256_storeu_x5_pd(v_ptr + e, x0, x1, x2, x3, x4);
        _mm256_storeu_x5_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 5, y0, y1, y2, y3, y4);
    }

    return SUCCESS;
}

// combsort h=21...23
static int combsort_h21to23_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 5 || h >= AVX2_DOUBLE_STRIDE * 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
    __m256d x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5;

    if (e > 0) {
        _mm256_maskload_x6_pd(v_ptr, a0, a1, a2, a3, a4, a5, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x6_pd(v_ptr + i + h, b0, b1, b2, b3, b4, b5, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);
            _mm256_cmpswap_pd(a5, b5, x5, y5);

            _mm256_maskstore_x6_pd(v_ptr + i, x0, x1, x2, x3, x4, x5, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
            a5 = y5;
        }
        _mm256_maskstore_x6_pd(v_ptr + i, a0, a1, a2, a3, a4, a5, mask);
    }
    {
        _mm256_maskload_x6_pd(v_ptr + e, a0, a1, a2, a3, a4, a5, mask);
        _mm256_maskload_x6_pd(v_ptr + e + h, b0, b1, b2, b3, b4, b5, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);

        _mm256_maskstore_x6_pd(v_ptr + e, x0, x1, x2, x3, x4, x5, mask);
        _mm256_maskstore_x6_pd(v_ptr + e + h, y0, y1, y2, y3, y4, y5, mask);
    }

    return SUCCESS;
}

// combsort h=24
static int combsort_h24_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 12) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 12;

    __m256d a0, a1, a2, a3, a4, a5, b0, b1, b2, b3, b4, b5;
    __m256d x0, x1, x2, x3, x4, x5, y0, y1, y2, y3, y4, y5;

    if (e > 0) {
        _mm256_loadu_x6_pd(v_ptr, a0, a1, a2, a3, a4, a5);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 6) {
            _mm256_loadu_x6_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 6, b0, b1, b2, b3, b4, b5);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);
            _mm256_cmpswap_pd(a5, b5, x5, y5);

            _mm256_storeu_x6_pd(v_ptr + i, x0, x1, x2, x3, x4, x5);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
            a5 = y5;
        }
        _mm256_storeu_x6_pd(v_ptr + i, a0, a1, a2, a3, a4, a5);
    }
    {
        _mm256_loadu_x6_pd(v_ptr + e, a0, a1, a2, a3, a4, a5);
        _mm256_loadu_x6_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 6, b0, b1, b2, b3, b4, b5);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);

        _mm256_storeu_x6_pd(v_ptr + e, x0, x1, x2, x3, x4, x5);
        _mm256_storeu_x6_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 6, y0, y1, y2, y3, y4, y5);
    }

    return SUCCESS;
}

// combsort h=25...27
static int combsort_h25to27_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 6 || h >= AVX2_DOUBLE_STRIDE * 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6;
    __m256d x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6;

    if (e > 0) {
        _mm256_maskload_x7_pd(v_ptr, a0, a1, a2, a3, a4, a5, a6, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x7_pd(v_ptr + i + h, b0, b1, b2, b3, b4, b5, b6, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);
            _mm256_cmpswap_pd(a5, b5, x5, y5);
            _mm256_cmpswap_pd(a6, b6, x6, y6);

            _mm256_maskstore_x7_pd(v_ptr + i, x0, x1, x2, x3, x4, x5, x6, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
            a5 = y5;
            a6 = y6;
        }
        _mm256_maskstore_x7_pd(v_ptr + i, a0, a1, a2, a3, a4, a5, a6, mask);
    }
    {
        _mm256_maskload_x7_pd(v_ptr + e, a0, a1, a2, a3, a4, a5, a6, mask);
        _mm256_maskload_x7_pd(v_ptr + e + h, b0, b1, b2, b3, b4, b5, b6, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);
        _mm256_cmpswap_pd(a6, b6, x6, y6);

        _mm256_maskstore_x7_pd(v_ptr + e, x0, x1, x2, x3, x4, x5, x6, mask);
        _mm256_maskstore_x7_pd(v_ptr + e + h, y0, y1, y2, y3, y4, y5, y6, mask);
    }

    return SUCCESS;
}

// combsort h=28
static int combsort_h28_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 14) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 14;

    __m256d a0, a1, a2, a3, a4, a5, a6, b0, b1, b2, b3, b4, b5, b6;
    __m256d x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6;

    if (e > 0) {
        _mm256_loadu_x7_pd(v_ptr, a0, a1, a2, a3, a4, a5, a6);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 7) {
            _mm256_loadu_x7_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 7, b0, b1, b2, b3, b4, b5, b6);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);
            _mm256_cmpswap_pd(a5, b5, x5, y5);
            _mm256_cmpswap_pd(a6, b6, x6, y6);

            _mm256_storeu_x7_pd(v_ptr + i, x0, x1, x2, x3, x4, x5, x6);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
            a5 = y5;
            a6 = y6;
        }
        _mm256_storeu_x7_pd(v_ptr + i, a0, a1, a2, a3, a4, a5, a6);
    }
    {
        _mm256_loadu_x7_pd(v_ptr + e, a0, a1, a2, a3, a4, a5, a6);
        _mm256_loadu_x7_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 7, b0, b1, b2, b3, b4, b5, b6);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);
        _mm256_cmpswap_pd(a6, b6, x6, y6);

        _mm256_storeu_x7_pd(v_ptr + e, x0, x1, x2, x3, x4, x5, x6);
        _mm256_storeu_x7_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 7, y0, y1, y2, y3, y4, y5, y6);
    }

    return SUCCESS;
}

// combsort h=29...31
static int combsort_h29to31_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 7 || h >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h * 2) {
        return SUCCESS;
    }

    const __m256i mask = _mm256_setmask_pd(h & AVX2_DOUBLE_REMAIN_MASK);

    uint e = n - h * 2;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
    __m256d x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7;

    if (e > 0) {
        _mm256_maskload_x8_pd(v_ptr, a0, a1, a2, a3, a4, a5, a6, a7, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x8_pd(v_ptr + i + h, b0, b1, b2, b3, b4, b5, b6, b7, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);
            _mm256_cmpswap_pd(a5, b5, x5, y5);
            _mm256_cmpswap_pd(a6, b6, x6, y6);
            _mm256_cmpswap_pd(a7, b7, x7, y7);

            _mm256_maskstore_x8_pd(v_ptr + i, x0, x1, x2, x3, x4, x5, x6, x7, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
            a5 = y5;
            a6 = y6;
            a7 = y7;
        }
        _mm256_maskstore_x8_pd(v_ptr + i, a0, a1, a2, a3, a4, a5, a6, a7, mask);
    }
    {
        _mm256_maskload_x8_pd(v_ptr + e, a0, a1, a2, a3, a4, a5, a6, a7, mask);
        _mm256_maskload_x8_pd(v_ptr + e + h, b0, b1, b2, b3, b4, b5, b6, b7, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);
        _mm256_cmpswap_pd(a6, b6, x6, y6);
        _mm256_cmpswap_pd(a7, b7, x7, y7);

        _mm256_maskstore_x8_pd(v_ptr + e, x0, x1, x2, x3, x4, x5, x6, x7, mask);
        _mm256_maskstore_x8_pd(v_ptr + e + h, y0, y1, y2, y3, y4, y5, y6, y7, mask);
    }

    return SUCCESS;
}

// combsort h=32
static int combsort_h32_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 16) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 16;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
    __m256d x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7;

    if (e > 0) {
        _mm256_loadu_x8_pd(v_ptr, a0, a1, a2, a3, a4, a5, a6, a7);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 8) {
            _mm256_loadu_x8_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 8, b0, b1, b2, b3, b4, b5, b6, b7);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);
            _mm256_cmpswap_pd(a4, b4, x4, y4);
            _mm256_cmpswap_pd(a5, b5, x5, y5);
            _mm256_cmpswap_pd(a6, b6, x6, y6);
            _mm256_cmpswap_pd(a7, b7, x7, y7);

            _mm256_storeu_x8_pd(v_ptr + i, x0, x1, x2, x3, x4, x5, x6, x7);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
            a4 = y4;
            a5 = y5;
            a6 = y6;
            a7 = y7;
        }
        _mm256_storeu_x8_pd(v_ptr + i, a0, a1, a2, a3, a4, a5, a6, a7);
    }
    {
        _mm256_loadu_x8_pd(v_ptr + e, a0, a1, a2, a3, a4, a5, a6, a7);
        _mm256_loadu_x8_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 8, b0, b1, b2, b3, b4, b5, b6, b7);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);
        _mm256_cmpswap_pd(a6, b6, x6, y6);
        _mm256_cmpswap_pd(a7, b7, x7, y7);

        _mm256_storeu_x8_pd(v_ptr + e, x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_x8_pd(v_ptr + e + AVX2_DOUBLE_STRIDE * 8, y0, y1, y2, y3, y4, y5, y6, y7);
    }

    return SUCCESS;
}

// combsort h>32
static int combsort_h33plus_d(const uint n, const uint h, double* v_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h + AVX2_DOUBLE_STRIDE * 8) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_DOUBLE_STRIDE * 8;

    __m256d a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;
    __m256d x0, x1, x2, x3, x4, x5, x6, x7, y0, y1, y2, y3, y4, y5, y6, y7;

    for (uint i = 0; i < e; i += AVX2_DOUBLE_STRIDE * 8) {
        _mm256_loadu_x8_pd(v_ptr + i, a0, a1, a2, a3, a4, a5, a6, a7);
        _mm256_loadu_x8_pd(v_ptr + i + h, b0, b1, b2, b3, b4, b5, b6, b7);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);
        _mm256_cmpswap_pd(a6, b6, x6, y6);
        _mm256_cmpswap_pd(a7, b7, x7, y7);

        _mm256_storeu_x8_pd(v_ptr + i, x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_x8_pd(v_ptr + i + h, y0, y1, y2, y3, y4, y5, y6, y7);
    }
    {
        _mm256_loadu_x8_pd(v_ptr + e, a0, a1, a2, a3, a4, a5, a6, a7);
        _mm256_loadu_x8_pd(v_ptr + e + h, b0, b1, b2, b3, b4, b5, b6, b7);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);
        _mm256_cmpswap_pd(a4, b4, x4, y4);
        _mm256_cmpswap_pd(a5, b5, x5, y5);
        _mm256_cmpswap_pd(a6, b6, x6, y6);
        _mm256_cmpswap_pd(a7, b7, x7, y7);

        _mm256_storeu_x8_pd(v_ptr + e, x0, x1, x2, x3, x4, x5, x6, x7);
        _mm256_storeu_x8_pd(v_ptr + e + h, y0, y1, y2, y3, y4, y5, y6, y7);
    }

    return SUCCESS;
}

#pragma endregion combsort

#pragma region backtracksort

// backtracksort 8 elems wise
__forceinline static int backtracksort_p8_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 4) {
        return SUCCESS;
    }

    uint i = 0, e = n - AVX2_DOUBLE_STRIDE * 4;

    __m256dx2 a, b, x, y;
    _mm256_loadu_x2_pd(v_ptr, a.imm0, a.imm1);
    _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, b.imm0, b.imm1);

    if (e <= 0) {
        _mm256_cmpswap_pd(a.imm0, b.imm0, x.imm0, y.imm0);
        _mm256_cmpswap_pd(a.imm1, b.imm1, x.imm1, y.imm1);

        _mm256_storeu_x2_pd(v_ptr, x.imm0, x.imm1);
        _mm256_storeu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, y.imm0, y.imm1);

        return SUCCESS;
    }

    while (true) {
        int indexes0 = _mm256_cmpswap_indexed_pd(a.imm0, b.imm0, x.imm0, y.imm0);
        int indexes1 = _mm256_cmpswap_indexed_pd(a.imm1, b.imm1, x.imm1, y.imm1);

        if ((indexes0 | indexes1) > 0) {
            _mm256_storeu_x2_pd(v_ptr + i, x.imm0, x.imm1);
            _mm256_storeu_x2_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 2, y.imm0, y.imm1);

            if (i >= AVX2_DOUBLE_STRIDE * 2) {
                i -= AVX2_DOUBLE_STRIDE * 2;
                _mm256_loadu_x2_pd(v_ptr + i, a.imm0, a.imm1);
                b = x;
                continue;
            }
            else if (i > 0) {
                i = 0;
                _mm256_loadu_x2_pd(v_ptr, a.imm0, a.imm1);
                _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 2, b.imm0, b.imm1);
                continue;
            }
            else {
                i = AVX2_DOUBLE_STRIDE * 2;
                if (i <= e) {
                    a = y;
                    _mm256_loadu_x2_pd(v_ptr + AVX2_DOUBLE_STRIDE * 4, b.imm0, b.imm1);
                    continue;
                }
            }
        }
        else if (i < e) {
            i += AVX2_DOUBLE_STRIDE * 2;

            if (i <= e) {
                a = b;
                _mm256_loadu_x2_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 2, b.imm0, b.imm1);
                continue;
            }
        }
        else {
            break;
        }

        i = e;
        _mm256_loadu_x2_pd(v_ptr + i, a.imm0, a.imm1);
        _mm256_loadu_x2_pd(v_ptr + i + AVX2_DOUBLE_STRIDE * 2, b.imm0, b.imm1);
    }

    return SUCCESS;
}

#pragma endregion backtracksort

#pragma region batchsort

// batchsort 8 elems wise
static int batchsort_p8_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 2) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 2;

    __m256dx2 x0, x1, x2, x3;
    __m256dx2 y0, y1, y2, y3;

    double* ve_ptr = v_ptr + e;

    for (int iter = 0; iter < 2; iter++) {
        {
            double* vc_ptr = v_ptr;
            uint r = n;

            while (r >= AVX2_DOUBLE_STRIDE * 8) {
                _mm256_loadu_x8_pd(vc_ptr, x0.imm0, x0.imm1, x1.imm0, x1.imm1, x2.imm0, x2.imm1, x3.imm0, x3.imm1);

                y0 = _mm256x2_sort_pd(x0);
                y1 = _mm256x2_sort_pd(x1);
                y2 = _mm256x2_sort_pd(x2);
                y3 = _mm256x2_sort_pd(x3);

                _mm256_storeu_x8_pd(vc_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1, y2.imm0, y2.imm1, y3.imm0, y3.imm1);

                vc_ptr += AVX2_DOUBLE_STRIDE * 8;
                r -= AVX2_DOUBLE_STRIDE * 8;
            }
            if (r >= AVX2_DOUBLE_STRIDE * 6) {
                _mm256_loadu_x6_pd(vc_ptr, x0.imm0, x0.imm1, x1.imm0, x1.imm1, x2.imm0, x2.imm1);

                y0 = _mm256x2_sort_pd(x0);
                y1 = _mm256x2_sort_pd(x1);
                y2 = _mm256x2_sort_pd(x2);

                _mm256_storeu_x6_pd(vc_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1, y2.imm0, y2.imm1);
            }
            else if (r >= AVX2_DOUBLE_STRIDE * 4) {
                _mm256_loadu_x4_pd(vc_ptr, x0.imm0, x0.imm1, x1.imm0, x1.imm1);

                y0 = _mm256x2_sort_pd(x0);
                y1 = _mm256x2_sort_pd(x1);

                _mm256_storeu_x4_pd(vc_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1);
            }
            else if (r >= AVX2_DOUBLE_STRIDE * 2) {
                _mm256_loadu_x2_pd(vc_ptr, x0.imm0, x0.imm1);

                y0 = _mm256x2_sort_pd(x0);

                _mm256_storeu_x2_pd(vc_ptr, y0.imm0, y0.imm1);
            }
            if (r > 0) {
                _mm256_loadu_x2_pd(ve_ptr, x0.imm0, x0.imm1);

                y0 = _mm256x2_sort_pd(x0);

                _mm256_storeu_x2_pd(ve_ptr, y0.imm0, y0.imm1);
            }
        }

        {
            double* vc_ptr = v_ptr + AVX2_DOUBLE_STRIDE;
            uint r = n - AVX2_DOUBLE_STRIDE;

            while (r >= AVX2_DOUBLE_STRIDE * 8) {
                _mm256_loadu_x8_pd(vc_ptr, x0.imm0, x0.imm1, x1.imm0, x1.imm1, x2.imm0, x2.imm1, x3.imm0, x3.imm1);

                y0 = _mm256x2_sort_pd(x0);
                y1 = _mm256x2_sort_pd(x1);
                y2 = _mm256x2_sort_pd(x2);
                y3 = _mm256x2_sort_pd(x3);

                _mm256_storeu_x8_pd(vc_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1, y2.imm0, y2.imm1, y3.imm0, y3.imm1);

                vc_ptr += AVX2_DOUBLE_STRIDE * 8;
                r -= AVX2_DOUBLE_STRIDE * 8;
            }
            if (r >= AVX2_DOUBLE_STRIDE * 6) {
                _mm256_loadu_x6_pd(vc_ptr, x0.imm0, x0.imm1, x1.imm0, x1.imm1, x2.imm0, x2.imm1);

                y0 = _mm256x2_sort_pd(x0);
                y1 = _mm256x2_sort_pd(x1);
                y2 = _mm256x2_sort_pd(x2);

                _mm256_storeu_x6_pd(vc_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1, y2.imm0, y2.imm1);
            }
            else if (r >= AVX2_DOUBLE_STRIDE * 4) {
                _mm256_loadu_x4_pd(vc_ptr, x0.imm0, x0.imm1, x1.imm0, x1.imm1);

                y0 = _mm256x2_sort_pd(x0);
                y1 = _mm256x2_sort_pd(x1);

                _mm256_storeu_x4_pd(vc_ptr, y0.imm0, y0.imm1, y1.imm0, y1.imm1);
            }
            else if (r >= AVX2_DOUBLE_STRIDE * 2) {
                _mm256_loadu_x2_pd(vc_ptr, x0.imm0, x0.imm1);

                y0 = _mm256x2_sort_pd(x0);

                _mm256_storeu_x2_pd(vc_ptr, y0.imm0, y0.imm1);
            }
            if (r > 0) {
                _mm256_loadu_x2_pd(ve_ptr, x0.imm0, x0.imm1);

                y0 = _mm256x2_sort_pd(x0);

                _mm256_storeu_x2_pd(ve_ptr, y0.imm0, y0.imm1);
            }
        }
    }

    return SUCCESS;
}

#pragma endregion batchsort

#pragma region scansort

// scansort 8 elems wise
__forceinline static int scansort_p8_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
#endif

    uint e = n - AVX2_DOUBLE_STRIDE * 2;

    __m256dx2 x, y;

    uint i = 0;
    while (true) {
        _mm256_loadu_x2_pd(v_ptr + i, x.imm0, x.imm1);

        if (_mm256x2_needssort_pd(x)) {
            y = _mm256x2_sort_pd(x);
            _mm256_storeu_x2_pd(v_ptr + i, y.imm0, y.imm1);

            if (i > 0) {
                uint indexes = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x.imm0, y.imm0));
                if ((indexes & 1) == 0) {
                    const uint back = AVX2_DOUBLE_STRIDE * 2 - 2;

                    i = (i > back) ? i - back : 0;
                    continue;
                }
            }
        }

        if (i < e) {
            i += AVX2_DOUBLE_STRIDE * 2 - 1;
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

// shortsort elems 8...16
__forceinline static int shortsort_n8to16_d(const uint n, double* v_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE * 2 || n > AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    backtracksort_p8_d(n, v_ptr);
    scansort_p8_d(n, v_ptr);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

// longsort
__forceinline static int longsort_d(const uint n, double* v_ptr) {
    uint h;

    for (h = (uint)(n * 10L / 13L); h > 32; h = (uint)(h * 10L / 13L)) {
        combsort_h33plus_d(n, h, v_ptr);
    }
    if (h >= 32) {
        combsort_h32_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 28; h = h * 10 / 13) {
        combsort_h29to31_d(n, h, v_ptr);
    }
    if (h >= 28) {
        combsort_h28_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 24; h = h * 10 / 13) {
        combsort_h25to27_d(n, h, v_ptr);
    }
    if (h >= 24) {
        combsort_h24_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 20; h = h * 10 / 13) {
        combsort_h21to23_d(n, h, v_ptr);
    }
    if (h >= 20) {
        combsort_h20_d(n, v_ptr);
        h = h * 10 / 13;
    }
    for (; h > 16; h = h * 10 / 13) {
        combsort_h17to19_d(n, h, v_ptr);
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
    }

    backtracksort_p8_d(n, v_ptr);
    batchsort_p8_d(n, v_ptr);
    scansort_p8_d(n, v_ptr);

    return SUCCESS;
}

#pragma endregion longsort

#pragma region sort

int sortasc_ignnan_s2_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s != 2 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s3_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s != 3) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s4_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s != 4 || ((size_t)v_ptr % AVX2_ALIGNMENT) != 0) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s5_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s != 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s6_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s != 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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

int sortasc_ignnan_s7_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s != 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

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


int sortasc_ignnan_s8to16_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s < AVX2_DOUBLE_STRIDE * 2 || s > AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    for (uint i = 0; i < n; i++) {
        shortsort_n8to16_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

int sortasc_ignnan_slong_d(const uint n, const uint s, double* v_ptr) {
#ifdef _DEBUG
    if (s < AVX2_DOUBLE_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    for (uint i = 0; i < n; i++) {
        longsort_d(s, v_ptr);
        v_ptr += s;
    }

    return SUCCESS;
}

#pragma endregion sort