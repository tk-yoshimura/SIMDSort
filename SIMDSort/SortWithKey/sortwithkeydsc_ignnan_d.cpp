#include "sortwithkey.h"
#include "../Inline/inline_cmp_d.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"
#include "../Inline/inline_loadstore_xn_epi64.hpp"
#include "../Inline/inline_blend_epi.hpp"

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
__forceinline static __m256dkv _mm256_sort2x2_pd(__m256dkv x) {
    __m256d yk = _mm256_permute4x64_pd(x.k, _MM_PERM_CDAB);
    __m256i yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_CDAB);
    __m256d c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_CCAA);
    __m256d zk = _mm256_blendv_pd(x.k, yk, c);
    __m256i zv = _mm256_blendv_epi64(x.v, yv, c);

    return __m256dkv(zk, zv);
}

// sort batches1 x elems3
__forceinline static __m256dkv _mm256_sort1x3_pd(__m256dkv x) {
    __m256d yk, c;
    __m256i yv;

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_DCAB);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_DCAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_DCAA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_DBCA);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_DBCA);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_DBBA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_DCAB);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_DCAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_DCAA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    return x;
}

// sort elems4
__forceinline static __m256dkv _mm256_sort_pd(__m256dkv x) {
    __m256d yk, c;
    __m256i yv;

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_CDAB);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_CDAB);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_CCAA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_BADC);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_BADC);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_BABA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_ABCD);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_ABCD);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_ABBA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    return x;
}

// sort elems4 (ho, lo sorted)
__forceinline static __m256dkv _mm256_halfsort_pd(__m256dkv x) {
    __m256d yk, c;
    __m256i yv;

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_BADC);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_BADC);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_BABA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    yk = _mm256_permute4x64_pd(x.k, _MM_PERM_ABCD);
    yv = _mm256_permute4x64_epi64(x.v, _MM_PERM_ABCD);
    c = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.k, yk), _MM_PERM_ABBA);
    x.k = _mm256_blendv_pd(x.k, yk, c);
    x.v = _mm256_blendv_epi64(x.v, yv, c);

    return x;
}

// sort elems8
__forceinline static __m256dkvx2 _mm256x2_sort_pd(__m256dkvx2 x) {
    __m256dkvx2 y;
    __m256d swaps, notswaps;

    swaps = _mm256_needsswap_pd(x.k0, x.k1);
    notswaps = _mm256_not_pd(swaps);
    y.k0 = _mm256_blendv_pd(x.k0, x.k1, swaps);
    y.k1 = _mm256_blendv_pd(x.k0, x.k1, notswaps);
    y.v0 = _mm256_blendv_epi64(x.v0, x.v1, swaps);
    y.v1 = _mm256_blendv_epi64(x.v0, x.v1, notswaps);
    x.k0 = y.k0;
    x.k1 = _mm256_permute4x64_pd(y.k1, _MM_PERM_ADCB);
    x.v0 = y.v0;
    x.v1 = _mm256_permute4x64_epi64(y.v1, _MM_PERM_ADCB);

    swaps = _mm256_needsswap_pd(x.k0, x.k1);
    notswaps = _mm256_not_pd(swaps);
    y.k0 = _mm256_blendv_pd(x.k0, x.k1, swaps);
    y.k1 = _mm256_blendv_pd(x.k0, x.k1, notswaps);
    y.v0 = _mm256_blendv_epi64(x.v0, x.v1, swaps);
    y.v1 = _mm256_blendv_epi64(x.v0, x.v1, notswaps);
    x.k0 = y.k0;
    x.k1 = _mm256_permute4x64_pd(y.k1, _MM_PERM_ADCB);
    x.v0 = y.v0;
    x.v1 = _mm256_permute4x64_epi64(y.v1, _MM_PERM_ADCB);

    swaps = _mm256_needsswap_pd(x.k0, x.k1);
    notswaps = _mm256_not_pd(swaps);
    y.k0 = _mm256_blendv_pd(x.k0, x.k1, swaps);
    y.k1 = _mm256_blendv_pd(x.k0, x.k1, notswaps);
    y.v0 = _mm256_blendv_epi64(x.v0, x.v1, swaps);
    y.v1 = _mm256_blendv_epi64(x.v0, x.v1, notswaps);
    x.k0 = y.k0;
    x.k1 = _mm256_permute4x64_pd(y.k1, _MM_PERM_ADCB);
    x.v0 = y.v0;
    x.v1 = _mm256_permute4x64_epi64(y.v1, _MM_PERM_ADCB);

    swaps = _mm256_needsswap_pd(x.k0, x.k1);
    notswaps = _mm256_not_pd(swaps);
    y.k0 = _mm256_blendv_pd(x.k0, x.k1, swaps);
    y.k1 = _mm256_blendv_pd(x.k0, x.k1, notswaps);
    y.v0 = _mm256_blendv_epi64(x.v0, x.v1, swaps);
    y.v1 = _mm256_blendv_epi64(x.v0, x.v1, notswaps);
    x = y;

    __m256dkv y0 = _mm256_sort_pd(__m256dkv(x.k0, x.v0));
    __m256dkv y1 = _mm256_sort_pd(__m256dkv(x.k1, x.v1));

    return __m256dkvx2(y0.k, y1.k, y0.v, y1.v);
}

#pragma endregion horizontal sort

#pragma region cmp and swap

// compare and swap
__forceinline static void _mm256_cmpswap_pd(__m256dkv a, __m256dkv b, __m256dkv& x, __m256dkv& y) {
    __m256d swaps = _mm256_needsswap_pd(a.k, b.k), notswaps = _mm256_not_pd(swaps);

    x.k = _mm256_blendv_pd(a.k, b.k, swaps);
    y.k = _mm256_blendv_pd(a.k, b.k, notswaps);
    x.v = _mm256_blendv_epi64(a.v, b.v, swaps);
    y.v = _mm256_blendv_epi64(a.v, b.v, notswaps);
}

// compare and swap
__forceinline static uint _mm256_cmpswap_indexed_pd(__m256dkv a, __m256dkv b, __m256dkv& x, __m256dkv& y) {
    __m256d swaps = _mm256_needsswap_pd(a.k, b.k), notswaps = _mm256_not_pd(swaps);

    uint index = _mm256_movemask_pd(swaps);

    x.k = _mm256_blendv_pd(a.k, b.k, swaps);
    y.k = _mm256_blendv_pd(a.k, b.k, notswaps);
    x.v = _mm256_blendv_epi64(a.v, b.v, swaps);
    y.v = _mm256_blendv_epi64(a.v, b.v, notswaps);

    return index;
}

// compare and swap with permutate
__forceinline static void _mm256_cmpswap_withperm_pd(__m256dkv a, __m256dkv b, __m256dkv& x, __m256dkv& y) {

    _mm256_cmpswap_pd(a, b, x, y);

    a.k = _mm256_permute4x64_pd(x.k, _MM_PERM_CBAD);
    a.v = _mm256_permute4x64_epi64(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_pd(a, b, x, y);

    a.k = _mm256_permute4x64_pd(x.k, _MM_PERM_CBAD);
    a.v = _mm256_permute4x64_epi64(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_pd(a, b, x, y);

    a.k = _mm256_permute4x64_pd(x.k, _MM_PERM_CBAD);
    a.v = _mm256_permute4x64_epi64(x.v, _MM_PERM_CBAD);
    b = y;
    _mm256_cmpswap_pd(a, b, x, y);
}

#pragma endregion cmp and swap

#pragma region combsort

// combsort h=5...7
static int combsort_h5to7_d(const uint n, const uint h, ulong* __restrict v_ptr, double* __restrict k_ptr) {
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

    __m256dkv a0, a1, b0, b1;
    __m256dkv x0, x1, y0, y1;

    if (e > 0) {
        _mm256_maskload_x2_pd(k_ptr, a0.k, a1.k, mask);
        _mm256_maskload_x2_epi64(v_ptr, a0.v, a1.v, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x2_pd(k_ptr + i + h, b0.k, b1.k, mask);
            _mm256_maskload_x2_epi64(v_ptr + i + h, b0.v, b1.v, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);

            _mm256_maskstore_x2_pd(k_ptr + i, x0.k, x1.k, mask);
            _mm256_maskstore_x2_epi64(v_ptr + i, x0.v, x1.v, mask);

            a0 = y0;
            a1 = y1;
        }
        _mm256_maskstore_x2_pd(k_ptr + i, a0.k, a1.k, mask);
        _mm256_maskstore_x2_epi64(v_ptr + i, a0.v, a1.v, mask);
    }
    {
        _mm256_maskload_x2_pd(k_ptr + e, a0.k, a1.k, mask);
        _mm256_maskload_x2_epi64(v_ptr + e, a0.v, a1.v, mask);
        _mm256_maskload_x2_pd(k_ptr + e + h, b0.k, b1.k, mask);
        _mm256_maskload_x2_epi64(v_ptr + e + h, b0.v, b1.v, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        _mm256_maskstore_x2_pd(k_ptr + e, x0.k, x1.k, mask);
        _mm256_maskstore_x2_epi64(v_ptr + e, x0.v, x1.v, mask);
        _mm256_maskstore_x2_pd(k_ptr + e + h, y0.k, y1.k, mask);
        _mm256_maskstore_x2_epi64(v_ptr + e + h, y0.v, y1.v, mask);
    }

    return SUCCESS;
}

// combsort h=8
static int combsort_h8_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 4;

    __m256dkv a0, a1, b0, b1;
    __m256dkv x0, x1, y0, y1;

    if (e > 0) {
        _mm256_loadu_x2_pd(k_ptr, a0.k, a1.k);
        _mm256_loadu_x2_epi64(v_ptr, a0.v, a1.v);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(k_ptr + i + AVX2_DOUBLE_STRIDE * 2, b0.k, b1.k);
            _mm256_loadu_x2_epi64(v_ptr + i + AVX2_EPI64_STRIDE * 2, b0.v, b1.v);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);

            _mm256_storeu_x2_pd(k_ptr + i, x0.k, x1.k);
            _mm256_storeu_x2_epi64(v_ptr + i, x0.v, x1.v);

            a0 = y0;
            a1 = y1;
        }
        _mm256_storeu_x2_pd(k_ptr + i, a0.k, a1.k);
        _mm256_storeu_x2_epi64(v_ptr + i, a0.v, a1.v);
    }
    {
        _mm256_loadu_x2_pd(k_ptr + e, a0.k, a1.k);
        _mm256_loadu_x2_epi64(v_ptr + e, a0.v, a1.v);
        _mm256_loadu_x2_pd(k_ptr + e + AVX2_DOUBLE_STRIDE * 2, b0.k, b1.k);
        _mm256_loadu_x2_epi64(v_ptr + e + AVX2_EPI64_STRIDE * 2, b0.v, b1.v);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        _mm256_storeu_x2_pd(k_ptr + e, x0.k, x1.k);
        _mm256_storeu_x2_epi64(v_ptr + e, x0.v, x1.v);
        _mm256_storeu_x2_pd(k_ptr + e + AVX2_DOUBLE_STRIDE * 2, y0.k, y1.k);
        _mm256_storeu_x2_epi64(v_ptr + e + AVX2_EPI64_STRIDE * 2, y0.v, y1.v);
    }

    return SUCCESS;
}

// combsort h=9...11
static int combsort_h9to11_d(const uint n, const uint h, ulong* __restrict v_ptr, double* __restrict k_ptr) {
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

    __m256dkv a0, a1, a2, b0, b1, b2;
    __m256dkv x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_maskload_x3_pd(k_ptr, a0.k, a1.k, a2.k, mask);
        _mm256_maskload_x3_epi64(v_ptr, a0.v, a1.v, a2.v, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x3_pd(k_ptr + i + h, b0.k, b1.k, b2.k, mask);
            _mm256_maskload_x3_epi64(v_ptr + i + h, b0.v, b1.v, b2.v, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);

            _mm256_maskstore_x3_pd(k_ptr + i, x0.k, x1.k, x2.k, mask);
            _mm256_maskstore_x3_epi64(v_ptr + i, x0.v, x1.v, x2.v, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_maskstore_x3_pd(k_ptr + i, a0.k, a1.k, a2.k, mask);
        _mm256_maskstore_x3_epi64(v_ptr + i, a0.v, a1.v, a2.v, mask);
    }
    {
        _mm256_maskload_x3_pd(k_ptr + e, a0.k, a1.k, a2.k, mask);
        _mm256_maskload_x3_epi64(v_ptr + e, a0.v, a1.v, a2.v, mask);
        _mm256_maskload_x3_pd(k_ptr + e + h, b0.k, b1.k, b2.k, mask);
        _mm256_maskload_x3_epi64(v_ptr + e + h, b0.v, b1.v, b2.v, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);

        _mm256_maskstore_x3_pd(k_ptr + e, x0.k, x1.k, x2.k, mask);
        _mm256_maskstore_x3_epi64(v_ptr + e, x0.v, x1.v, x2.v, mask);
        _mm256_maskstore_x3_pd(k_ptr + e + h, y0.k, y1.k, y2.k, mask);
        _mm256_maskstore_x3_epi64(v_ptr + e + h, y0.v, y1.v, y2.v, mask);
    }

    return SUCCESS;
}

// combsort h=12
static int combsort_h12_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 6) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 6;

    __m256dkv a0, a1, a2, b0, b1, b2;
    __m256dkv x0, x1, x2, y0, y1, y2;

    if (e > 0) {
        _mm256_loadu_x3_pd(k_ptr, a0.k, a1.k, a2.k);
        _mm256_loadu_x3_epi64(v_ptr, a0.v, a1.v, a2.v);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(k_ptr + i + AVX2_DOUBLE_STRIDE * 3, b0.k, b1.k, b2.k);
            _mm256_loadu_x3_epi64(v_ptr + i + AVX2_EPI64_STRIDE * 3, b0.v, b1.v, b2.v);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);

            _mm256_storeu_x3_pd(k_ptr + i, x0.k, x1.k, x2.k);
            _mm256_storeu_x3_epi64(v_ptr + i, x0.v, x1.v, x2.v);

            a0 = y0;
            a1 = y1;
            a2 = y2;
        }
        _mm256_storeu_x3_pd(k_ptr + i, a0.k, a1.k, a2.k);
        _mm256_storeu_x3_epi64(v_ptr + i, a0.v, a1.v, a2.v);
    }
    {
        _mm256_loadu_x3_pd(k_ptr + e, a0.k, a1.k, a2.k);
        _mm256_loadu_x3_epi64(v_ptr + e, a0.v, a1.v, a2.v);
        _mm256_loadu_x3_pd(k_ptr + e + AVX2_DOUBLE_STRIDE * 3, b0.k, b1.k, b2.k);
        _mm256_loadu_x3_epi64(v_ptr + e + AVX2_EPI64_STRIDE * 3, b0.v, b1.v, b2.v);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);

        _mm256_storeu_x3_pd(k_ptr + e, x0.k, x1.k, x2.k);
        _mm256_storeu_x3_epi64(v_ptr + e, x0.v, x1.v, x2.v);
        _mm256_storeu_x3_pd(k_ptr + e + AVX2_DOUBLE_STRIDE * 3, y0.k, y1.k, y2.k);
        _mm256_storeu_x3_epi64(v_ptr + e + AVX2_EPI64_STRIDE * 3, y0.v, y1.v, y2.v);
    }

    return SUCCESS;
}

// combsort h=13...15
static int combsort_h13to15_d(const uint n, const uint h, ulong* __restrict v_ptr, double* __restrict k_ptr) {
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

    __m256dkv a0, a1, a2, a3, b0, b1, b2, b3;
    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_maskload_x4_pd(k_ptr, a0.k, a1.k, a2.k, a3.k, mask);
        _mm256_maskload_x4_epi64(v_ptr, a0.v, a1.v, a2.v, a3.v, mask);

        uint i = 0;
        for (; i < e; i += h) {
            _mm256_maskload_x4_pd(k_ptr + i + h, b0.k, b1.k, b2.k, b3.k, mask);
            _mm256_maskload_x4_epi64(v_ptr + i + h, b0.v, b1.v, b2.v, b3.v, mask);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);

            _mm256_maskstore_x4_pd(k_ptr + i, x0.k, x1.k, x2.k, x3.k, mask);
            _mm256_maskstore_x4_epi64(v_ptr + i, x0.v, x1.v, x2.v, x3.v, mask);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_maskstore_x4_pd(k_ptr + i, a0.k, a1.k, a2.k, a3.k, mask);
        _mm256_maskstore_x4_epi64(v_ptr + i, a0.v, a1.v, a2.v, a3.v, mask);
    }
    {
        _mm256_maskload_x4_pd(k_ptr + e, a0.k, a1.k, a2.k, a3.k, mask);
        _mm256_maskload_x4_epi64(v_ptr + e, a0.v, a1.v, a2.v, a3.v, mask);
        _mm256_maskload_x4_pd(k_ptr + e + h, b0.k, b1.k, b2.k, b3.k, mask);
        _mm256_maskload_x4_epi64(v_ptr + e + h, b0.v, b1.v, b2.v, b3.v, mask);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_maskstore_x4_pd(k_ptr + e, x0.k, x1.k, x2.k, x3.k, mask);
        _mm256_maskstore_x4_epi64(v_ptr + e, x0.v, x1.v, x2.v, x3.v, mask);
        _mm256_maskstore_x4_pd(k_ptr + e + h, y0.k, y1.k, y2.k, y3.k, mask);
        _mm256_maskstore_x4_epi64(v_ptr + e + h, y0.v, y1.v, y2.v, y3.v, mask);
    }

    return SUCCESS;
}

// combsort h=16
static int combsort_h16_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 8) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 8;

    __m256dkv a0, a1, a2, a3, b0, b1, b2, b3;
    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    if (e > 0) {
        _mm256_loadu_x4_pd(k_ptr, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi64(v_ptr, a0.v, a1.v, a2.v, a3.v);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(k_ptr + i + AVX2_DOUBLE_STRIDE * 4, b0.k, b1.k, b2.k, b3.k);
            _mm256_loadu_x4_epi64(v_ptr + i + AVX2_EPI64_STRIDE * 4, b0.v, b1.v, b2.v, b3.v);

            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
            _mm256_cmpswap_pd(a2, b2, x2, y2);
            _mm256_cmpswap_pd(a3, b3, x3, y3);

            _mm256_storeu_x4_pd(k_ptr + i, x0.k, x1.k, x2.k, x3.k);
            _mm256_storeu_x4_epi64(v_ptr + i, x0.v, x1.v, x2.v, x3.v);

            a0 = y0;
            a1 = y1;
            a2 = y2;
            a3 = y3;
        }
        _mm256_storeu_x4_pd(k_ptr + i, a0.k, a1.k, a2.k, a3.k);
        _mm256_storeu_x4_epi64(v_ptr + i, a0.v, a1.v, a2.v, a3.v);
    }
    {
        _mm256_loadu_x4_pd(k_ptr + e, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi64(v_ptr + e, a0.v, a1.v, a2.v, a3.v);
        _mm256_loadu_x4_pd(k_ptr + e + AVX2_DOUBLE_STRIDE * 4, b0.k, b1.k, b2.k, b3.k);
        _mm256_loadu_x4_epi64(v_ptr + e + AVX2_EPI64_STRIDE * 4, b0.v, b1.v, b2.v, b3.v);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_storeu_x4_pd(k_ptr + e, x0.k, x1.k, x2.k, x3.k);
        _mm256_storeu_x4_epi64(v_ptr + e, x0.v, x1.v, x2.v, x3.v);
        _mm256_storeu_x4_pd(k_ptr + e + AVX2_DOUBLE_STRIDE * 4, y0.k, y1.k, y2.k, y3.k);
        _mm256_storeu_x4_epi64(v_ptr + e + AVX2_EPI64_STRIDE * 4, y0.v, y1.v, y2.v, y3.v);
    }

    return SUCCESS;
}

// combsort h>16
static int combsort_h17plus_d(const uint n, const uint h, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (h <= AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    if (n < h + AVX2_DOUBLE_STRIDE * 4) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_DOUBLE_STRIDE * 4;

    __m256dkv a0, a1, a2, a3, b0, b1, b2, b3;
    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    for (uint i = 0; i < e; i += AVX2_DOUBLE_STRIDE * 4) {
        _mm256_loadu_x4_pd(k_ptr + i, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi64(v_ptr + i, a0.v, a1.v, a2.v, a3.v);
        _mm256_loadu_x4_pd(k_ptr + i + h, b0.k, b1.k, b2.k, b3.k);
        _mm256_loadu_x4_epi64(v_ptr + i + h, b0.v, b1.v, b2.v, b3.v);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_storeu_x4_pd(k_ptr + i, x0.k, x1.k, x2.k, x3.k);
        _mm256_storeu_x4_epi64(v_ptr + i, x0.v, x1.v, x2.v, x3.v);
        _mm256_storeu_x4_pd(k_ptr + i + h, y0.k, y1.k, y2.k, y3.k);
        _mm256_storeu_x4_epi64(v_ptr + i + h, y0.v, y1.v, y2.v, y3.v);
    }
    {
        _mm256_loadu_x4_pd(k_ptr + e, a0.k, a1.k, a2.k, a3.k);
        _mm256_loadu_x4_epi64(v_ptr + e, a0.v, a1.v, a2.v, a3.v);
        _mm256_loadu_x4_pd(k_ptr + e + h, b0.k, b1.k, b2.k, b3.k);
        _mm256_loadu_x4_epi64(v_ptr + e + h, b0.v, b1.v, b2.v, b3.v);

        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);
        _mm256_cmpswap_pd(a2, b2, x2, y2);
        _mm256_cmpswap_pd(a3, b3, x3, y3);

        _mm256_storeu_x4_pd(k_ptr + e, x0.k, x1.k, x2.k, x3.k);
        _mm256_storeu_x4_epi64(v_ptr + e, x0.v, x1.v, x2.v, x3.v);
        _mm256_storeu_x4_pd(k_ptr + e + h, y0.k, y1.k, y2.k, y3.k);
        _mm256_storeu_x4_epi64(v_ptr + e + h, y0.v, y1.v, y2.v, y3.v);
    }

    return SUCCESS;
}

#pragma endregion combsort

#pragma region paracombsort

// paracombsort 2x4
static int paracombsort_p2x4_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    const uint e = n - AVX2_DOUBLE_STRIDE * 4;
    const uint c = n % AVX2_DOUBLE_STRIDE;

    __m256dkv a0, a1, b0, b1;
    __m256dkv x0, x1, y0, y1;

    for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
        a1.k = _mm256_loadu_pd(k_ptr + i);
        a1.v = _mm256_loadu_epi64(v_ptr + i);
        b1.k = _mm256_loadu_pd(k_ptr + i + AVX2_DOUBLE_STRIDE);
        b1.v = _mm256_loadu_epi64(v_ptr + i + AVX2_EPI64_STRIDE);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        a0 = x1;
        a1 = y1;
        b1.k = _mm256_loadu_pd(k_ptr + i + AVX2_DOUBLE_STRIDE * 2);
        b1.v = _mm256_loadu_epi64(v_ptr + i + AVX2_EPI64_STRIDE * 2);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        b0.k = _mm256_permute4x64_pd(x1.k, _MM_PERM_ABDC);
        b0.v = _mm256_permute4x64_epi64(x1.v, _MM_PERM_ABDC);
        a1 = y1;
        b1.k = _mm256_loadu_pd(k_ptr + i + AVX2_DOUBLE_STRIDE * 3);
        b1.v = _mm256_loadu_epi64(v_ptr + i + AVX2_EPI64_STRIDE * 3);
        _mm256_cmpswap_pd(a0, b0, x0, y0);
        _mm256_cmpswap_pd(a1, b1, x1, y1);

        for (j = i; j + AVX2_DOUBLE_STRIDE <= e; j += AVX2_DOUBLE_STRIDE) {
            _mm256_storeu_pd(k_ptr + j, x0.k);
            _mm256_storeu_epi64(v_ptr + j, x0.v);
            a0 = y0;
            b0.k = _mm256_permute4x64_pd(x1.k, _MM_PERM_ABDC);
            b0.v = _mm256_permute4x64_epi64(x1.v, _MM_PERM_ABDC);
            a1 = y1;
            b1.k = _mm256_loadu_pd(k_ptr + j + AVX2_DOUBLE_STRIDE * 4);
            b1.v = _mm256_loadu_epi64(v_ptr + j + AVX2_EPI64_STRIDE * 4);
            _mm256_cmpswap_pd(a0, b0, x0, y0);
            _mm256_cmpswap_pd(a1, b1, x1, y1);
        }

        _mm256_storeu_pd(k_ptr + j, x0.k);
        _mm256_storeu_epi64(v_ptr + j, x0.v);
        a0 = y0;
        b0.k = _mm256_permute4x64_pd(x1.k, _MM_PERM_ABDC);
        b0.v = _mm256_permute4x64_epi64(x1.v, _MM_PERM_ABDC);
        _mm256_cmpswap_pd(a0, b0, x0, y0);
        j += AVX2_DOUBLE_STRIDE;

        _mm256_storeu_pd(k_ptr + j, x0.k);
        _mm256_storeu_epi64(v_ptr + j, x0.v);
        a0 = y0;
        b0 = y1;
        _mm256_cmpswap_pd(a0, b0, x0, y0);
        j += AVX2_DOUBLE_STRIDE;

        _mm256_storeu_pd(k_ptr + j, x0.k);
        _mm256_storeu_epi64(v_ptr + j, x0.v);
        _mm256_storeu_pd(k_ptr + j + AVX2_DOUBLE_STRIDE, y0.k);
        _mm256_storeu_epi64(v_ptr + j + AVX2_EPI64_STRIDE, y0.v);
    }

    return SUCCESS;
}

#pragma endregion paracombsort

#pragma region backtracksort

// backtracksort 4 elems wise
__forceinline static int backtracksort_p4_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 2) {
        return SUCCESS;
    }

    uint i = 0, e = n - AVX2_DOUBLE_STRIDE * 2;

    __m256dkv a, b, x, y;
    a.k = _mm256_loadu_pd(k_ptr);
    a.v = _mm256_loadu_epi64(v_ptr);
    b.k = _mm256_loadu_pd(k_ptr + AVX2_DOUBLE_STRIDE);
    b.v = _mm256_loadu_epi64(v_ptr + AVX2_EPI64_STRIDE);

    if (e <= 0) {
        _mm256_cmpswap_pd(a, b, x, y);

        _mm256_storeu_pd(k_ptr, x.k);
        _mm256_storeu_epi64(v_ptr, x.v);
        _mm256_storeu_pd(k_ptr + AVX2_DOUBLE_STRIDE, y.k);
        _mm256_storeu_epi64(v_ptr + AVX2_EPI64_STRIDE, y.v);

        return SUCCESS;
    }

    while (true) {
        int indexes = _mm256_cmpswap_indexed_pd(a, b, x, y);

        if (indexes > 0) {
            _mm256_storeu_pd(k_ptr + i, x.k);
            _mm256_storeu_epi64(v_ptr + i, x.v);
            _mm256_storeu_pd(k_ptr + i + AVX2_DOUBLE_STRIDE, y.k);
            _mm256_storeu_epi64(v_ptr + i + AVX2_EPI64_STRIDE, y.v);

            if (i >= AVX2_DOUBLE_STRIDE) {
                i -= AVX2_DOUBLE_STRIDE;
                a.k = _mm256_loadu_pd(k_ptr + i);
                a.v = _mm256_loadu_epi64(v_ptr + i);
                b = x;
                continue;
            }
            else if (i > 0) {
                i = 0;
                a.k = _mm256_loadu_pd(k_ptr);
                a.v = _mm256_loadu_epi64(v_ptr);
                b.k = _mm256_loadu_pd(k_ptr + AVX2_DOUBLE_STRIDE);
                b.v = _mm256_loadu_epi64(v_ptr + AVX2_EPI64_STRIDE);
                continue;
            }
            else {
                i = AVX2_DOUBLE_STRIDE;
                if (i <= e) {
                    a = y;
                    b.k = _mm256_loadu_pd(k_ptr + AVX2_DOUBLE_STRIDE * 2);
                    b.v = _mm256_loadu_epi64(v_ptr + AVX2_EPI64_STRIDE * 2);
                    continue;
                }
            }
        }
        else if (i < e) {
            i += AVX2_DOUBLE_STRIDE;

            if (i <= e) {
                a = b;
                b.k = _mm256_loadu_pd(k_ptr + i + AVX2_DOUBLE_STRIDE);
                b.v = _mm256_loadu_epi64(v_ptr + i + AVX2_EPI64_STRIDE);
                continue;
            }
        }
        else {
            break;
        }

        i = e;
        a.k = _mm256_loadu_pd(k_ptr + i);
        a.v = _mm256_loadu_epi64(v_ptr + i);
        b.k = _mm256_loadu_pd(k_ptr + i + AVX2_DOUBLE_STRIDE);
        b.v = _mm256_loadu_epi64(v_ptr + i + AVX2_EPI64_STRIDE);
    }

    return SUCCESS;
}

#pragma endregion backtracksort

#pragma region batchsort

// batchsort 4 elems wise
static int batchsort_p4_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (n < AVX2_DOUBLE_STRIDE) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE;

    __m256dkv x0, x1, x2, x3;
    __m256dkv y0, y1, y2, y3;

    double* const ke_ptr = k_ptr + e;
    ulong* const ve_ptr = v_ptr + e;

    {
        double* kc_ptr = k_ptr;
        ulong* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi64(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);
            y2 = _mm256_sort_pd(x2);
            y3 = _mm256_sort_pd(x3);

            _mm256_storeu_x4_pd(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi64(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            kc_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi64(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);
            y2 = _mm256_sort_pd(x2);

            _mm256_storeu_x3_pd(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi64(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi64(vc_ptr, x0.v, x1.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);

            _mm256_storeu_x2_pd(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi64(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(kc_ptr, x0.k);
            _mm256_loadu_x1_epi64(vc_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(kc_ptr, y0.k);
            _mm256_storeu_x1_epi64(vc_ptr, y0.v);
        }
        if ((r & AVX2_DOUBLE_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ke_ptr, x0.k);
            _mm256_loadu_x1_epi64(ve_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ke_ptr, y0.k);
            _mm256_storeu_x1_epi64(ve_ptr, y0.v);
        }
    }
    {
        double* kc_ptr = k_ptr + AVX2_DOUBLE_STRIDE / 2;
        ulong* vc_ptr = v_ptr + AVX2_EPI64_STRIDE / 2;
        uint r = n - AVX2_DOUBLE_STRIDE / 2;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi64(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);
            y3 = _mm256_halfsort_pd(x3);

            _mm256_storeu_x4_pd(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi64(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            kc_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi64(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);

            _mm256_storeu_x3_pd(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi64(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi64(vc_ptr, x0.v, x1.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);

            _mm256_storeu_x2_pd(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi64(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(kc_ptr, x0.k);
            _mm256_loadu_x1_epi64(vc_ptr, x0.v);

            y0 = _mm256_halfsort_pd(x0);

            _mm256_storeu_x1_pd(kc_ptr, y0.k);
            _mm256_storeu_x1_epi64(vc_ptr, y0.v);
        }
        if ((r & AVX2_DOUBLE_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ke_ptr, x0.k);
            _mm256_loadu_x1_epi64(ve_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ke_ptr, y0.k);
            _mm256_storeu_x1_epi64(ve_ptr, y0.v);
        }
    }
    {
        double* kc_ptr = k_ptr;
        ulong* vc_ptr = v_ptr;
        uint r = n;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi64(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);
            y3 = _mm256_halfsort_pd(x3);

            _mm256_storeu_x4_pd(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi64(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            kc_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi64(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);

            _mm256_storeu_x3_pd(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi64(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi64(vc_ptr, x0.v, x1.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);

            _mm256_storeu_x2_pd(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi64(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(kc_ptr, x0.k);
            _mm256_loadu_x1_epi64(vc_ptr, x0.v);

            y0 = _mm256_halfsort_pd(x0);

            _mm256_storeu_x1_pd(kc_ptr, y0.k);
            _mm256_storeu_x1_epi64(vc_ptr, y0.v);
        }
        if ((r & AVX2_DOUBLE_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ke_ptr, x0.k);
            _mm256_loadu_x1_epi64(ve_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ke_ptr, y0.k);
            _mm256_storeu_x1_epi64(ve_ptr, y0.v);
        }
    }
    {
        double* kc_ptr = k_ptr + AVX2_DOUBLE_STRIDE / 2;
        ulong* vc_ptr = v_ptr + AVX2_EPI64_STRIDE / 2;
        uint r = n - AVX2_DOUBLE_STRIDE / 2;

        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(kc_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi64(vc_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);
            y3 = _mm256_halfsort_pd(x3);

            _mm256_storeu_x4_pd(kc_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi64(vc_ptr, y0.v, y1.v, y2.v, y3.v);

            vc_ptr += AVX2_DOUBLE_STRIDE * 4;
            kc_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 3) {
            _mm256_loadu_x3_pd(kc_ptr, x0.k, x1.k, x2.k);
            _mm256_loadu_x3_epi64(vc_ptr, x0.v, x1.v, x2.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);
            y2 = _mm256_halfsort_pd(x2);

            _mm256_storeu_x3_pd(kc_ptr, y0.k, y1.k, y2.k);
            _mm256_storeu_x3_epi64(vc_ptr, y0.v, y1.v, y2.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(kc_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi64(vc_ptr, x0.v, x1.v);

            y0 = _mm256_halfsort_pd(x0);
            y1 = _mm256_halfsort_pd(x1);

            _mm256_storeu_x2_pd(kc_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi64(vc_ptr, y0.v, y1.v);
        }
        else if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(kc_ptr, x0.k);
            _mm256_loadu_x1_epi64(vc_ptr, x0.v);

            y0 = _mm256_halfsort_pd(x0);

            _mm256_storeu_x1_pd(kc_ptr, y0.k);
            _mm256_storeu_x1_epi64(vc_ptr, y0.v);
        }
        if ((r & AVX2_DOUBLE_REMAIN_MASK) > 0) {
            _mm256_loadu_x1_pd(ke_ptr, x0.k);
            _mm256_loadu_x1_epi64(ve_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(ke_ptr, y0.k);
            _mm256_storeu_x1_epi64(ve_ptr, y0.v);
        }
    }

    return SUCCESS;
}

#pragma endregion batchsort

#pragma region scansort

// scansort 4 elems wise
__forceinline static int scansort_p4_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif

    uint e = n - AVX2_DOUBLE_STRIDE;

    uint indexes;
    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    {
        uint i = 0;
        while (true) {
            if (i + AVX2_DOUBLE_STRIDE * 4 + 1 <= n) {
                _mm256_loadu_x4_pd(k_ptr + i, x0.k, x1.k, x2.k, x3.k);
                _mm256_loadu_x4_pd(k_ptr + i + 1, y0.k, y1.k, y2.k, y3.k);

                uint i0 = _mm256_movemask_pd(_mm256_needsswap_pd(x0.k, y0.k));
                uint i1 = _mm256_movemask_pd(_mm256_needsswap_pd(x1.k, y1.k));
                uint i2 = _mm256_movemask_pd(_mm256_needsswap_pd(x2.k, y2.k));
                uint i3 = _mm256_movemask_pd(_mm256_needsswap_pd(x3.k, y3.k));

                indexes = (i0) | (i1 << (AVX2_DOUBLE_STRIDE)) | (i2 << (AVX2_DOUBLE_STRIDE * 2)) | (i3 << (AVX2_DOUBLE_STRIDE * 3));

                if (indexes == 0u) {
                    i += AVX2_DOUBLE_STRIDE * 4;
                    continue;
                }
            }
            else if (i + AVX2_DOUBLE_STRIDE * 2 + 1 <= n) {
                _mm256_loadu_x2_pd(k_ptr + i, x0.k, x1.k);
                _mm256_loadu_x2_pd(k_ptr + i + 1, y0.k, y1.k);

                uint i0 = _mm256_movemask_pd(_mm256_needsswap_pd(x0.k, y0.k));
                uint i1 = _mm256_movemask_pd(_mm256_needsswap_pd(x1.k, y1.k));

                indexes = (i0) | (i1 << (AVX2_DOUBLE_STRIDE));

                if (indexes == 0u) {
                    i += AVX2_DOUBLE_STRIDE * 2;
                    continue;
                }
            }
            else if (i + AVX2_DOUBLE_STRIDE + 1 <= n) {
                _mm256_loadu_x1_pd(k_ptr + i, x0.k);
                _mm256_loadu_x1_pd(k_ptr + i + 1, y0.k);

                indexes = (uint)_mm256_movemask_pd(_mm256_needsswap_pd(x0.k, y0.k));

                if (indexes == 0u) {
                    i += AVX2_DOUBLE_STRIDE;
                    continue;
                }
            }
            else {
                i = e;

                x0.k = _mm256_loadu_pd(k_ptr + i);

                if (!_mm256_needssort_pd(x0.k)) {
                    break;
                }

                x0.v = _mm256_loadu_epi64(v_ptr + i);

                y0 = _mm256_sort_pd(x0);
                _mm256_storeu_pd(k_ptr + i, y0.k);
                _mm256_storeu_epi64(v_ptr + i, y0.v);

                indexes = 0xFu - (uint)_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0.k, y0.k));

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

            x0.k = _mm256_loadu_pd(k_ptr + i);
            x0.v = _mm256_loadu_epi64(v_ptr + i);

            while (true) {
                y0 = _mm256_sort_pd(x0);
                _mm256_storeu_pd(k_ptr + i, y0.k);
                _mm256_storeu_epi64(v_ptr + i, y0.v);

                indexes = (uint)_mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x0.k, y0.k));
                if ((indexes & 1u) == 0u && i > 0u) {
                    uint backward = AVX2_DOUBLE_STRIDE - 2;
                    i = (i > backward) ? i - backward : 0;

                    x0.k = _mm256_loadu_pd(k_ptr + i);

                    if (_mm256_needssort_pd(x0.k)) {
                        x0.v = _mm256_loadu_epi64(v_ptr + i);

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
__forceinline static int permcombsort_p4_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_DOUBLE_STRIDE * 4 || n >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_DOUBLE_STRIDE;

    __m256dkv a, b, x, y;

    for (uint h = c > 2 ? 2 : 1; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a.k = _mm256_loadu_pd(k_ptr + i * AVX2_DOUBLE_STRIDE);
            a.v = _mm256_loadu_epi64(v_ptr + i * AVX2_EPI64_STRIDE);
            b.k = _mm256_loadu_pd(k_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b.v = _mm256_loadu_epi64(v_ptr + (i + h) * AVX2_EPI64_STRIDE);

            _mm256_cmpswap_withperm_pd(a, b, x, y);

            _mm256_storeu_pd(k_ptr + i * AVX2_DOUBLE_STRIDE, x.k);
            _mm256_storeu_epi64(v_ptr + i * AVX2_EPI64_STRIDE, x.v);
            _mm256_storeu_pd(k_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y.k);
            _mm256_storeu_epi64(v_ptr + (i + h) * AVX2_EPI64_STRIDE, y.v);
        }
    }

    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0) {
        b.k = _mm256_loadu_pd(k_ptr + (n - AVX2_DOUBLE_STRIDE));
        b.v = _mm256_loadu_epi64(v_ptr + (n - AVX2_EPI64_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a.k = _mm256_loadu_pd(k_ptr + i * AVX2_DOUBLE_STRIDE);
            a.v = _mm256_loadu_epi64(v_ptr + i * AVX2_EPI64_STRIDE);

            _mm256_cmpswap_pd(a, b, x, y);
            x = _mm256_sort_pd(x);
            b = y;

            _mm256_storeu_pd(k_ptr + i * AVX2_DOUBLE_STRIDE, x.k);
            _mm256_storeu_epi64(v_ptr + i * AVX2_EPI64_STRIDE, x.v);
        }

        a.k = _mm256_loadu_pd(k_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a.v = _mm256_loadu_epi64(v_ptr + (n - AVX2_EPI64_STRIDE * 2));

        _mm256_cmpswap_withperm_pd(a, b, x, y);

        x = _mm256_sort_pd(x);
        y = _mm256_sort_pd(y);

        _mm256_storeu_pd(k_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x.k);
        _mm256_storeu_epi64(v_ptr + (n - AVX2_EPI64_STRIDE * 2), x.v);
        _mm256_storeu_pd(k_ptr + (n - AVX2_DOUBLE_STRIDE), y.k);
        _mm256_storeu_epi64(v_ptr + (n - AVX2_EPI64_STRIDE), y.v);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a.k = _mm256_loadu_pd(k_ptr + i * AVX2_DOUBLE_STRIDE);
            a.v = _mm256_loadu_epi64(v_ptr + i * AVX2_EPI64_STRIDE);
            a = _mm256_sort_pd(a);
            _mm256_storeu_pd(k_ptr + i * AVX2_DOUBLE_STRIDE, a.k);
            _mm256_storeu_epi64(v_ptr + i * AVX2_EPI64_STRIDE, a.v);
        }
    }

    return SUCCESS;
}

// permcombsort 4 elems wise 4 batches
__forceinline static int permcombsort_p4x4_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n <= AVX2_DOUBLE_STRIDE * 4 || n >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }
#endif

    const uint c = n / AVX2_DOUBLE_STRIDE;

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + n;
    double* k2_ptr = k_ptr + n * 2;
    double* k3_ptr = k_ptr + n * 3;
    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + n;
    ulong* v2_ptr = v_ptr + n * 2;
    ulong* v3_ptr = v_ptr + n * 3;

    __m256dkv a0, b0, x0, y0, a1, b1, x1, y1, a2, b2, x2, y2, a3, b3, x3, y3;

    for (uint h = c > 4 ? 4 : 2; h >= 1; h /= 2) {
        for (uint i = 0; i < c - h; i++) {
            a0.k = _mm256_loadu_pd(k0_ptr + i * AVX2_DOUBLE_STRIDE);
            a1.k = _mm256_loadu_pd(k1_ptr + i * AVX2_DOUBLE_STRIDE);
            a2.k = _mm256_loadu_pd(k2_ptr + i * AVX2_DOUBLE_STRIDE);
            a3.k = _mm256_loadu_pd(k3_ptr + i * AVX2_DOUBLE_STRIDE);
            b0.k = _mm256_loadu_pd(k0_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b1.k = _mm256_loadu_pd(k1_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b2.k = _mm256_loadu_pd(k2_ptr + (i + h) * AVX2_DOUBLE_STRIDE);
            b3.k = _mm256_loadu_pd(k3_ptr + (i + h) * AVX2_DOUBLE_STRIDE);

            a0.v = _mm256_loadu_epi64(v0_ptr + i * AVX2_EPI64_STRIDE);
            a1.v = _mm256_loadu_epi64(v1_ptr + i * AVX2_EPI64_STRIDE);
            a2.v = _mm256_loadu_epi64(v2_ptr + i * AVX2_EPI64_STRIDE);
            a3.v = _mm256_loadu_epi64(v3_ptr + i * AVX2_EPI64_STRIDE);
            b0.v = _mm256_loadu_epi64(v0_ptr + (i + h) * AVX2_EPI64_STRIDE);
            b1.v = _mm256_loadu_epi64(v1_ptr + (i + h) * AVX2_EPI64_STRIDE);
            b2.v = _mm256_loadu_epi64(v2_ptr + (i + h) * AVX2_EPI64_STRIDE);
            b3.v = _mm256_loadu_epi64(v3_ptr + (i + h) * AVX2_EPI64_STRIDE);

            _mm256_cmpswap_withperm_pd(a0, b0, x0, y0);
            _mm256_cmpswap_withperm_pd(a1, b1, x1, y1);
            _mm256_cmpswap_withperm_pd(a2, b2, x2, y2);
            _mm256_cmpswap_withperm_pd(a3, b3, x3, y3);

            _mm256_storeu_pd(k0_ptr + i * AVX2_DOUBLE_STRIDE, x0.k);
            _mm256_storeu_pd(k1_ptr + i * AVX2_DOUBLE_STRIDE, x1.k);
            _mm256_storeu_pd(k2_ptr + i * AVX2_DOUBLE_STRIDE, x2.k);
            _mm256_storeu_pd(k3_ptr + i * AVX2_DOUBLE_STRIDE, x3.k);
            _mm256_storeu_pd(k0_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y0.k);
            _mm256_storeu_pd(k1_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y1.k);
            _mm256_storeu_pd(k2_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y2.k);
            _mm256_storeu_pd(k3_ptr + (i + h) * AVX2_DOUBLE_STRIDE, y3.k);

            _mm256_storeu_epi64(v0_ptr + i * AVX2_EPI64_STRIDE, x0.v);
            _mm256_storeu_epi64(v1_ptr + i * AVX2_EPI64_STRIDE, x1.v);
            _mm256_storeu_epi64(v2_ptr + i * AVX2_EPI64_STRIDE, x2.v);
            _mm256_storeu_epi64(v3_ptr + i * AVX2_EPI64_STRIDE, x3.v);
            _mm256_storeu_epi64(v0_ptr + (i + h) * AVX2_EPI64_STRIDE, y0.v);
            _mm256_storeu_epi64(v1_ptr + (i + h) * AVX2_EPI64_STRIDE, y1.v);
            _mm256_storeu_epi64(v2_ptr + (i + h) * AVX2_EPI64_STRIDE, y2.v);
            _mm256_storeu_epi64(v3_ptr + (i + h) * AVX2_EPI64_STRIDE, y3.v);
        }
    }

    if ((n & AVX2_DOUBLE_REMAIN_MASK) != 0) {
        b0.k = _mm256_loadu_pd(k0_ptr + (n - AVX2_DOUBLE_STRIDE));
        b1.k = _mm256_loadu_pd(k1_ptr + (n - AVX2_DOUBLE_STRIDE));
        b2.k = _mm256_loadu_pd(k2_ptr + (n - AVX2_DOUBLE_STRIDE));
        b3.k = _mm256_loadu_pd(k3_ptr + (n - AVX2_DOUBLE_STRIDE));

        b0.v = _mm256_loadu_epi64(v0_ptr + (n - AVX2_EPI64_STRIDE));
        b1.v = _mm256_loadu_epi64(v1_ptr + (n - AVX2_EPI64_STRIDE));
        b2.v = _mm256_loadu_epi64(v2_ptr + (n - AVX2_EPI64_STRIDE));
        b3.v = _mm256_loadu_epi64(v3_ptr + (n - AVX2_EPI64_STRIDE));

        for (uint i = 0; i < c - 1; i++) {
            a0.k = _mm256_loadu_pd(k0_ptr + i * AVX2_DOUBLE_STRIDE);
            a1.k = _mm256_loadu_pd(k1_ptr + i * AVX2_DOUBLE_STRIDE);
            a2.k = _mm256_loadu_pd(k2_ptr + i * AVX2_DOUBLE_STRIDE);
            a3.k = _mm256_loadu_pd(k3_ptr + i * AVX2_DOUBLE_STRIDE);

            a0.v = _mm256_loadu_epi64(v0_ptr + i * AVX2_EPI64_STRIDE);
            a1.v = _mm256_loadu_epi64(v1_ptr + i * AVX2_EPI64_STRIDE);
            a2.v = _mm256_loadu_epi64(v2_ptr + i * AVX2_EPI64_STRIDE);
            a3.v = _mm256_loadu_epi64(v3_ptr + i * AVX2_EPI64_STRIDE);

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

            _mm256_storeu_pd(k0_ptr + i * AVX2_DOUBLE_STRIDE, x0.k);
            _mm256_storeu_pd(k1_ptr + i * AVX2_DOUBLE_STRIDE, x1.k);
            _mm256_storeu_pd(k2_ptr + i * AVX2_DOUBLE_STRIDE, x2.k);
            _mm256_storeu_pd(k3_ptr + i * AVX2_DOUBLE_STRIDE, x3.k);

            _mm256_storeu_epi64(v0_ptr + i * AVX2_EPI64_STRIDE, x0.v);
            _mm256_storeu_epi64(v1_ptr + i * AVX2_EPI64_STRIDE, x1.v);
            _mm256_storeu_epi64(v2_ptr + i * AVX2_EPI64_STRIDE, x2.v);
            _mm256_storeu_epi64(v3_ptr + i * AVX2_EPI64_STRIDE, x3.v);
        }

        a0.k = _mm256_loadu_pd(k0_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a1.k = _mm256_loadu_pd(k1_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a2.k = _mm256_loadu_pd(k2_ptr + (n - AVX2_DOUBLE_STRIDE * 2));
        a3.k = _mm256_loadu_pd(k3_ptr + (n - AVX2_DOUBLE_STRIDE * 2));

        a0.v = _mm256_loadu_epi64(v0_ptr + (n - AVX2_EPI64_STRIDE * 2));
        a1.v = _mm256_loadu_epi64(v1_ptr + (n - AVX2_EPI64_STRIDE * 2));
        a2.v = _mm256_loadu_epi64(v2_ptr + (n - AVX2_EPI64_STRIDE * 2));
        a3.v = _mm256_loadu_epi64(v3_ptr + (n - AVX2_EPI64_STRIDE * 2));

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

        _mm256_storeu_pd(k0_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x0.k);
        _mm256_storeu_pd(k1_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x1.k);
        _mm256_storeu_pd(k2_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x2.k);
        _mm256_storeu_pd(k3_ptr + (n - AVX2_DOUBLE_STRIDE * 2), x3.k);
        _mm256_storeu_pd(k0_ptr + (n - AVX2_DOUBLE_STRIDE), y0.k);
        _mm256_storeu_pd(k1_ptr + (n - AVX2_DOUBLE_STRIDE), y1.k);
        _mm256_storeu_pd(k2_ptr + (n - AVX2_DOUBLE_STRIDE), y2.k);
        _mm256_storeu_pd(k3_ptr + (n - AVX2_DOUBLE_STRIDE), y3.k);

        _mm256_storeu_epi64(v0_ptr + (n - AVX2_EPI64_STRIDE * 2), x0.v);
        _mm256_storeu_epi64(v1_ptr + (n - AVX2_EPI64_STRIDE * 2), x1.v);
        _mm256_storeu_epi64(v2_ptr + (n - AVX2_EPI64_STRIDE * 2), x2.v);
        _mm256_storeu_epi64(v3_ptr + (n - AVX2_EPI64_STRIDE * 2), x3.v);
        _mm256_storeu_epi64(v0_ptr + (n - AVX2_EPI64_STRIDE), y0.v);
        _mm256_storeu_epi64(v1_ptr + (n - AVX2_EPI64_STRIDE), y1.v);
        _mm256_storeu_epi64(v2_ptr + (n - AVX2_EPI64_STRIDE), y2.v);
        _mm256_storeu_epi64(v3_ptr + (n - AVX2_EPI64_STRIDE), y3.v);
    }
    else {
        for (uint i = 0; i < c; i++) {
            a0.k = _mm256_loadu_pd(k0_ptr + i * AVX2_DOUBLE_STRIDE);
            a1.k = _mm256_loadu_pd(k1_ptr + i * AVX2_DOUBLE_STRIDE);
            a2.k = _mm256_loadu_pd(k2_ptr + i * AVX2_DOUBLE_STRIDE);
            a3.k = _mm256_loadu_pd(k3_ptr + i * AVX2_DOUBLE_STRIDE);
            a0.v = _mm256_loadu_epi64(v0_ptr + i * AVX2_EPI64_STRIDE);
            a1.v = _mm256_loadu_epi64(v1_ptr + i * AVX2_EPI64_STRIDE);
            a2.v = _mm256_loadu_epi64(v2_ptr + i * AVX2_EPI64_STRIDE);
            a3.v = _mm256_loadu_epi64(v3_ptr + i * AVX2_EPI64_STRIDE);

            a0 = _mm256_sort_pd(a0);
            a1 = _mm256_sort_pd(a1);
            a2 = _mm256_sort_pd(a2);
            a3 = _mm256_sort_pd(a3);

            _mm256_storeu_pd(k0_ptr + i * AVX2_DOUBLE_STRIDE, a0.k);
            _mm256_storeu_pd(k1_ptr + i * AVX2_DOUBLE_STRIDE, a1.k);
            _mm256_storeu_pd(k2_ptr + i * AVX2_DOUBLE_STRIDE, a2.k);
            _mm256_storeu_pd(k3_ptr + i * AVX2_DOUBLE_STRIDE, a3.k);
            _mm256_storeu_epi64(v0_ptr + i * AVX2_EPI64_STRIDE, a0.v);
            _mm256_storeu_epi64(v1_ptr + i * AVX2_EPI64_STRIDE, a1.v);
            _mm256_storeu_epi64(v2_ptr + i * AVX2_EPI64_STRIDE, a2.v);
            _mm256_storeu_epi64(v3_ptr + i * AVX2_EPI64_STRIDE, a3.v);
        }
    }

    return SUCCESS;
}

#pragma endregion permcombsort

#pragma region shortsort

// shortsort elems5
__forceinline static int shortsort_n5_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkv x, y;

    x.k = _mm256_loadu_pd(k_ptr);
    x.v = _mm256_loadu_epi64(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr, y.k);
    _mm256_storeu_epi64(v_ptr, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 1);
    x.v = _mm256_loadu_epi64(v_ptr + 1);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 1, y.k);
    _mm256_storeu_epi64(v_ptr + 1, y.v);

    x.k = _mm256_loadu_pd(k_ptr);
    x.v = _mm256_loadu_epi64(v_ptr);
    y = _mm256_sort2x2_pd(x);
    _mm256_storeu_pd(k_ptr, y.k);
    _mm256_storeu_epi64(v_ptr, y.v);


    return SUCCESS;
}

// shortsort batches4 x elems5
__forceinline static int shortsort_n4x5_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 5) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 5;
    double* k2_ptr = k_ptr + 10;
    double* k3_ptr = k_ptr + 15;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 5;
    ulong* v2_ptr = v_ptr + 10;
    ulong* v3_ptr = v_ptr + 15;

    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_pd(k0_ptr);
    x1.k = _mm256_loadu_pd(k1_ptr);
    x2.k = _mm256_loadu_pd(k2_ptr);
    x3.k = _mm256_loadu_pd(k3_ptr);
    x0.v = _mm256_loadu_epi64(v0_ptr);
    x1.v = _mm256_loadu_epi64(v1_ptr);
    x2.v = _mm256_loadu_epi64(v2_ptr);
    x3.v = _mm256_loadu_epi64(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr, y0.k);
    _mm256_storeu_pd(k1_ptr, y1.k);
    _mm256_storeu_pd(k2_ptr, y2.k);
    _mm256_storeu_pd(k3_ptr, y3.k);
    _mm256_storeu_epi64(v0_ptr, y0.v);
    _mm256_storeu_epi64(v1_ptr, y1.v);
    _mm256_storeu_epi64(v2_ptr, y2.v);
    _mm256_storeu_epi64(v3_ptr, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 1);
    x1.k = _mm256_loadu_pd(k1_ptr + 1);
    x2.k = _mm256_loadu_pd(k2_ptr + 1);
    x3.k = _mm256_loadu_pd(k3_ptr + 1);
    x0.v = _mm256_loadu_epi64(v0_ptr + 1);
    x1.v = _mm256_loadu_epi64(v1_ptr + 1);
    x2.v = _mm256_loadu_epi64(v2_ptr + 1);
    x3.v = _mm256_loadu_epi64(v3_ptr + 1);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 1, y0.k);
    _mm256_storeu_pd(k1_ptr + 1, y1.k);
    _mm256_storeu_pd(k2_ptr + 1, y2.k);
    _mm256_storeu_pd(k3_ptr + 1, y3.k);
    _mm256_storeu_epi64(v0_ptr + 1, y0.v);
    _mm256_storeu_epi64(v1_ptr + 1, y1.v);
    _mm256_storeu_epi64(v2_ptr + 1, y2.v);
    _mm256_storeu_epi64(v3_ptr + 1, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr);
    x1.k = _mm256_loadu_pd(k1_ptr);
    x2.k = _mm256_loadu_pd(k2_ptr);
    x3.k = _mm256_loadu_pd(k3_ptr);
    x0.v = _mm256_loadu_epi64(v0_ptr);
    x1.v = _mm256_loadu_epi64(v1_ptr);
    x2.v = _mm256_loadu_epi64(v2_ptr);
    x3.v = _mm256_loadu_epi64(v3_ptr);
    y0 = _mm256_sort2x2_pd(x0);
    y1 = _mm256_sort2x2_pd(x1);
    y2 = _mm256_sort2x2_pd(x2);
    y3 = _mm256_sort2x2_pd(x3);
    _mm256_storeu_pd(k0_ptr, y0.k);
    _mm256_storeu_pd(k1_ptr, y1.k);
    _mm256_storeu_pd(k2_ptr, y2.k);
    _mm256_storeu_pd(k3_ptr, y3.k);
    _mm256_storeu_epi64(v0_ptr, y0.v);
    _mm256_storeu_epi64(v1_ptr, y1.v);
    _mm256_storeu_epi64(v2_ptr, y2.v);
    _mm256_storeu_epi64(v3_ptr, y3.v);

    return SUCCESS;
}

// shortsort elems6
__forceinline static int shortsort_n6_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkv x, y;

    x.k = _mm256_loadu_pd(k_ptr);
    x.v = _mm256_loadu_epi64(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr, y.k);
    _mm256_storeu_epi64(v_ptr, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 2);
    x.v = _mm256_loadu_epi64(v_ptr + 2);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 2, y.k);
    _mm256_storeu_epi64(v_ptr + 2, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 1);
    x.v = _mm256_loadu_epi64(v_ptr + 1);
    y = _mm256_sort2x2_pd(x);
    _mm256_storeu_pd(k_ptr + 1, y.k);
    _mm256_storeu_epi64(v_ptr + 1, y.v);

    x.k = _mm256_loadu_pd(k_ptr);
    x.v = _mm256_loadu_epi64(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr, y.k);
    _mm256_storeu_epi64(v_ptr, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 2);
    x.v = _mm256_loadu_epi64(v_ptr + 2);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 2, y.k);
    _mm256_storeu_epi64(v_ptr + 2, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems6
__forceinline static int shortsort_n4x6_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 6) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 6;
    double* k2_ptr = k_ptr + 12;
    double* k3_ptr = k_ptr + 18;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 6;
    ulong* v2_ptr = v_ptr + 12;
    ulong* v3_ptr = v_ptr + 18;

    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_pd(k0_ptr);
    x1.k = _mm256_loadu_pd(k1_ptr);
    x2.k = _mm256_loadu_pd(k2_ptr);
    x3.k = _mm256_loadu_pd(k3_ptr);
    x0.v = _mm256_loadu_epi64(v0_ptr);
    x1.v = _mm256_loadu_epi64(v1_ptr);
    x2.v = _mm256_loadu_epi64(v2_ptr);
    x3.v = _mm256_loadu_epi64(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr, y0.k);
    _mm256_storeu_pd(k1_ptr, y1.k);
    _mm256_storeu_pd(k2_ptr, y2.k);
    _mm256_storeu_pd(k3_ptr, y3.k);
    _mm256_storeu_epi64(v0_ptr, y0.v);
    _mm256_storeu_epi64(v1_ptr, y1.v);
    _mm256_storeu_epi64(v2_ptr, y2.v);
    _mm256_storeu_epi64(v3_ptr, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 2);
    x1.k = _mm256_loadu_pd(k1_ptr + 2);
    x2.k = _mm256_loadu_pd(k2_ptr + 2);
    x3.k = _mm256_loadu_pd(k3_ptr + 2);
    x0.v = _mm256_loadu_epi64(v0_ptr + 2);
    x1.v = _mm256_loadu_epi64(v1_ptr + 2);
    x2.v = _mm256_loadu_epi64(v2_ptr + 2);
    x3.v = _mm256_loadu_epi64(v3_ptr + 2);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 2, y0.k);
    _mm256_storeu_pd(k1_ptr + 2, y1.k);
    _mm256_storeu_pd(k2_ptr + 2, y2.k);
    _mm256_storeu_pd(k3_ptr + 2, y3.k);
    _mm256_storeu_epi64(v0_ptr + 2, y0.v);
    _mm256_storeu_epi64(v1_ptr + 2, y1.v);
    _mm256_storeu_epi64(v2_ptr + 2, y2.v);
    _mm256_storeu_epi64(v3_ptr + 2, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 1);
    x1.k = _mm256_loadu_pd(k1_ptr + 1);
    x2.k = _mm256_loadu_pd(k2_ptr + 1);
    x3.k = _mm256_loadu_pd(k3_ptr + 1);
    x0.v = _mm256_loadu_epi64(v0_ptr + 1);
    x1.v = _mm256_loadu_epi64(v1_ptr + 1);
    x2.v = _mm256_loadu_epi64(v2_ptr + 1);
    x3.v = _mm256_loadu_epi64(v3_ptr + 1);
    y0 = _mm256_sort2x2_pd(x0);
    y1 = _mm256_sort2x2_pd(x1);
    y2 = _mm256_sort2x2_pd(x2);
    y3 = _mm256_sort2x2_pd(x3);
    _mm256_storeu_pd(k0_ptr + 1, y0.k);
    _mm256_storeu_pd(k1_ptr + 1, y1.k);
    _mm256_storeu_pd(k2_ptr + 1, y2.k);
    _mm256_storeu_pd(k3_ptr + 1, y3.k);
    _mm256_storeu_epi64(v0_ptr + 1, y0.v);
    _mm256_storeu_epi64(v1_ptr + 1, y1.v);
    _mm256_storeu_epi64(v2_ptr + 1, y2.v);
    _mm256_storeu_epi64(v3_ptr + 1, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr);
    x1.k = _mm256_loadu_pd(k1_ptr);
    x2.k = _mm256_loadu_pd(k2_ptr);
    x3.k = _mm256_loadu_pd(k3_ptr);
    x0.v = _mm256_loadu_epi64(v0_ptr);
    x1.v = _mm256_loadu_epi64(v1_ptr);
    x2.v = _mm256_loadu_epi64(v2_ptr);
    x3.v = _mm256_loadu_epi64(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr, y0.k);
    _mm256_storeu_pd(k1_ptr, y1.k);
    _mm256_storeu_pd(k2_ptr, y2.k);
    _mm256_storeu_pd(k3_ptr, y3.k);
    _mm256_storeu_epi64(v0_ptr, y0.v);
    _mm256_storeu_epi64(v1_ptr, y1.v);
    _mm256_storeu_epi64(v2_ptr, y2.v);
    _mm256_storeu_epi64(v3_ptr, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 2);
    x1.k = _mm256_loadu_pd(k1_ptr + 2);
    x2.k = _mm256_loadu_pd(k2_ptr + 2);
    x3.k = _mm256_loadu_pd(k3_ptr + 2);
    x0.v = _mm256_loadu_epi64(v0_ptr + 2);
    x1.v = _mm256_loadu_epi64(v1_ptr + 2);
    x2.v = _mm256_loadu_epi64(v2_ptr + 2);
    x3.v = _mm256_loadu_epi64(v3_ptr + 2);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 2, y0.k);
    _mm256_storeu_pd(k1_ptr + 2, y1.k);
    _mm256_storeu_pd(k2_ptr + 2, y2.k);
    _mm256_storeu_pd(k3_ptr + 2, y3.k);
    _mm256_storeu_epi64(v0_ptr + 2, y0.v);
    _mm256_storeu_epi64(v1_ptr + 2, y1.v);
    _mm256_storeu_epi64(v2_ptr + 2, y2.v);
    _mm256_storeu_epi64(v3_ptr + 2, y3.v);

    return SUCCESS;
}

// shortsort elems7
__forceinline static int shortsort_n7_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkv x, y;

    x.k = _mm256_loadu_pd(k_ptr);
    x.v = _mm256_loadu_epi64(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr, y.k);
    _mm256_storeu_epi64(v_ptr, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 3);
    x.v = _mm256_loadu_epi64(v_ptr + 3);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 3, y.k);
    _mm256_storeu_epi64(v_ptr + 3, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 1);
    x.v = _mm256_loadu_epi64(v_ptr + 1);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 1, y.k);
    _mm256_storeu_epi64(v_ptr + 1, y.v);

    x.k = _mm256_loadu_pd(k_ptr);
    x.v = _mm256_loadu_epi64(v_ptr);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr, y.k);
    _mm256_storeu_epi64(v_ptr, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 3);
    x.v = _mm256_loadu_epi64(v_ptr + 3);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 3, y.k);
    _mm256_storeu_epi64(v_ptr + 3, y.v);

    x.k = _mm256_loadu_pd(k_ptr + 2);
    x.v = _mm256_loadu_epi64(v_ptr + 2);
    y = _mm256_sort_pd(x);
    _mm256_storeu_pd(k_ptr + 2, y.k);
    _mm256_storeu_epi64(v_ptr + 2, y.v);

    return SUCCESS;
}

// shortsort batches4 x elems7
__forceinline static int shortsort_n4x7_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 7) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 7;
    double* k2_ptr = k_ptr + 14;
    double* k3_ptr = k_ptr + 21;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 7;
    ulong* v2_ptr = v_ptr + 14;
    ulong* v3_ptr = v_ptr + 21;

    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    x0.k = _mm256_loadu_pd(k0_ptr);
    x1.k = _mm256_loadu_pd(k1_ptr);
    x2.k = _mm256_loadu_pd(k2_ptr);
    x3.k = _mm256_loadu_pd(k3_ptr);
    x0.v = _mm256_loadu_epi64(v0_ptr);
    x1.v = _mm256_loadu_epi64(v1_ptr);
    x2.v = _mm256_loadu_epi64(v2_ptr);
    x3.v = _mm256_loadu_epi64(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr, y0.k);
    _mm256_storeu_pd(k1_ptr, y1.k);
    _mm256_storeu_pd(k2_ptr, y2.k);
    _mm256_storeu_pd(k3_ptr, y3.k);
    _mm256_storeu_epi64(v0_ptr, y0.v);
    _mm256_storeu_epi64(v1_ptr, y1.v);
    _mm256_storeu_epi64(v2_ptr, y2.v);
    _mm256_storeu_epi64(v3_ptr, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 3);
    x1.k = _mm256_loadu_pd(k1_ptr + 3);
    x2.k = _mm256_loadu_pd(k2_ptr + 3);
    x3.k = _mm256_loadu_pd(k3_ptr + 3);
    x0.v = _mm256_loadu_epi64(v0_ptr + 3);
    x1.v = _mm256_loadu_epi64(v1_ptr + 3);
    x2.v = _mm256_loadu_epi64(v2_ptr + 3);
    x3.v = _mm256_loadu_epi64(v3_ptr + 3);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 3, y0.k);
    _mm256_storeu_pd(k1_ptr + 3, y1.k);
    _mm256_storeu_pd(k2_ptr + 3, y2.k);
    _mm256_storeu_pd(k3_ptr + 3, y3.k);
    _mm256_storeu_epi64(v0_ptr + 3, y0.v);
    _mm256_storeu_epi64(v1_ptr + 3, y1.v);
    _mm256_storeu_epi64(v2_ptr + 3, y2.v);
    _mm256_storeu_epi64(v3_ptr + 3, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 1);
    x1.k = _mm256_loadu_pd(k1_ptr + 1);
    x2.k = _mm256_loadu_pd(k2_ptr + 1);
    x3.k = _mm256_loadu_pd(k3_ptr + 1);
    x0.v = _mm256_loadu_epi64(v0_ptr + 1);
    x1.v = _mm256_loadu_epi64(v1_ptr + 1);
    x2.v = _mm256_loadu_epi64(v2_ptr + 1);
    x3.v = _mm256_loadu_epi64(v3_ptr + 1);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 1, y0.k);
    _mm256_storeu_pd(k1_ptr + 1, y1.k);
    _mm256_storeu_pd(k2_ptr + 1, y2.k);
    _mm256_storeu_pd(k3_ptr + 1, y3.k);
    _mm256_storeu_epi64(v0_ptr + 1, y0.v);
    _mm256_storeu_epi64(v1_ptr + 1, y1.v);
    _mm256_storeu_epi64(v2_ptr + 1, y2.v);
    _mm256_storeu_epi64(v3_ptr + 1, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr);
    x1.k = _mm256_loadu_pd(k1_ptr);
    x2.k = _mm256_loadu_pd(k2_ptr);
    x3.k = _mm256_loadu_pd(k3_ptr);
    x0.v = _mm256_loadu_epi64(v0_ptr);
    x1.v = _mm256_loadu_epi64(v1_ptr);
    x2.v = _mm256_loadu_epi64(v2_ptr);
    x3.v = _mm256_loadu_epi64(v3_ptr);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr, y0.k);
    _mm256_storeu_pd(k1_ptr, y1.k);
    _mm256_storeu_pd(k2_ptr, y2.k);
    _mm256_storeu_pd(k3_ptr, y3.k);
    _mm256_storeu_epi64(v0_ptr, y0.v);
    _mm256_storeu_epi64(v1_ptr, y1.v);
    _mm256_storeu_epi64(v2_ptr, y2.v);
    _mm256_storeu_epi64(v3_ptr, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 3);
    x1.k = _mm256_loadu_pd(k1_ptr + 3);
    x2.k = _mm256_loadu_pd(k2_ptr + 3);
    x3.k = _mm256_loadu_pd(k3_ptr + 3);
    x0.v = _mm256_loadu_epi64(v0_ptr + 3);
    x1.v = _mm256_loadu_epi64(v1_ptr + 3);
    x2.v = _mm256_loadu_epi64(v2_ptr + 3);
    x3.v = _mm256_loadu_epi64(v3_ptr + 3);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 3, y0.k);
    _mm256_storeu_pd(k1_ptr + 3, y1.k);
    _mm256_storeu_pd(k2_ptr + 3, y2.k);
    _mm256_storeu_pd(k3_ptr + 3, y3.k);
    _mm256_storeu_epi64(v0_ptr + 3, y0.v);
    _mm256_storeu_epi64(v1_ptr + 3, y1.v);
    _mm256_storeu_epi64(v2_ptr + 3, y2.v);
    _mm256_storeu_epi64(v3_ptr + 3, y3.v);

    x0.k = _mm256_loadu_pd(k0_ptr + 2);
    x1.k = _mm256_loadu_pd(k1_ptr + 2);
    x2.k = _mm256_loadu_pd(k2_ptr + 2);
    x3.k = _mm256_loadu_pd(k3_ptr + 2);
    x0.v = _mm256_loadu_epi64(v0_ptr + 2);
    x1.v = _mm256_loadu_epi64(v1_ptr + 2);
    x2.v = _mm256_loadu_epi64(v2_ptr + 2);
    x3.v = _mm256_loadu_epi64(v3_ptr + 2);
    y0 = _mm256_sort_pd(x0);
    y1 = _mm256_sort_pd(x1);
    y2 = _mm256_sort_pd(x2);
    y3 = _mm256_sort_pd(x3);
    _mm256_storeu_pd(k0_ptr + 2, y0.k);
    _mm256_storeu_pd(k1_ptr + 2, y1.k);
    _mm256_storeu_pd(k2_ptr + 2, y2.k);
    _mm256_storeu_pd(k3_ptr + 2, y3.k);
    _mm256_storeu_epi64(v0_ptr + 2, y0.v);
    _mm256_storeu_epi64(v1_ptr + 2, y1.v);
    _mm256_storeu_epi64(v2_ptr + 2, y2.v);
    _mm256_storeu_epi64(v3_ptr + 2, y3.v);

    return SUCCESS;
}

// shortsort elems9
__forceinline static int shortsort_n9_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;
    __m256dkv z;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 1, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 1, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 1, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 1, y.v0, y.v1);

    z.k = _mm256_loadu_pd(k_ptr);
    z.v = _mm256_loadu_epi64(v_ptr);
    z = _mm256_sort2x2_pd(z);
    _mm256_storeu_pd(k_ptr, z.k);
    _mm256_storeu_epi64(v_ptr, z.v);

    return SUCCESS;
}

// shortsort batches4 x elems9
__forceinline static int shortsort_n4x9_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 9) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 9;
    double* k2_ptr = k_ptr + 18;
    double* k3_ptr = k_ptr + 27;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 9;
    ulong* v2_ptr = v_ptr + 18;
    ulong* v3_ptr = v_ptr + 27;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256dkv z0, z1, z2, z3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 1, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 1, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 1, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 1, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 1, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 1, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 1, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 1, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 1, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 1, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 1, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 1, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 1, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 1, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 1, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 1, y3.v0, y3.v1);

    z0.k = _mm256_loadu_pd(k0_ptr);
    z1.k = _mm256_loadu_pd(k1_ptr);
    z2.k = _mm256_loadu_pd(k2_ptr);
    z3.k = _mm256_loadu_pd(k3_ptr);
    z0.v = _mm256_loadu_epi64(v0_ptr);
    z1.v = _mm256_loadu_epi64(v1_ptr);
    z2.v = _mm256_loadu_epi64(v2_ptr);
    z3.v = _mm256_loadu_epi64(v3_ptr);
    z0 = _mm256_sort2x2_pd(z0);
    z1 = _mm256_sort2x2_pd(z1);
    z2 = _mm256_sort2x2_pd(z2);
    z3 = _mm256_sort2x2_pd(z3);
    _mm256_storeu_pd(k0_ptr, z0.k);
    _mm256_storeu_pd(k1_ptr, z1.k);
    _mm256_storeu_pd(k2_ptr, z2.k);
    _mm256_storeu_pd(k3_ptr, z3.k);
    _mm256_storeu_epi64(v0_ptr, z0.v);
    _mm256_storeu_epi64(v1_ptr, z1.v);
    _mm256_storeu_epi64(v2_ptr, z2.v);
    _mm256_storeu_epi64(v3_ptr, z3.v);

    return SUCCESS;
}

// shortsort elems10
__forceinline static int shortsort_n10_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;
    __m256dkv z;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 2, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 2, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 2, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 2, y.v0, y.v1);

    z.k = _mm256_loadu_pd(k_ptr);
    z.v = _mm256_loadu_epi64(v_ptr);
    z = _mm256_sort_pd(z);
    _mm256_storeu_pd(k_ptr, z.k);
    _mm256_storeu_epi64(v_ptr, z.v);

    return SUCCESS;
}

// shortsort batches4 x elems10
__forceinline static int shortsort_n4x10_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 10) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 10;
    double* k2_ptr = k_ptr + 20;
    double* k3_ptr = k_ptr + 30;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 10;
    ulong* v2_ptr = v_ptr + 20;
    ulong* v3_ptr = v_ptr + 30;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256dkv z0, z1, z2, z3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 2, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 2, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 2, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 2, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 2, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 2, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 2, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 2, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 2, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 2, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 2, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 2, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 2, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 2, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 2, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 2, y3.v0, y3.v1);

    z0.k = _mm256_loadu_pd(k0_ptr);
    z1.k = _mm256_loadu_pd(k1_ptr);
    z2.k = _mm256_loadu_pd(k2_ptr);
    z3.k = _mm256_loadu_pd(k3_ptr);
    z0.v = _mm256_loadu_epi64(v0_ptr);
    z1.v = _mm256_loadu_epi64(v1_ptr);
    z2.v = _mm256_loadu_epi64(v2_ptr);
    z3.v = _mm256_loadu_epi64(v3_ptr);
    z0 = _mm256_sort_pd(z0);
    z1 = _mm256_sort_pd(z1);
    z2 = _mm256_sort_pd(z2);
    z3 = _mm256_sort_pd(z3);
    _mm256_storeu_pd(k0_ptr, z0.k);
    _mm256_storeu_pd(k1_ptr, z1.k);
    _mm256_storeu_pd(k2_ptr, z2.k);
    _mm256_storeu_pd(k3_ptr, z3.k);
    _mm256_storeu_epi64(v0_ptr, z0.v);
    _mm256_storeu_epi64(v1_ptr, z1.v);
    _mm256_storeu_epi64(v2_ptr, z2.v);
    _mm256_storeu_epi64(v3_ptr, z3.v);

    return SUCCESS;
}

// shortsort elems11
__forceinline static int shortsort_n11_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 3, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 3, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 3, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 3, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    return SUCCESS;
}

// shortsort batches4 x elems11
__forceinline static int shortsort_n4x11_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 11) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 11;
    double* k2_ptr = k_ptr + 22;
    double* k3_ptr = k_ptr + 33;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 11;
    ulong* v2_ptr = v_ptr + 22;
    ulong* v3_ptr = v_ptr + 33;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 3, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 3, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 3, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 3, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 3, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 3, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 3, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 3, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 3, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 3, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 3, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 3, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 3, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 3, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 3, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 3, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    return SUCCESS;
}

// shortsort elems12
__forceinline static int shortsort_n12_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 4, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 4, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 4, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 4, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    return SUCCESS;
}

// shortsort batches4 x elems12
__forceinline static int shortsort_n4x12_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 12) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 12;
    double* k2_ptr = k_ptr + 24;
    double* k3_ptr = k_ptr + 36;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 12;
    ulong* v2_ptr = v_ptr + 24;
    ulong* v3_ptr = v_ptr + 36;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 4, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 4, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 4, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 4, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 4, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 4, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 4, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 4, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 4, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 4, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 4, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 4, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 4, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 4, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 4, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 4, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    return SUCCESS;
}

// shortsort elems13
__forceinline static int shortsort_n13_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 5, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 5, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 5, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 5, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 2, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 2, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 2, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 2, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 5, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 5, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 5, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 5, y.v0, y.v1);

    return SUCCESS;
}

// shortsort batches4 x elems13
__forceinline static int shortsort_n4x13_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 13) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 13;
    double* k2_ptr = k_ptr + 26;
    double* k3_ptr = k_ptr + 39;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 13;
    ulong* v2_ptr = v_ptr + 26;
    ulong* v3_ptr = v_ptr + 39;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 5, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 5, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 5, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 5, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 5, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 5, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 5, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 5, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 5, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 5, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 5, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 5, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 5, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 5, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 5, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 5, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 2, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 2, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 2, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 2, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 2, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 2, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 2, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 2, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 2, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 2, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 2, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 2, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 2, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 2, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 2, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 2, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 5, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 5, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 5, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 5, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 5, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 5, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 5, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 5, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 5, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 5, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 5, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 5, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 5, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 5, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 5, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 5, y3.v0, y3.v1);

    return SUCCESS;
}

// shortsort elems14
__forceinline static int shortsort_n14_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;
    __m256dkv z;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 6, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 6, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 6, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 6, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 3, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 3, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 3, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 3, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 6, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 6, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 6, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 6, y.v0, y.v1);

    z.k = _mm256_loadu_pd(k_ptr + 5);
    z.v = _mm256_loadu_epi64(v_ptr + 5);
    z = _mm256_sort2x2_pd(z);
    _mm256_storeu_pd(k_ptr + 5, z.k);
    _mm256_storeu_epi64(v_ptr + 5, z.v);

    return SUCCESS;
}

// shortsort batches4 x elems14
__forceinline static int shortsort_n4x14_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 14) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 14;
    double* k2_ptr = k_ptr + 28;
    double* k3_ptr = k_ptr + 42;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 14;
    ulong* v2_ptr = v_ptr + 28;
    ulong* v3_ptr = v_ptr + 42;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;
    __m256dkv z0, z1, z2, z3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 6, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 6, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 6, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 6, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 6, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 6, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 6, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 6, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 6, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 6, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 6, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 6, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 6, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 6, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 6, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 6, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 3, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 3, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 3, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 3, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 3, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 3, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 3, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 3, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 3, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 3, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 3, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 3, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 3, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 3, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 3, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 3, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 6, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 6, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 6, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 6, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 6, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 6, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 6, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 6, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 6, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 6, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 6, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 6, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 6, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 6, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 6, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 6, y3.v0, y3.v1);

    z0.k = _mm256_loadu_pd(k0_ptr + 5);
    z1.k = _mm256_loadu_pd(k1_ptr + 5);
    z2.k = _mm256_loadu_pd(k2_ptr + 5);
    z3.k = _mm256_loadu_pd(k3_ptr + 5);
    z0.v = _mm256_loadu_epi64(v0_ptr + 5);
    z1.v = _mm256_loadu_epi64(v1_ptr + 5);
    z2.v = _mm256_loadu_epi64(v2_ptr + 5);
    z3.v = _mm256_loadu_epi64(v3_ptr + 5);
    z0 = _mm256_sort2x2_pd(z0);
    z1 = _mm256_sort2x2_pd(z1);
    z2 = _mm256_sort2x2_pd(z2);
    z3 = _mm256_sort2x2_pd(z3);
    _mm256_storeu_pd(k0_ptr + 5, z0.k);
    _mm256_storeu_pd(k1_ptr + 5, z1.k);
    _mm256_storeu_pd(k2_ptr + 5, z2.k);
    _mm256_storeu_pd(k3_ptr + 5, z3.k);
    _mm256_storeu_epi64(v0_ptr + 5, z0.v);
    _mm256_storeu_epi64(v1_ptr + 5, z1.v);
    _mm256_storeu_epi64(v2_ptr + 5, z2.v);
    _mm256_storeu_epi64(v3_ptr + 5, z3.v);

    return SUCCESS;
}

// shortsort elems15
__forceinline static int shortsort_n15_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 7, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 7, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 7, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 7, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 3, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 3, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 3, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 3, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 7, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 7, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 7, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 7, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + 4, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + 4, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + 4, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + 4, y.v0, y.v1);

    return SUCCESS;
}

// shortsort batches4 x elems15
__forceinline static int shortsort_n4x15_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != 15) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 15;
    double* k2_ptr = k_ptr + 30;
    double* k3_ptr = k_ptr + 45;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 15;
    ulong* v2_ptr = v_ptr + 30;
    ulong* v3_ptr = v_ptr + 45;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 7, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 7, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 7, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 7, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 7, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 7, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 7, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 7, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 7, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 7, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 7, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 7, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 7, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 7, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 7, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 7, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 3, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 3, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 3, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 3, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 3, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 3, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 3, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 3, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 3, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 3, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 3, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 3, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 3, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 3, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 3, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 3, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 7, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 7, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 7, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 7, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 7, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 7, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 7, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 7, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 7, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 7, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 7, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 7, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 7, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 7, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 7, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 7, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + 4, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + 4, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + 4, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + 4, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + 4, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + 4, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + 4, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + 4, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + 4, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + 4, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + 4, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + 4, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + 4, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + 4, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + 4, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + 4, y3.v0, y3.v1);

    return SUCCESS;
}

// shortsort elems16
__forceinline static int shortsort_n16_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    __m256dkvx2 x, y;

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE * 2, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE * 2, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE * 2, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE * 2, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE * 2, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE * 2, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE * 2, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE * 2, y.v0, y.v1);

    _mm256_loadu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE, x.k0, x.k1);
    _mm256_loadu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE, x.v0, x.v1);
    y = _mm256x2_sort_pd(x);
    _mm256_storeu_x2_pd(k_ptr + AVX2_DOUBLE_STRIDE, y.k0, y.k1);
    _mm256_storeu_x2_epi64(v_ptr + AVX2_EPI64_STRIDE, y.v0, y.v1);

    return SUCCESS;
}

// shortsort batches4 x elems16
__forceinline static int shortsort_n4x16_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n != AVX2_DOUBLE_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    double* k0_ptr = k_ptr;
    double* k1_ptr = k_ptr + 16;
    double* k2_ptr = k_ptr + 32;
    double* k3_ptr = k_ptr + 48;

    ulong* v0_ptr = v_ptr;
    ulong* v1_ptr = v_ptr + 16;
    ulong* v2_ptr = v_ptr + 32;
    ulong* v3_ptr = v_ptr + 48;

    __m256dkvx2 x0, x1, x2, x3, y0, y1, y2, y3;

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE * 2, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE * 2, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE * 2, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE * 2, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE * 2, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE * 2, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE * 2, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE * 2, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE * 2, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE * 2, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE * 2, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE * 2, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE * 2, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE * 2, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE * 2, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE * 2, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE * 2, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE * 2, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE * 2, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE * 2, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE * 2, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE * 2, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE * 2, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE * 2, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE * 2, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE * 2, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE * 2, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE * 2, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE * 2, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE * 2, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE * 2, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE * 2, y3.v0, y3.v1);

    _mm256_loadu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE, x0.k0, x0.k1);
    _mm256_loadu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE, x1.k0, x1.k1);
    _mm256_loadu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE, x2.k0, x2.k1);
    _mm256_loadu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE, x3.k0, x3.k1);
    _mm256_loadu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE, x0.v0, x0.v1);
    _mm256_loadu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE, x1.v0, x1.v1);
    _mm256_loadu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE, x2.v0, x2.v1);
    _mm256_loadu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE, x3.v0, x3.v1);
    y0 = _mm256x2_sort_pd(x0);
    y1 = _mm256x2_sort_pd(x1);
    y2 = _mm256x2_sort_pd(x2);
    y3 = _mm256x2_sort_pd(x3);
    _mm256_storeu_x2_pd(k0_ptr + AVX2_DOUBLE_STRIDE, y0.k0, y0.k1);
    _mm256_storeu_x2_pd(k1_ptr + AVX2_DOUBLE_STRIDE, y1.k0, y1.k1);
    _mm256_storeu_x2_pd(k2_ptr + AVX2_DOUBLE_STRIDE, y2.k0, y2.k1);
    _mm256_storeu_x2_pd(k3_ptr + AVX2_DOUBLE_STRIDE, y3.k0, y3.k1);
    _mm256_storeu_x2_epi64(v0_ptr + AVX2_EPI64_STRIDE, y0.v0, y0.v1);
    _mm256_storeu_x2_epi64(v1_ptr + AVX2_EPI64_STRIDE, y1.v0, y1.v1);
    _mm256_storeu_x2_epi64(v2_ptr + AVX2_EPI64_STRIDE, y2.v0, y2.v1);
    _mm256_storeu_x2_epi64(v3_ptr + AVX2_EPI64_STRIDE, y3.v0, y3.v1);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

// longsort elems 32+
__forceinline static int longsort_n32plus_d(const uint n, ulong* __restrict v_ptr, double* __restrict k_ptr) {
#ifdef _DEBUG
    if (n < AVX2_DOUBLE_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }
#endif //_DEBUG

    uint h;

    for (h = (uint)(n * 10uLL / 13uLL); h > 33; h = (uint)(h * 10uLL / 13uLL)) {
        combsort_h17plus_d(n, h, v_ptr, k_ptr);
    }
    if (h >= 16) {
        combsort_h16_d(n, v_ptr, k_ptr);
        h = h * 10 / 13;
    }
    for (; h > 12; h = h * 10 / 13) {
        combsort_h13to15_d(n, h, v_ptr, k_ptr);
    }
    if (h >= 12) {
        combsort_h12_d(n, v_ptr, k_ptr);
        h = h * 10 / 13;
    }
    for (; h > 8; h = h * 10 / 13) {
        combsort_h9to11_d(n, h, v_ptr, k_ptr);
    }
    if (h >= 8) {
        combsort_h8_d(n, v_ptr, k_ptr);
        h = h * 10 / 13;
    }
    for (; h > 4; h = h * 10 / 13) {
        combsort_h5to7_d(n, h, v_ptr, k_ptr);
    }

    paracombsort_p2x4_d(n, v_ptr, k_ptr);

    batchsort_p4_d(n, v_ptr, k_ptr);
    scansort_p4_d(n, v_ptr, k_ptr);

    return SUCCESS;
}

#pragma endregion longsort

#pragma region sort

int sortwithkeydsc_ignnan_s2_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 2) {
        return FAILURE_BADPARAM;
    }

    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)k_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= AVX2_DOUBLE_STRIDE * 4 / 2) {
            _mm256_loadu_x4_pd(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi64(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort2x2_pd(x0);
            y1 = _mm256_sort2x2_pd(x1);
            y2 = _mm256_sort2x2_pd(x2);
            y3 = _mm256_sort2x2_pd(x3);

            _mm256_storeu_x4_pd(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi64(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 4;
            v_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4 / 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x2_pd(k_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi64(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort2x2_pd(x0);
            y1 = _mm256_sort2x2_pd(x1);

            _mm256_storeu_x2_pd(k_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi64(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 2;
            v_ptr += AVX2_EPI64_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r >= AVX2_DOUBLE_STRIDE / 2) {
            _mm256_loadu_x1_pd(k_ptr, x0.k);
            _mm256_loadu_x1_epi64(v_ptr, x0.v);

            y0 = _mm256_sort2x2_pd(x0);

            _mm256_storeu_x1_pd(k_ptr, y0.k);
            _mm256_storeu_x1_epi64(v_ptr, y0.v);

            k_ptr += AVX2_DOUBLE_STRIDE;
            v_ptr += AVX2_EPI64_STRIDE;
            r -= AVX2_DOUBLE_STRIDE / 2;
        }
    }
    else {
        while (r >= AVX2_DOUBLE_STRIDE * 4 / 2) {
            _mm256_load_x4_pd(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_load_x4_epi64(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort2x2_pd(x0);
            y1 = _mm256_sort2x2_pd(x1);
            y2 = _mm256_sort2x2_pd(x2);
            y3 = _mm256_sort2x2_pd(x3);

            _mm256_stream_x4_pd(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_stream_x4_epi64(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 4;
            v_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4 / 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x2_pd(k_ptr, x0.k, x1.k);
            _mm256_load_x2_epi64(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort2x2_pd(x0);
            y1 = _mm256_sort2x2_pd(x1);

            _mm256_stream_x2_pd(k_ptr, y0.k, y1.k);
            _mm256_stream_x2_epi64(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 2;
            v_ptr += AVX2_EPI64_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r >= AVX2_DOUBLE_STRIDE / 2) {
            _mm256_load_x1_pd(k_ptr, x0.k);
            _mm256_load_x1_epi64(v_ptr, x0.v);

            y0 = _mm256_sort2x2_pd(x0);

            _mm256_stream_x1_pd(k_ptr, y0.k);
            _mm256_stream_x1_epi64(v_ptr, y0.v);

            k_ptr += AVX2_DOUBLE_STRIDE;
            v_ptr += AVX2_EPI64_STRIDE;
            r -= AVX2_DOUBLE_STRIDE / 2;
        }
    }
    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd((r * 2) & AVX2_DOUBLE_REMAIN_MASK);

        _mm256_maskload_x1_pd(k_ptr, x0.k, mask);
        _mm256_maskload_x1_epi64(v_ptr, x0.v, mask);

        y0 = _mm256_sort2x2_pd(x0);

        _mm256_maskstore_x1_pd(k_ptr, y0.k, mask);
        _mm256_maskstore_x1_epi64(v_ptr, y0.v, mask);
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s3_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 3) {
        return FAILURE_BADPARAM;
    }

    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    while (r >= 5) {
        x0.k = _mm256_loadu_pd(k_ptr);
        x1.k = _mm256_loadu_pd(k_ptr + 3);
        x2.k = _mm256_loadu_pd(k_ptr + 6);
        x3.k = _mm256_loadu_pd(k_ptr + 9);
        x0.v = _mm256_loadu_epi64(v_ptr);
        x1.v = _mm256_loadu_epi64(v_ptr + 3);
        x2.v = _mm256_loadu_epi64(v_ptr + 6);
        x3.v = _mm256_loadu_epi64(v_ptr + 9);

        y0 = _mm256_sort1x3_pd(x0);
        y1 = _mm256_sort1x3_pd(x1);
        y2 = _mm256_sort1x3_pd(x2);
        y3 = _mm256_sort1x3_pd(x3);

        _mm256_storeu_pd(k_ptr, y0.k);
        _mm256_storeu_pd(k_ptr + 3, y1.k);
        _mm256_storeu_pd(k_ptr + 6, y2.k);
        _mm256_storeu_pd(k_ptr + 9, y3.k);
        _mm256_storeu_epi64(v_ptr, y0.v);
        _mm256_storeu_epi64(v_ptr + 3, y1.v);
        _mm256_storeu_epi64(v_ptr + 6, y2.v);
        _mm256_storeu_epi64(v_ptr + 9, y3.v);

        k_ptr += AVX2_DOUBLE_STRIDE * 3;
        v_ptr += AVX2_EPI64_STRIDE * 3;
        r -= 4;
    }
    while (r >= 1) {
        const __m256i mask = _mm256_setmask_pd(3);

        x0.k = _mm256_maskload_pd(k_ptr, mask);
        x0.v = _mm256_maskload_epi64(v_ptr, mask);

        y0 = _mm256_sort1x3_pd(x0);

        _mm256_maskstore_pd(k_ptr, mask, y0.k);
        _mm256_maskstore_epi64(v_ptr, mask, y0.v);

        k_ptr += 3;
        v_ptr += 3;
        r -= 1;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s4_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 4) {
        return FAILURE_BADPARAM;
    }

    __m256dkv x0, x1, x2, x3, y0, y1, y2, y3;

    uint r = n;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)k_ptr % AVX2_ALIGNMENT) != 0) {
        while (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x4_pd(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_loadu_x4_epi64(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);
            y2 = _mm256_sort_pd(x2);
            y3 = _mm256_sort_pd(x3);

            _mm256_storeu_x4_pd(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_storeu_x4_epi64(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 4;
            v_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r >= AVX2_DOUBLE_STRIDE / 2) {
            _mm256_loadu_x2_pd(k_ptr, x0.k, x1.k);
            _mm256_loadu_x2_epi64(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);

            _mm256_storeu_x2_pd(k_ptr, y0.k, y1.k);
            _mm256_storeu_x2_epi64(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 2;
            v_ptr += AVX2_EPI64_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE / 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE / 4) {
            _mm256_loadu_x1_pd(k_ptr, x0.k);
            _mm256_loadu_x1_epi64(v_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_storeu_x1_pd(k_ptr, y0.k);
            _mm256_storeu_x1_epi64(v_ptr, y0.v);
        }
    }
    else {
        while (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x4_pd(k_ptr, x0.k, x1.k, x2.k, x3.k);
            _mm256_load_x4_epi64(v_ptr, x0.v, x1.v, x2.v, x3.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);
            y2 = _mm256_sort_pd(x2);
            y3 = _mm256_sort_pd(x3);

            _mm256_stream_x4_pd(k_ptr, y0.k, y1.k, y2.k, y3.k);
            _mm256_stream_x4_epi64(v_ptr, y0.v, y1.v, y2.v, y3.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 4;
            v_ptr += AVX2_EPI64_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE;
        }
        if (r >= AVX2_DOUBLE_STRIDE / 2) {
            _mm256_load_x2_pd(k_ptr, x0.k, x1.k);
            _mm256_load_x2_epi64(v_ptr, x0.v, x1.v);

            y0 = _mm256_sort_pd(x0);
            y1 = _mm256_sort_pd(x1);

            _mm256_stream_x2_pd(k_ptr, y0.k, y1.k);
            _mm256_stream_x2_epi64(v_ptr, y0.v, y1.v);

            k_ptr += AVX2_DOUBLE_STRIDE * 2;
            v_ptr += AVX2_EPI64_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE / 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE / 4) {
            _mm256_load_x1_pd(k_ptr, x0.k);
            _mm256_load_x1_epi64(v_ptr, x0.v);

            y0 = _mm256_sort_pd(x0);

            _mm256_stream_x1_pd(k_ptr, y0.k);
            _mm256_stream_x1_epi64(v_ptr, y0.v);
        }
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s5_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 5) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x5_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n5_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s6_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 6) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x6_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n6_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s7_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 7) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x7_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n7_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s8_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != AVX2_DOUBLE_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }

    __m256dkv s0, s1, s2, s3, s4, s5, s6, s7;

    if (((size_t)v_ptr % AVX2_ALIGNMENT) != 0 || ((size_t)k_ptr % AVX2_ALIGNMENT) != 0) {
        for (uint i = 0; i < (n & (~3u)); i += 4u) {
            _mm256_loadu_x4_pd(k_ptr, s0.k, s1.k, s2.k, s3.k);
            _mm256_loadu_x4_pd(k_ptr + AVX2_DOUBLE_STRIDE * 4, s4.k, s5.k, s6.k, s7.k);
            _mm256_loadu_x4_epi64(v_ptr, s0.v, s1.v, s2.v, s3.v);
            _mm256_loadu_x4_epi64(v_ptr + AVX2_EPI64_STRIDE * 4, s4.v, s5.v, s6.v, s7.v);

            __m256dkvx2 x0 = __m256dkvx2(s0.k, s1.k, s0.v, s1.v);
            __m256dkvx2 x1 = __m256dkvx2(s2.k, s3.k, s2.v, s3.v);
            __m256dkvx2 x2 = __m256dkvx2(s4.k, s5.k, s4.v, s5.v);
            __m256dkvx2 x3 = __m256dkvx2(s6.k, s7.k, s6.v, s7.v);

            __m256dkvx2 y0 = _mm256x2_sort_pd(x0);
            __m256dkvx2 y1 = _mm256x2_sort_pd(x1);
            __m256dkvx2 y2 = _mm256x2_sort_pd(x2);
            __m256dkvx2 y3 = _mm256x2_sort_pd(x3);

            _mm256_storeu_x4_pd(k_ptr, y0.k0, y0.k1, y1.k0, y1.k1);
            _mm256_storeu_x4_pd(k_ptr + AVX2_DOUBLE_STRIDE * 4, y2.k0, y2.k1, y3.k0, y3.k1);
            _mm256_storeu_x4_epi64(v_ptr, y0.v0, y0.v1, y1.v0, y1.v1);
            _mm256_storeu_x4_epi64(v_ptr + AVX2_EPI64_STRIDE * 4, y2.v0, y2.v1, y3.v0, y3.v1);

            k_ptr += s * 4;
            v_ptr += s * 4;
        }
        for (uint i = (n & (~3u)); i < n; i++) {
            _mm256_loadu_x2_pd(k_ptr, s0.k, s1.k);
            _mm256_loadu_x2_epi64(v_ptr, s0.v, s1.v);

            __m256dkvx2 x0 = __m256dkvx2(s0.k, s1.k, s0.v, s1.v);

            __m256dkvx2 y0 = _mm256x2_sort_pd(x0);

            _mm256_storeu_x2_pd(k_ptr, y0.k0, y0.k1);
            _mm256_storeu_x2_epi64(v_ptr, y0.v0, y0.v1);

            k_ptr += s;
            v_ptr += s;
        }
    }
    else {
        for (uint i = 0; i < (n & (~3u)); i += 4u) {
            _mm256_load_x4_pd(k_ptr, s0.k, s1.k, s2.k, s3.k);
            _mm256_load_x4_pd(k_ptr + AVX2_DOUBLE_STRIDE * 4, s4.k, s5.k, s6.k, s7.k);
            _mm256_load_x4_epi64(v_ptr, s0.v, s1.v, s2.v, s3.v);
            _mm256_load_x4_epi64(v_ptr + AVX2_EPI64_STRIDE * 4, s4.v, s5.v, s6.v, s7.v);

            __m256dkvx2 x0 = __m256dkvx2(s0.k, s1.k, s0.v, s1.v);
            __m256dkvx2 x1 = __m256dkvx2(s2.k, s3.k, s2.v, s3.v);
            __m256dkvx2 x2 = __m256dkvx2(s4.k, s5.k, s4.v, s5.v);
            __m256dkvx2 x3 = __m256dkvx2(s6.k, s7.k, s6.v, s7.v);

            __m256dkvx2 y0 = _mm256x2_sort_pd(x0);
            __m256dkvx2 y1 = _mm256x2_sort_pd(x1);
            __m256dkvx2 y2 = _mm256x2_sort_pd(x2);
            __m256dkvx2 y3 = _mm256x2_sort_pd(x3);

            _mm256_store_x4_pd(k_ptr, y0.k0, y0.k1, y1.k0, y1.k1);
            _mm256_store_x4_pd(k_ptr + AVX2_DOUBLE_STRIDE * 4, y2.k0, y2.k1, y3.k0, y3.k1);
            _mm256_store_x4_epi64(v_ptr, y0.v0, y0.v1, y1.v0, y1.v1);
            _mm256_store_x4_epi64(v_ptr + AVX2_EPI64_STRIDE * 4, y2.v0, y2.v1, y3.v0, y3.v1);

            k_ptr += s * 4;
            v_ptr += s * 4;
        }
        for (uint i = (n & (~3u)); i < n; i++) {
            _mm256_load_x2_pd(k_ptr, s0.k, s1.k);
            _mm256_load_x2_epi64(v_ptr, s0.v, s1.v);

            __m256dkvx2 x0 = __m256dkvx2(s0.k, s1.k, s0.v, s1.v);

            __m256dkvx2 y0 = _mm256x2_sort_pd(x0);

            _mm256_store_x2_pd(k_ptr, y0.k0, y0.k1);
            _mm256_store_x2_epi64(v_ptr, y0.v0, y0.v1);

            k_ptr += s;
            v_ptr += s;
        }
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s9_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 9) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x9_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n9_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s10_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 10) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x10_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n10_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s11_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 11) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x11_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n11_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s12_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 12) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x12_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n12_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s13_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 13) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x13_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n13_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s14_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 14) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x14_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n14_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s15_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 15) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x15_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n15_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s16_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s != 16) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        shortsort_n4x16_d(s, v_ptr, k_ptr);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        shortsort_n16_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s17to31_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s <= AVX2_DOUBLE_STRIDE * 4 || s >= AVX2_DOUBLE_STRIDE * 8) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < (n & (~3u)); i += 4u) {
        permcombsort_p4x4_d(s, v_ptr, k_ptr);
        scansort_p4_d(s, v_ptr, k_ptr);
        scansort_p4_d(s, v_ptr + s, k_ptr + s);
        scansort_p4_d(s, v_ptr + s * 2, k_ptr + s * 2);
        scansort_p4_d(s, v_ptr + s * 3, k_ptr + s * 3);
        k_ptr += s * 4;
        v_ptr += s * 4;
    }
    for (uint i = (n & (~3u)); i < n; i++) {
        permcombsort_p4_d(s, v_ptr, k_ptr);
        scansort_p4_d(s, v_ptr, k_ptr);

        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

int sortwithkeydsc_ignnan_s32plus_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s < AVX2_DOUBLE_STRIDE * 8 || n > MAX_SORT_STRIDE) {
        return FAILURE_BADPARAM;
    }

    for (uint i = 0; i < n; i++) {
        longsort_n32plus_d(s, v_ptr, k_ptr);
        k_ptr += s;
        v_ptr += s;
    }

    return SUCCESS;
}

#pragma endregion sort

#pragma region sort_allstride

int sortwithkeydsc_ignnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s <= 1) {
        return SUCCESS;
    }
    else if (s <= 2) {
        return sortwithkeydsc_ignnan_s2_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 3) {
        return sortwithkeydsc_ignnan_s3_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 4) {
        return sortwithkeydsc_ignnan_s4_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 5) {
        return sortwithkeydsc_ignnan_s5_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 6) {
        return sortwithkeydsc_ignnan_s6_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 7) {
        return sortwithkeydsc_ignnan_s7_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 8) {
        return sortwithkeydsc_ignnan_s8_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 9) {
        return sortwithkeydsc_ignnan_s9_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 10) {
        return sortwithkeydsc_ignnan_s10_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 11) {
        return sortwithkeydsc_ignnan_s11_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 12) {
        return sortwithkeydsc_ignnan_s12_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 13) {
        return sortwithkeydsc_ignnan_s13_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 14) {
        return sortwithkeydsc_ignnan_s14_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 15) {
        return sortwithkeydsc_ignnan_s15_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 16) {
        return sortwithkeydsc_ignnan_s16_d(n, s, v_ptr, k_ptr);
    }
    else if (s < 32) {
        return sortwithkeydsc_ignnan_s17to31_d(n, s, v_ptr, k_ptr);
    }
    else {
        return sortwithkeydsc_ignnan_s32plus_d(n, s, v_ptr, k_ptr);
    }
}

#pragma endregion sort_allstride