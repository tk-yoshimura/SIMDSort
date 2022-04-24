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

// sort batches1 x elems4
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

#pragma endregion horizontal sort

#pragma region cmp and swap

// compare and swap
__forceinline static void _mm256_cmpswap_pd(__m256d a, __m256d b, __m256d& x, __m256d& y) {
    __m256d swapd = _mm256_needsswap_pd(a, b);

    x = _mm256_blendv_pd(a, b, swapd);
    y = _mm256_blendv_pd(a, b, _mm256_not_pd(swapd));
}

// compare and swap
__forceinline static uint _mm256_cmpswap_indexed_pd(__m256d a, __m256d b, __m256d& x, __m256d& y) {
    __m256d swapd = _mm256_needsswap_pd(a, b);

    uint index = _mm256_movemask_pd(swapd);

    x = _mm256_blendv_pd(a, b, swapd);
    y = _mm256_blendv_pd(a, b, _mm256_not_pd(swapd));

    return index;
}

#pragma endregion cmp and swap

#pragma region combsort

// combsort h=4
static int combsort_h4_d(const uint n, double* v_ptr) {
    if (n < AVX2_DOUBLE_STRIDE * 2) {
        return SUCCESS;
    }

    uint e = n - AVX2_DOUBLE_STRIDE * 2;

    __m256d a0, b0;
    __m256d x0, y0;

    if (e > 0) {
        _mm256_loadu_x1_pd(v_ptr, a0);

        uint i = 0;
        for (; i < e; i += AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(v_ptr + i + AVX2_DOUBLE_STRIDE, b0);

            _mm256_cmpswap_pd(a0, b0, x0, y0);

            _mm256_storeu_x1_pd(v_ptr + i, x0);

            a0 = y0;
        }
        _mm256_storeu_x1_pd(v_ptr + i, a0);
    }
    {
        _mm256_loadu_x1_pd(v_ptr + e, a0);
        _mm256_loadu_x1_pd(v_ptr + e + AVX2_DOUBLE_STRIDE, b0);

        _mm256_cmpswap_pd(a0, b0, x0, y0);

        _mm256_storeu_x1_pd(v_ptr + e, x0);
        _mm256_storeu_x1_pd(v_ptr + e + AVX2_DOUBLE_STRIDE, y0);
    }

    return SUCCESS;
}

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

// combsort h>17
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

    double* ve_ptr = v_ptr + e;

    for (int iter = 0; iter < 2; iter++) {
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
            if (r > 0) {
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
            if (r > 0) {
                _mm256_loadu_x1_pd(ve_ptr, x0);

                y0 = _mm256_sort_pd(x0);

                _mm256_storeu_x1_pd(ve_ptr, y0);
            }
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

    uint i = 0;
    while (true) {
        __m256d x = _mm256_loadu_pd(v_ptr + i);

        if (_mm256_needssort_pd(x)) {
            __m256d y = _mm256_sort_pd(x);
            _mm256_storeu_pd(v_ptr + i, y);

            if (i > 0) {
                uint indexes = _mm256_movemask_pd(_mm256_cmpeq_ignnan_pd(x, y));
                if ((indexes & 1) == 0) {
                    const uint back = AVX2_DOUBLE_STRIDE - 2;

                    i = (i > back) ? i - back : 0;
                    continue;
                }
            }
        }

        if (i < e) {
            i += AVX2_DOUBLE_STRIDE - 1;
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

    backtracksort_p4_d(n, v_ptr);
    scansort_p4_d(n, v_ptr);

    return SUCCESS;
}

#pragma endregion shortsort

#pragma region longsort

// longsort
__forceinline static int longsort_d(const uint n, double* v_ptr) {
    uint h;

    for (h = (uint)(n * 10L / 13L); h > 16; h = (uint)(h * 10L / 13L)) {
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
    if (h >= 4) {
        combsort_h8_d(n, v_ptr);
    }

    backtracksort_p4_d(n, v_ptr);
    batchsort_p4_d(n, v_ptr);
    scansort_p4_d(n, v_ptr);

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