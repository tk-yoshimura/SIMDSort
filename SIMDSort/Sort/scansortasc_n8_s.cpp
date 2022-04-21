#include <immintrin.h>
#include "../constants.h"
#include "../simdsort.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_cmp_s.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_sort_s.hpp"

void scansortasciter_n8_s(const uint i, const uint n, outfloats v_ptr) {
    if (i > n) {
        return;
    }

    v_ptr += i;

    uint r = n - i;

    __m256 x0, x1, x2, x3;
    __m256 y0, y1, y2, y3;

    while (r >= AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(v_ptr, x0, x1, x2, x3);

        y0 = _mm256_sortasc_ps(x0);
        y1 = _mm256_sortasc_ps(x1);
        y2 = _mm256_sortasc_ps(x2);
        y3 = _mm256_sortasc_ps(x3);

        _mm256_storeu_x4_ps(v_ptr, y0, y1, y2, y3);

        v_ptr += AVX2_FLOAT_STRIDE * 4;
        r -= AVX2_FLOAT_STRIDE * 4;
    }
    if (r >= AVX2_FLOAT_STRIDE * 3) {
        _mm256_loadu_x3_ps(v_ptr, x0, x1, x2);

        y0 = _mm256_sortasc_ps(x0);
        y1 = _mm256_sortasc_ps(x1);
        y2 = _mm256_sortasc_ps(x2);

        _mm256_storeu_x3_ps(v_ptr, y0, y1, y2);
    }
    else if (r >= AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x2_ps(v_ptr, x0, x1);

        y0 = _mm256_sortasc_ps(x0);
        y1 = _mm256_sortasc_ps(x1);

        _mm256_storeu_x2_ps(v_ptr, y0, y1);
    }
    else if (r >= AVX2_FLOAT_STRIDE) {
        _mm256_loadu_x1_ps(v_ptr, x0);

        y0 = _mm256_sortasc_ps(x0);

        _mm256_storeu_x1_ps(v_ptr, y0);
    }
}

int scansortasc_n8_s(const uint n, outfloats v_ptr) {
    if (n < AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }

    scansortasciter_n8_s(0, n, v_ptr);
    scansortasciter_n8_s(AVX2_FLOAT_STRIDE / 2, n, v_ptr);

    uint e = n - AVX2_FLOAT_STRIDE;

    uint i = 0;
    while (true) {
        __m256 x = _mm256_loadu_ps(v_ptr + i);
        
        if (_mm256_needssortasc_ps(x)) {
            __m256 y = _mm256_sortasc_ps(x);
            _mm256_storeu_ps(v_ptr + i, y);

            uint indexes = 0xFFu - _mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(x, y));

            uint index = bsf(indexes);

            uint back = AVX2_FLOAT_STRIDE - index - 1;

            i = (i > back) ? i - back : 0;
        }
        else if (i < e) {
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