#include <immintrin.h>
#include "../constants.h"
#include "../simdsort.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_cmp_s.hpp"

int combsortasc_n32_s(const uint n, const uint h, outfloats v_ptr) {
    if (h < AVX2_FLOAT_STRIDE * 4) {
        return FAILURE_BADPARAM;
    }
    if (n < AVX2_FLOAT_STRIDE * 4 || n - AVX2_FLOAT_STRIDE * 4 < h) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_FLOAT_STRIDE * 4;

    __m256 a0, a1, a2, a3, b0, b1, b2, b3;
    __m256 x0, x1, x2, x3, y0, y1, y2, y3;

    for (uint i = 0; i < e; i += AVX2_FLOAT_STRIDE * 4) {
        _mm256_loadu_x4_ps(v_ptr + i,     a0, a1, a2, a3);
        _mm256_loadu_x4_ps(v_ptr + i + h, b0, b1, b2, b3);

        _mm256_cmpgtswap_ps(a0, b0, x0, y0);
        _mm256_cmpgtswap_ps(a1, b1, x1, y1);
        _mm256_cmpgtswap_ps(a2, b2, x2, y2);
        _mm256_cmpgtswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(v_ptr + i,     x0, x1, x2, x3);
        _mm256_storeu_x4_ps(v_ptr + i + h, y0, y1, y2, y3);
    }
    {
        _mm256_loadu_x4_ps(v_ptr + e,     a0, a1, a2, a3);
        _mm256_loadu_x4_ps(v_ptr + e + h, b0, b1, b2, b3);

        _mm256_cmpgtswap_ps(a0, b0, x0, y0);
        _mm256_cmpgtswap_ps(a1, b1, x1, y1);
        _mm256_cmpgtswap_ps(a2, b2, x2, y2);
        _mm256_cmpgtswap_ps(a3, b3, x3, y3);

        _mm256_storeu_x4_ps(v_ptr + e,     x0, x1, x2, x3);
        _mm256_storeu_x4_ps(v_ptr + e + h, y0, y1, y2, y3);
    }

    return SUCCESS;
}