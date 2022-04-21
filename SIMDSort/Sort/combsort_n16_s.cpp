#include <immintrin.h>
#include "../constants.h"
#include "../simdsort.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_cmp_s.hpp"

int combsortasc_n16_s(const uint n, const uint h, outfloats v_ptr) {
    if (h < AVX2_FLOAT_STRIDE * 2) {
        return FAILURE_BADPARAM;
    }
    if (n < AVX2_FLOAT_STRIDE * 2 || n - AVX2_FLOAT_STRIDE * 2 < h) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_FLOAT_STRIDE * 2;

    __m256 a0, a1, b0, b1;
    __m256 x0, x1, y0, y1;

    for (uint i = 0; i < e; i += AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x2_ps(v_ptr + i,     a0, a1);
        _mm256_loadu_x2_ps(v_ptr + i + h, b0, b1);

        _mm256_cmpgtswap_ps(a0, b0, x0, y0);
        _mm256_cmpgtswap_ps(a1, b1, x1, y1);

        _mm256_storeu_x2_ps(v_ptr + i,     x0, x1);
        _mm256_storeu_x2_ps(v_ptr + i + h, y0, y1);
    }
    {
        _mm256_loadu_x2_ps(v_ptr + e,     a0, a1);
        _mm256_loadu_x2_ps(v_ptr + e + h, b0, b1);

        _mm256_cmpgtswap_ps(a0, b0, x0, y0);
        _mm256_cmpgtswap_ps(a1, b1, x1, y1);

        _mm256_storeu_x2_ps(v_ptr + e,     x0, x1);
        _mm256_storeu_x2_ps(v_ptr + e + h, y0, y1);
    }

    return SUCCESS;
}