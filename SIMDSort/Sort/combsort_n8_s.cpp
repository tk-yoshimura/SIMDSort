#include <immintrin.h>
#include "../constants.h"
#include "../simdsort.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_cmp_s.hpp"

int combsortasc_n8_s(const uint n, const uint h, outfloats v_ptr) {
    if (h < AVX2_FLOAT_STRIDE) {
        return FAILURE_BADPARAM;
    }
    if (n < AVX2_FLOAT_STRIDE || n - AVX2_FLOAT_STRIDE < h) {
        return SUCCESS;
    }

    uint e = n - h - AVX2_FLOAT_STRIDE;

    __m256 x0, y0;
    __m256 a0, b0;

    for (uint i = 0; i < e; i += AVX2_FLOAT_STRIDE * 2) {
        _mm256_loadu_x1_ps(v_ptr + i,     a0);
        _mm256_loadu_x1_ps(v_ptr + i + h, b0);

        _mm256_cmpgtswap_ps(a0, b0, x0, y0);

        _mm256_storeu_x1_ps(v_ptr + i,     x0);
        _mm256_storeu_x1_ps(v_ptr + i + h, y0);
    }
    {
        _mm256_loadu_x1_ps(v_ptr + e,     a0);
        _mm256_loadu_x1_ps(v_ptr + e + h, b0);

        _mm256_cmpgtswap_ps(a0, b0, x0, y0);

        _mm256_storeu_x1_ps(v_ptr + e,     x0);
        _mm256_storeu_x1_ps(v_ptr + e + h, y0);
    }

    return SUCCESS;
}