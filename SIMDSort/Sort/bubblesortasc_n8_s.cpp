#include <immintrin.h>
#include "../constants.h"
#include "../simdsort.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_cmp_s.hpp"
#include "../Inline/inline_misc.hpp"
#include "../Inline/inline_sort_s.hpp"

int bubblesortasc_n8_s(const uint n, outfloats v_ptr) {
    if (n < AVX2_FLOAT_STRIDE * 2) {
        return SUCCESS;
    }

    uint i = 0, e = n - 2 * AVX2_FLOAT_STRIDE;
    __m256 a = _mm256_loadu_ps(v_ptr), b, x, y;

    if (e <= 0) {
        b = _mm256_loadu_ps(v_ptr + AVX2_FLOAT_STRIDE);
        _mm256_cmpgtswap_ps(a, b, x, y);

        _mm256_storeu_ps(v_ptr, x);
        _mm256_storeu_ps(v_ptr + AVX2_FLOAT_STRIDE, y);
        
        return SUCCESS;
    }

    while (true) {
        b = _mm256_loadu_ps(v_ptr + i + AVX2_FLOAT_STRIDE);

        uint indexes = _mm256_cmpgtswap_indexed_ps(a, b, x, y);

        if (indexes > 0u) {
            uint index = bsf(indexes);

            _mm256_storeu_ps(v_ptr + i, x);
            _mm256_storeu_ps(v_ptr + i + AVX2_FLOAT_STRIDE, y);

            if (i == 0) {
                i = AVX2_FLOAT_STRIDE;
                if (i <= e) {
                    a = y;
                    continue;
                }
                else {
                    i = e;
                }
            }
            else {
                uint back = AVX2_FLOAT_STRIDE - index;

                i = i >= back ? i - back : 0;
            }
        }
        else if (i < e) {
            i += AVX2_FLOAT_STRIDE;

            if (i <= e) {
                a = b;
                continue;
            }

            i = e;
        }
        else {
            break;
        }

        a = _mm256_loadu_ps(v_ptr + i);
    }

    return SUCCESS;
}