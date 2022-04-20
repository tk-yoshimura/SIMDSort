#include "../constants.h"
#include "../utils.h"
#include "../Inline/inline_loadstore_xn_s.hpp"
#include "../Inline/inline_loadstore_xn_d.hpp"
#include "../Inline/inline_ope_s.hpp"
#include "../Inline/inline_ope_d.hpp"

#pragma unmanaged

bool contains_nan_s(const uint n, infloats x_ptr) {
    uint r = n;

    __m256 x0, x1, x2, x3;
    __m256 isnan = _mm256_setzero_ps();

    if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_load_x4_ps(x_ptr, x0, x1, x2, x3);

            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x1));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x2));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x3));

            if (_mm256_movemask_ps(isnan)) {
                return true;
            }
    
            x_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_load_x2_ps(x_ptr, x0, x1);

            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x1));
     
            x_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_load_x1_ps(x_ptr, x0);

            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));
    
            x_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
    }
    else {
        while (r >= AVX2_FLOAT_STRIDE * 4) {
            _mm256_loadu_x4_ps(x_ptr, x0, x1, x2, x3);

            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x1));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x2));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x3));

            if (_mm256_movemask_ps(isnan)) {
                return true;
            }

            x_ptr += AVX2_FLOAT_STRIDE * 4;
            r -= AVX2_FLOAT_STRIDE * 4;
        }
        if (r >= AVX2_FLOAT_STRIDE * 2) {
            _mm256_loadu_x2_ps(x_ptr, x0, x1);

            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));
            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x1));

            x_ptr += AVX2_FLOAT_STRIDE * 2;
            r -= AVX2_FLOAT_STRIDE * 2;
        }
        if (r >= AVX2_FLOAT_STRIDE) {
            _mm256_loadu_x1_ps(x_ptr, x0);

            isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));

            x_ptr += AVX2_FLOAT_STRIDE;
            r -= AVX2_FLOAT_STRIDE;
        }
    }

    if (r > 0) {
        const __m256i mask = _mm256_setmask_ps(r);

        x0 = _mm256_maskload_ps(x_ptr, mask);

        isnan = _mm256_or_ps(isnan, _mm256_isnan_ps(x0));
    }

    if (_mm256_movemask_ps(isnan)) {
        return true;
    }

    return false;
}

bool contains_nan_d(const uint n, indoubles x_ptr) {
    uint r = n;

    __m256d x0, x1, x2, x3;
    __m256d isnan = _mm256_setzero_pd();

    if (((size_t)x_ptr % AVX2_ALIGNMENT) == 0) {
        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_load_x4_pd(x_ptr, x0, x1, x2, x3);

            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x1));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x2));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x3));

            if (_mm256_movemask_pd(isnan)) {
                return true;
            }

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_load_x2_pd(x_ptr, x0, x1);

            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x1));

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_load_x1_pd(x_ptr, x0);

            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));

            x_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
    }
    else {
        while (r >= AVX2_DOUBLE_STRIDE * 4) {
            _mm256_loadu_x4_pd(x_ptr, x0, x1, x2, x3);

            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x1));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x2));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x3));

            if (_mm256_movemask_pd(isnan)) {
                return true;
            }

            x_ptr += AVX2_DOUBLE_STRIDE * 4;
            r -= AVX2_DOUBLE_STRIDE * 4;
        }
        if (r >= AVX2_DOUBLE_STRIDE * 2) {
            _mm256_loadu_x2_pd(x_ptr, x0, x1);

            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));
            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x1));

            x_ptr += AVX2_DOUBLE_STRIDE * 2;
            r -= AVX2_DOUBLE_STRIDE * 2;
        }
        if (r >= AVX2_DOUBLE_STRIDE) {
            _mm256_loadu_x1_pd(x_ptr, x0);

            isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));

            x_ptr += AVX2_DOUBLE_STRIDE;
            r -= AVX2_DOUBLE_STRIDE;
        }
    }

    if (r > 0) {
        const __m256i mask = _mm256_setmask_pd(r);

        x0 = _mm256_maskload_pd(x_ptr, mask);

        isnan = _mm256_or_pd(isnan, _mm256_isnan_pd(x0));
    }

    if (_mm256_movemask_pd(isnan)) {
        return true;
    }

    return false;
}