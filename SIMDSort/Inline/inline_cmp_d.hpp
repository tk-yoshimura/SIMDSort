#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_d.hpp"

#pragma region greater than (x > y)

// compare x > y (ignore nan)
__forceinline __m256d _mm256_cmpgt_ignnan_pd(__m256d x, __m256d y) {
    __m256d gt = _mm256_cmp_pd(x, y, _CMP_GT_OQ);

    return gt;
}

// compare !(x <= y) and not isnan(x) ... nan < minf < nval < pinf
__forceinline __m256d _mm256_cmpgt_minnan_pd(__m256d x, __m256d y) {
    __m256d gt = _mm256_cmp_pd(x, y, _CMP_NLE_UQ);
    __m256d xisnan = _mm256_isnan_pd(x);

    __m256d ret = _mm256_andnot_pd(xisnan, gt);

    return ret;
}

// compare !(x <= y) and not isnan(y) ... minf < nval < pinf < nan
__forceinline __m256d _mm256_cmpgt_maxnan_pd(__m256d x, __m256d y) {
    __m256d gt = _mm256_cmp_pd(x, y, _CMP_NLE_UQ);
    __m256d yisnan = _mm256_isnan_pd(y);

    __m256d ret = _mm256_andnot_pd(yisnan, gt);

    return ret;
}

#pragma endregion greater than (x > y)

#pragma region less than (x < y)

// compare x < y (ignore nan)
__forceinline __m256d _mm256_cmplt_ignnan_pd(__m256d x, __m256d y) {
    __m256d lt = _mm256_cmp_pd(x, y, _CMP_LT_OQ);

    return lt;
}

// compare !(x >= y) and not isnan(x) ... pinf > nval > minf > nan
__forceinline __m256d _mm256_cmplt_minnan_pd(__m256d x, __m256d y) {
    __m256d lt = _mm256_cmp_pd(x, y, _CMP_NGE_UQ);
    __m256d yisnan = _mm256_isnan_pd(y);

    __m256d ret = _mm256_andnot_pd(yisnan, lt);

    return ret;
}

// compare !(x >= y) and not isnan(x) ... nan > pinf > nval > minf
__forceinline __m256d _mm256_cmplt_maxnan_pd(__m256d x, __m256d y) {
    __m256d lt = _mm256_cmp_pd(x, y, _CMP_NGE_UQ);
    __m256d xisnan = _mm256_isnan_pd(x);

    __m256d ret = _mm256_andnot_pd(xisnan, lt);

    return ret;
}

#pragma endregion less than (x < y)

#pragma region equals

// compare x == y (ignore nan)
__forceinline __m256d _mm256_cmpeq_ignnan_pd(__m256d x, __m256d y) {
    __m256d eqflag = _mm256_castsi256_pd(
        _mm256_cmpeq_epi64(
            _mm256_castpd_si256(x),
            _mm256_castpd_si256(y)
        )
    );

    return eqflag;
}

#pragma endregion equals