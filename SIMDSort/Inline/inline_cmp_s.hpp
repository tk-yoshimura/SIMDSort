#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"

#pragma region greater than (x > y)

// compare x > y (ignore nan)
__forceinline __m256 _mm256_cmpgt_ignnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);

    return gt;
}

// compare !(x <= y) and not isnan(x) ... nan < minf < nval < pinf
__forceinline __m256 _mm256_cmpgt_minnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_NLE_UQ);
    __m256 xisnan = _mm256_isnan_ps(x);

    __m256 ret = _mm256_andnot_ps(xisnan, gt);

    return ret;
}

// compare !(x <= y) and not isnan(y) ... minf < nval < pinf < nan
__forceinline __m256 _mm256_cmpgt_maxnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_NLE_UQ);
    __m256 yisnan = _mm256_isnan_ps(y);

    __m256 ret = _mm256_andnot_ps(yisnan, gt);

    return ret;
}

#pragma endregion greater than (x > y)

#pragma region less than (x < y)

// compare x < y (ignore nan)
__forceinline __m256 _mm256_cmplt_ignnan_ps(__m256 x, __m256 y) {
    __m256 lt = _mm256_cmp_ps(x, y, _CMP_LT_OQ);

    return lt;
}

// compare !(x >= y) and not isnan(x) ... pinf > nval > minf > nan
__forceinline __m256 _mm256_cmplt_minnan_ps(__m256 x, __m256 y) {
    __m256 lt = _mm256_cmp_ps(x, y, _CMP_NGE_UQ);
    __m256 yisnan = _mm256_isnan_ps(y);

    __m256 ret = _mm256_andnot_ps(yisnan, lt);

    return ret;
}

// compare !(x >= y) and not isnan(x) ... nan > pinf > nval > minf
__forceinline __m256 _mm256_cmplt_maxnan_ps(__m256 x, __m256 y) {
    __m256 lt = _mm256_cmp_ps(x, y, _CMP_NGE_UQ);
    __m256 xisnan = _mm256_isnan_ps(x);

    __m256 ret = _mm256_andnot_ps(xisnan, lt);

    return ret;
}

#pragma endregion less than (x < y)

#pragma region equals

// compare x == y (ignore nan)
__forceinline __m256 _mm256_cmpeq_ignnan_ps(__m256 x, __m256 y) {
    __m256 eqflag = _mm256_castsi256_ps(
        _mm256_cmpeq_epi32(
            _mm256_castps_si256(x),
            _mm256_castps_si256(y)
        )
    );

    return eqflag;
}

#pragma endregion equals