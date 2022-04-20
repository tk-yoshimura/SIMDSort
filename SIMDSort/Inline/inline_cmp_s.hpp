#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"

#pragma region greater than (x > y)

// compare x > y (ignore nan)
__forceinline __m256 _mm256_cmpgt_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);

    return gt;
}

// compare !(x <= y) and not isnan(x) ... nan < minf < nval < pinf
__forceinline __m256 _mm256_cmpgt_minnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    __m256 xisnan = _mm256_isnan_ps(x);

    __m256 ret = _mm256_andnot_ps(xisnan, gt);

    return gt;
}

// compare !(x <= y) and not isnan(y) ... minf < nval < pinf < nan
__forceinline __m256 _mm256_cmpgt_maxnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    __m256 yisnan = _mm256_isnan_ps(y);

    __m256 ret = _mm256_andnot_ps(yisnan, gt);

    return gt;
}

// compare and swap x > y ? y : x (ignore nan)
__forceinline void _mm256_cmpgtswap_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_ps(a, b);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));
}

// compare and swap x > y ? y : x (ignore nan)
__forceinline int _mm256_cmpgtswap_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_ps(a, b);

    int index = _mm256_movemask_ps(gtflag);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return index;
}

// compare and swap x > y ? y : x ... nan < minf < nval < pinf
__forceinline void _mm256_cmpgtswap_minnan_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_minnan_ps(a, b);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));
}

// compare and swap x > y ? y : x ... nan < minf < nval < pinf
__forceinline int _mm256_cmpgtswap_minnan_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_minnan_ps(a, b);

    int index = _mm256_movemask_ps(gtflag);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return index;
}

// compare and swap x > y ? y : x ... minf < nval < pinf < nan
__forceinline void _mm256_cmpgtswap_maxnan_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_maxnan_ps(a, b);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));
}

// compare and swap x > y ? y : x ... minf < nval < pinf < nan
__forceinline int _mm256_cmpgtswap_maxnan_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_maxnan_ps(a, b);

    int index = _mm256_movemask_ps(gtflag);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return index;
}

#pragma endregion greater than (x > y)

#pragma region less than (x < y)

// compare x < y (ignore nan)
__forceinline __m256 _mm256_cmplt_ps(__m256 x, __m256 y) {
    __m256 lt = _mm256_cmp_ps(x, y, _CMP_LT_OQ);

    return lt;
}

// compare !(x >= y) and not isnan(x) ... pinf > nval > minf > nan
__forceinline __m256 _mm256_cmplt_minnan_ps(__m256 x, __m256 y) {
    __m256 lt = _mm256_cmp_ps(x, y, _CMP_LT_OQ);
    __m256 yisnan = _mm256_isnan_ps(y);

    __m256 ret = _mm256_andnot_ps(yisnan, lt);

    return lt;
}

// compare !(x >= y) and not isnan(x) ... nan > pinf > nval > minf
__forceinline __m256 _mm256_cmplt_maxnan_ps(__m256 x, __m256 y) {
    __m256 lt = _mm256_cmp_ps(x, y, _CMP_LT_OQ);
    __m256 xisnan = _mm256_isnan_ps(x);

    __m256 ret = _mm256_andnot_ps(xisnan, lt);

    return lt;
}

// compare and swap x < y ? y : x (ignore nan)
__forceinline void _mm256_cmpltswap_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 ltflag = _mm256_cmplt_ps(a, b);

    x = _mm256_blendv_ps(a, b, ltflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(ltflag));
}

// compare and swap x < y ? y : x (ignore nan)
__forceinline int _mm256_cmpltswap_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 ltflag = _mm256_cmplt_ps(a, b);

    int index = _mm256_movemask_ps(ltflag);

    x = _mm256_blendv_ps(a, b, ltflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(ltflag));

    return index;
}

// compare and swap x < y ? y : x ... pinf > nval > minf > nan
__forceinline void _mm256_cmpltswap_minnan_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 ltflag = _mm256_cmplt_minnan_ps(a, b);

    x = _mm256_blendv_ps(a, b, ltflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(ltflag));
}

// compare and swap x < y ? y : x ... pinf > nval > minf > nan
__forceinline int _mm256_cmpltswap_minnan_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 ltflag = _mm256_cmplt_minnan_ps(a, b);

    int index = _mm256_movemask_ps(ltflag);

    x = _mm256_blendv_ps(a, b, ltflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(ltflag));

    return index;
}

// compare and swap x < y ? y : x ... nan > pinf > nval > minf
__forceinline void _mm256_cmpltswap_maxnan_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 ltflag = _mm256_cmplt_maxnan_ps(a, b);

    x = _mm256_blendv_ps(a, b, ltflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(ltflag));
}

// compare and swap x < y ? y : x ... nan > pinf > nval > minf
__forceinline int _mm256_cmpltswap_maxnan_indexed_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 ltflag = _mm256_cmplt_maxnan_ps(a, b);

    int index = _mm256_movemask_ps(ltflag);

    x = _mm256_blendv_ps(a, b, ltflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(ltflag));

    return index;
}

#pragma endregion less than (x < y)

#pragma region equals

// compare x == y (ignore nan)
__forceinline int _mm256_cmpeq_indexed_ps(__m256 x, __m256 y) {
    __m256i eqflag = _mm256_cmpeq_epi32(_mm256_castps_si256(x), _mm256_castps_si256(y));

    int index = _mm256_movemask_ps(_mm256_castsi256_ps(eqflag));

    return index;
}

#pragma endregion equals