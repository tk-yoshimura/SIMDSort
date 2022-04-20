#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"

__forceinline __m256 _mm256_cmpgt_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);

    return gt;
}

__forceinline __m256 _mm256_cmpgt_minnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    __m256 xisnan = _mm256_isnan_ps(x);
    __m256 yisnan = _mm256_isnan_ps(y);
    __m256 bothnan = _mm256_and_ps(xisnan, yisnan);

    __m256 ret = _mm256_andnot_ps(bothnan, _mm256_or_ps(gt, yisnan));

    return ret;
}

__forceinline __m256 _mm256_cmpgt_maxnan_ps(__m256 x, __m256 y) {
    __m256 gt = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    __m256 xisnan = _mm256_isnan_ps(x);
    __m256 yisnan = _mm256_isnan_ps(y);
    __m256 bothnan = _mm256_and_ps(xisnan, yisnan);

    __m256 ret = _mm256_andnot_ps(bothnan, _mm256_or_ps(gt, xisnan));

    return ret;
}

__forceinline void _mm256_cmpgtswap_minnan_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_minnan_ps(a, b);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));
}

__forceinline bool _mm256_cmpgtswap_minnan_signal_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_minnan_ps(a, b);

    bool swaped = _mm256_movemask_ps(gtflag) > 0;

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return swaped;
}

__forceinline bool _mm256_cmpgtswap_minnan_masksignal_ps(__m256 a, __m256 b, __m256i mask, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_and_ps(_mm256_cmpgt_minnan_ps(a, b), _mm256_castsi256_ps(mask));

    bool swaped = _mm256_movemask_ps(gtflag) > 0;

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return swaped;
}

__forceinline void _mm256_cmpgtswap_maxnan_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_maxnan_ps(a, b);

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));
}

__forceinline bool _mm256_cmpgtswap_maxnan_signal_ps(__m256 a, __m256 b, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_cmpgt_maxnan_ps(a, b);

    bool swaped = _mm256_movemask_ps(gtflag) > 0;

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return swaped;
}

__forceinline bool _mm256_cmpgtswap_maxnan_masksignal_ps(__m256 a, __m256 b, __m256i mask, __m256& x, __m256& y) {
    __m256 gtflag = _mm256_and_ps(_mm256_cmpgt_maxnan_ps(a, b), _mm256_castsi256_ps(mask));

    bool swaped = _mm256_movemask_ps(gtflag) > 0;

    x = _mm256_blendv_ps(a, b, gtflag);
    y = _mm256_blendv_ps(a, b, _mm256_not_ps(gtflag));

    return swaped;
}