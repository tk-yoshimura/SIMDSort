#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"
#include "inline_cmp_s.hpp"

#pragma region ascending sort

// ascending sort evens (ignore nan)
__forceinline __m128 _mm_evensortasc_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmpgt_ignnan_ps(x, y), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// ascending sort odds (ignore nan)
__forceinline __m128 _mm_oddsortasc_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmpgt_ignnan_ps(x, y), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// ascending sort evens (ignore nan)
__forceinline __m256 _mm256_evensortasc_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmpgt_ignnan_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// ascending sort odds (ignore nan)
__forceinline __m256 _mm256_oddsortasc_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmpgt_ignnan_ps(x, y), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// ascending sort (ignore nan)
__forceinline __m128 _mm_sortasc_ps(__m128 x) {
    x = _mm_oddsortasc_ps(x);
    x = _mm_evensortasc_ps(x);
    x = _mm_oddsortasc_ps(x);

    return x;
}

// ascending sort (ignore nan)
__forceinline __m256 _mm256_sortasc_ps(__m256 x) {
    x = _mm256_oddsortasc_ps(x);
    x = _mm256_evensortasc_ps(x);
    x = _mm256_oddsortasc_ps(x);
    x = _mm256_evensortasc_ps(x);
    x = _mm256_oddsortasc_ps(x);
    x = _mm256_evensortasc_ps(x);
    x = _mm256_oddsortasc_ps(x);

    return x;
}

// needs ascending sort (ignore nan)
__forceinline bool _mm_needssortasc_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_cmpgt_ignnan_ps(x, y)) > 0;

    return needssort;
}

// needs ascending sort (ignore nan)
__forceinline bool _mm256_needssortasc_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_cmpgt_ignnan_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion ascending sort

#pragma region ascending sort minnan

// ascending sort evens (min nan)
__forceinline __m128 _mm_evensortasc_minnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmpgt_minnan_ps(x, y), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// ascending sort odds (min nan)
__forceinline __m128 _mm_oddsortasc_minnan_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmpgt_minnan_ps(x, y), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// ascending sort evens (min nan)
__forceinline __m256 _mm256_evensortasc_minnan_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmpgt_minnan_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// ascending sort odds (min nan)
__forceinline __m256 _mm256_oddsortasc_minnan_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmpgt_minnan_ps(x, y), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// ascending sort (min nan)
__forceinline __m128 _mm_sortasc_minnan_ps(__m128 x) {
    x = _mm_oddsortasc_minnan_ps(x);
    x = _mm_evensortasc_minnan_ps(x);
    x = _mm_oddsortasc_minnan_ps(x);

    return x;
}

// ascending sort (min nan)
__forceinline __m256 _mm256_sortasc_minnan_ps(__m256 x) {
    x = _mm256_oddsortasc_minnan_ps(x);
    x = _mm256_evensortasc_minnan_ps(x);
    x = _mm256_oddsortasc_minnan_ps(x);
    x = _mm256_evensortasc_minnan_ps(x);
    x = _mm256_oddsortasc_minnan_ps(x);
    x = _mm256_evensortasc_minnan_ps(x);
    x = _mm256_oddsortasc_minnan_ps(x);

    return x;
}

// needs ascending sort (min nan)
__forceinline bool _mm_needssortasc_minnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_cmpgt_minnan_ps(x, y)) > 0;

    return needssort;
}

// needs ascending sort (min nan)
__forceinline bool _mm256_needssortasc_minnan_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_cmpgt_minnan_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion ascending sort minnan

#pragma region ascending sort maxnan

// ascending sort evens (max nan)
__forceinline __m128 _mm_evensortasc_maxnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmpgt_maxnan_ps(x, y), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// ascending sort odds (max nan)
__forceinline __m128 _mm_oddsortasc_maxnan_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmpgt_maxnan_ps(x, y), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// ascending sort evens (max nan)
__forceinline __m256 _mm256_evensortasc_maxnan_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmpgt_maxnan_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// ascending sort odds (max nan)
__forceinline __m256 _mm256_oddsortasc_maxnan_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmpgt_maxnan_ps(x, y), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// ascending sort (max nan)
__forceinline __m128 _mm_sortasc_maxnan_ps(__m128 x) {
    x = _mm_oddsortasc_maxnan_ps(x);
    x = _mm_evensortasc_maxnan_ps(x);
    x = _mm_oddsortasc_maxnan_ps(x);

    return x;
}

// ascending sort (max nan)
__forceinline __m256 _mm256_sortasc_maxnan_ps(__m256 x) {
    x = _mm256_oddsortasc_maxnan_ps(x);
    x = _mm256_evensortasc_maxnan_ps(x);
    x = _mm256_oddsortasc_maxnan_ps(x);
    x = _mm256_evensortasc_maxnan_ps(x);
    x = _mm256_oddsortasc_maxnan_ps(x);
    x = _mm256_evensortasc_maxnan_ps(x);
    x = _mm256_oddsortasc_maxnan_ps(x);

    return x;
}

// needs ascending sort (max nan)
__forceinline bool _mm_needssortasc_maxnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_cmpgt_maxnan_ps(x, y)) > 0;

    return needssort;
}

// needs ascending sort (max nan)
__forceinline bool _mm256_needssortasc_maxnan_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion ascending sort maxnan


#pragma region descending sort

// descending sort evens (ignore nan)
__forceinline __m128 _mm_evensortdsc_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmplt_ignnan_ps(x, y), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// descending sort odds (ignore nan)
__forceinline __m128 _mm_oddsortdsc_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmplt_ignnan_ps(x, y), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// descending sort evens (ignore nan)
__forceinline __m256 _mm256_evensortdsc_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmplt_ignnan_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// descending sort odds (ignore nan)
__forceinline __m256 _mm256_oddsortdsc_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmplt_ignnan_ps(x, y), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// descending sort (ignore nan)
__forceinline __m128 _mm_sortdsc_ps(__m128 x) {
    x = _mm_oddsortdsc_ps(x);
    x = _mm_evensortdsc_ps(x);
    x = _mm_oddsortdsc_ps(x);

    return x;
}

// descending sort (ignore nan)
__forceinline __m256 _mm256_sortdsc_ps(__m256 x) {
    x = _mm256_oddsortdsc_ps(x);
    x = _mm256_evensortdsc_ps(x);
    x = _mm256_oddsortdsc_ps(x);
    x = _mm256_evensortdsc_ps(x);
    x = _mm256_oddsortdsc_ps(x);
    x = _mm256_evensortdsc_ps(x);
    x = _mm256_oddsortdsc_ps(x);

    return x;
}

// needs descending sort (ignore nan)
__forceinline bool _mm_needssortdsc_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_cmplt_ignnan_ps(x, y)) > 0;

    return needssort;
}

// needs descending sort (ignore nan)
__forceinline bool _mm256_needssortdsc_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_cmplt_ignnan_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion descending sort

#pragma region descending sort minnan

// descending sort evens (min nan)
__forceinline __m128 _mm_evensortdsc_minnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmplt_minnan_ps(x, y), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// descending sort odds (min nan)
__forceinline __m128 _mm_oddsortdsc_minnan_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmplt_minnan_ps(x, y), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// descending sort evens (min nan)
__forceinline __m256 _mm256_evensortdsc_minnan_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmplt_minnan_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// descending sort odds (min nan)
__forceinline __m256 _mm256_oddsortdsc_minnan_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmplt_minnan_ps(x, y), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// descending sort (min nan)
__forceinline __m128 _mm_sortdsc_minnan_ps(__m128 x) {
    x = _mm_oddsortdsc_minnan_ps(x);
    x = _mm_evensortdsc_minnan_ps(x);
    x = _mm_oddsortdsc_minnan_ps(x);

    return x;
}

// descending sort (min nan)
__forceinline __m256 _mm256_sortdsc_minnan_ps(__m256 x) {
    x = _mm256_oddsortdsc_minnan_ps(x);
    x = _mm256_evensortdsc_minnan_ps(x);
    x = _mm256_oddsortdsc_minnan_ps(x);
    x = _mm256_evensortdsc_minnan_ps(x);
    x = _mm256_oddsortdsc_minnan_ps(x);
    x = _mm256_evensortdsc_minnan_ps(x);
    x = _mm256_oddsortdsc_minnan_ps(x);

    return x;
}

// needs descending sort (min nan)
__forceinline bool _mm_needssortdsc_minnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_cmplt_minnan_ps(x, y)) > 0;

    return needssort;
}

// needs descending sort (min nan)
__forceinline bool _mm256_needssortdsc_minnan_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_cmplt_minnan_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion descending sort minnan

#pragma region descending sort maxnan

// descending sort evens (max nan)
__forceinline __m128 _mm_evensortdsc_maxnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_CDAB);
    __m128 c = _mm_permute_ps(_mm_cmplt_maxnan_ps(x, y), _MM_PERM_CCAA);
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// descending sort odds (max nan)
__forceinline __m128 _mm_oddsortdsc_maxnan_ps(__m128 x) {
    const __m128 xormask = _mm_castsi128_ps(_mm_setr_epi32(~0u, 0, 0, ~0u));

    __m128 y = _mm_permute_ps(x, _MM_PERM_ABCD);
    __m128 c = _mm_xor_ps(xormask, _mm_permute_ps(_mm_cmplt_maxnan_ps(x, y), _MM_PERM_DBBD));
    __m128 z = _mm_blendv_ps(x, y, c);

    return z;
}

// descending sort evens (max nan)
__forceinline __m256 _mm256_evensortdsc_maxnan_ps(__m256 x) {
    __m256 y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    __m256 c = _mm256_permute_ps(_mm256_cmplt_maxnan_ps(x, y), _MM_PERM_CCAA);
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// descending sort odds (max nan)
__forceinline __m256 _mm256_oddsortdsc_maxnan_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);
    __m256 c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_cmplt_maxnan_ps(x, y), permcmp));
    __m256 z = _mm256_blendv_ps(x, y, c);

    return z;
}

// descending sort (max nan)
__forceinline __m128 _mm_sortdsc_maxnan_ps(__m128 x) {
    x = _mm_oddsortdsc_maxnan_ps(x);
    x = _mm_evensortdsc_maxnan_ps(x);
    x = _mm_oddsortdsc_maxnan_ps(x);

    return x;
}

// descending sort (max nan)
__forceinline __m256 _mm256_sortdsc_maxnan_ps(__m256 x) {
    x = _mm256_oddsortdsc_maxnan_ps(x);
    x = _mm256_evensortdsc_maxnan_ps(x);
    x = _mm256_oddsortdsc_maxnan_ps(x);
    x = _mm256_evensortdsc_maxnan_ps(x);
    x = _mm256_oddsortdsc_maxnan_ps(x);
    x = _mm256_evensortdsc_maxnan_ps(x);
    x = _mm256_oddsortdsc_maxnan_ps(x);

    return x;
}

// needs descending sort (max nan)
__forceinline bool _mm_needssortdsc_maxnan_ps(__m128 x) {
    __m128 y = _mm_permute_ps(x, _MM_PERM_DDCB);

    bool needssort = _mm_movemask_ps(_mm_cmplt_maxnan_ps(x, y)) > 0;

    return needssort;
}

// needs descending sort (max nan)
__forceinline bool _mm256_needssortdsc_maxnan_ps(__m256 x) {
    const __m256i perm = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 7);

    __m256 y = _mm256_permutevar8x32_ps(x, perm);

    bool needssort = _mm256_movemask_ps(_mm256_cmplt_maxnan_ps(x, y)) > 0;

    return needssort;
}

#pragma endregion descending sort maxnan