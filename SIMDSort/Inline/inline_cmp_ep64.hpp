#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"

#pragma region greater than (x > y)

// compare x > y
__forceinline __m256i _mm256_cmpgt_epu64(__m256i x, __m256i y) {
    __m256i gt = _mm256_xor_si256(_mm256_cmpeq_epi64(_mm256_max_epu64(x, y), y), _mm256_set1_epi32(~0u));

    return gt;
}

#pragma endregion greater than (x > y)

#pragma region less than (x < y)

// compare x < y
__forceinline __m256i _mm256_cmplt_epi64(__m256i x, __m256i y) {
    __m256i lt = _mm256_cmpgt_epi64(y, x);

    return lt;
}

// compare x < y
__forceinline __m256i _mm256_cmplt_epu64(__m256i x, __m256i y) {
    __m256i lt = _mm256_cmpgt_epu64(y, x);

    return lt;
}

#pragma endregion less than (x < y)

#pragma region greater than or equal (x >= y)

// compare x >= y
__forceinline __m256i _mm256_cmpge_epi64(__m256i x, __m256i y) {
    __m256i ge = _mm256_cmpeq_epi64(_mm256_max_epi64(x, y), x);

    return ge;
}

// compare x >= y
__forceinline __m256i _mm256_cmpge_epu64(__m256i x, __m256i y) {
    __m256i ge = _mm256_cmpeq_epi64(_mm256_max_epu64(x, y), x);

    return ge;
}

#pragma endregion greater than or equal (x >= y)

#pragma region less than (x <= y)

// compare x <= y
__forceinline __m256i _mm256_cmple_epi64(__m256i x, __m256i y) {
    __m256i le = _mm256_cmpge_epi64(y, x);

    return le;
}

// compare x <= y
__forceinline __m256i _mm256_cmple_epu64(__m256i x, __m256i y) {
    __m256i le = _mm256_cmpge_epu64(y, x);

    return le;
}

#pragma endregion less than (x <= y)

#pragma region equals

// compare x == y
__forceinline __m256i _mm256_cmpeq_ep64(__m256i x, __m256i y) {
    __m256i ep = _mm256_cmpeq_epi64(x, y);

    return ep;
}

#pragma endregion equals