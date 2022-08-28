#pragma once
#pragma unmanaged

#include "../utils.h"
#include "inline_ope_s.hpp"

#pragma region greater than (x > y)

// compare x > y
__forceinline __m256i _mm256_cmpgt_epu32(__m256i x, __m256i y) {
    __m256i gt = _mm256_xor_si256(_mm256_cmpeq_epi32(_mm256_max_epu32(x, y), y), _mm256_set1_epi32(~0u));

    return gt;
}

#pragma endregion greater than (x > y)

#pragma region less than (x < y)

// compare x < y
__forceinline __m256i _mm256_cmplt_epi32(__m256i x, __m256i y) {
    __m256i lt = _mm256_cmpgt_epi32(y, x);

    return lt;
}

// compare x < y
__forceinline __m256i _mm256_cmplt_epu32(__m256i x, __m256i y) {
    __m256i lt = _mm256_cmpgt_epu32(y, x);

    return lt;
}

#pragma endregion less than (x < y)

#pragma region equals

// compare x == y
__forceinline __m256i _mm256_cmpeq_ep32(__m256i x, __m256i y) {
    __m256i ep = _mm256_cmpeq_epi32(x, y);

    return ep;
}

#pragma endregion equals