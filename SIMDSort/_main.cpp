#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_s.hpp"

__forceinline __m256 _mm256_reverse_ps(__m256 x) {
    __m256 z = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(
        _mm256_permute_ps(x, _MM_PERM_ABCD)), _MM_PERM_BADC)
    );

    return z;
}

__forceinline __m256 _mm256_reflect_ps(__m256 x) {
    __m256 z = _mm256_permute2f128_ps(x, _mm256_permute_ps(x, _MM_PERM_ABCD), 0b00100000);

    return z;
}

int main() {
    sortwithkeyasc_test_s();
    sortwithkeyasc_test_d();

    printf("end");
    return getchar();
}