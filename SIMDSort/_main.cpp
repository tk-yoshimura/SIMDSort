#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_s.hpp"

// needs swap (sort order definition)
__forceinline static __m256 _mm256_needsswap_ps(__m256 x, __m256 y) {
    return _mm256_cmpgt_ignnan_ps(x, y);
}

// sort elems8
__forceinline static __m256 _mm256_sort_ps(__m256 x) {
    const __m256 xormask = _mm256_castsi256_ps(_mm256_setr_epi32(~0u, 0, 0, 0, 0, 0, 0, ~0u));
    const __m256i perm = _mm256_setr_epi32(7, 2, 1, 4, 3, 6, 5, 0);
    const __m256i permcmp = _mm256_setr_epi32(7, 1, 1, 3, 3, 5, 5, 7);

    __m256 y, c;

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permute_ps(x, _MM_PERM_CDAB);
    c = _mm256_permute_ps(_mm256_needsswap_ps(x, y), _MM_PERM_CCAA);
    x = _mm256_blendv_ps(x, y, c);

    y = _mm256_permutevar8x32_ps(x, perm);
    c = _mm256_xor_ps(xormask, _mm256_permutevar8x32_ps(_mm256_needsswap_ps(x, y), permcmp));
    x = _mm256_blendv_ps(x, y, c);

    return x;
}

int main() {

    const uint s = 8;

    std::vector<float> v(s);
    for (uint i = 0; i < s; i++) {
        v[i] = (float)((i + 1) % s + 1);
    }

    uint c = 0;

    float* t = (float*)_aligned_malloc((s + 4) * sizeof(float), AVX2_ALIGNMENT);
    if (t == nullptr) {
        return FAILURE_BADALLOC;
    }

    do {
        memcpy_s(t, s * sizeof(float), v.data(), s * sizeof(float));

        for (uint i = s; i < s + 4; i++) {
            t[i] = ((i + c) * 31) % s;
        }

        __m256 x = _mm256_loadu_ps(t);
        __m256 y = _mm256_sort_ps(x);
        _mm256_storeu_ps(t, y);

        for (uint i = 1; i < s; i++) {
            if (t[i - 1u] >= t[i]) {
                throw std::exception("err");
            }
        }
        for (uint i = s; i < s + 4; i++) {
            if (t[i] != ((i + c) * 31) % s) {
                throw std::exception("err");
            }
        }

        c++;

        if ((c % 1000) == 0 && c > 0) {
            printf(".");
        }

    } while (std::next_permutation(v.begin(), v.end()));

    _aligned_free(t);

    //sortasc_test_s();
    //sortdsc_test_s();
    //sort_ndist_speed_test_s();
    //sort_random_speed_test_s();
    //sort_reverse_speed_test_s();
    //sort_inbalance_speed_test_s();

    //for (uint n = 32; n <= 0x8000000u; n *= 2) {
    //    printf("%d : ", n);
    //
    //    for (uint h = (uint)(n * 10L / 13L); h > 1; h = (uint)(h * 10L / 13L)) {
    //        printf("%d -> ", h);
    //    }
    //
    //    printf("\n");
    //}

    //sortasc_test_d();
    //sort_ndist_speed_test_d();
    //sort_random_speed_test_d();
    //sort_reverse_speed_test_d();
    //sort_inbalance_speed_test_d();

    printf("end");
    return getchar();
}