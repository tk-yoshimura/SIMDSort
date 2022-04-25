#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_d.hpp"

// needs swap (sort order definition)
__forceinline static __m256d _mm256_needsswap_pd(__m256d x, __m256d y) {
    return _mm256_cmpgt_ignnan_pd(x, y);
}

__forceinline __m256dx2 _m256x2_sort_pd(__m256dx2 x) {
    const __m256d xormask = _mm256_castsi256_pd(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, ~0u, ~0u));

    __m256d y0, y1, z0, z1, c0, c1, d0, d1;

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    z0 = _mm256_blend_pd(y0, y1, 0b1001);
    z1 = _mm256_blend_pd(y0, y1, 0b0110);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, z0), _MM_PERM_DBBA);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, z1), _MM_PERM_DBBA));
    d0 = _mm256_blend_pd(c0, _mm256_permute4x64_pd(c1, _MM_PERM_ACBD), 0b0001);
    d1 = _mm256_blend_pd(c1, _mm256_permute4x64_pd(c0, _MM_PERM_ACBD), 0b0001);
    x.imm0 = _mm256_blendv_pd(x.imm0, z0, d0);
    x.imm1 = _mm256_blendv_pd(x.imm1, z1, d1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_CDAB);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_CDAB);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_CCAA);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_CCAA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    z0 = _mm256_blend_pd(y0, y1, 0b1001);
    z1 = _mm256_blend_pd(y0, y1, 0b0110);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, z0), _MM_PERM_DBBA);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, z1), _MM_PERM_DBBA));
    d0 = _mm256_blend_pd(c0, _mm256_permute4x64_pd(c1, _MM_PERM_ACBD), 0b0001);
    d1 = _mm256_blend_pd(c1, _mm256_permute4x64_pd(c0, _MM_PERM_ACBD), 0b0001);
    x.imm0 = _mm256_blendv_pd(x.imm0, z0, d0);
    x.imm1 = _mm256_blendv_pd(x.imm1, z1, d1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_CDAB);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_CDAB);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_CCAA);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_CCAA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    z0 = _mm256_blend_pd(y0, y1, 0b1001);
    z1 = _mm256_blend_pd(y0, y1, 0b0110);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, z0), _MM_PERM_DBBA);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, z1), _MM_PERM_DBBA));
    d0 = _mm256_blend_pd(c0, _mm256_permute4x64_pd(c1, _MM_PERM_ACBD), 0b0001);
    d1 = _mm256_blend_pd(c1, _mm256_permute4x64_pd(c0, _MM_PERM_ACBD), 0b0001);
    x.imm0 = _mm256_blendv_pd(x.imm0, z0, d0);
    x.imm1 = _mm256_blendv_pd(x.imm1, z1, d1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_CDAB);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_CDAB);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_CCAA);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_CCAA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    z0 = _mm256_blend_pd(y0, y1, 0b1001);
    z1 = _mm256_blend_pd(y0, y1, 0b0110);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, z0), _MM_PERM_DBBA);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, z1), _MM_PERM_DBBA));
    d0 = _mm256_blend_pd(c0, _mm256_permute4x64_pd(c1, _MM_PERM_ACBD), 0b0001);
    d1 = _mm256_blend_pd(c1, _mm256_permute4x64_pd(c0, _MM_PERM_ACBD), 0b0001);
    x.imm0 = _mm256_blendv_pd(x.imm0, z0, d0);
    x.imm1 = _mm256_blendv_pd(x.imm1, z1, d1);

    return x;
}

__forceinline __m256dx2 _m256x2_sort_pd_r2(__m256dx2 x) {
    const __m256d xormask = _mm256_castsi256_pd(_mm256_setr_epi32(~0u, ~0u, 0, 0, 0, 0, ~0u, ~0u));

    __m256d y0, y1, c0, c1, swaps;

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, _mm256_permute4x64_pd(y1, _MM_PERM_ADCB));

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, _mm256_permute4x64_pd(y1, _MM_PERM_ADCB));

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, _mm256_permute4x64_pd(y1, _MM_PERM_ADCB));

    swaps = _mm256_needsswap_pd(x.imm0, x.imm1);
    y0 = _mm256_blendv_pd(x.imm0, x.imm1, swaps);
    y1 = _mm256_blendv_pd(x.imm0, x.imm1, _mm256_not_pd(swaps));
    x = __m256dx2(y0, y1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    c0 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_DBBD));
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_DBBD));
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_CDAB);
    c0 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_CCAA);
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_CDAB);
    c1 = _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_CCAA);
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    y0 = _mm256_permute4x64_pd(x.imm0, _MM_PERM_ABCD);
    c0 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm0, y0), _MM_PERM_DBBD));
    x.imm0 = _mm256_blendv_pd(x.imm0, y0, c0);
    y1 = _mm256_permute4x64_pd(x.imm1, _MM_PERM_ABCD);
    c1 = _mm256_xor_pd(xormask, _mm256_permute4x64_pd(_mm256_needsswap_pd(x.imm1, y1), _MM_PERM_DBBD));
    x.imm1 = _mm256_blendv_pd(x.imm1, y1, c1);

    return x;
}

int main() {
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

    const uint s = 8;

    std::vector<double> v(s);
    for (uint i = 0; i < s; i++) {
        v[i] = (double)((i + 1) % s + 1);
    }

    uint c = 0;

    double* t = (double*)_aligned_malloc((s + 4) * sizeof(double), AVX2_ALIGNMENT);
    if (t == nullptr) {
        return FAILURE_BADALLOC;
    }

    do {
        memcpy_s(t, s * sizeof(double), v.data(), s * sizeof(double));

        for (uint i = s; i < s + 4; i++) {
            t[i] = ((i + c) * 31) % s;
        }

        __m256dx2 x = __m256dx2(_mm256_loadu_pd(t), _mm256_loadu_pd(t + AVX2_DOUBLE_STRIDE));
        __m256dx2 y = _m256x2_sort_pd_r2(x);
        _mm256_storeu_pd(t, y.imm0);
        _mm256_storeu_pd(t + AVX2_DOUBLE_STRIDE, y.imm1);

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

        if ((c % 100) == 0 && c > 0) {
            printf(".");
        }

    } while (std::next_permutation(v.begin(), v.end()));

    _aligned_free(t);

    printf("end");
    return getchar();
}