#include <stdio.h>
#include <intrin.h>
#include "simdsort.h"
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

    __m256d x0 = _mm256_setr_pd(8, 7, 6, 5);
    __m256d x1 = _mm256_setr_pd(4, 3, 2, 1);
    __m256dx2 x(x0, x1);

    __m256dx2 y = _m256x2_sort_pd(x);


    printf("end");
    return getchar();
}