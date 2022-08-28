#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_ep32.hpp"

int main() {
    __m256i a = _mm256_setr_epi32(-2147483648, -2147483647, -256, -1, 0, 1, 255, 2147483647);
    __m256i b = _mm256_setr_epi32(-2147483648, -2147483647, -256, -1, 0, 1, 255, 2147483647);
    __m256i c = _mm256_setr_epi32(0, 1, 255, 2147483647, -2147483648, -2147483647, -256, -1);
    __m256i d = _mm256_setr_epi32(-256, -1, 0, 1, 255, -2147483647, 2147483647, -2147483648);
    __m256i e = _mm256_setr_epi32(1, -2147483647, 2147483647, -2147483648, 255, -256, -1, 0);

    __m256i ab = _mm256_cmpgt_epu32(a, b);
    __m256i ac = _mm256_cmpgt_epu32(a, c);
    __m256i ad = _mm256_cmpgt_epu32(a, d);
    __m256i ae = _mm256_cmpgt_epu32(a, e);

    __m256i ba = _mm256_cmpgt_epu32(b, a);
    __m256i ca = _mm256_cmpgt_epu32(c, a);
    __m256i da = _mm256_cmpgt_epu32(d, a);
    __m256i ea = _mm256_cmpgt_epu32(e, a);

    for (int j = 0; j < 32; j++) {
        for (int i = 0; i < 32; i++) {
            unsigned __int32 a = (1 << i) - 1;
            unsigned __int32 b = (1 << i);
            unsigned __int32 c = (1 << i) + 1;
            unsigned __int32 d = (1 << j) - 1;
            unsigned __int32 e = (1 << j);
            unsigned __int32 f = (1 << j) + 1;

            __m256i x1 = _mm256_set1_epi32(a);
            __m256i y1 = _mm256_set1_epi32(b);
            __m256i z1 = _mm256_set1_epi32(c);
            __m256i x2 = _mm256_set1_epi32(d);
            __m256i y2 = _mm256_set1_epi32(e);
            __m256i z2 = _mm256_set1_epi32(f);

            __m256i x12 = _mm256_cmplt_epu32(x1, x2);
            __m256i y12 = _mm256_cmplt_epu32(y1, y2);
            __m256i z12 = _mm256_cmplt_epu32(z1, z2);

            bool xr12 = _mm256_movemask_epi8(x12) != 0;
            bool yr12 = _mm256_movemask_epi8(y12) != 0;
            bool zr12 = _mm256_movemask_epi8(z12) != 0;
 
            if ((a < d) != xr12) {
                printf("%u > %u\n", a, d);
            }

            if ((b < e) != yr12) {
                printf("%u > %u\n", b, e);
            }

            if ((c < f) != zr12) {
                printf("%u > %u\n", c, f);
            }
        }
    }

    printf("end");
    return getchar();
}