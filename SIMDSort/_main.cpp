#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_ep32.hpp"

int main() {
    printf("uint32\n");

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

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmpge_epu32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmpge_epu32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmpge_epu32(z1, z2)) != 0;

                if ((a >= d) != xr12) {
                    printf("%u >= %u\n", a, d);
                }

                if ((b >= e) != yr12) {
                    printf("%u >= %u\n", b, e);
                }

                if ((c >= f) != zr12) {
                    printf("%u >= %u\n", c, f);
                }
            }

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmple_epu32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmple_epu32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmple_epu32(z1, z2)) != 0;

                if ((a <= d) != xr12) {
                    printf("%u <= %u\n", a, d);
                }

                if ((b <= e) != yr12) {
                    printf("%u <= %u\n", b, e);
                }

                if ((c <= f) != zr12) {
                    printf("%u <= %u\n", c, f);
                }
            }

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmpgt_epu32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmpgt_epu32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmpgt_epu32(z1, z2)) != 0;

                if ((a > d) != xr12) {
                    printf("%u > %u\n", a, d);
                }

                if ((b > e) != yr12) {
                    printf("%u > %u\n", b, e);
                }

                if ((c > f) != zr12) {
                    printf("%u > %u\n", c, f);
                }
            }

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmplt_epu32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmplt_epu32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmplt_epu32(z1, z2)) != 0;

                if ((a < d) != xr12) {
                    printf("%u < %u\n", a, d);
                }

                if ((b < e) != yr12) {
                    printf("%u < %u\n", b, e);
                }

                if ((c < f) != zr12) {
                    printf("%u < %u\n", c, f);
                }
            }
        }
    }

    printf("int32\n");

    for (int j = 0; j < 32; j++) {
        for (int i = 0; i < 32; i++) {
            __int32 a = (1 << i) - 1;
            __int32 b = (1 << i);
            __int32 c = (1 << i) + 1;
            __int32 d = (1 << j) - 1;
            __int32 e = (1 << j);
            __int32 f = (1 << j) + 1;

            __m256i x1 = _mm256_set1_epi32(a);
            __m256i y1 = _mm256_set1_epi32(b);
            __m256i z1 = _mm256_set1_epi32(c);
            __m256i x2 = _mm256_set1_epi32(d);
            __m256i y2 = _mm256_set1_epi32(e);
            __m256i z2 = _mm256_set1_epi32(f);

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmpge_epi32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmpge_epi32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmpge_epi32(z1, z2)) != 0;

                if ((a >= d) != xr12) {
                    printf("%u >= %u\n", a, d);
                }

                if ((b >= e) != yr12) {
                    printf("%u >= %u\n", b, e);
                }

                if ((c >= f) != zr12) {
                    printf("%u >= %u\n", c, f);
                }
            }

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmple_epi32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmple_epi32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmple_epi32(z1, z2)) != 0;

                if ((a <= d) != xr12) {
                    printf("%u <= %u\n", a, d);
                }

                if ((b <= e) != yr12) {
                    printf("%u <= %u\n", b, e);
                }

                if ((c <= f) != zr12) {
                    printf("%u <= %u\n", c, f);
                }
            }

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmpgt_epi32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmpgt_epi32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmpgt_epi32(z1, z2)) != 0;

                if ((a > d) != xr12) {
                    printf("%u > %u\n", a, d);
                }

                if ((b > e) != yr12) {
                    printf("%u > %u\n", b, e);
                }

                if ((c > f) != zr12) {
                    printf("%u > %u\n", c, f);
                }
            }

            {
                bool xr12 = _mm256_movemask_epi8(_mm256_cmplt_epi32(x1, x2)) != 0;
                bool yr12 = _mm256_movemask_epi8(_mm256_cmplt_epi32(y1, y2)) != 0;
                bool zr12 = _mm256_movemask_epi8(_mm256_cmplt_epi32(z1, z2)) != 0;

                if ((a < d) != xr12) {
                    printf("%u < %u\n", a, d);
                }

                if ((b < e) != yr12) {
                    printf("%u < %u\n", b, e);
                }

                if ((c < f) != zr12) {
                    printf("%u < %u\n", c, f);
                }
            }
        }
    }

    printf("end");
    return getchar();
}