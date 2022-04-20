#include <immintrin.h>
#include <vector>
#include <algorithm>
#include "../simdsort.h"
#include "../Inline/inline_sort_s.hpp"

int mm_sort_test_s() {
    std::vector<float> v4{ 1, 2, 3, 4 };

    do {
        __m128 x = _mm_loadu_ps(v4.data());
        __m128 y = _mm_sort_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] != 1 || t[1] != 2 || t[2] != 3 || t[3] != 4) {
            printf("err");
            return -1;
        }

    } while (std::next_permutation(v4.begin(), v4.end()));

    return 0;
}

int mm256_sort_test_s() {
    std::vector<float> v8{ 1, 2, 3, 4, 5, 6, 7, 8 };

    do {
        __m256 x = _mm256_loadu_ps(v8.data());
        __m256 y = _mm256_sort_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] != 1 || t[1] != 2 || t[2] != 3 || t[3] != 4 || t[4] != 5 || t[5] != 6 || t[6] != 7 || t[7] != 8) {
            printf("err");
            return -1;
        }

    } while (std::next_permutation(v8.begin(), v8.end()));

    return 0;
}
