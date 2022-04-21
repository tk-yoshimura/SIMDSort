#include <immintrin.h>
#include <vector>
#include <algorithm>
#include "../simdsort.h"
#include "../Inline/inline_sort_s.hpp"

int mm_sort_test_s() {
    std::vector<float> v1{ 3, 4, 1, 2 };
    std::vector<float> v2{ 1, 3, NAN, 2 };

    do {
        __m128 x = _mm_loadu_ps(v1.data());
        __m128 y = _mm_sortasc_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] != 1 || t[1] != 2 || t[2] != 3 || t[3] != 4) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v1.begin(), v1.end()));

    do {
        __m128 x = _mm_loadu_ps(v1.data());
        __m128 y = _mm_sortdsc_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] != 4 || t[1] != 3 || t[2] != 2 || t[3] != 1) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v1.begin(), v1.end()));

    do {
        __m128 x = _mm_loadu_ps(v2.data());
        __m128 y = _mm_sortasc_minnan_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] == t[0] || t[1] != 1 || t[2] != 2 || t[3] != 3) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    do {
        __m128 x = _mm_loadu_ps(v2.data());
        __m128 y = _mm_sortdsc_minnan_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] != 3 || t[1] != 2 || t[2] != 1 || t[3] == t[3]) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    do {
        __m128 x = _mm_loadu_ps(v2.data());
        __m128 y = _mm_sortasc_maxnan_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] != 1 || t[1] != 2 || t[2] != 3 || t[3] == t[3]) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    do {
        __m128 x = _mm_loadu_ps(v2.data());
        __m128 y = _mm_sortdsc_maxnan_ps(x);

        float t[4];
        _mm_storeu_ps(t, y);

        printf("%f %f %f %f\n", t[0], t[1], t[2], t[3]);

        if (t[0] == t[0] || t[1] != 3 || t[2] != 2 || t[3] != 1) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    return 0;
}

int mm256_sort_test_s() {
    std::vector<float> v1{ 3, 4, 1, 2, 7, 8, 5, 6 };
    std::vector<float> v2{ 3, 4, 1, 2, 7, NAN, 5, 6 };

    do {
        __m256 x = _mm256_loadu_ps(v1.data());
        __m256 y = _mm256_sortasc_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] != 1 || t[1] != 2 || t[2] != 3 || t[3] != 4 || t[4] != 5 || t[5] != 6 || t[6] != 7 || t[7] != 8) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v1.begin(), v1.end()));

    do {
        __m256 x = _mm256_loadu_ps(v1.data());
        __m256 y = _mm256_sortdsc_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] != 8 || t[1] != 7 || t[2] != 6 || t[3] != 5 || t[4] != 4 || t[5] != 3 || t[6] != 2 || t[7] != 1) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v1.begin(), v1.end()));

    do {
        __m256 x = _mm256_loadu_ps(v2.data());
        __m256 y = _mm256_sortasc_minnan_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] == t[0] || t[1] != 1 || t[2] != 2 || t[3] != 3 || t[4] != 4 || t[5] != 5 || t[6] != 6 || t[7] != 7) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    do {
        __m256 x = _mm256_loadu_ps(v2.data());
        __m256 y = _mm256_sortdsc_minnan_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] != 7 || t[1] != 6 || t[2] != 5 || t[3] != 4 || t[4] != 3 || t[5] != 2 || t[6] != 1 || t[7] == t[7]) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    do {
        __m256 x = _mm256_loadu_ps(v2.data());
        __m256 y = _mm256_sortasc_maxnan_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] != 1 || t[1] != 2 || t[2] != 3 || t[3] != 4 || t[4] != 5 || t[5] != 6 || t[6] != 7 || t[7] == t[7]) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    do {
        __m256 x = _mm256_loadu_ps(v2.data());
        __m256 y = _mm256_sortdsc_maxnan_ps(x);

        float t[8];
        _mm256_storeu_ps(t, y);

        printf("%f %f %f %f %f %f %f %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);

        if (t[0] == t[0] || t[1] != 7 || t[2] != 6 || t[3] != 5 || t[4] != 4 || t[5] != 3 || t[6] != 2 || t[7] != 1) {
            throw std::exception("err");
            return -1;
        }

    } while (std::next_permutation(v2.begin(), v2.end()));

    return 0;
}
