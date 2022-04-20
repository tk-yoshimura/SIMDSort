#include <immintrin.h>
#include <vector>
#include <algorithm>
#include "../simdsort.h"
#include "../Inline/inline_cmp_s.hpp"

int cmpgt_test_s() {
    __m256 a = _mm256_setr_ps(1,          2,         3, 4, 5, 6, 7, 8);
    __m256 b = _mm256_setr_ps(NAN, INFINITY, -INFINITY, 4, 3, 2, 1, 0);

    if (_mm256_movemask_ps(_mm256_cmpgt_ps(a, b)) != 0b11110100) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_ps(b, a)) != 0b00000010) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_ps(a, a)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_ps(b, b)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(a, b)) != 0b11110101) {     
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(b, a)) != 0b00000010) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(a, a)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(b, b)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(a, b)) != 0b11110100) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(b, a)) != 0b00000011) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(a, a)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(b, b)) != 0) {
        printf("err");
        return -1;
    }

    return 0;
}


int cmplt_test_s() {
    __m256 a = _mm256_setr_ps(1,          2,         3, 4, 5, 6, 7, 8);
    __m256 b = _mm256_setr_ps(NAN, INFINITY, -INFINITY, 4, 3, 2, 1, 0);

    if (_mm256_movemask_ps(_mm256_cmplt_ps(a, b)) != 0b00000010) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_ps(b, a)) != 0b11110100) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_ps(a, a)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_ps(b, b)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(a, b)) != 0b00000010) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(b, a)) != 0b11110101) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(a, a)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(b, b)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(a, b)) != 0b00000011) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(b, a)) != 0b11110100) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(a, a)) != 0) {
        printf("err");
        return -1;
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(b, b)) != 0) {
        printf("err");
        return -1;
    }

    return 0;
}