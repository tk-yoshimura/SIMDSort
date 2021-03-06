#include <immintrin.h>
#include <vector>
#include <algorithm>
#include "../simdsort.h"
#include "../Inline/inline_cmp_s.hpp"

int mm256_cmpgt_test_s() {
    __m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
    __m256 b = _mm256_setr_ps(NAN, INFINITY, -INFINITY, 4, 3, 2, 1, 0);

    if (_mm256_movemask_ps(_mm256_cmpgt_ignnan_ps(a, b)) != 0b11110100) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_ignnan_ps(b, a)) != 0b00000010) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_ignnan_ps(a, a)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_ignnan_ps(b, b)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(a, b)) != 0b11110101) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(b, a)) != 0b00000010) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(a, a)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_minnan_ps(b, b)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(a, b)) != 0b11110100) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(b, a)) != 0b00000011) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(a, a)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpgt_maxnan_ps(b, b)) != 0) {
        throw std::exception("err");
    }

    return 0;
}


int mm256_cmplt_test_s() {
    __m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
    __m256 b = _mm256_setr_ps(NAN, INFINITY, -INFINITY, 4, 3, 2, 1, 0);

    if (_mm256_movemask_ps(_mm256_cmplt_ignnan_ps(a, b)) != 0b00000010) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_ignnan_ps(b, a)) != 0b11110100) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_ignnan_ps(a, a)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_ignnan_ps(b, b)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(a, b)) != 0b00000010) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(b, a)) != 0b11110101) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(a, a)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_minnan_ps(b, b)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(a, b)) != 0b00000011) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(b, a)) != 0b11110100) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(a, a)) != 0) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmplt_maxnan_ps(b, b)) != 0) {
        throw std::exception("err");
    }

    return 0;
}

int mm256_cmpeq_test_s() {
    __m256 a = _mm256_setr_ps(1, 2, 3, 4, 5, 6, 7, 8);
    __m256 b = _mm256_setr_ps(NAN, INFINITY, -INFINITY, 4, 3, 2, 1, 0);

    if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(a, b)) != 0b00001000) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(b, a)) != 0b00001000) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(a, a)) != 0b11111111) {
        throw std::exception("err");
    }

    if (_mm256_movemask_ps(_mm256_cmpeq_ignnan_ps(b, b)) != 0b11111111) {
        throw std::exception("err");
    }

    return 0;
}