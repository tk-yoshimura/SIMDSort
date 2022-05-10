#pragma once

#include <immintrin.h>

struct __m256kv {
    __m256 k;
    __m256i v;

    __m256kv()
        : k(__m256()), v(__m256i()) { }

    constexpr __m256kv(__m256 k, __m256i v)
        : k(k), v(v) { }
};

struct __m256dkv {
    __m256d k;
    __m256i v;

    __m256dkv()
        : k(__m256d()), v(__m256i()) { }

    constexpr __m256dkv(__m256d k, __m256i v)
        : k(k), v(v) { }
};

struct __m256dkvx2 {
    __m256d k0, k1;
    __m256i v0, v1;

    __m256dkvx2()
        : k0(__m256d()), k1(__m256d()), v0(__m256i()), v1(__m256i()) { }

    constexpr __m256dkvx2(__m256d k0, __m256d k1, __m256i v0, __m256i v1)
        : k0(k0), k1(k1), v0(v0), v1(v1) { }
};