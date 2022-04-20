#pragma once

#include <immintrin.h>

typedef unsigned int uint;

typedef const float* __restrict infloats;
typedef const double* __restrict indoubles;
typedef float* __restrict outfloats;
typedef double* __restrict outdoubles;

static_assert(sizeof(unsigned char) == 1, "sizeof byte must be 1");

static_assert(sizeof(float) == 4, "sizeof float must be 4");
static_assert(sizeof(double) == 8, "sizeof float must be 8");
static_assert(sizeof(uint) == 4, "sizeof uint must be 4");

union _m32 {
    float f;
    unsigned __int32 i;

    constexpr _m32(unsigned __int32 i) : i(i) { }
};

union _m64 {
    double f;
    unsigned __int64 i;

    constexpr _m64(unsigned __int64 i) : i(i) { }
};

struct __m256x2 {
    __m256 imm0, imm1;

    constexpr __m256x2(__m256 imm0, __m256 imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct __m256dx2 {
    __m256d imm0, imm1;

    constexpr __m256dx2(__m256d imm0, __m256d imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct __m256x3 {
    __m256 imm0, imm1, imm2;

    constexpr __m256x3(__m256 imm0, __m256 imm1, __m256 imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct __m256dx3 {
    __m256d imm0, imm1, imm2;

    constexpr __m256dx3(__m256d imm0, __m256d imm1, __m256d imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct __m256x4 {
    __m256 imm0, imm1, imm2, imm3;

    constexpr __m256x4(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct __m256dx4 {
    __m256d imm0, imm1, imm2, imm3;

    constexpr __m256dx4(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct __m256x5 {
    __m256 imm0, imm1, imm2, imm3, imm4;

    constexpr __m256x5(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3, __m256 imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct __m256dx5 {
    __m256d imm0, imm1, imm2, imm3, imm4;

    constexpr __m256dx5(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3, __m256d imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct __m256x6 {
    __m256 imm0, imm1, imm2, imm3, imm4, imm5;

    constexpr __m256x6(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3, __m256 imm4, __m256 imm5)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5) { }
};

struct __m256dx6 {
    __m256d imm0, imm1, imm2, imm3, imm4, imm5;

    constexpr __m256dx6(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3, __m256d imm4, __m256d imm5)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5) { }
};

struct __m256x7 {
    __m256 imm0, imm1, imm2, imm3, imm4, imm5, imm6;

    constexpr __m256x7(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3, __m256 imm4, __m256 imm5, __m256 imm6)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6) { }
};

struct __m256dx7 {
    __m256d imm0, imm1, imm2, imm3, imm4, imm5, imm6;

    constexpr __m256dx7(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3, __m256d imm4, __m256d imm5, __m256d imm6)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6) { }
};

struct __m256x8 {
    __m256 imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7;

    constexpr __m256x8(__m256 imm0, __m256 imm1, __m256 imm2, __m256 imm3, __m256 imm4, __m256 imm5, __m256 imm6, __m256 imm7)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6), imm7(imm7) { }
};

struct __m256dx8 {
    __m256d imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7;

    constexpr __m256dx8(__m256d imm0, __m256d imm1, __m256d imm2, __m256d imm3, __m256d imm4, __m256d imm5, __m256d imm6, __m256d imm7)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6), imm7(imm7) { }
};

struct __m128x2 {
    __m128 imm0, imm1;

    constexpr __m128x2(__m128 imm0, __m128 imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct __m128dx2 {
    __m128d imm0, imm1;

    constexpr __m128dx2(__m128d imm0, __m128d imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct __m128x3 {
    __m128 imm0, imm1, imm2;

    constexpr __m128x3(__m128 imm0, __m128 imm1, __m128 imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct __m128dx3 {
    __m128d imm0, imm1, imm2;

    constexpr __m128dx3(__m128d imm0, __m128d imm1, __m128d imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct __m128x4 {
    __m128 imm0, imm1, imm2, imm3;

    constexpr __m128x4(__m128 imm0, __m128 imm1, __m128 imm2, __m128 imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct __m128dx4 {
    __m128d imm0, imm1, imm2, imm3;

    constexpr __m128dx4(__m128d imm0, __m128d imm1, __m128d imm2, __m128d imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct __m128x5 {
    __m128 imm0, imm1, imm2, imm3, imm4;

    constexpr __m128x5(__m128 imm0, __m128 imm1, __m128 imm2, __m128 imm3, __m128 imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct __m128dx5 {
    __m128d imm0, imm1, imm2, imm3, imm4;

    constexpr __m128dx5(__m128d imm0, __m128d imm1, __m128d imm2, __m128d imm3, __m128d imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct __m128x6 {
    __m128 imm0, imm1, imm2, imm3, imm4, imm5;

    constexpr __m128x6(__m128 imm0, __m128 imm1, __m128 imm2, __m128 imm3, __m128 imm4, __m128 imm5)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5) { }
};

struct __m128dx6 {
    __m128d imm0, imm1, imm2, imm3, imm4, imm5;

    constexpr __m128dx6(__m128d imm0, __m128d imm1, __m128d imm2, __m128d imm3, __m128d imm4, __m128d imm5)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5) { }
};

struct __m128x7 {
    __m128 imm0, imm1, imm2, imm3, imm4, imm5, imm6;

    constexpr __m128x7(__m128 imm0, __m128 imm1, __m128 imm2, __m128 imm3, __m128 imm4, __m128 imm5, __m128 imm6)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6) { }
};

struct __m128dx7 {
    __m128d imm0, imm1, imm2, imm3, imm4, imm5, imm6;

    constexpr __m128dx7(__m128d imm0, __m128d imm1, __m128d imm2, __m128d imm3, __m128d imm4, __m128d imm5, __m128d imm6)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6) { }
};

struct __m128x8 {
    __m128 imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7;

    constexpr __m128x8(__m128 imm0, __m128 imm1, __m128 imm2, __m128 imm3, __m128 imm4, __m128 imm5, __m128 imm6, __m128 imm7)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6), imm7(imm7) { }
};

struct __m128dx8 {
    __m128d imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7;

    constexpr __m128dx8(__m128d imm0, __m128d imm1, __m128d imm2, __m128d imm3, __m128d imm4, __m128d imm5, __m128d imm6, __m128d imm7)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6), imm7(imm7) { }
};

struct floatx2 {
    float imm0, imm1;

    constexpr floatx2(float imm0, float imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct doublex2 {
    double imm0, imm1;

    constexpr doublex2(double imm0, double imm1)
        : imm0(imm0), imm1(imm1) { }
};

struct floatx3 {
    float imm0, imm1, imm2;

    constexpr floatx3(float imm0, float imm1, float imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct doublex3 {
    double imm0, imm1, imm2;

    constexpr doublex3(double imm0, double imm1, double imm2)
        : imm0(imm0), imm1(imm1), imm2(imm2) { }
};

struct floatx4 {
    float imm0, imm1, imm2, imm3;

    constexpr floatx4(float imm0, float imm1, float imm2, float imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct doublex4 {
    double imm0, imm1, imm2, imm3;

    constexpr doublex4(double imm0, double imm1, double imm2, double imm3)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3) { }
};

struct floatx5 {
    float imm0, imm1, imm2, imm3, imm4;

    constexpr floatx5(float imm0, float imm1, float imm2, float imm3, float imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct doublex5 {
    double imm0, imm1, imm2, imm3, imm4;

    constexpr doublex5(double imm0, double imm1, double imm2, double imm3, double imm4)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4) { }
};

struct floatx6 {
    float imm0, imm1, imm2, imm3, imm4, imm5;

    constexpr floatx6(float imm0, float imm1, float imm2, float imm3, float imm4, float imm5)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5) { }
};

struct doublex6 {
    double imm0, imm1, imm2, imm3, imm4, imm5;

    constexpr doublex6(double imm0, double imm1, double imm2, double imm3, double imm4, double imm5)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5) { }
};

struct floatx7 {
    float imm0, imm1, imm2, imm3, imm4, imm5, imm6;

    constexpr floatx7(float imm0, float imm1, float imm2, float imm3, float imm4, float imm5, float imm6)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6) { }
};

struct doublex7 {
    double imm0, imm1, imm2, imm3, imm4, imm5, imm6;

    constexpr doublex7(double imm0, double imm1, double imm2, double imm3, double imm4, double imm5, double imm6)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6) { }
};

struct floatx8 {
    float imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7;

    constexpr floatx8(float imm0, float imm1, float imm2, float imm3, float imm4, float imm5, float imm6, float imm7)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6), imm7(imm7) { }
};

struct doublex8 {
    double imm0, imm1, imm2, imm3, imm4, imm5, imm6, imm7;

    constexpr doublex8(double imm0, double imm1, double imm2, double imm3, double imm4, double imm5, double imm6, double imm7)
        : imm0(imm0), imm1(imm1), imm2(imm2), imm3(imm3), imm4(imm4), imm5(imm5), imm6(imm6), imm7(imm7) { }
};