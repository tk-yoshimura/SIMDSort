#pragma once

#pragma unmanaged

#include <immintrin.h>
#include "types.h"

extern __m128i _mm_setmask_ps(const uint n);
extern __m256i _mm256_setmask_ps(const uint n);
extern __m128i _mm_setmask_pd(const uint n);
extern __m256i _mm256_setmask_pd(const uint n);

extern bool contains_nan_s(const uint n, infloats x_ptr);
extern bool contains_nan_d(const uint n, indoubles x_ptr);