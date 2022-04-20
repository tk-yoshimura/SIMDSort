#include "../utils.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

#pragma unmanaged

__forceinline int fz(uint i, uint n) {
    return (i < n) ? ~0u : 0u;
}

__m128i _mm_setmask_ps(const uint n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_setr_epi32(fz(0, n), fz(1, n), fz(2, n), fz(3, n));
}

__m256i _mm256_setmask_ps(const uint n) {
#ifdef _DEBUG
    if (n >= 8) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_setr_epi32(fz(0, n), fz(1, n), fz(2, n), fz(3, n), fz(4, n), fz(5, n), fz(6, n), fz(7, n));
}

__m128i _mm_setmask_pd(const uint n) {
#ifdef _DEBUG
    if (n >= 2) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm_setr_epi32(fz(0, n), fz(0, n), fz(1, n), fz(1, n));
}

__m256i _mm256_setmask_pd(const uint n) {
#ifdef _DEBUG
    if (n >= 4) {
        throw std::exception();
    }
#endif // _DEBUG

    return _mm256_setr_epi32(fz(0, n), fz(0, n), fz(1, n), fz(1, n), fz(2, n), fz(2, n), fz(3, n), fz(3, n));
}