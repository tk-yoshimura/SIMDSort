#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

static_assert(sizeof(float) == sizeof(uint), "mismatch sizeof float and uint");

__forceinline __m256i _mm256_load_epi32(const uint* ptr) {
    return _mm256_castps_si256(_mm256_load_ps((float*)(void*)ptr));
}

__forceinline __m256i _mm256_loadu_epi32(const uint* ptr) {
    return _mm256_castps_si256(_mm256_loadu_ps((float*)(void*)ptr));
}

__forceinline __m256i _mm256_maskload_epi32(const uint* ptr, __m256i mask) {
    return _mm256_castps_si256(_mm256_maskload_ps((float*)(void*)ptr, mask));
}

__forceinline void _mm256_store_epi32(const uint* ptr, __m256i v) {
    _mm256_store_ps((float*)(void*)ptr, _mm256_castsi256_ps(v));
}

__forceinline void _mm256_storeu_epi32(const uint* ptr, __m256i v) {
    _mm256_storeu_ps((float*)(void*)ptr, _mm256_castsi256_ps(v));
}

__forceinline void _mm256_maskstore_epi32(const uint* ptr, __m256i mask, __m256i v) {
    _mm256_maskstore_ps((float*)(void*)ptr, mask, _mm256_castsi256_ps(v));
}

__forceinline void _mm256_stream_epi32(const uint* ptr, __m256i v) {
    _mm256_stream_ps((float*)(void*)ptr, _mm256_castsi256_ps(v));
}

__forceinline void _mm256_load_x1_epi32(inuints ptr, __m256i& x0) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
}

__forceinline void _mm256_load_x2_epi32(inuints ptr, __m256i& x0, __m256i& x1) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
}

__forceinline void _mm256_load_x3_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 2);
}

__forceinline void _mm256_load_x4_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 3);
}

__forceinline void _mm256_load_x5_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 4);
}

__forceinline void _mm256_load_x6_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 5);
}

__forceinline void _mm256_load_x7_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 5);
    x6 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 6);
}

__forceinline void _mm256_load_x8_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6, __m256i& x7) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_epi32(ptr);
    x1 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 5);
    x6 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 6);
    x7 = _mm256_load_epi32(ptr + AVX2_EPI32_STRIDE * 7);
}

__forceinline void _mm256_loadu_x1_epi32(inuints ptr, __m256i& x0) {
    x0 = _mm256_loadu_epi32(ptr);
}

__forceinline void _mm256_loadu_x2_epi32(inuints ptr, __m256i& x0, __m256i& x1) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
}

__forceinline void _mm256_loadu_x3_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
}

__forceinline void _mm256_loadu_x4_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
}

__forceinline void _mm256_loadu_x5_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
}

__forceinline void _mm256_loadu_x6_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 5);
}

__forceinline void _mm256_loadu_x7_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 5);
    x6 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 6);
}

__forceinline void _mm256_loadu_x8_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6, __m256i& x7) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 5);
    x6 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 6);
    x7 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 7);
}

__forceinline void _mm256_maskload_x1_epi32(inuints ptr, __m256i& x0, const __m256i mask) {
    x0 = _mm256_maskload_epi32(ptr, mask);
}

__forceinline void _mm256_maskload_x2_epi32(inuints ptr, __m256i& x0, __m256i& x1, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE, mask);
}

__forceinline void _mm256_maskload_x3_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE * 2, mask);
}

__forceinline void _mm256_maskload_x4_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE * 3, mask);
}

__forceinline void _mm256_maskload_x5_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE * 4, mask);
}

__forceinline void _mm256_maskload_x6_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE * 5, mask);
}

__forceinline void _mm256_maskload_x7_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 5);
    x6 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE * 6, mask);
}

__forceinline void _mm256_maskload_x8_epi32(inuints ptr, __m256i& x0, __m256i& x1, __m256i& x2, __m256i& x3, __m256i& x4, __m256i& x5, __m256i& x6, __m256i& x7, const __m256i mask) {
    x0 = _mm256_loadu_epi32(ptr);
    x1 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE);
    x2 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 2);
    x3 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 3);
    x4 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 4);
    x5 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 5);
    x6 = _mm256_loadu_epi32(ptr + AVX2_EPI32_STRIDE * 6);
    x7 = _mm256_maskload_epi32(ptr + AVX2_EPI32_STRIDE * 7, mask);
}

__forceinline void _mm256_store_x1_epi32(outuints ptr, __m256i x0) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
}

__forceinline void _mm256_store_x2_epi32(outuints ptr, __m256i x0, __m256i x1) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
}

__forceinline void _mm256_store_x3_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
}

__forceinline void _mm256_store_x4_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
}

__forceinline void _mm256_store_x5_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
}

__forceinline void _mm256_store_x6_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
}

__forceinline void _mm256_store_x7_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
}

__forceinline void _mm256_store_x8_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6, __m256i x7) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_epi32(ptr, x0);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
    _mm256_store_epi32(ptr + AVX2_EPI32_STRIDE * 7, x7);
}

__forceinline void _mm256_storeu_x1_epi32(outuints ptr, __m256i x0) {
    _mm256_storeu_epi32(ptr, x0);
}

__forceinline void _mm256_storeu_x2_epi32(outuints ptr, __m256i x0, __m256i x1) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
}

__forceinline void _mm256_storeu_x3_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
}

__forceinline void _mm256_storeu_x4_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
}

__forceinline void _mm256_storeu_x5_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
}

__forceinline void _mm256_storeu_x6_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
}

__forceinline void _mm256_storeu_x7_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
}

__forceinline void _mm256_storeu_x8_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6, __m256i x7) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 7, x7);
}

__forceinline void _mm256_maskstore_x1_epi32(outuints ptr, __m256i x0, const __m256i mask) {
    _mm256_maskstore_epi32(ptr, mask, x0);
}

__forceinline void _mm256_maskstore_x2_epi32(outuints ptr, __m256i x0, __m256i x1, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE, mask, x1);
}

__forceinline void _mm256_maskstore_x3_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE * 2, mask, x2);
}

__forceinline void _mm256_maskstore_x4_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE * 3, mask, x3);
}

__forceinline void _mm256_maskstore_x5_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE * 4, mask, x4);
}

__forceinline void _mm256_maskstore_x6_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE * 5, mask, x5);
}

__forceinline void _mm256_maskstore_x7_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE * 6, mask, x6);
}

__forceinline void _mm256_maskstore_x8_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6, __m256i x7, const __m256i mask) {
    _mm256_storeu_epi32(ptr, x0);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_storeu_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
    _mm256_maskstore_epi32(ptr + AVX2_EPI32_STRIDE * 7, mask, x7);
}

__forceinline void _mm256_stream_x1_epi32(outuints ptr, __m256i x0) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
}

__forceinline void _mm256_stream_x2_epi32(outuints ptr, __m256i x0, __m256i x1) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
}

__forceinline void _mm256_stream_x3_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
}

__forceinline void _mm256_stream_x4_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
}

__forceinline void _mm256_stream_x5_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
}

__forceinline void _mm256_stream_x6_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
}

__forceinline void _mm256_stream_x7_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
}

__forceinline void _mm256_stream_x8_epi32(outuints ptr, __m256i x0, __m256i x1, __m256i x2, __m256i x3, __m256i x4, __m256i x5, __m256i x6, __m256i x7) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_epi32(ptr, x0);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE, x1);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 2, x2);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 3, x3);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 4, x4);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 5, x5);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 6, x6);
    _mm256_stream_epi32(ptr + AVX2_EPI32_STRIDE * 7, x7);
}