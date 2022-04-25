#pragma once
#pragma unmanaged

#include "../utils.h"
#include "../constants.h"

#ifdef _DEBUG
#include <exception>
#endif // _DEBUG

__forceinline void _mm256_load_x1_pd(indoubles ptr, __m256d& x0) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
}

__forceinline void _mm256_load_x2_pd(indoubles ptr, __m256d& x0, __m256d& x1) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
}

__forceinline void _mm256_load_x3_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
}

__forceinline void _mm256_load_x4_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
}

__forceinline void _mm256_load_x5_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
}

__forceinline void _mm256_load_x6_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
}

__forceinline void _mm256_load_x7_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
    x6 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 6);
}

__forceinline void _mm256_load_x8_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6, __m256d& x7) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    x0 = _mm256_load_pd(ptr);
    x1 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
    x6 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 6);
    x7 = _mm256_load_pd(ptr + AVX2_DOUBLE_STRIDE * 7);
}

__forceinline void _mm256_loadu_x1_pd(indoubles ptr, __m256d& x0) {
    x0 = _mm256_loadu_pd(ptr);
}

__forceinline void _mm256_loadu_x2_pd(indoubles ptr, __m256d& x0, __m256d& x1) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
}

__forceinline void _mm256_loadu_x3_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
}

__forceinline void _mm256_loadu_x4_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
}

__forceinline void _mm256_loadu_x5_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
}

__forceinline void _mm256_loadu_x6_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
}

__forceinline void _mm256_loadu_x7_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
    x6 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 6);
}

__forceinline void _mm256_loadu_x8_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6, __m256d& x7) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
    x6 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 6);
    x7 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 7);
}

__forceinline void _mm256_maskload_x1_pd(indoubles ptr, __m256d& x0, const __m256i mask) {
    x0 = _mm256_maskload_pd(ptr, mask);
}

__forceinline void _mm256_maskload_x2_pd(indoubles ptr, __m256d& x0, __m256d& x1, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE, mask);
}

__forceinline void _mm256_maskload_x3_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE * 2, mask);
}

__forceinline void _mm256_maskload_x4_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE * 3, mask);
}

__forceinline void _mm256_maskload_x5_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE * 4, mask);
}

__forceinline void _mm256_maskload_x6_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE * 5, mask);
}

__forceinline void _mm256_maskload_x7_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
    x6 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE * 6, mask);
}

__forceinline void _mm256_maskload_x8_pd(indoubles ptr, __m256d& x0, __m256d& x1, __m256d& x2, __m256d& x3, __m256d& x4, __m256d& x5, __m256d& x6, __m256d& x7, const __m256i mask) {
    x0 = _mm256_loadu_pd(ptr);
    x1 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE);
    x2 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 2);
    x3 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 3);
    x4 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 4);
    x5 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 5);
    x6 = _mm256_loadu_pd(ptr + AVX2_DOUBLE_STRIDE * 6);
    x7 = _mm256_maskload_pd(ptr + AVX2_DOUBLE_STRIDE * 7, mask);
}

__forceinline void _mm256_store_x1_pd(outdoubles ptr, __m256d x0) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
}

__forceinline void _mm256_store_x2_pd(outdoubles ptr, __m256d x0, __m256d x1) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
}

__forceinline void _mm256_store_x3_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
}

__forceinline void _mm256_store_x4_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
}

__forceinline void _mm256_store_x5_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
}

__forceinline void _mm256_store_x6_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
}

__forceinline void _mm256_store_x7_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
}

__forceinline void _mm256_store_x8_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6, __m256d x7) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_store_pd(ptr, x0);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
    _mm256_store_pd(ptr + AVX2_DOUBLE_STRIDE * 7, x7);
}

__forceinline void _mm256_storeu_x1_pd(outdoubles ptr, __m256d x0) {
    _mm256_storeu_pd(ptr, x0);
}

__forceinline void _mm256_storeu_x2_pd(outdoubles ptr, __m256d x0, __m256d x1) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
}

__forceinline void _mm256_storeu_x3_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
}

__forceinline void _mm256_storeu_x4_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
}

__forceinline void _mm256_storeu_x5_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
}

__forceinline void _mm256_storeu_x6_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
}

__forceinline void _mm256_storeu_x7_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
}

__forceinline void _mm256_storeu_x8_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6, __m256d x7) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 7, x7);
}

__forceinline void _mm256_maskstore_x1_pd(outdoubles ptr, __m256d x0, const __m256i mask) {
    _mm256_maskstore_pd(ptr, mask, x0);
}

__forceinline void _mm256_maskstore_x2_pd(outdoubles ptr, __m256d x0, __m256d x1, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE, mask, x1);
}

__forceinline void _mm256_maskstore_x3_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE * 2, mask, x2);
}

__forceinline void _mm256_maskstore_x4_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE * 3, mask, x3);
}

__forceinline void _mm256_maskstore_x5_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE * 4, mask, x4);
}

__forceinline void _mm256_maskstore_x6_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE * 5, mask, x5);
}

__forceinline void _mm256_maskstore_x7_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE * 6, mask, x6);
}

__forceinline void _mm256_maskstore_x8_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6, __m256d x7, const __m256i mask) {
    _mm256_storeu_pd(ptr, x0);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_storeu_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
    _mm256_maskstore_pd(ptr + AVX2_DOUBLE_STRIDE * 7, mask, x7);
}

__forceinline void _mm256_stream_x1_pd(outdoubles ptr, __m256d x0) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
}

__forceinline void _mm256_stream_x2_pd(outdoubles ptr, __m256d x0, __m256d x1) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
}

__forceinline void _mm256_stream_x3_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
}

__forceinline void _mm256_stream_x4_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
}

__forceinline void _mm256_stream_x5_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
}

__forceinline void _mm256_stream_x6_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
}

__forceinline void _mm256_stream_x7_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
}

__forceinline void _mm256_stream_x8_pd(outdoubles ptr, __m256d x0, __m256d x1, __m256d x2, __m256d x3, __m256d x4, __m256d x5, __m256d x6, __m256d x7) {
#ifdef _DEBUG
    if (((size_t)ptr % AVX2_ALIGNMENT) != 0) {
        throw std::exception();
    }
#endif // _DEBUG

    _mm256_stream_pd(ptr, x0);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE, x1);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 2, x2);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 3, x3);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 4, x4);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 5, x5);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 6, x6);
    _mm256_stream_pd(ptr + AVX2_DOUBLE_STRIDE * 7, x7);
}