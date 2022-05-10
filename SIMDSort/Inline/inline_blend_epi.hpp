#include <immintrin.h>

__forceinline __m256i _mm256_blendv_epi32(__m256i a, __m256i b, __m256 c) {
    return _mm256_castps_si256(_mm256_blendv_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), c));
}

__forceinline __m256i _mm256_blendv_epi64(__m256i a, __m256i b, __m256d c) {
    return _mm256_castpd_si256(_mm256_blendv_pd(_mm256_castsi256_pd(a), _mm256_castsi256_pd(b), c));
}