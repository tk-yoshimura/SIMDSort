#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include "../simdsort.h"
#include "../SortWithKey/sortwithkey.h"

static int sortwithkeyasc_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr) {
    if (s <= 1) {
        return SUCCESS;
    }
    else if (s <= 2) {
        return sortwithkeyasc_ignnan_s2_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 3) {
        return sortwithkeyasc_ignnan_s3_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 4) {
        return sortwithkeyasc_ignnan_s4_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 5) {
        return sortwithkeyasc_ignnan_s5_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 6) {
        return sortwithkeyasc_ignnan_s6_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 7) {
        return sortwithkeyasc_ignnan_s7_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 8) {
        return sortwithkeyasc_ignnan_s8_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 9) {
        return sortwithkeyasc_ignnan_s9_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 10) {
        return sortwithkeyasc_ignnan_s10_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 11) {
        return sortwithkeyasc_ignnan_s11_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 12) {
        return sortwithkeyasc_ignnan_s12_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 13) {
        return sortwithkeyasc_ignnan_s13_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 14) {
        return sortwithkeyasc_ignnan_s14_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 15) {
        return sortwithkeyasc_ignnan_s15_s(n, s, v_ptr, k_ptr);
    }
    else if (s <= 16) {
        return sortwithkeyasc_ignnan_s16_s(n, s, v_ptr, k_ptr);
    }
    else if (s < 32) {
        return sortwithkeyasc_ignnan_s17to31_s(n, s, v_ptr, k_ptr);
    }
    else if (s < 64) {
        return sortwithkeyasc_ignnan_s32to63_s(n, s, v_ptr, k_ptr);
    }
    else {
        return sortwithkeyasc_ignnan_s64plus_s(n, s, v_ptr, k_ptr);
    }
}

int sortwithkeyasc_test_s() {
    std::mt19937 mt(1234);

    for (uint s = 2; s <= 128; s++) {
        for (uint n = 1; n <= 64; n++) {
            float* k = (float*)_aligned_malloc(s * n * sizeof(float), AVX2_ALIGNMENT);
            uint* v = (uint*)_aligned_malloc(s * n * sizeof(uint), AVX2_ALIGNMENT);

            if (k == nullptr || v == nullptr) {
                return FAILURE_BADALLOC;
            }

            for (uint test = 0; test < 64; test++) {
                for (uint i = 0; i < s * n; i++) {
                    uint r = mt();
                    k[i] = r / (float)(~0u);
                    v[i] = i;
                }

                std::vector<float> tk(s * n), tc;

                memcpy_s(tk.data(), s * n * sizeof(float), k, s * n * sizeof(float));

                tc = tk;

                for (uint j = 0; j < n; j++) {
                    std::sort(tk.begin() + j * s, tk.begin() + (j + 1) * s);
                }

                sortwithkeyasc_s(n, s, v, k);

                for (uint i = 0; i < s * n; i++) {
                    if (tk[i] != k[i] || tc[v[i]] != k[i]) {
                        printf("random ng n=%d s=%d\n", n, s);
                        throw std::exception("err");
                    }
                }


                printf("random ok n=%d s=%d\n", n, s);
            }

            _aligned_free(k);
            _aligned_free(v);
        }

        for (uint n = 1; n <= 16; n++) {
            float* k = (float*)malloc(s * n * sizeof(float));
            uint* v = (uint*)malloc(s * n * sizeof(uint));

            if (k == nullptr || v == nullptr) {
                return FAILURE_BADALLOC;
            }

            for (uint test = 0; test < 64; test++) {
                for (uint i = 0; i < s * n; i++) {
                    uint r = mt();
                    k[i] = r / (float)(~0u);
                    v[i] = i;
                }

                std::vector<float> tk(s * n), tc;

                memcpy_s(tk.data(), s * n * sizeof(float), k, s * n * sizeof(float));

                tc = tk;

                for (uint j = 0; j < n; j++) {
                    std::sort(tk.begin() + j * s, tk.begin() + (j + 1) * s);
                }

                sortwithkeyasc_s(n, s, v, k);

                for (uint i = 0; i < s * n; i++) {
                    if (tk[i] != k[i] || tc[v[i]] != k[i]) {
                        printf("random ng n=%d s=%d\n", n, s);
                        throw std::exception("err");
                    }
                }


                printf("random ok n=%d s=%d\n", n, s);
            }

            free(k);
            free(v);
        }
    }

    return 0;
}