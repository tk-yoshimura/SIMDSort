#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include "../simdsort.h"
#include "../SortWithKey/sortwithkey.h"

static int sortwithkeyasc_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr) {
    if (s <= 1) {
        return SUCCESS;
    }
    else if (s <= 2) {
        return sortwithkeyasc_ignnan_s2_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 3) {
        return sortwithkeyasc_ignnan_s3_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 4) {
        return sortwithkeyasc_ignnan_s4_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 5) {
        return sortwithkeyasc_ignnan_s5_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 6) {
        return sortwithkeyasc_ignnan_s6_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 7) {
        return sortwithkeyasc_ignnan_s7_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 8) {
        return sortwithkeyasc_ignnan_s8_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 9) {
        return sortwithkeyasc_ignnan_s9_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 10) {
        return sortwithkeyasc_ignnan_s10_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 11) {
        return sortwithkeyasc_ignnan_s11_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 12) {
        return sortwithkeyasc_ignnan_s12_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 13) {
        return sortwithkeyasc_ignnan_s13_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 14) {
        return sortwithkeyasc_ignnan_s14_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 15) {
        return sortwithkeyasc_ignnan_s15_d(n, s, v_ptr, k_ptr);
    }
    else if (s <= 16) {
        return sortwithkeyasc_ignnan_s16_d(n, s, v_ptr, k_ptr);
    }
    else if (s < 32) {
        return sortwithkeyasc_ignnan_s17to31_d(n, s, v_ptr, k_ptr);
    }
    else {
        return sortwithkeyasc_ignnan_s32plus_d(n, s, v_ptr, k_ptr);
    }
}

int sortwithkeyasc_test_d() {
    std::mt19937 mt(1234);

    for (uint s = 2; s <= 128; s++) {
        for (uint n = 1; n <= 64; n++) {
            double* k = (double*)_aligned_malloc(s * n * sizeof(double), AVX2_ALIGNMENT);
            ulong* v = (ulong*)_aligned_malloc(s * n * sizeof(ulong), AVX2_ALIGNMENT);

            if (k == nullptr || v == nullptr) {
                return FAILURE_BADALLOC;
            }

            for (uint test = 0; test < 64; test++) {
                for (uint i = 0; i < s * n; i++) {
                    uint r = mt();
                    k[i] = r / (double)(~0u);
                    v[i] = i;
                }

                std::vector<double> tk(s * n), tc;

                memcpy_s(tk.data(), s * n * sizeof(double), k, s * n * sizeof(double));

                tc = tk;

                for (uint j = 0; j < n; j++) {
                    std::sort(tk.begin() + j * s, tk.begin() + (j + 1) * s);
                }

                sortwithkeyasc_d(n, s, v, k);

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

        for (uint n = 1; n <= 64; n++) {
            double* k = (double*)malloc(s * n * sizeof(double));
            ulong* v = (ulong*)malloc(s * n * sizeof(ulong));

            if (k == nullptr || v == nullptr) {
                return FAILURE_BADALLOC;
            }

            for (uint test = 0; test < 64; test++) {
                for (uint i = 0; i < s * n; i++) {
                    uint r = mt();
                    k[i] = r / (double)(~0u);
                    v[i] = i;
                }

                std::vector<double> tk(s * n), tc;

                memcpy_s(tk.data(), s * n * sizeof(double), k, s * n * sizeof(double));

                tc = tk;

                for (uint j = 0; j < n; j++) {
                    std::sort(tk.begin() + j * s, tk.begin() + (j + 1) * s);
                }

                sortwithkeyasc_d(n, s, v, k);

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