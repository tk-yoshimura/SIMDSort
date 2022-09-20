#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include "../simdsort.h"
#include "../SortWithKey/sortwithkey.h"

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

                sortwithkeyasc_ignnan_s(n, s, v, k);

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

                sortwithkeyasc_ignnan_s(n, s, v, k);

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