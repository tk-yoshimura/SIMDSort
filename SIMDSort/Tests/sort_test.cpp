#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include "../simdsort.h"
#include "../Sort/sort.h"

int sort_s(const uint n, const uint s, outfloats v_ptr) {
    if (s <= 1) {
        return SUCCESS;
    }
    else if(s <= 2){
        return sortasc_ignnan_s2_s(n, s, v_ptr);
    }
    else if (s <= 3) {
        return sortasc_ignnan_s3_s(n, s, v_ptr);
    }
    else if (s <= 4) {
        return sortasc_ignnan_s4_s(n, s, v_ptr);
    }
    else if (s <= 5) {
        return sortasc_ignnan_s5_s(n, s, v_ptr);
    }
    else if (s <= 6) {
        return sortasc_ignnan_s6_s(n, s, v_ptr);
    }
    else if (s <= 7) {
        return sortasc_ignnan_s7_s(n, s, v_ptr);
    }
    else if (s <= 8) {
        return sortasc_ignnan_s8_s(n, s, v_ptr);
    }
    else if (s <= 11) {
        return sortasc_ignnan_s9to11_s(n, s, v_ptr);
    }
    else if (s <= 12) {
        return sortasc_ignnan_s12_s(n, s, v_ptr);
    }
    else if (s <= 15) {
        return sortasc_ignnan_s13to15_s(n, s, v_ptr);
    }
    else if (s <= 32) {
        return sortasc_ignnan_s16to32_s(n, s, v_ptr);
    }
    else {
        return sortasc_ignnan_slong_s(n, s, v_ptr);
    }
}

int sort_test_s() {
    std::mt19937 mt(1234);

    for (uint s = 2; s <= 128; s++) {
        for (uint n = 1; n <= 64; n++) {
            for (uint test = 0; test < 20; test++) {

                float* v = (float*)_aligned_malloc(s * n * sizeof(float), AVX2_ALIGNMENT);
                if (v == nullptr) {
                    return FAILURE_BADALLOC;
                }

                for (uint i = 0; i < s * n; i++) {
                    uint r = mt();
                    v[i] = r / (float)(~0u);
                }

                std::vector<float> t(s * n);
                for (uint i = 0; i < s * n; i++) {
                    t[i] = v[i];
                }

                for (uint j = 0; j < n; j++) {
                    std::sort(t.begin() + j * s, t.begin() + (j + 1) * s);
                }

                sort_s(n, s, v);

                for (uint i = 0; i < s * n; i++) {
                    if (t[i] != v[i]) {
                        printf("random ng n=%d s=%d\n", n, s);
                        throw std::exception("err");
                    }
                }

                _aligned_free(v);

                printf("random ok n=%d s=%d\n", n, s);

            }
        }
    }

    for (uint s = 2; s <= 12; s++) {
        std::vector<float> v(s);
        for (uint i = 0; i < s; i++) {
            v[i] = (float)((i + 1) % s + 1);
        }

        uint c = 0;

        do {
            float* t = (float*)_aligned_malloc((s + 4) * sizeof(float), AVX2_ALIGNMENT);
            if (t == nullptr) {
                return FAILURE_BADALLOC;
            }

            memcpy_s(t, s * sizeof(float), v.data(), s * sizeof(float));

            for (uint i = s; i < s + 4; i++) {
                t[i] = ((i + c) * 31) % s;
            }

            sort_s(1, s, t);

            for (uint i = 1; i < s; i++) {
                if (t[i - 1u] >= t[i]) {
                    throw std::exception("err");
                }
            }
            for (uint i = s; i < s + 4; i++) {
                if (t[i] != ((i + c) * 31) % s) {
                    throw std::exception("err");
                }
            }

            c++;

            if ((c % 10000) == 0 && c > 0) {
                printf(".");
            }

            _aligned_free(t);

        } while (std::next_permutation(v.begin(), v.end()));

        printf("\npermute ok s=%d\n", s);
    }

    return 0;
}