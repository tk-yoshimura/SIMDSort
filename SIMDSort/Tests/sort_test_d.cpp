#include <immintrin.h>
#include <vector>
#include <algorithm>
#include <random>
#include "../simdsort.h"
#include "../Sort/sort.h"

int sortasc_d(const uint n, const uint s, double* v_ptr) {
    if (s <= 1) {
        return SUCCESS;
    }
    else if (s <= 2) {
        return sortasc_ignnan_s2_d(n, s, v_ptr);
    }
    else if (s <= 3) {
        return sortasc_ignnan_s3_d(n, s, v_ptr);
    }
    else if (s <= 4) {
        return sortasc_ignnan_s4_d(n, s, v_ptr);
    }
    else if (s <= 5) {
        return sortasc_ignnan_s5_d(n, s, v_ptr);
    }
    else if (s <= 6) {
        return sortasc_ignnan_s6_d(n, s, v_ptr);
    }
    else if (s <= 7) {
        return sortasc_ignnan_s7_d(n, s, v_ptr);
    }
    else if (s <= 16) {
        return sortasc_ignnan_s8to16_d(n, s, v_ptr);
    }
    else {
        return sortasc_ignnan_slong_d(n, s, v_ptr);
    }
}

int sortasc_test_d() {
    std::mt19937 mt(1234);

    for (uint s = 16; s <= 128; s++) {
        for (uint n = 1; n <= 64; n++) {
            double* v = (double*)_aligned_malloc(s * n * sizeof(double), AVX2_ALIGNMENT);
            if (v == nullptr) {
                return FAILURE_BADALLOC;
            }

            for (uint test = 0; test < 64; test++) {
                for (uint i = 0; i < s * n; i++) {
                    uint r = mt();
                    v[i] = r / (double)(~0u);
                }

                std::vector<double> t(s * n);
                for (uint i = 0; i < s * n; i++) {
                    t[i] = v[i];
                }

                for (uint j = 0; j < n; j++) {
                    std::sort(t.begin() + j * s, t.begin() + (j + 1) * s);
                }

                sortasc_d(n, s, v);

                for (uint i = 0; i < s * n; i++) {
                    if (t[i] != v[i]) {
                        std::vector<double> u(s * n);
                        memcpy_s(u.data(), s * n * sizeof(double), v, s * n * sizeof(double));

                        printf("random ng n=%d s=%d\n", n, s);
                        throw std::exception("err");
                    }
                }


                printf("random ok n=%d s=%d\n", n, s);
            }

            _aligned_free(v);
        }
    }

    return 0;
}

int sortasc_perm_test_d() {
    for (uint s = 2; s <= 7; s++) {
        std::vector<double> v(s);
        for (uint i = 0; i < s; i++) {
            v[i] = (double)((i + 1) % s + 1);
        }

        uint c = 0;

        double* t = (double*)_aligned_malloc((s + 4) * sizeof(double), AVX2_ALIGNMENT);
        if (t == nullptr) {
            return FAILURE_BADALLOC;
        }

        do {

            memcpy_s(t, s * sizeof(double), v.data(), s * sizeof(double));

            for (uint i = s; i < s + 4; i++) {
                t[i] = ((i + c) * 31) % s;
            }

            sortasc_d(1, s, t);

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

            if ((c % 1000000) == 0 && c > 0) {
                printf(".");
            }

        } while (std::next_permutation(v.begin(), v.end()));
        
        _aligned_free(t);

        printf("\npermute ok s=%d\n", s);
    }

    return 0;
}
