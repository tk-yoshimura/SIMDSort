#include <immintrin.h>
#include <vector>
#include <exception>
#include "../constants.h"
#include "../simdsort.h"

int scansort_n8_test_s() {
    for (uint n = AVX2_FLOAT_STRIDE * 2; n <= 64; n++) {
        std::vector<float> v(n + 4);

        for (uint i = 0; i < n; i++) {
            v[i] = rand() / (float)RAND_MAX;
        }

        for (uint i = n; i < n + 4; i++) {
            v[i] = NAN;
        }

        scansortasc_n8_s(n, v.data());

        for (uint i = 0; i < n - 1; i++) {
            if (v[i] > v[i + 1]) {
                throw std::exception("sort failure");
            }
        }

        printf("ok n = %d\n", n);
    }

    return 0;
}