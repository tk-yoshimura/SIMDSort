#include <immintrin.h>
#include <vector>
#include <exception>
#include "../constants.h"
#include "../simdsort.h"
#include "../Sort/sort.h"

int sort_n8_test_s() {
    for (uint n = AVX2_FLOAT_STRIDE; n <= 64; n++) {
        std::vector<float> v(n + 4);

        for (uint i = 0; i < n; i++) {
            v[i] = rand() / (float)RAND_MAX;
        }

        for (uint i = n; i < n + 4; i++) {
            v[i] = NAN;
        }

        sortasc_ignnan_slong_s(1, n, v.data());

        for (uint i = 0; i < n - 1; i++) {
            if (v[i] > v[i + 1]) {
                throw std::exception("sort failure");
            }
        }

        printf("ok n = %d\n", n);
    }

    return 0;
}