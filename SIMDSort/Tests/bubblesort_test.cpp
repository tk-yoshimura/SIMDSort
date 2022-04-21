#include <immintrin.h>
#include <vector>
#include <exception>
#include "../constants.h"
#include "../simdsort.h"

int bubblesort_n8_test_s() {
    for (uint n = AVX2_FLOAT_STRIDE * 2; n <= 64; n++) {
        std::vector<float> v(n + 4);
        
        for (uint i = 0; i < n; i++) {
            v[i] = rand() / (float)RAND_MAX;
        }

        for (uint i = n; i < n + 4; i++) {
            v[i] = NAN;
        }

        bubblesortasc_n8_s(n, v.data());

        for (uint i = 0; i < n - AVX2_FLOAT_STRIDE; i++) {
            if (v[i] > v[i + AVX2_FLOAT_STRIDE]) {
                throw std::exception("sort failure");
            }
        }

        printf("ok n = %d\n", n);
    }

    return 0;
}