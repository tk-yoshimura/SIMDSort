#include <vector>
#include <algorithm>
#include "../simdsort.h"

void check_dup_s(const uint n, const float* ptr, const char* message) {
    std::vector<float> v(n);
    memcpy_s(v.data(), n * sizeof(float), ptr, n * sizeof(float));

    std::sort(v.begin(), v.end());

    for (uint i = 1; i < n; i++) {
        if (v[i - 1] >= v[i]) {
            throw std::exception(message);
        }
    }
}

void check_dup_d(const uint n, const double* ptr, const char* message) {
    std::vector<double> v(n);
    memcpy_s(v.data(), n * sizeof(double), ptr, n * sizeof(double));

    std::sort(v.begin(), v.end());

    for (uint i = 1; i < n; i++) {
        if (v[i - 1] >= v[i]) {
            throw std::exception(message);
        }
    }
}