#include <vector>
#include <random>
#include <chrono>
#include "../constants.h"
#include "../simdsort.h"
#include "../Sort/sort.h"

int sort_limit_random_test_d() {
    std::mt19937 mt(1234);

    const uint n = MAX_SORT_STRIDE;

    std::vector<double> x(n);

    for (uint i = 0; i < n; i++) {
        x[i] = mt() / (double)(~0u);
    }

    std::vector<double> y = x;

    auto stdsortclock = std::chrono::system_clock::now();

    std::sort(x.begin(), x.end());
        
    auto stdsorttime = std::chrono::system_clock::now() - stdsortclock;

    auto stdsortusec = std::chrono::duration_cast<std::chrono::microseconds>(stdsorttime).count();
    printf_s("%d, std=%lld\n", n, stdsortusec);

    auto avxsortclock = std::chrono::system_clock::now();

    sortasc_ignnan_d(1, n, y.data());

    auto avxsorttime = std::chrono::system_clock::now() - avxsortclock;

    auto avxsortusec = std::chrono::duration_cast<std::chrono::microseconds>(avxsorttime).count();
    printf_s("%d, avx=%lld\n", n, avxsortusec);

    for (uint i = 0; i < n; i++) {
        if (x[i] != y[i]) {
            throw std::exception("mismatch");
        }
    }

    return SUCCESS;
}
