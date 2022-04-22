#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include "../constants.h"
#include "../simdsort.h"
#include "../Sort/sort.h"

int sort_speed_test() {
    std::ofstream ofs;
    ofs.open("bin/sort_speed.txt");

    ofs << "n,std,avx" << std::endl;

    std::mt19937 mt(1234);

    for (uint n = 32; n <= 0x8000000u; n *= 2) {

        std::vector<long long> stdsorttimes, avxsorttimes;

        for (uint tests = 0; tests < 32; tests++) {
            std::vector<float> x(n);

            for (uint i = 0; i < n; i++) {
                x[i] = mt() / (float)(~0u);
            }

            std::vector<float> y = x;

            auto stdsortclock = std::chrono::system_clock::now();

            std::sort(x.begin(), x.end());
        
            auto stdsorttime = std::chrono::system_clock::now() - stdsortclock;
            
            auto avxsortclock = std::chrono::system_clock::now();

            sortasc_ignnan_slong_s(1, n, y.data());

            auto avxsorttime = std::chrono::system_clock::now() - avxsortclock;

            auto stdsortusec = std::chrono::duration_cast<std::chrono::microseconds>(stdsorttime).count();
            auto avxsortusec = std::chrono::duration_cast<std::chrono::microseconds>(avxsorttime).count();

            stdsorttimes.push_back(stdsortusec);
            avxsorttimes.push_back(avxsortusec);

            printf_s("%d, std=%lld, avx=%lld\n", n, stdsortusec, avxsortusec);

            for (uint i = 0; i < n; i++) {
                if (x[i] != y[i]) {
                    throw std::exception("mismatch");
                }
            }
        }

        for (uint tests = 0; tests < 32; tests++) {
            ofs << n << ',' << stdsorttimes[tests] << ',' << avxsorttimes[tests] << std::endl;
        }

        ofs.flush();
    }

    return SUCCESS;
}
