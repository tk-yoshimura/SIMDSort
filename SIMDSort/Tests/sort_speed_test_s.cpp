#include <vector>
#include <random>
#include <fstream>
#include <chrono>
#include "../constants.h"
#include "../simdsort.h"
#include "../Sort/sort.h"

int sort_random_speed_test_s() {
    std::ofstream ofs;
    ofs.open("bin/sorts_random_speed.txt");

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

int sort_inbalance_speed_test_s() {
    std::ofstream ofs;
    ofs.open("bin/sorts_inbalance_speed.txt");

    ofs << "n,std,avx" << std::endl;

    std::mt19937 mt(1234);

    for (uint n = 32; n <= 0x8000000u; n *= 2) {

        std::vector<long long> stdsorttimes, avxsorttimes;

        for (uint tests = 0; tests < 32; tests++) {
            std::vector<float> x(n);

            for (uint i = 0; i < n; i++) {
                float u1 = mt() / (float)(~0u);
                float u2 = mt() / (float)(~0u);

                x[i] = u1 < 0.01f ? u2 : (u2 * 0.01f);
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

int sort_reverse_speed_test_s() {
    std::ofstream ofs;
    ofs.open("bin/sorts_reverse_speed.txt");

    ofs << "n,std,avx" << std::endl;

    for (uint n = 32; n <= 0x8000000u; n *= 2) {

        std::vector<long long> stdsorttimes, avxsorttimes;

        for (uint tests = 0; tests < 32; tests++) {
            std::vector<float> x(n);

            for (uint i = 0; i < n; i++) {
                float u = 1 - i / (float)(n - 1);

                x[i] = u;
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

int sort_ndist_speed_test_s() {
    std::ofstream ofs;
    ofs.open("bin/sorts_ndist_speed.txt");

    ofs << "n,std,avx" << std::endl;

    std::random_device seed_gen;
    std::normal_distribution<float> ndist(0, 1);
    std::default_random_engine engine(seed_gen());

    for (uint n = 32; n <= 0x8000000u; n *= 2) {

        std::vector<long long> stdsorttimes, avxsorttimes;

        for (uint tests = 0; tests < 32; tests++) {
            std::vector<float> x(n);

            for (uint i = 0; i < n; i++) {
                float u = ndist(engine);

                x[i] = u;
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
