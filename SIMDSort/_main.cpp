#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_s.hpp"

int main() {
    sortasc_test_s();
    //sortdsc_test_s();
    sort_long_ndist_speed_test_s();
    sort_long_random_speed_test_s();
    sort_long_reverse_speed_test_s();
    sort_long_inbalance_speed_test_s();

    //for (uint n = 32; n <= 0x8000000u; n *= 2) {
    //    printf("%d : ", n);
    //
    //    for (uint h = (uint)(n * 10L / 13L); h > 1; h = (uint)(h * 10L / 13L)) {
    //        printf("%d -> ", h);
    //    }
    //
    //    printf("\n");
    //}

    //sortasc_test_d();
    //sort_long_ndist_speed_test_d();
    //sort_long_random_speed_test_d();
    //sort_long_reverse_speed_test_d();
    //sort_long_inbalance_speed_test_d();

    printf("end");
    return getchar();
}