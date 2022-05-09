#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_s.hpp"

int main() {
    //sortasc_test_s();
    //sortdsc_test_s();
    //sortasc_test_d();
    //
    sort_short_random_speed_test_d();
    sort_short_ndist_speed_test_d();
    sort_short_reverse_speed_test_d();
    sort_short_inbalance_speed_test_d();

    //sort_short_random_speed_test_s();
    //sort_short_ndist_speed_test_s();
    //sort_short_reverse_speed_test_s();
    //sort_short_inbalance_speed_test_s();


    printf("end");
    return getchar();
}