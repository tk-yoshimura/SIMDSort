#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_ep32.hpp"
#include "Inline/inline_cmp_ep64.hpp"

int main() {
    mm256_cmpgt_test_s();
    mm256_cmplt_test_s();
    mm256_cmpeq_test_s();

    sortasc_test_s();
    sortdsc_test_s();
    sortasc_perm_test_s();

    sortasc_test_d();
    sortdsc_test_d();
    sortasc_perm_test_d();

    sortwithkeyasc_test_s();
    sortwithkeyasc_test_d();

    sort_long_random_speed_test_s();
    sort_long_inbalance_speed_test_s();
    sort_long_reverse_speed_test_s();
    sort_long_ndist_speed_test_s();

    sort_long_random_speed_test_d();
    sort_long_inbalance_speed_test_d();
    sort_long_reverse_speed_test_d();
    sort_long_ndist_speed_test_d();

    sort_short_random_speed_test_s();
    sort_short_inbalance_speed_test_s();
    sort_short_reverse_speed_test_s();
    sort_short_ndist_speed_test_s();

    sort_short_random_speed_test_d();
    sort_short_inbalance_speed_test_d();
    sort_short_reverse_speed_test_d();
    sort_short_ndist_speed_test_d();

    sort_limit_random_test_s();
    sort_limit_random_test_d();

    printf("end");
    return getchar();
}