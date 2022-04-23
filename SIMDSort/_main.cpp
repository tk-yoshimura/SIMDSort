#include <stdio.h>
#include <intrin.h>
#include "simdsort.h"

int main() {
    //sort_test_s();
    sort_ndist_speed_test_s();
    sort_random_speed_test_s();
    sort_reverse_speed_test_s();
    sort_inbalance_speed_test_s();

    printf("end");
    return getchar();
}