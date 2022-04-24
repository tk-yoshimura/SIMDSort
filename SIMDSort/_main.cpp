#include <stdio.h>
#include <intrin.h>
#include "simdsort.h"

int main() {
    sort_test_s();
    //sort_ndist_speed_test_s();
    //sort_random_speed_test_s();
    //sort_reverse_speed_test_s();
    //sort_inbalance_speed_test_s();

    //for (uint n = 32; n <= 0x8000000u; n *= 2) {
    //    printf("%d : ", n);
    //
    //    for (uint h = (uint)(n * 10L / 13L); h > 1; h = (uint)(h * 10L / 13L)) {
    //        printf("%d -> ", h);
    //    }
    //
    //    printf("\n");
    //}

    printf("end");
    return getchar();
}