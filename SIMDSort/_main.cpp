#include <stdio.h>
#include <intrin.h>
#include "simdsort.h"

int main() {
    //mm_cmpgt_test_s();
    //mm_cmplt_test_s();
    //mm_cmpeq_test_s();
    //
    //mm256_cmpgt_test_s();
    //mm256_cmplt_test_s();
    //mm256_cmpeq_test_s();
    //
    //sort_n8_test_s();
    //
    
    //sort_test_s();

    sort_speed_test_s();

    //for (uint n = 32; n <= 0x8000000u; n *= 2) {
    //    printf("%d : ", n);
    //    
    //    for (uint h = (uint)(n * 10L / 13L); h > 1; h = (uint)(h * 10L / 13L)) {
    //        if (h < 512) {
    //            printf("%d -> ", h);
    //        }
    //    }
    //
    //    printf("\n");
    //}

    printf("end");
    return getchar();
}