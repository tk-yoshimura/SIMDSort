#include <stdio.h>
#include <intrin.h>
#include "simdsort.h"
#include "Inline/inline_misc.hpp"

int main(){
    //mm_sort_test_s();
    //mm256_sort_test_s();
    //
    //mm_cmpgt_test_s();
    //mm_cmplt_test_s();
    //mm_cmpeq_test_s();
    //mm256_cmpgt_test_s();
    //mm256_cmplt_test_s();
    //mm256_cmpeq_test_s();

    bubblesort_n8_test_s();
    scansort_n8_test_s();

    printf("end");
    return getchar();
}