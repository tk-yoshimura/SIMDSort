#include <stdio.h>
#include "simdsort.h"

int main(){
    mm_sort_test_s();
    mm256_sort_test_s();

    printf("end");
    return getchar();
}