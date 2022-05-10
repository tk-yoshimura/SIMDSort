#include <stdio.h>
#include <intrin.h>
#include <vector>
#include <algorithm>
#include "simdsort.h"
#include "constants.h"
#include "Inline/inline_cmp_s.hpp"

int main() {
    
    sort_limit_random_test_s();
    sort_limit_random_test_d();

    printf("end");
    return getchar();
}