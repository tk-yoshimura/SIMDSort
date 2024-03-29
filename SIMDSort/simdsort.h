#pragma once

#include "types.h"
#include <stdio.h>

extern int mm256_cmpgt_test_s();
extern int mm256_cmplt_test_s();
extern int mm256_cmpeq_test_s();

extern int sortasc_test_s();
extern int sortdsc_test_s();
extern int sortasc_perm_test_s();

extern int sortasc_test_d();
extern int sortdsc_test_d();
extern int sortasc_perm_test_d();

extern int sortwithkeyasc_test_s();
extern int sortwithkeyasc_test_d();

extern int sort_long_random_speed_test_s();
extern int sort_long_inbalance_speed_test_s();
extern int sort_long_reverse_speed_test_s();
extern int sort_long_ndist_speed_test_s();

extern int sort_long_random_speed_test_d();
extern int sort_long_inbalance_speed_test_d();
extern int sort_long_reverse_speed_test_d();
extern int sort_long_ndist_speed_test_d();

extern int sort_short_random_speed_test_s();
extern int sort_short_inbalance_speed_test_s();
extern int sort_short_reverse_speed_test_s();
extern int sort_short_ndist_speed_test_s();

extern int sort_short_random_speed_test_d();
extern int sort_short_inbalance_speed_test_d();
extern int sort_short_reverse_speed_test_d();
extern int sort_short_ndist_speed_test_d();

extern int sort_limit_random_test_s();
extern int sort_limit_random_test_d();

extern void check_dup_s(const uint n, const float* ptr, const char* message);
extern void check_dup_d(const uint n, const double* ptr, const char* message);