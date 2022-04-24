#pragma once

#include "types.h"

extern int mm256_cmpgt_test_s();
extern int mm256_cmplt_test_s();
extern int mm256_cmpeq_test_s();

extern int sortasc_test_s();
extern int sortdsc_test_s();
extern int sortasc_perm_test_s();

extern int sort_random_speed_test_s();
extern int sort_inbalance_speed_test_s();
extern int sort_reverse_speed_test_s();
extern int sort_ndist_speed_test_s();