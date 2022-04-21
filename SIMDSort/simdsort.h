#pragma once

#include "types.h"

extern int mm_sort_test_s();
extern int mm256_sort_test_s();

extern int mm_cmpgt_test_s();
extern int mm_cmplt_test_s();
extern int mm_cmpeq_test_s();

extern int mm256_cmpgt_test_s();
extern int mm256_cmplt_test_s();
extern int mm256_cmpeq_test_s();

extern int bubblesort_n8_test_s();
extern int scansort_n8_test_s();

extern int combsortasc_n32_s(const uint n, const uint h, outfloats v_ptr);
extern int combsortasc_n24_s(const uint n, const uint h, outfloats v_ptr);
extern int combsortasc_n16_s(const uint n, const uint h, outfloats v_ptr);
extern int combsortasc_n8_s (const uint n, const uint h, outfloats v_ptr);
extern int bubblesortasc_n8_s(const uint n, outfloats v_ptr);
extern int scansortasc_n8_s(const uint n, outfloats v_ptr);