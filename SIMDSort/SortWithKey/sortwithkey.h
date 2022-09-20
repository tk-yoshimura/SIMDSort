#pragma once

#include "../types.h"
#include "../sortkv_types.h"
#include "../constants.h"

extern int sortwithkeyasc_ignnan_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr);
extern int sortwithkeyasc_minnan_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr);
extern int sortwithkeyasc_maxnan_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr);
extern int sortwithkeydsc_ignnan_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr);
extern int sortwithkeydsc_minnan_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr);
extern int sortwithkeydsc_maxnan_s(const uint n, const uint s, uint* __restrict v_ptr, float* __restrict k_ptr);
extern int sortwithkeyasc_ignnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr);
extern int sortwithkeyasc_minnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr);
extern int sortwithkeyasc_maxnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr);
extern int sortwithkeydsc_ignnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr);
extern int sortwithkeydsc_minnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr);
extern int sortwithkeydsc_maxnan_d(const uint n, const uint s, ulong* __restrict v_ptr, double* __restrict k_ptr);