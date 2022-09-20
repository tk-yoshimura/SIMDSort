#pragma once

#include "../types.h"
#include "../constants.h"

extern int sortasc_ignnan_s(const uint n, const uint s, float* v_ptr);
extern int sortasc_minnan_s(const uint n, const uint s, float* v_ptr);
extern int sortasc_maxnan_s(const uint n, const uint s, float* v_ptr);
extern int sortdsc_ignnan_s(const uint n, const uint s, float* v_ptr);
extern int sortdsc_minnan_s(const uint n, const uint s, float* v_ptr);
extern int sortdsc_maxnan_s(const uint n, const uint s, float* v_ptr);

extern int sortasc_ignnan_d(const uint n, const uint s, double* v_ptr);
extern int sortasc_minnan_d(const uint n, const uint s, double* v_ptr);
extern int sortasc_maxnan_d(const uint n, const uint s, double* v_ptr);
extern int sortdsc_ignnan_d(const uint n, const uint s, double* v_ptr);
extern int sortdsc_minnan_d(const uint n, const uint s, double* v_ptr);
extern int sortdsc_maxnan_d(const uint n, const uint s, double* v_ptr);