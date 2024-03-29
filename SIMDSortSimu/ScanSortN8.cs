﻿using System;

namespace SIMDSortSimu {
    public static class ScanSortN8 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE) {
                Array.Sort(vs);
                return 0;
            }

            int swaps = 0;

            uint e = n - MM256.AVX2_FLOAT_STRIDE;

            {
                uint i = 0;
                while (true) {
                    MM256 x = MM256.Load(vs, i);

                    if (MM256.NeedsSort(x)) {
                        MM256 y = MM256.Sort(x);
                        MM256.Store(vs, i, y);
                        swaps++;

                        if (i > 0) {
                            (_, uint index) = MM256.CmpNeq(x, y);
                            if ((index & 1) == 1) {
                                uint back = MM256.AVX2_FLOAT_STRIDE - 2;

                                i = (i > back) ? i - back : 0;
                                continue;
                            }
                        }
                    }

                    if (i < e) {
                        i += MM256.AVX2_FLOAT_STRIDE - 1;
                        if (i > e) {
                            i = e;
                        }
                    }
                    else {
                        break;
                    }
                }
            }

            return swaps;
        }
    }
}
