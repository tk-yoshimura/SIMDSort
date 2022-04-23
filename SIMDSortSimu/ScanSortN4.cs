using System;

namespace SIMDSortSimu {
    public static class ScanSortN4 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE) {
                Array.Sort(vs);
                return 0;
            }

            int swaps = 0;

            uint e = n - MM128.AVX1_FLOAT_STRIDE;
            for (uint i = 0; i < e; i += MM128.AVX1_FLOAT_STRIDE) {
                MM128 x = MM128.Load(vs, i);
                MM128 y = MM128.Sort(x);
                MM128.Store(vs, i, y);

                swaps++;
            }
            for (uint i = MM128.AVX1_FLOAT_STRIDE / 2; i < e; i += MM128.AVX1_FLOAT_STRIDE) {
                MM128 x = MM128.Load(vs, i);
                MM128 y = MM128.Sort(x);
                MM128.Store(vs, i, y);

                swaps++;
            }
            {
                uint i = 0;
                while (true) {
                    MM128 x = MM128.Load(vs, i);

                    if (MM128.NeedsSort(x)) {
                        MM128 y = MM128.Sort(x);
                        MM128.Store(vs, i, y);
                        swaps++;

                        if (i > 0) {
                            (_, uint index) = MM128.CmpEq(x, y);
                            if (index == 0) {
                                uint back = MM128.AVX1_FLOAT_STRIDE - 2;

                                i = (i > back) ? i - back : 0;
                                continue;
                            }
                        }
                    }

                    if (i < e) {
                        i += MM128.AVX1_FLOAT_STRIDE - 1;
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
