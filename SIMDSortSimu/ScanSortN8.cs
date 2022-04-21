using System;

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
            for (uint i = 0; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                MM256 x = MM256.Load(vs, i);
                MM256 y = MM256.Sort(x);
                MM256.Store(vs, i, y);

                swaps++;
            }
            for (uint i = MM256.AVX2_FLOAT_STRIDE / 2; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                MM256 x = MM256.Load(vs, i);
                MM256 y = MM256.Sort(x);
                MM256.Store(vs, i, y);

                swaps++;
            }
            {
                uint i = 0;
                while (true) {
                    MM256 x = MM256.Load(vs, i);

                    if (MM256.NeedsSort(x)) {
                        MM256 y = MM256.Sort(x);
                        MM256.Store(vs, i, y);
                        swaps++;

                        if (i == 0) {
                            i = MM256.AVX2_FLOAT_STRIDE - 1;
                            if (i > e) {
                                i = e;
                            }
                            continue;
                        }

                        (_, uint index) = MM256.CmpEq(x, y);
                        
                        uint back = MM256.AVX2_FLOAT_STRIDE - index - 1;

                        i = (i > back) ? i - back : 0;

                    }
                    else if (i < e) {
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
