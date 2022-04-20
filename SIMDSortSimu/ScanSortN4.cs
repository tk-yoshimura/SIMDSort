using System;

namespace SIMDSortSimu {
    public static class ScanSortN4 {
        public static void Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE) {
                Array.Sort(vs);
                return;
            }

            uint e = n - MM128.AVX1_FLOAT_STRIDE;
            for (uint i = 0; i < e; i += MM128.AVX1_FLOAT_STRIDE) {
                MM128 x = MM128.Load(vs, i);
                MM128 y = MM128.Sort(x);
                MM128.Store(vs, i, y);
            }
            {
                MM128 x = MM128.Load(vs, e);
                MM128 y = MM128.Sort(x);
                MM128.Store(vs, e, y);
            }
            {
                uint i = 0;
                while (true) {
                    MM128 x = MM128.Load(vs, i);
                    MM128 y = MM128.Sort(x);
                    MM128.Store(vs, i, y);

                    (_, uint index) = MM128.CmpEq(x, y);

                    if (index < MM128.AVX1_FLOAT_STRIDE) {
                        uint back = MM128.AVX1_FLOAT_STRIDE - index - 1;

                        i = (i > back) ? i - back : 0;
                    }
                    else if (i < e) {
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
        }
    }
}
