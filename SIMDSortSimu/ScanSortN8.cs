using System;
using System.IO;

namespace SIMDSortSimu {
    public static class ScanSortN8 {
        public static void Iter(float[] vs) {
            uint n = (uint)vs.Length;

            using StreamWriter sw = new($"../../history/swap_{n}_1.csv");

            if (n < MM256.AVX2_FLOAT_STRIDE) {
                Array.Sort(vs);
                return;
            }

            uint e = n - MM256.AVX2_FLOAT_STRIDE;
            for (uint i = 0; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                MM256 x = MM256.Load(vs, i);
                MM256 y = MM256.Sort(x);
                MM256.Store(vs, i, y);
            }
            {
                MM256 x = MM256.Load(vs, e);
                MM256 y = MM256.Sort(x);
                MM256.Store(vs, e, y);
            }
            {
                uint i = 0;
                while (true) {
                    MM256 x = MM256.Load(vs, i);
                    MM256 y = MM256.Sort(x);
                    MM256.Store(vs, i, y);

                    (_, uint index) = MM256.CmpEq(x, y);

                    sw.WriteLine($"{i} {index}");

                    if (index < MM256.AVX2_FLOAT_STRIDE) {
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
        }
    }
}
