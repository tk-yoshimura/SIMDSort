using System;
using System.IO;

namespace SIMDSortSimu {
    static class RewindSortN4 {

        public static void Iter(float[] vs, uint h) {
            if (h < MM128.AVX1_FLOAT_STRIDE) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            using StreamWriter sw = new($"../../history/swap_{n}_{h}.csv");

            if (n >= MM128.AVX1_FLOAT_STRIDE && n - MM128.AVX1_FLOAT_STRIDE >= h) {
                uint i = 0, e = n - h, f = e - MM128.AVX1_FLOAT_STRIDE;
                MM128 x, y;

                while (i < e) {
                    if (i > f) {
                        i = f;
                    }

                    x = MM128.Load(vs, i);
                    y = MM128.Load(vs, i + h);

                    (_, uint index, MM128 a, MM128 b) = MM128.CmpSwapGt(x, y);

                    sw.WriteLine($"{i} <-> {i + h} {index}");

                    if (index >= MM128.AVX1_FLOAT_STRIDE) {
                        i += MM128.AVX1_FLOAT_STRIDE;
                    }
                    else {
                        MM128.Store(vs, i, a);
                        MM128.Store(vs, i + h, b);

                        uint back = h - index;

                        i = i >= back ? i - back : 0;
                    }
                }
            }
        }
    }
}
