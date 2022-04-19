using System;
using System.IO;

namespace SIMDSortSimu {
    static class RewindSortN8 {

        public static void Iter(float[] vs, uint h) {
            if (h < MM256.AVX2_FLOAT_STRIDE) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            using StreamWriter sw = new($"../../history/swap_{n}_{h}.csv");

            if (n >= MM256.AVX2_FLOAT_STRIDE && n - MM256.AVX2_FLOAT_STRIDE >= h) {
                uint i = 0, e = n - h, f = e - MM256.AVX2_FLOAT_STRIDE;
                MM256 x, y;

                while (i < e) {
                    if (i > f) {
                        i = f;
                    }

                    x = MM256.Load(vs, i);
                    y = MM256.Load(vs, i + h);

                    (_, uint index, MM256 a, MM256 b) = MM256.CmpSwapGt(x, y);

                    sw.WriteLine($"{i} <-> {i + h} {index}");

                    if (index >= MM256.AVX2_FLOAT_STRIDE) {
                        i += MM256.AVX2_FLOAT_STRIDE;
                    }
                    else {
                        MM256.Store(vs, i, a);
                        MM256.Store(vs, i + h, b);

                        uint back = h - index;

                        i = i >= back ? i - back : 0;
                    }
                }
            }
        }
    }
}
