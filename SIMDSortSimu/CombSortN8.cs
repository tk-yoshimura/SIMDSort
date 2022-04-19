using System;

namespace SIMDSortSimu {
    static class CombSortN8 {

        public static void Iter(float[] vs, uint h) {
            if (h < MM256.AVX2_FLOAT_STRIDE) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            if (n >= MM256.AVX2_FLOAT_STRIDE && n - MM256.AVX2_FLOAT_STRIDE >= h) {
                uint e = n - h - MM256.AVX2_FLOAT_STRIDE;

                for (uint i = 0; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                    MM256 x = MM256.Load(vs, i);
                    MM256 y = MM256.Load(vs, i + h);

                    (_, _, MM256 a, MM256 b) = MM256.CmpSwapGt(x, y);

                    MM256.Store(vs, i, a);
                    MM256.Store(vs, i + h, b);
                }
                {
                    MM256 x = MM256.Load(vs, e);
                    MM256 y = MM256.Load(vs, e + h);

                    (_, _, MM256 a, MM256 b) = MM256.CmpSwapGt(x, y);

                    MM256.Store(vs, e, a);
                    MM256.Store(vs, e + h, b);
                }
            }
        }
    }
}
