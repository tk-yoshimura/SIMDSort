using System;

namespace SIMDSortSimu {
    public static class CombSortN4 {

        public static void Iter(float[] vs, uint h) {
            if (h < MM128.AVX1_FLOAT_STRIDE) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE || n - MM128.AVX1_FLOAT_STRIDE < h) {
                return;
            }

            uint e = n - h - MM128.AVX1_FLOAT_STRIDE;

            for (uint i = 0; i < e; i += MM128.AVX1_FLOAT_STRIDE) {
                MM128 x = MM128.Load(vs, i);
                MM128 y = MM128.Load(vs, i + h);

                (_, _, MM128 a, MM128 b) = MM128.CmpSwapGt(x, y);

                MM128.Store(vs, i, a);
                MM128.Store(vs, i + h, b);
            }
            {
                MM128 x = MM128.Load(vs, e);
                MM128 y = MM128.Load(vs, e + h);

                (_, _, MM128 a, MM128 b) = MM128.CmpSwapGt(x, y);

                MM128.Store(vs, e, a);
                MM128.Store(vs, e + h, b);
            }
        }
    }
}
