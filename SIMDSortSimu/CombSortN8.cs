using System;

namespace SIMDSortSimu {
    public static class CombSortN8 {

        public static int Iter(float[] vs, uint h) {
            if (h < MM256.AVX2_FLOAT_STRIDE) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE || n - MM256.AVX2_FLOAT_STRIDE < h) {
                return 0;
            }

            uint e = n - h - MM256.AVX2_FLOAT_STRIDE;

            int swaps = 0;

            for (uint i = 0; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                MM256 a = MM256.Load(vs, i);
                MM256 b = MM256.Load(vs, i + h);

                (_, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                MM256.Store(vs, i, x);
                MM256.Store(vs, i + h, y);

                swaps++;
            }
            {
                MM256 a = MM256.Load(vs, e);
                MM256 b = MM256.Load(vs, e + h);

                (_, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                MM256.Store(vs, e, x);
                MM256.Store(vs, e + h, y);

                swaps++;
            }

            return swaps;
        }
    }
}
