using System;

namespace SIMDSortSimu {
    public static class CombSortH33plus {

        public static int Iter(float[] vs, uint h) {
            if (h <= MM256.AVX2_FLOAT_STRIDE * 4) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 4 || n - MM256.AVX2_FLOAT_STRIDE * 4 < h) {
                return 0;
            }

            uint e = n - h - MM256.AVX2_FLOAT_STRIDE * 4;

            int swaps = 0;

            MM256 a0, a1, a2, a3, b0, b1, b2, b3;

            for (uint i = 0; i < e; i += MM256.AVX2_FLOAT_STRIDE * 4) {
                a0 = MM256.Load(vs, i);
                a1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                a2 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 2);
                a3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 3);

                b0 = MM256.Load(vs, i + h);
                b1 = MM256.Load(vs, i + h + MM256.AVX2_FLOAT_STRIDE);
                b2 = MM256.Load(vs, i + h + MM256.AVX2_FLOAT_STRIDE * 2);
                b3 = MM256.Load(vs, i + h + MM256.AVX2_FLOAT_STRIDE * 3);

                (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, MM256 x2, MM256 y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, MM256 x3, MM256 y3) = MM256.CmpSwapGt(a3, b3);

                MM256.Store(vs, i, x0);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, x1);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE * 2, x2);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE * 3, x3);

                MM256.Store(vs, i + h, y0);
                MM256.Store(vs, i + h + MM256.AVX2_FLOAT_STRIDE, y1);
                MM256.Store(vs, i + h + MM256.AVX2_FLOAT_STRIDE * 2, y2);
                MM256.Store(vs, i + h + MM256.AVX2_FLOAT_STRIDE * 3, y3);

                swaps++;
            }
            {
                a0 = MM256.Load(vs, e);
                a1 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE);
                a2 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 2);
                a3 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 3);

                b0 = MM256.Load(vs, e + h);
                b1 = MM256.Load(vs, e + h + MM256.AVX2_FLOAT_STRIDE);
                b2 = MM256.Load(vs, e + h + MM256.AVX2_FLOAT_STRIDE * 2);
                b3 = MM256.Load(vs, e + h + MM256.AVX2_FLOAT_STRIDE * 3);

                (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, MM256 x2, MM256 y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, MM256 x3, MM256 y3) = MM256.CmpSwapGt(a3, b3);

                MM256.Store(vs, e, x0);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE, x1);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 2, x2);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 3, x3);

                MM256.Store(vs, e + h, y0);
                MM256.Store(vs, e + h + MM256.AVX2_FLOAT_STRIDE, y1);
                MM256.Store(vs, e + h + MM256.AVX2_FLOAT_STRIDE * 2, y2);
                MM256.Store(vs, e + h + MM256.AVX2_FLOAT_STRIDE * 3, y3);

                swaps++;
            }

            return swaps;
        }
    }
}
