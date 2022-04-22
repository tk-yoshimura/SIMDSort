using System;

namespace SIMDSortSimu {
    public static class CombSortH9to15 {

        public static int Iter(float[] vs, uint h) {
            if (h <= MM256.AVX2_FLOAT_STRIDE || h >= MM256.AVX2_FLOAT_STRIDE * 2) {
                throw new ArgumentException(null, nameof(h));
            }

            uint n = (uint)vs.Length;

            if (n < h * 2) {
                return 0;
            }

            uint e = n - h * 2;

            int swaps = 0;

            MM256 a0, a1, b0, b1;

            if (e > 0) {
                a0 = MM256.Load(vs, 0);
                a1 = MM256.MaskLoad(vs, MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE);

                uint i = 0;
                for (; i < e; i += h) {
                    b0 = MM256.Load(vs, i + h);
                    b1 = MM256.MaskLoad(vs, i + h + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE);

                    (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                    (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);

                    MM256.Store(vs, i, x0);
                    MM256.MaskStore(vs, i + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE, x1);
                    a0 = y0;
                    a1 = y1;

                    swaps++;
                }
                MM256.Store(vs, i, a0);
                MM256.MaskStore(vs, i + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE, a1);
            }

            {
                a0 = MM256.Load(vs, e);
                a1 = MM256.MaskLoad(vs, e + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE);
                b0 = MM256.Load(vs, e + h);
                b1 = MM256.MaskLoad(vs, e + h + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE);

                (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);

                MM256.Store(vs, e, x0);
                MM256.MaskStore(vs, e + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE, x1);
                MM256.Store(vs, e + h, y0);
                MM256.MaskStore(vs, e + h + MM256.AVX2_FLOAT_STRIDE, h - MM256.AVX2_FLOAT_STRIDE, y1);

                swaps++;
            }

            return swaps;
        }
    }
}
