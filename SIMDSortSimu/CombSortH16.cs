namespace SIMDSortSimu {
    public static class CombSortH16 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 4) {
                return 0;
            }

            uint e = n - MM256.AVX2_FLOAT_STRIDE * 4;

            int swaps = 0;

            MM256 a0, a1, b0, b1;

            if (e > 0) {
                a0 = MM256.Load(vs, 0);
                a1 = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE);

                uint i = 0;
                for (; i < e; i += MM256.AVX2_FLOAT_STRIDE * 2) {
                    b0 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 2);
                    b1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 3);

                    (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                    (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);

                    MM256.Store(vs, i, x0);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, x1);
                    a0 = y0;
                    a1 = y1;

                    swaps++;
                }
                MM256.Store(vs, i, a0);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, a1);
            }

            {
                a0 = MM256.Load(vs, e);
                a1 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE);
                b0 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 2);
                b1 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 3);

                (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);

                MM256.Store(vs, e, x0);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE, x1);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 2, y0);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 3, y1);

                swaps++;
            }

            return swaps;
        }
    }
}
