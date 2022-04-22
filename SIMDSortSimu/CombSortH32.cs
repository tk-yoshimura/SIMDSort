namespace SIMDSortSimu {
    public static class CombSortH32 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 8) {
                return 0;
            }

            uint e = n - MM256.AVX2_FLOAT_STRIDE * 8;

            int swaps = 0;

            MM256 a0, a1, a2, a3, b0, b1, b2, b3;

            if (e > 0) {
                a0 = MM256.Load(vs, 0);
                a1 = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE);
                a2 = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE * 2);
                a3 = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE * 3);

                uint i = 0;
                for (; i < e; i += MM256.AVX2_FLOAT_STRIDE * 4) {
                    b0 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 4);
                    b1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 5);
                    b2 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 6);
                    b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 7);

                    (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                    (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);
                    (_, _, MM256 x2, MM256 y2) = MM256.CmpSwapGt(a2, b2);
                    (_, _, MM256 x3, MM256 y3) = MM256.CmpSwapGt(a3, b3);

                    MM256.Store(vs, i, x0);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, x1);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE * 2, x2);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE * 3, x3);
                    a0 = y0;
                    a1 = y1;
                    a2 = y2;
                    a3 = y3;

                    swaps++;
                }
                MM256.Store(vs, i, a0);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, a1);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE * 2, a2);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE * 3, a3);
            }

            {
                a0 = MM256.Load(vs, e);
                a1 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE);
                a2 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 2);
                a3 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 3);
                b0 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 4);
                b1 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 5);
                b2 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 6);
                b3 = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE * 7);

                (_, _, MM256 x0, MM256 y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, MM256 x1, MM256 y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, MM256 x2, MM256 y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, MM256 x3, MM256 y3) = MM256.CmpSwapGt(a3, b3);

                MM256.Store(vs, e, x0);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE, x1);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 2, x2);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 3, x3);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 4, y0);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 5, y1);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 6, y2);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE * 7, y3);

                swaps++;
            }

            return swaps;
        }
    }
}
