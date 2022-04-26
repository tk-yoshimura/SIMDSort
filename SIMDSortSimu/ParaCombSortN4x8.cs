namespace SIMDSortSimu {
    public static class ParaCombSortN4x8 {
        public static void Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 8) {
                return;
            }

            uint e = n - MM256.AVX2_FLOAT_STRIDE * 8;
            uint c = n % MM256.AVX2_FLOAT_STRIDE;

            MM256 a0, a1, a2, a3, b0, b1, b2, b3;
            MM256 x0, x1, x2, x3, y0, y1, y2, y3;

            for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
                a3 = MM256.Load(vs, i);
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                a2 = x3;
                a3 = y3;
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 2);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                b2 = x3;
                a3 = y3;
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 3);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                a1 = x2;
                a2 = y2;
                b2 = x3;
                a3 = y3;
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 4);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                b1 = x2;
                a2 = y2;
                b2 = x3;
                a3 = y3;
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 5);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                a0 = x1;
                a1 = y1;
                b1 = x2;
                a2 = y2;
                b2 = x3;
                a3 = y3;
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 6);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                b0 = x1;
                a1 = y1;
                b1 = x2;
                a2 = y2;
                b2 = x3;
                a3 = y3;
                b3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 7);
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);

                for (j = i; j + MM256.AVX2_FLOAT_STRIDE <= e; j += MM256.AVX2_FLOAT_STRIDE) {
                    MM256.Store(vs, j, x0);
                    a0 = y0;
                    b0 = x1;
                    a1 = y1;
                    b1 = x2;
                    a2 = y2;
                    b2 = x3;
                    a3 = y3;
                    b3 = MM256.Load(vs, j + MM256.AVX2_FLOAT_STRIDE * 8);
                    (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                    (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                    (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                    (_, _, x3, y3) = MM256.CmpSwapGt(a3, b3);
                }

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = x1;
                a1 = y1;
                b1 = x2;
                a2 = y2;
                b2 = x3;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                j += MM256.AVX2_FLOAT_STRIDE;

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = x1;
                a1 = y1;
                b1 = x2;
                a2 = y2;
                b2 = y3;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM256.CmpSwapGt(a2, b2);
                j += MM256.AVX2_FLOAT_STRIDE;

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = x1;
                a1 = y1;
                b1 = x2;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                j += MM256.AVX2_FLOAT_STRIDE;

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = x1;
                a1 = y1;
                b1 = y2;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                j += MM256.AVX2_FLOAT_STRIDE;

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = x1;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                j += MM256.AVX2_FLOAT_STRIDE;

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = y1;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                j += MM256.AVX2_FLOAT_STRIDE;

                MM256.Store(vs, j, x0);
                MM256.Store(vs, j + MM256.AVX2_FLOAT_STRIDE, y0);
            }
        }
    }
}
