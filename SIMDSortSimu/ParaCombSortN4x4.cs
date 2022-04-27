namespace SIMDSortSimu {
    public static class ParaCombSortN4x4 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE * 8) {
                return 0;
            }

            int swaps = 0;

            uint e = n - MM128.AVX1_FLOAT_STRIDE * 8;
            uint c = n % MM128.AVX1_FLOAT_STRIDE;

            MM128 a0, a1, a2, a3, b0, b1, b2, b3;
            MM128 x0, x1, x2, x3, y0, y1, y2, y3;

            for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
                a3 = MM128.Load(vs, i);
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                a2 = x3;
                a3 = y3;
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 2);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                b2 = MM128.Perm(x3);
                a3 = y3;
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 3);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                a1 = x2;
                a2 = y2;
                b2 = MM128.Perm(x3);
                a3 = y3;
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 4);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                b1 = MM128.Perm(x2);
                a2 = y2;
                b2 = MM128.Perm(x3);
                a3 = y3;
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 5);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                a0 = x1;
                a1 = y1;
                b1 = MM128.Perm(x2);
                a2 = y2;
                b2 = MM128.Perm(x3);
                a3 = y3;
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 6);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                b0 = MM128.Perm(x1);
                a1 = y1;
                b1 = MM128.Perm(x2);
                a2 = y2;
                b2 = MM128.Perm(x3);
                a3 = y3;
                b3 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 7);
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                swaps++;

                for (j = i; j + MM128.AVX1_FLOAT_STRIDE <= e; j += MM128.AVX1_FLOAT_STRIDE) {
                    MM128.Store(vs, j, x0);
                    a0 = y0;
                    b0 = MM128.Perm(x1);
                    a1 = y1;
                    b1 = MM128.Perm(x2);
                    a2 = y2;
                    b2 = MM128.Perm(x3);
                    a3 = y3;
                    b3 = MM128.Load(vs, j + MM128.AVX1_FLOAT_STRIDE * 8);
                    (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                    (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                    (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                    (_, _, x3, y3) = MM128.CmpSwapGt(a3, b3);
                    swaps++;
                }

                MM128.Store(vs, j, x0);
                a0 = y0;
                b0 = MM128.Perm(x1);
                a1 = y1;
                b1 = MM128.Perm(x2);
                a2 = y2;
                b2 = MM128.Perm(x3);
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                j += MM128.AVX1_FLOAT_STRIDE;
                swaps++;

                MM128.Store(vs, j, x0);
                a0 = y0;
                b0 = MM128.Perm(x1);
                a1 = y1;
                b1 = MM128.Perm(x2);
                a2 = y2;
                b2 = y3;
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                (_, _, x2, y2) = MM128.CmpSwapGt(a2, b2);
                j += MM128.AVX1_FLOAT_STRIDE;
                swaps++;

                MM128.Store(vs, j, x0);
                a0 = y0;
                b0 = MM128.Perm(x1);
                a1 = y1;
                b1 = MM128.Perm(x2);
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                j += MM128.AVX1_FLOAT_STRIDE;
                swaps++;

                MM128.Store(vs, j, x0);
                a0 = y0;
                b0 = MM128.Perm(x1);
                a1 = y1;
                b1 = y2;
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                j += MM128.AVX1_FLOAT_STRIDE;
                swaps++;

                MM128.Store(vs, j, x0);
                a0 = y0;
                b0 = MM128.Perm(x1);
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                j += MM128.AVX1_FLOAT_STRIDE;
                swaps++;

                MM128.Store(vs, j, x0);
                a0 = y0;
                b0 = y1;
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                j += MM128.AVX1_FLOAT_STRIDE;
                swaps++;

                MM128.Store(vs, j, x0);
                MM128.Store(vs, j + MM128.AVX1_FLOAT_STRIDE, y0);
            }

            return swaps;
        }
    }
}
