namespace SIMDSortSimu {
    public static class ParaCombSortN2x4 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE * 4) {
                return 0;
            }

            int swaps = 0;

            uint e = n - MM128.AVX1_FLOAT_STRIDE * 4;
            uint c = n % MM128.AVX1_FLOAT_STRIDE;

            MM128 a0, a1, b0, b1;
            MM128 x0, x1, y0, y1;

            for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
                a1 = MM128.Load(vs, i);
                b1 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                swaps++;

                a0 = x1;
                a1 = y1;
                b1 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 2);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                swaps++;

                b0 = MM128.Perm(x1);
                a1 = y1;
                b1 = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE * 3);
                (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                swaps++;

                for (j = i; j + MM128.AVX1_FLOAT_STRIDE <= e; j += MM128.AVX1_FLOAT_STRIDE) {
                    MM128.Store(vs, j, x0);
                    a0 = y0;
                    b0 = MM128.Perm(x1);
                    a1 = y1;
                    b1 = MM128.Load(vs, j + MM128.AVX1_FLOAT_STRIDE * 4);
                    (_, _, x0, y0) = MM128.CmpSwapGt(a0, b0);
                    (_, _, x1, y1) = MM128.CmpSwapGt(a1, b1);
                    swaps++;
                }

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
