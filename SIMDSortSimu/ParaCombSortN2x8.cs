namespace SIMDSortSimu {
    public static class ParaCombSortN2x8 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 4) {
                return 0;
            }

            int swaps = 0;

            uint e = n - MM256.AVX2_FLOAT_STRIDE * 4;
            uint c = n % MM256.AVX2_FLOAT_STRIDE;

            MM256 a0, a1, b0, b1;
            MM256 x0, x1, y0, y1;

            for (uint k = 0, i = 0, j; k < 2; k++, i += c) {
                a1 = MM256.Load(vs, i);
                b1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                swaps++;

                a0 = x1;
                a1 = y1;
                b1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 2);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                swaps++;

                b0 = MM256.Perm(x1);
                a1 = y1;
                b1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 3);
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                swaps++;

                for (j = i; j + MM256.AVX2_FLOAT_STRIDE <= e; j += MM256.AVX2_FLOAT_STRIDE) {
                    MM256.Store(vs, j, x0);
                    a0 = y0;
                    b0 = MM256.Perm(x1);
                    a1 = y1;
                    b1 = MM256.Load(vs, j + MM256.AVX2_FLOAT_STRIDE * 4);
                    (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                    (_, _, x1, y1) = MM256.CmpSwapGt(a1, b1);
                    swaps++;
                }

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = MM256.Perm(x1);
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                j += MM256.AVX2_FLOAT_STRIDE;
                swaps++;

                MM256.Store(vs, j, x0);
                a0 = y0;
                b0 = y1;
                (_, _, x0, y0) = MM256.CmpSwapGt(a0, b0);
                j += MM256.AVX2_FLOAT_STRIDE;
                swaps++;

                MM256.Store(vs, j, x0);
                MM256.Store(vs, j + MM256.AVX2_FLOAT_STRIDE, y0);
            }

            return swaps;
        }
    }
}
