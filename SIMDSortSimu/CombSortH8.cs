namespace SIMDSortSimu {
    public static class CombSortH8 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 2) {
                return 0;
            }

            uint e = n - MM256.AVX2_FLOAT_STRIDE * 2;

            int swaps = 0;

            MM256 a = MM256.Load(vs, 0), b;

            uint i = 0;
            for (; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                b = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);

                (_, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                MM256.Store(vs, i, x);
                a = y;

                swaps++;
            }
            MM256.Store(vs, i, a);

            {
                a = MM256.Load(vs, e);
                b = MM256.Load(vs, e + MM256.AVX2_FLOAT_STRIDE);

                (_, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                MM256.Store(vs, e, x);
                MM256.Store(vs, e + MM256.AVX2_FLOAT_STRIDE, y);

                swaps++;
            }

            return swaps;
        }
    }
}
