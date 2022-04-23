namespace SIMDSortSimu {
    public static class BatchSortN8 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE) {
                return 0;
            }

            int swaps = 0;
            uint e = n - MM256.AVX2_FLOAT_STRIDE;

            MM256 x, y;

            for (int iter = 0; iter < 2; iter++) {
                for (uint i = 0; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                    x = MM256.Load(vs, i);
                    y = MM256.Sort(x);
                    MM256.Store(vs, i, y);

                    swaps++;
                }
                {
                    x = MM256.Load(vs, e);
                    y = MM256.Sort(x);
                    MM256.Store(vs, e, y);

                    swaps++;
                }

                for (uint i = 4; i < e; i += MM256.AVX2_FLOAT_STRIDE) {
                    x = MM256.Load(vs, i);
                    y = MM256.Sort(x);
                    MM256.Store(vs, i, y);

                    swaps++;
                }
                {
                    x = MM256.Load(vs, e);
                    y = MM256.Sort(x);
                    MM256.Store(vs, e, y);

                    swaps++;
                }
            }

            return swaps;
        }
    }
}
