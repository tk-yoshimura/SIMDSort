namespace SIMDSortSimu {
    public static class BatchSortN4 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE) {
                return 0;
            }

            int swaps = 0;
            uint e = n - MM128.AVX1_FLOAT_STRIDE;

            MM128 x, y;

            for (int iter = 0; iter < 2; iter++) {
                for (uint i = 0; i < e; i += MM128.AVX1_FLOAT_STRIDE) {
                    x = MM128.Load(vs, i);
                    y = MM128.Sort(x);
                    MM128.Store(vs, i, y);

                    swaps++;
                }
                {
                    x = MM128.Load(vs, e);
                    y = MM128.Sort(x);
                    MM128.Store(vs, e, y);

                    swaps++;
                }

                for (uint i = 4; i < e; i += MM128.AVX1_FLOAT_STRIDE) {
                    x = MM128.Load(vs, i);
                    y = MM128.Sort(x);
                    MM128.Store(vs, i, y);

                    swaps++;
                }
                {
                    x = MM128.Load(vs, e);
                    y = MM128.Sort(x);
                    MM128.Store(vs, e, y);

                    swaps++;
                }
            }

            return swaps;
        }
    }
}
