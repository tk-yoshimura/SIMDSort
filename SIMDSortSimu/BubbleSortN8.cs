namespace SIMDSortSimu {
    public static class BubbleSortN8 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 2) {
                return 0;
            }

            uint i = 0, e = n - 2 * MM256.AVX2_FLOAT_STRIDE;
            MM256 x = MM256.Load(vs, 0), y;

            int swaps = 0;

            while (true) {
                y = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);

                (_, uint index, MM256 a, MM256 b) = MM256.CmpSwapGt(x, y);

                swaps++;

                if (index < MM256.AVX2_FLOAT_STRIDE) {
                    MM256.Store(vs, i, a);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, b);

                    uint back = MM256.AVX2_FLOAT_STRIDE - index;

                    i = i >= back ? i - back : 0;

                    x = MM256.Load(vs, i);
                }
                else if (i < e) {
                    i += MM256.AVX2_FLOAT_STRIDE;

                    if (i <= e) {
                        x = y;
                    }
                    else {
                        i = e;
                        x = MM256.Load(vs, i);
                    }
                }
                else {
                    break;
                }
            }

            return swaps;
        }
    }
}
