namespace SIMDSortSimu {
    public static class BubbleSortN4 {

        public static void Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE * 2) {
                return;
            }

            uint i = 0, e = n - 2 * MM128.AVX1_FLOAT_STRIDE;
            MM128 x = MM128.Load(vs, 0), y;

            while (true) {
                y = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);

                (_, uint index, MM128 a, MM128 b) = MM128.CmpSwapGt(x, y);

                if (index < MM128.AVX1_FLOAT_STRIDE) {
                    MM128.Store(vs, i, a);
                    MM128.Store(vs, i + MM128.AVX1_FLOAT_STRIDE, b);

                    uint back = MM128.AVX1_FLOAT_STRIDE - index;

                    i = i >= back ? i - back : 0;

                    x = MM128.Load(vs, i);
                }
                else if (i < e) {
                    i += MM128.AVX1_FLOAT_STRIDE;

                    if (i <= e) {
                        x = y;
                    }
                    else {
                        i = e;
                        x = MM128.Load(vs, i);
                    }
                }
                else {
                    break;
                }
            }
        }
    }
}
