namespace SIMDSortSimu {
    public static class BacktrackSortN4 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE * 2) {
                return 0;
            }

            uint i = 0, e = n - 2 * MM128.AVX1_FLOAT_STRIDE;
            MM128 a = MM128.Load(vs, 0), b;

            int swaps = 0;

            if (e <= 0) {
                b = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);
                (_, _, MM128 x, MM128 y) = MM128.CmpSwapGt(a, b);

                MM128.Store(vs, i, x);
                MM128.Store(vs, i + MM128.AVX1_FLOAT_STRIDE, y);

                swaps++;

                return swaps;
            }

            while (true) {
                b = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);

                (_, uint index, MM128 x, MM128 y) = MM128.CmpSwapGt(a, b);

                swaps++;

                if (index < MM128.AVX1_FLOAT_STRIDE) {
                    MM128.Store(vs, i, x);
                    MM128.Store(vs, i + MM128.AVX1_FLOAT_STRIDE, y);

                    if (i == 0) {
                        i = MM128.AVX1_FLOAT_STRIDE;
                        if (i <= e) {
                            a = y;
                            continue;
                        }
                        else {
                            i = e;
                        }
                    }
                    else {
                        uint back = MM128.AVX1_FLOAT_STRIDE - index;

                        i = i >= back ? i - back : 0;
                    }
                }
                else if (i < e) {
                    i += MM128.AVX1_FLOAT_STRIDE;

                    if (i <= e) {
                        a = b;
                        continue;
                    }
                    else {
                        i = e;
                    }
                }
                else {
                    break;
                }

                a = MM128.Load(vs, i);
            }

            return swaps;
        }
    }
}
