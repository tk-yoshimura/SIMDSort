namespace SIMDSortSimu {
    public static class BacktrackSortN4 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM128.AVX1_FLOAT_STRIDE * 2) {
                return 0;
            }

            uint i = 0, e = n - 2 * MM128.AVX1_FLOAT_STRIDE;
            MM128 a = MM128.Load(vs, 0), b = MM128.Load(vs, MM128.AVX1_FLOAT_STRIDE);

            int swaps = 0;

            if (e <= 0) {
                (_, _, MM128 x, MM128 y) = MM128.CmpSwapGt(a, b);

                MM128.Store(vs, 0, x);
                MM128.Store(vs, MM128.AVX1_FLOAT_STRIDE, y);

                swaps++;

                return swaps;
            }

            while (true) {
                (bool swaped, _, MM128 x, MM128 y) = MM128.CmpSwapGt(a, b);

                swaps++;

                if (swaped) {
                    MM128.Store(vs, i, x);
                    MM128.Store(vs, i + MM128.AVX1_FLOAT_STRIDE, y);

                    if (i >= MM128.AVX1_FLOAT_STRIDE) {
                        i -= MM128.AVX1_FLOAT_STRIDE;
                        a = MM128.Load(vs, i);
                        b = x;
                        continue;
                    }
                    else if (i > 0) {
                        i = 0;
                        a = MM128.Load(vs, 0);
                        b = MM128.Load(vs, MM128.AVX1_FLOAT_STRIDE);
                        continue;
                    }
                    else {
                        i = MM128.AVX1_FLOAT_STRIDE;
                        if (i <= e) {
                            a = y;
                            b = MM128.Load(vs, MM128.AVX1_FLOAT_STRIDE * 2);
                            continue;
                        }
                    }
                }
                else if (i < e) {
                    i += MM128.AVX1_FLOAT_STRIDE;

                    if (i <= e) {
                        a = b;
                        b = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);
                        continue;
                    }
                }
                else {
                    break;
                }

                i = e;
                a = MM128.Load(vs, i);
                b = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);
            }

            return swaps;
        }
    }
}
