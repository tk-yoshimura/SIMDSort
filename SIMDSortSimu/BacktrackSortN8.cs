namespace SIMDSortSimu {
    public static class BacktrackSortN8 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 2) {
                return 0;
            }

            uint i = 0, e = n - 2 * MM256.AVX2_FLOAT_STRIDE;
            MM256 a = MM256.Load(vs, 0), b = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE);

            int swaps = 0;

            if (e <= 0) {
                (_, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                MM256.Store(vs, 0, x);
                MM256.Store(vs, MM256.AVX2_FLOAT_STRIDE, y);

                swaps++;

                return swaps;
            }

            while (true) {
                (bool swaped, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                swaps++;

                if (swaped) {
                    MM256.Store(vs, i, x);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, y);

                    if (i >= MM256.AVX2_FLOAT_STRIDE) {
                        i -= MM256.AVX2_FLOAT_STRIDE;
                        a = MM256.Load(vs, i);
                        b = x;
                        continue;
                    }
                    else if (i > 0) {
                        i = 0;
                        a = MM256.Load(vs, 0);
                        b = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE);
                        continue;
                    }
                    else {
                        i = MM256.AVX2_FLOAT_STRIDE;
                        if (i <= e) {
                            a = y;
                            b = MM256.Load(vs, MM256.AVX2_FLOAT_STRIDE * 2);
                            continue;
                        }
                    }
                }
                else if (i < e) {
                    i += MM256.AVX2_FLOAT_STRIDE;

                    if (i <= e) {
                        a = b;
                        b = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                        continue;
                    }
                }
                else {
                    break;
                }

                i = e;
                a = MM256.Load(vs, i);
                b = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
            }

            return swaps;
        }
    }
}
