namespace SIMDSortSimu {
    public static class BacktrackSortN8 {

        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE * 2) {
                return 0;
            }

            uint i = 0, e = n - 2 * MM256.AVX2_FLOAT_STRIDE;
            MM256 a = MM256.Load(vs, 0), b;

            int swaps = 0;

            if (e <= 0) { 
                b = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                (_, _, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                MM256.Store(vs, i, x);
                MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, y);

                swaps++;

                return swaps;
            }

            while (true) {
                b = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);

                (_, uint index, MM256 x, MM256 y) = MM256.CmpSwapGt(a, b);

                swaps++;

                if (index < MM256.AVX2_FLOAT_STRIDE) {
                    MM256.Store(vs, i, x);
                    MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, y);

                    if (i == 0) {
                        i = MM256.AVX2_FLOAT_STRIDE;
                        if (i <= e) {
                            a = y;
                            continue;
                        }
                        else {
                            i = e;
                        }
                    }
                    else {
                        uint back = MM256.AVX2_FLOAT_STRIDE - index;

                        i = i >= back ? i - back : 0;
                    }
                }
                else if (i < e) {
                    i += MM256.AVX2_FLOAT_STRIDE;

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

                a = MM256.Load(vs, i);
            }

            return swaps;
        }
    }
}
