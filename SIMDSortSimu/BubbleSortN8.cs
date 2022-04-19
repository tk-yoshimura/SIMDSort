using System.IO;

namespace SIMDSortSimu {
    static class BubbleSortN8 {

        public static void Iter(float[] vs) {
            uint n = (uint)vs.Length;

            using StreamWriter sw = new($"../../history/swap_{n}_8.csv");

            if (n >= MM256.AVX2_FLOAT_STRIDE * 2) {
                uint i = 0, e = n - MM256.AVX2_FLOAT_STRIDE, f = e - MM256.AVX2_FLOAT_STRIDE;
                MM256 x = MM256.Load(vs, 0), y;

                while (i < e) {
                    if (i > f) {
                        i = f;
                        x = MM256.Load(vs, i);
                    }

                    y = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);

                    (_, uint index, MM256 a, MM256 b) = MM256.CmpSwapGt(x, y);

                    sw.WriteLine($"{i} <-> {i + MM256.AVX2_FLOAT_STRIDE} {index}");

                    if (index >= MM256.AVX2_FLOAT_STRIDE) {
                        i += MM256.AVX2_FLOAT_STRIDE;

                        x = y;
                    }
                    else {
                        MM256.Store(vs, i, a);
                        MM256.Store(vs, i + MM256.AVX2_FLOAT_STRIDE, b);

                        uint back = MM256.AVX2_FLOAT_STRIDE - index;

                        i = i >= back ? i - back : 0;

                        x = MM256.Load(vs, i);
                    }
                }
            }
        }
    }
}
