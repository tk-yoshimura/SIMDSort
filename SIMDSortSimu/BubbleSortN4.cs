using System.IO;

namespace SIMDSortSimu {
    static class BubbleSortN4 {

        public static void Iter(float[] vs) {
            uint n = (uint)vs.Length;

            using StreamWriter sw = new($"../../history/swap_{n}_4.csv");

            if (n >= MM128.AVX1_FLOAT_STRIDE * 2) {
                uint i = 0, e = n - MM128.AVX1_FLOAT_STRIDE, f = e - MM128.AVX1_FLOAT_STRIDE;
                MM128 x = MM128.Load(vs, 0), y;

                while (i < e) {
                    if (i > f) {
                        i = f;
                        x = MM128.Load(vs, i);
                    }

                    y = MM128.Load(vs, i + MM128.AVX1_FLOAT_STRIDE);

                    (_, uint index, MM128 a, MM128 b) = MM128.CmpSwapGt(x, y);

                    sw.WriteLine($"{i} <-> {i + MM128.AVX1_FLOAT_STRIDE} {index}");

                    if (index >= MM128.AVX1_FLOAT_STRIDE) {
                        i += MM128.AVX1_FLOAT_STRIDE;

                        x = y;
                    }
                    else {
                        MM128.Store(vs, i, a);
                        MM128.Store(vs, i + MM128.AVX1_FLOAT_STRIDE, b);

                        uint back = MM128.AVX1_FLOAT_STRIDE - index;

                        i = i >= back ? i - back : 0;

                        x = MM128.Load(vs, i);
                    }
                }
            }
        }
    }
}
