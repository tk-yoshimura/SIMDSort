namespace SIMDSortSimu {
    static class CombSort {

        public static void Iter(float[] vs, uint h) {
            uint n = (uint)vs.Length;

            if (n >= h * 2) {
                uint e = n - h * 2;

                for (uint i = 0; i < e; i += h) {
                    CmpSwap.ElementN(vs, h, i, h);
                }
                CmpSwap.ElementN(vs, h, e, h);
            }
        }
    }
}
