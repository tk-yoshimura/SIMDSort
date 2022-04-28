namespace SIMDSortSimu {
    public static class AvxSortN8 {
        public static (int swaps, int sorts) Sort(float[] vs) {
            uint n = (uint)vs.Length;
            uint h;

            int swaps = 0, sorts = 0;

            if (n >= MM256.AVX2_FLOAT_STRIDE * 8) {
                for (h = (uint)(n * 10L / 13L); h > 32; h = (uint)(h * 10L / 13L)) {
                    swaps += CombSortH33plus.Iter(vs, h);
                }
                swaps += CombSortH32.Iter(vs);
                swaps += ParaCombSortN4x8.Iter(vs);

                sorts += BatchSortN8.Iter(vs) / 4;
                sorts += ScanSortN8R2.Iter(vs);
            }
            else {
                swaps += ParaCombSortN2x8.Iter(vs);
                swaps += BacktrackSortN8.Iter(vs);

                sorts += BatchSortN8.Iter(vs) / 4;
                sorts += ScanSortN8R2.Iter(vs);
            }

            return (swaps, sorts);
        }
    }
}
