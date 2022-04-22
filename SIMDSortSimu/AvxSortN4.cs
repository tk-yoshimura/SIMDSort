namespace SIMDSortSimu {
    public static class AvxSortN4 {
        public static (int swaps, int sorts) Sort(float[] vs) {
            uint n = (uint)vs.Length;

            int swaps = 0, sorts = 0;

            for (uint h = n * 10 / 13; h > 4; h = h * 10 / 13) {
                swaps += CombSortN4.Iter(vs, h);
            }

            swaps += BacktrackSortN4.Iter(vs);
            sorts += ScanSortN4.Iter(vs);

            return (swaps, sorts);
        }
    }
}
