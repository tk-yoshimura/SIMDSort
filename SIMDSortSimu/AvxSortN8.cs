namespace SIMDSortSimu {
    public static class AvxSortN8 {
        public static (int swaps, int sorts) Sort(float[] vs) {
            uint n = (uint)vs.Length;
            uint h;

            int swaps = 0, sorts = 0;

            for (h = (uint)(n * 10L / 13L); h > 33; h = (uint)(h * 10L / 13L)) {
                swaps += CombSortH33plus.Iter(vs, h);
            }
            if (h >= 32) {
                swaps += CombSortH32.Iter(vs);
                h = h * 10 / 13;
            }
            for (; h > 25; h = h * 10 / 13) {
                swaps += CombSortH25to31.Iter(vs, h);
            }
            if (h >= 24) {
                swaps += CombSortH24.Iter(vs);
                h = h * 10 / 13;
            }
            for (; h > 17; h = h * 10 / 13) {
                swaps += CombSortH17to23.Iter(vs, h);
            }
            if (h >= 16) {
                swaps += CombSortH16.Iter(vs);
                h = h * 10 / 13;
            }
            for (; h > 9; h = h * 10 / 13) {
                swaps += CombSortH9to15.Iter(vs, h);
            }
            if (h >= 8) {
                swaps += CombSortH8.Iter(vs);
            }

            swaps += BacktrackSortN8.Iter(vs);
            sorts += BatchSortN8.Iter(vs);
            sorts += ScanSortN8.Iter(vs);

            return (swaps, sorts);
        }
    }
}
