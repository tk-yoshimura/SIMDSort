namespace SIMDSortSimu {
    public static class AvxSortN8 {
        public static (int swaps, int sorts) Sort(float[] vs) {
            uint n = (uint)vs.Length;
            
            int swaps = 0, sorts = 0;

            for (uint h = n * 10 / 13; h > 8; h = h * 10 / 13) {
                swaps += CombSortN8.Iter(vs, h);
            }
            
            swaps += BacktrackSortN8.Iter(vs);
            sorts += ScanSortN8.Iter(vs);

            return (swaps, sorts);
        }
    }
}
