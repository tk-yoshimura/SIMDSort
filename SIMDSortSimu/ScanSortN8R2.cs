using System;

namespace SIMDSortSimu {
    public static class ScanSortN8R2 {
        public static int Iter(float[] vs) {
            uint n = (uint)vs.Length;

            if (n < MM256.AVX2_FLOAT_STRIDE) {
                Array.Sort(vs);
                return 0;
            }

            int sorts = 0;

            uint e = n - MM256.AVX2_FLOAT_STRIDE;

            uint indexes;
            MM256 x0, x1, x2, x3, y0, y1, y2, y3;

            {
                uint i = 0;
                while (true) {
                    if (i + MM256.AVX2_FLOAT_STRIDE * 4 + 1 <= n) {
                        x0 = MM256.Load(vs, i);
                        x1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                        x2 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 2);
                        x3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 3);
                        y0 = MM256.Load(vs, i + 1);
                        y1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE + 1);
                        y2 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 2 + 1);
                        y3 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE * 3 + 1);

                        indexes = MM256.NeedSwap(x0, y0).index | MM256.NeedSwap(x1, y1).index << 8
                                | MM256.NeedSwap(x2, y2).index << 16 | MM256.NeedSwap(x3, y3).index << 24;

                        if (indexes == 0u) {
                            i += MM256.AVX2_FLOAT_STRIDE * 4;
                            continue;
                        }
                    }
                    else if (i + MM256.AVX2_FLOAT_STRIDE * 2 + 1 <= n) {
                        x0 = MM256.Load(vs, i);
                        x1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE);
                        y0 = MM256.Load(vs, i + 1);
                        y1 = MM256.Load(vs, i + MM256.AVX2_FLOAT_STRIDE + 1);

                        indexes = MM256.NeedSwap(x0, y0).index | MM256.NeedSwap(x1, y1).index << 8;

                        if (indexes == 0u) {
                            i += MM256.AVX2_FLOAT_STRIDE * 2;
                            continue;
                        }
                    }
                    else if (i + MM256.AVX2_FLOAT_STRIDE + 1 <= n) {
                        x0 = MM256.Load(vs, i);
                        y0 = MM256.Load(vs, i + 1);

                        indexes = MM256.NeedSwap(x0, y0).index;

                        if (indexes == 0u) {
                            i += MM256.AVX2_FLOAT_STRIDE;
                            continue;
                        }
                    }
                    else {
                        i = e;

                        x0 = MM256.Load(vs, i);

                        if (!MM256.NeedsSort(x0)) {
                            break;
                        }

                        y0 = MM256.Sort(x0);
                        MM256.Store(vs, i, y0);
                        sorts++;

                        indexes = MM256.CmpNeq(x0, y0).index;

                        if ((indexes & 1) == 0 || i == 0) {
                            break;
                        }
                    }

                    uint index = Util.Bsf(indexes);

                    if (index >= MM256.AVX2_FLOAT_STRIDE - 2) {
                        uint forward = index - (MM256.AVX2_FLOAT_STRIDE - 2);
                        i += forward;
                    }
                    else {
                        uint backward = (MM256.AVX2_FLOAT_STRIDE - 2) - index;
                        i = (i > backward) ? i - backward : 0;
                    }

                    x0 = MM256.Load(vs, i);

                    while (true) {
                        y0 = MM256.Sort(x0);
                        MM256.Store(vs, i, y0);

                        sorts++;

                        indexes = MM256.CmpNeq(x0, y0).index;
                        if ((indexes & 1u) == 1u && i > 0) {
                            uint backward = MM256.AVX2_FLOAT_STRIDE - 2;
                            i = (i > backward) ? i - backward : 0;

                            x0 = MM256.Load(vs, i);
                        
                            if (MM256.NeedsSort(x0)) {
                                continue;
                            }
                        }

                        uint forward = MM256.AVX2_FLOAT_STRIDE - 1;
                        i += forward;
                        break;
                    }
                }
            }

            return sorts;
        }
    }
}
