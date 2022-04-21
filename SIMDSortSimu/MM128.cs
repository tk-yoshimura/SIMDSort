using System;

namespace SIMDSortSimu {
    public class MM128 {
#pragma warning disable IDE1006
        public const uint AVX1_FLOAT_STRIDE = 4;
#pragma warning restore IDE1006

        readonly float[] vs;

        internal MM128() {
            this.vs = new float[AVX1_FLOAT_STRIDE];
        }

        internal MM128(float[] vs, uint i, uint count) : this() {
            Array.Copy(vs, i, this.vs, 0, count);
        }

        public static MM128 Load(float[] vs, uint i) {
            return new MM128(vs, i, AVX1_FLOAT_STRIDE);
        }

        public static MM128 MaskLoad(float[] vs, uint i, uint count) {
            return new MM128(vs, i, count);
        }

        public static void Store(float[] vs, uint i, MM128 x) {
            Array.Copy(x.vs, 0, vs, i, AVX1_FLOAT_STRIDE);
        }

        public static void MaskStore(float[] vs, uint i, uint count, MM128 x) {
            Array.Copy(x.vs, 0, vs, i, count);
        }

        public static MM128 Sort(MM128 x) {
            MM128 y = new();
            Array.Copy(x.vs, y.vs, AVX1_FLOAT_STRIDE);

            Array.Sort(y.vs);

            return y;
        }

        public static bool NeedsSort(MM128 x) {
            for (uint i = 0; i < AVX1_FLOAT_STRIDE - 1; i++) {
                if (!(x.vs[i] <= x.vs[i + 1]) && !float.IsNaN(x.vs[i])) {
                    return true;
                }
            }

            return false;
        }

        public static (bool swaped, uint index, MM128 a, MM128 b) CmpSwapGt(MM128 x, MM128 y) {
            bool swaped = false;
            uint index = AVX1_FLOAT_STRIDE;
            MM128 a = new(), b = new();

            for (uint i = 0; i < AVX1_FLOAT_STRIDE; i++) {
                if (!(x.vs[i] <= y.vs[i]) && !float.IsNaN(x.vs[i])) {
                    a.vs[i] = y.vs[i];
                    b.vs[i] = x.vs[i];

                    if (!swaped) {
                        index = i;
                    }
                    swaped = true;
                }
                else {
                    a.vs[i] = x.vs[i];
                    b.vs[i] = y.vs[i];
                }
            }

            return (swaped, index, a, b);
        }

        public static (bool ismatch, uint index) CmpEq(MM128 x, MM128 y) {
            for (uint i = 0; i < AVX1_FLOAT_STRIDE; i++) {
                if (x.vs[i] != y.vs[i] && !float.IsNaN(x.vs[i])) {
                    return (false, i);
                }
            }

            return (true, AVX1_FLOAT_STRIDE);
        }
    }
}
