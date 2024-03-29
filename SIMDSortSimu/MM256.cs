﻿using System;

namespace SIMDSortSimu {
    public class MM256 {
#pragma warning disable IDE1006
        public const uint AVX2_FLOAT_STRIDE = 8;
#pragma warning restore IDE1006

        readonly float[] vs;

        internal MM256() {
            this.vs = new float[AVX2_FLOAT_STRIDE];
        }

        internal MM256(float[] vs, uint i, uint count) : this() {
            Array.Copy(vs, i, this.vs, 0, count);
        }

        public static MM256 Load(float[] vs, uint i) {
            return new MM256(vs, i, AVX2_FLOAT_STRIDE);
        }

        public static MM256 MaskLoad(float[] vs, uint i, uint count) {
            return new MM256(vs, i, count);
        }

        public static void Store(float[] vs, uint i, MM256 x) {
            Array.Copy(x.vs, 0, vs, i, AVX2_FLOAT_STRIDE);
        }

        public static void MaskStore(float[] vs, uint i, uint count, MM256 x) {
            Array.Copy(x.vs, 0, vs, i, count);
        }

        public static MM256 Sort(MM256 x) {
            MM256 y = new();
            Array.Copy(x.vs, y.vs, AVX2_FLOAT_STRIDE);

            Array.Sort(y.vs);

            return y;
        }

        public static bool NeedsSort(MM256 x) {
            for (uint i = 0; i < AVX2_FLOAT_STRIDE - 1; i++) {
                if (!(x.vs[i] <= x.vs[i + 1]) && !float.IsNaN(x.vs[i])) {
                    return true;
                }
            }

            return false;
        }

        public static MM256 Perm(MM256 x) {
            MM256 y = new();
            Array.Copy(x.vs, 1, y.vs, 0, AVX2_FLOAT_STRIDE - 1);
            y.vs[^1] = x.vs[0];

            return y;
        }

        public static (bool swaped, uint index, MM256 a, MM256 b) CmpSwapGt(MM256 x, MM256 y) {
            bool swaped = false;
            uint index = 0;
            MM256 a = new(), b = new();

            for (uint i = 0; i < AVX2_FLOAT_STRIDE; i++) {
                if (!(x.vs[i] <= y.vs[i]) && !float.IsNaN(x.vs[i])) {
                    a.vs[i] = y.vs[i];
                    b.vs[i] = x.vs[i];

                    index |= (uint)(1 << (int)i);
                    swaped = true;
                }
                else {
                    a.vs[i] = x.vs[i];
                    b.vs[i] = y.vs[i];
                }
            }

            return (swaped, index, a, b);
        }

        public static (bool ismatch, uint index) CmpNeq(MM256 x, MM256 y) {
            uint index = 0;

            for (uint i = 0; i < AVX2_FLOAT_STRIDE; i++) {
                if (x.vs[i] != y.vs[i] && !float.IsNaN(x.vs[i])) {
                    index |= (uint)(1 << (int)i);
                }
            }

            return (index > 0, index);
        }

        public static (bool needsswap, uint index) NeedSwap(MM256 x, MM256 y) {
            uint index = 0;

            for (uint i = 0; i < AVX2_FLOAT_STRIDE; i++) {
                if (!(x.vs[i] <= y.vs[i]) && !float.IsNaN(x.vs[i])) {
                    index |= (uint)(1 << (int)i);
                }
            }

            return (index > 0, index);
        }

        public override string ToString() {
            return string.Join(", ", vs);
        }
    }
}
