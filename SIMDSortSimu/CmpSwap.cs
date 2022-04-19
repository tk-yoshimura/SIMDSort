using System;

namespace SIMDSortSimu {
    static class CmpSwap {
        public static bool Element1(float[] vs, uint i, uint h) {
            float a = vs[i], b = vs[i + h];

            if (a > b) {
                vs[i] = b;
                vs[i + h] = a;

                return true;
            }

            return false;
        }

        public static bool ElementN(float[] vs, uint n, uint i, uint h) {
            if (h < n) {
                throw new ArgumentException(nameof(h));
            }

            bool swaped = false;

            for (uint j = 0; j < n; j++) {
                swaped |= Element1(vs, i + j, h);
            }

            return swaped;
        }

        public static uint ElementIndexed(float[] vs, uint n, uint i, uint h) {
            if (h < n) {
                throw new ArgumentException(nameof(h));
            }

            uint index = n;

            for (uint j = 0; j < n; j++) {
                if (Element1(vs, i + j, h) && index >= n) {
                    index = j;
                }
            }

            return index;
        }
    }
}
