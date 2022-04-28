using System;

namespace SIMDSortSimu {
    public static class Util {
        public static uint Bsf(uint n) {
            for (uint i = 0; i < 32; i++) {
                if ((n & 1u) == 1u) {
                    return i;
                }

                n >>= 1;
            }

            throw new ArgumentException(nameof(n));
        }
    }
}
