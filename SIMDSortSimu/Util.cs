using System;

namespace SIMDSortSimu {
    static class Util {
        public static uint OverPower2(uint n) {
            for (uint i = 1; i <= 0x80000000u; i *= 2) {
                if (n <= i) {
                    return i;
                }
            }

            throw new OverflowException();
        }
    }
}
