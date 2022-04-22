using Microsoft.VisualStudio.TestTools.UnitTesting;
using SIMDSortSimu;
using System;
using System.Linq;

namespace SIMDSortSimuTest {
    [TestClass]
    public class BacktrackSortN8Test {
        [TestMethod]
        public void RandomSortTest() {
            Random random = new();

            for (uint n = 16; n <= 64; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();

                    BacktrackSortN8.Iter(vs);

                    for (uint j = 0; j < n - 8; j++) {
                        Assert.IsTrue(vs[j] <= vs[j + 8]);
                    }
                }
            }
        }

        [TestMethod]
        public void InverseSortTest() {
            for (uint n = 16; n <= 64; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)idx).Reverse().ToArray();

                    BacktrackSortN8.Iter(vs);

                    for (uint j = 0; j < n - 8; j++) {
                        Assert.IsTrue(vs[j] <= vs[j + 8]);
                    }
                }
            }
        }
    }
}
