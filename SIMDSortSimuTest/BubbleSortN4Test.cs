using Microsoft.VisualStudio.TestTools.UnitTesting;
using SIMDSortSimu;
using System;
using System.Linq;

namespace SIMDSortSimuTest {
    [TestClass]
    public class BubbleSortN4Test {
        [TestMethod]
        public void RandomSortTest() {
            Random random = new();

            for (uint n = 8; n <= 32; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();

                    BubbleSortN4.Iter(vs);

                    for (uint j = 0; j < n - 4; j++) {
                        Assert.IsTrue(vs[j] <= vs[j + 4]);
                    }
                }
            }
        }

        [TestMethod]
        public void InverseSortTest() {
            for (uint n = 8; n <= 32; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)idx).Reverse().ToArray();

                    BubbleSortN4.Iter(vs);

                    for (uint j = 0; j < n - 4; j++) {
                        Assert.IsTrue(vs[j] <= vs[j + 4]);
                    }
                }
            }
        }
    }
}
