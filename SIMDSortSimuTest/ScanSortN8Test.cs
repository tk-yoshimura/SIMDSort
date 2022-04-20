using Microsoft.VisualStudio.TestTools.UnitTesting;
using SIMDSortSimu;
using System;
using System.Linq;

namespace SIMDSortSimuTest {
    [TestClass]
    public class ScanSortN8Test {
        [TestMethod]
        public void RandomSortTest() {
            Random random = new();

            for (uint n = 8; n <= 64; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    float[] us = (float[])vs.Clone();

                    ScanSortN8.Iter(vs);
                    Array.Sort(us);

                    CollectionAssert.AreEqual(us, vs, $"n = {n}");
                }
            }
        }

        [TestMethod]
        public void InverseSortTest() {
            for (uint n = 8; n <= 64; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)idx).Reverse().ToArray();
                    float[] us = (float[])vs.Clone();

                    ScanSortN8.Iter(vs);
                    Array.Sort(us);

                    CollectionAssert.AreEqual(us, vs, $"n = {n}");
                }
            }
        }
    }
}
