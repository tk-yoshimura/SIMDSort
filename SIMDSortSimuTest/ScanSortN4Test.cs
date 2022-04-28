using Microsoft.VisualStudio.TestTools.UnitTesting;
using SIMDSortSimu;
using System;
using System.Linq;

namespace SIMDSortSimuTest {
    [TestClass]
    public class ScanSortN4Test {
        [TestMethod]
        public void RandomSortTest() {
            Random random = new();

            for (uint n = 4; n <= 32; n++) {
                for (uint i = 0; i < 8; i++) {

                    float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                    float[] us = (float[])vs.Clone();

                    ScanSortN4.Iter(vs);
                    Array.Sort(us);

                    CollectionAssert.AreEqual(us, vs, $"n = {n}");
                }
            }
        }

        [TestMethod]
        public void InverseSortTest() {
            for (uint n = 4; n <= 32; n++) {

                float[] vs = (new float[n]).Select((_, idx) => (float)idx).Reverse().ToArray();
                float[] us = (float[])vs.Clone();

                ScanSortN4.Iter(vs);
                Array.Sort(us);

                CollectionAssert.AreEqual(us, vs, $"n = {n}");
            }
        }
    }
}
