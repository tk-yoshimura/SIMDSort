using Microsoft.VisualStudio.TestTools.UnitTesting;
using SIMDSortSimu;
using System;
using System.Linq;

namespace SIMDSortSimuTest {
    [TestClass]
    public class ParaCombSortN4x8Test {
        [TestMethod]
        public void RandomSortTest() {
            Random random = new();

            for (uint n = 64; n <= 1024; n++) {

                float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                float[] us = (float[])vs.Clone();

                ParaCombSortN4x8.Iter(vs);
                Array.Sort(us);
                Array.Sort(vs);

                CollectionAssert.AreEqual(us, vs, $"n = {n}");
            }
        }

        [TestMethod]
        public void InverseSortTest() {
            for (uint n = 64; n <= 1024; n++) {

                float[] vs = (new float[n]).Select((_, idx) => (float)idx).Reverse().ToArray();
                float[] us = (float[])vs.Clone();

                ParaCombSortN4x8.Iter(vs);
                Array.Sort(us);
                Array.Sort(vs);

                CollectionAssert.AreEqual(us, vs, $"n = {n}");
            }
        }
    }
}
