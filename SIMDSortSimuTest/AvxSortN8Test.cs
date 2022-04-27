using Microsoft.VisualStudio.TestTools.UnitTesting;
using SIMDSortSimu;
using System;
using System.Linq;

namespace SIMDSortSimuTest {
    [TestClass]
    public class AvxSortN8Test {
        [TestMethod]
        public void RandomSortTest() {
            Random random = new();

            Console.WriteLine("n,swaps,sorts");

            for (uint n = 8; n <= 2048; n++) {

                float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();
                float[] us = (float[])vs.Clone();

                (int swaps, int sorts) = AvxSortN8.Sort(vs);
                Array.Sort(us);

                CollectionAssert.AreEqual(us, vs, $"n = {n}");

                Console.WriteLine($"{n},{swaps},{sorts}");
            }
        }

        [TestMethod]
        public void InverseSortTest() {
            Console.WriteLine("n,swaps,sorts");

            for (uint n = 8; n <= 2048; n++) {

                float[] vs = (new float[n]).Select((_, idx) => (float)idx).Reverse().ToArray();
                float[] us = (float[])vs.Clone();

                (int swaps, int sorts) = AvxSortN8.Sort(vs);
                Array.Sort(us);

                CollectionAssert.AreEqual(us, vs, $"n = {n}");

                Console.WriteLine($"{n},{swaps},{sorts}");
            }
        }

        [TestMethod]
        public void TriWaveTest() {
            Console.WriteLine("n,swaps,sorts");

            for (uint n = 8; n <= 1024; n++) {

                float[] vs = (new float[n]).Select((_, idx) => (float)(idx % 8 + idx / 8)).ToArray();
                float[] us = (float[])vs.Clone();

                (int swaps, int sorts) = AvxSortN8.Sort(vs);
                Array.Sort(us);

                CollectionAssert.AreEqual(us, vs, $"n = {n}");

                Console.WriteLine($"{n},{swaps},{sorts}");

            }
        }
    }
}
