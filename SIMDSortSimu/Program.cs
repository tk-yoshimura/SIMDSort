using System;
using System.IO;
using System.Linq;

namespace SIMDSortSimu {
    internal class Program {
        static void Main(string[] args) {
            //uint n = 777;
            //Random random = new Random();
            //
            //Directory.CreateDirectory("../../history/");
            //
            //float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();
            
            //for (uint h = n * 10 / 13; h > 8; h = h * 10 / 13) {
            //    CombSortN8.Iter(vs, h);
            //
            //    using StreamWriter sw = new($"../../history/iter_{n}_{h}.csv");
            //
            //    for (int i = 0; i < n; i++) {
            //        sw.WriteLine(vs[i]);
            //    }
            //}
            //
            //foreach (uint h in new[] { 8 }) {
            //    BubbleSortN8.Iter(vs);
            //
            //    using StreamWriter sw = new($"../../history/iter_{n}_{h}.csv");
            //
            //    for (int i = 0; i < n; i++) {
            //        sw.WriteLine(vs[i]);
            //    }
            //}
            //
            //foreach (uint h in new[] { 6 }) {
            //    CombSortN4.Iter(vs, h);
            //
            //    using StreamWriter sw = new($"../../history/iter_{n}_{h}.csv");
            //
            //    for (int i = 0; i < n; i++) {
            //        sw.WriteLine(vs[i]);
            //    }
            //}
            //
            //foreach (uint h in new[] { 4 }) {
            //    BubbleSortN4.Iter(vs);
            //
            //    using StreamWriter sw = new($"../../history/iter_{n}_{h}.csv");
            //
            //    for (int i = 0; i < n; i++) {
            //        sw.WriteLine(vs[i]);
            //    }
            //}
            //
            //foreach (uint h in new[] { 1 }) {
            //    ScanSortN8.Iter(vs);
            //
            //    using StreamWriter sw = new($"../../history/iter_{n}_{h}.csv");
            //
            //    for (int i = 0; i < n; i++) {
            //        sw.WriteLine(vs[i]);
            //    }
            //}

            uint n = 1023;
            Random random = new Random();

            Directory.CreateDirectory("../../history/");

            float[] vs = (new float[n]).Select((_, idx) => (float)random.NextDouble()).ToArray();
            float[] us = (float[])vs.Clone();

            Array.Sort(us);

            foreach (uint h in new[] { 1 }) {
                ScanSortN4.Iter(vs);

                using StreamWriter sw = new($"../../history/iter_{n}_{h}.csv");

                for (int i = 0; i < n; i++) {
                    sw.WriteLine($"{vs[i]}, {us[i]}");

                    if (vs[i] != us[i]) { 
                        Console.WriteLine("ERR");
                        Console.Read();
                    }
                }
            }


            Console.WriteLine("END");
            Console.Read();
        }
    }
}
