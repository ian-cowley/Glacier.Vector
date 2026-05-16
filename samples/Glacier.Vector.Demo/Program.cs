using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

namespace Glacier.Vector.Demo
{
    internal class Program
    {
        private const int Dimensions = 1536; // Standard LLM embedding size
        private const int VectorCount = 100_000; // ~600MB of RAM required

        static void Main(string[] args)
        {
            Console.WriteLine("==========================================");
            Console.WriteLine(" Glacier.Vector | SIMD Performance Engine ");
            Console.WriteLine("==========================================\n");

            // Initialize storage and search index
            using var storage = new InMemoryVectorStorage(Dimensions);
            using var index = new VectorIndex(storage);

            Console.WriteLine($"[1] Initializing In-Memory Storage...");
            Console.WriteLine($"    Dimensions: {Dimensions}");
            Console.WriteLine($"    Target Count: {VectorCount:N0}\n");

            Console.WriteLine("[2] Generating and loading synthetic vectors...");
            var sw = Stopwatch.StartNew();

            // We use a fixed seed so the benchmark is deterministic
            var rng = new Random(42);
            float[] tempVector = new float[Dimensions];

            for (int i = 0; i < VectorCount; i++)
            {
                // Generate a random vector
                float sumSq = 0f;
                for (int d = 0; d < Dimensions; d++)
                {
                    float val = (float)(rng.NextDouble() * 2.0 - 1.0);
                    tempVector[d] = val;
                    sumSq += val * val;
                }

                // Normalize the vector (so Dot Product == Cosine Similarity)
                float length = (float)Math.Sqrt(sumSq);
                for (int d = 0; d < Dimensions; d++)
                {
                    tempVector[d] /= length;
                }

                // Add to the database
                index.Add(tempVector, $"Document_Chunk_{i}");

                if (i > 0 && i % 25_000 == 0)
                {
                    Console.WriteLine($"    Loaded {i:N0} vectors...");
                }
            }
            sw.Stop();
            Console.WriteLine($"    Done! Loaded {VectorCount:N0} vectors in {sw.ElapsedMilliseconds} ms.\n");

            Console.WriteLine("[3] Preparing search query...");
            // Create a specific query (we'll just use a normalized random vector)
            float[] query = new float[Dimensions];
            float qSumSq = 0f;
            for (int d = 0; d < Dimensions; d++)
            {
                query[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
                qSumSq += query[d] * query[d];
            }
            float qLen = (float)Math.Sqrt(qSumSq);
            for (int d = 0; d < Dimensions; d++) query[d] /= qLen;

            Console.WriteLine("[4] Executing SIMD brute-force search...");

            // WARMUP PASS: JIT Compilation
            // We run it once and throw away the result so the JIT compiles the AVX2 instructions
            _ = index.Search(query, topK: 5);

            // REAL BENCHMARK PASS
            sw.Restart();
            var results = index.Search(query, topK: 5);
            sw.Stop();

            Console.WriteLine($"\n==========================================");
            Console.WriteLine($" SEARCH COMPLETED IN: {sw.Elapsed.TotalMilliseconds:F3} ms");
            Console.WriteLine($" Vectors scanned:     {index.Count:N0}");
            Console.WriteLine($" Operations/sec:      {(index.Count / sw.Elapsed.TotalSeconds):N0}");
            Console.WriteLine($"==========================================\n");

            Console.WriteLine("Top 5 Results:");
            for (int i = 0; i < results.Length; i++)
            {
                Console.WriteLine($"  Rank {i + 1} | Score: {results[i].Score:F4} | ID: {results[i].Id} | Meta: {results[i].Metadata}");
            }

            Console.WriteLine("\nPress any key to exit...");
            Console.ReadLine();
        }
    }
}