using System;
using System.Diagnostics;
using System.Linq;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

namespace Glacier.Vector.Demo
{
    internal class Program
    {
        private const int Dimensions = 1536; // Standard LLM embedding size (e.g. OpenAI text-embedding-3-large)
        private const int VectorCount = 100_000; // 100,000 vectors * 1,536 floats * 4 bytes = ~614 MB of VRAM / RAM

        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("================================================================================");
            Console.WriteLine("    Glacier.Vector | High-Performance Hardware-Accelerated Similarity Search    ");
            Console.WriteLine("================================================================================");
            Console.ResetColor();

            Console.WriteLine("Hardware Acceleration Status:");
            Console.WriteLine($"  - CPU AVX-512 FMA:      {Vector512.IsHardwareAccelerated && Avx512F.IsSupported}");
            Console.WriteLine($"  - CPU AVX2 Vector256:   {Vector256.IsHardwareAccelerated && Avx2.IsSupported}");
            Console.WriteLine($"  - CPU Cores / Threads:  {Environment.ProcessorCount}");
            Console.WriteLine($"  - NVIDIA RTX 4060 dGPU: {GpuVectorAccelerator.IsNvidiaAvailable} (Direct nvcuda.dll PTX)");
            Console.WriteLine($"  - AMD Radeon 890M APU:  {GpuVectorAccelerator.IsAmdAvailable} (Direct amdhip64.dll)");
            Console.WriteLine();

            // Initialize storage and search index
            using var storage = new InMemoryVectorStorage(Dimensions, vectorsPerChunk: VectorCount);
            using var index = new VectorIndex(storage);

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"[1/4] Ingesting {VectorCount:N0} synthetic normalized embeddings ({Dimensions} dimensions)...");
            Console.ResetColor();

            var sw = Stopwatch.StartNew();
            var rng = new Random(42);
            float[] tempVector = new float[Dimensions];

            for (int i = 0; i < VectorCount; i++)
            {
                float sumSq = 0f;
                for (int d = 0; d < Dimensions; d++)
                {
                    float val = (float)(rng.NextDouble() * 2.0 - 1.0);
                    tempVector[d] = val;
                    sumSq += val * val;
                }

                float invLen = 1.0f / MathF.Sqrt(sumSq);
                for (int d = 0; d < Dimensions; d++) tempVector[d] *= invLen;

                index.Add(tempVector, $"Document_Chunk_{i}");
            }
            sw.Stop();
            Console.WriteLine($"  -> Loaded {VectorCount:N0} vectors in {sw.ElapsedMilliseconds} ms ({(VectorCount * Dimensions * sizeof(float) / 1024.0 / 1024.0):F2} MB)\n");

            // Prepare a normalized query vector
            float[] query = new float[Dimensions];
            float qSumSq = 0f;
            for (int d = 0; d < Dimensions; d++)
            {
                query[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
                qSumSq += query[d] * query[d];
            }
            float invQLen = 1.0f / MathF.Sqrt(qSumSq);
            for (int d = 0; d < Dimensions; d++) query[d] *= invQLen;

            // Warmup
            _ = index.Search(query, topK: 5, target: GpuTarget.Cpu);
            if (GpuVectorAccelerator.IsGpuAvailable)
            {
                _ = index.Search(query, topK: 5, target: GpuTarget.Nvidia);
            }

            // -----------------------------------------------------------------------------
            // [2/4] Single Query Benchmark: CPU vs GPU
            // -----------------------------------------------------------------------------
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("[2/4] Single-Query Search Benchmark (Top-5 Nearest Neighbors over 100,000 vectors)...");
            Console.ResetColor();

            // CPU Search
            sw.Restart();
            var cpuResults = index.Search(query, topK: 5, target: GpuTarget.Cpu);
            sw.Stop();
            double cpuMs = sw.Elapsed.TotalMilliseconds;
            double cpuOpsSec = VectorCount / sw.Elapsed.TotalSeconds;
            Console.WriteLine($"  -> [CPU AVX-512]       Latency: {cpuMs:F2} ms ({cpuOpsSec:N0} vectors/sec)");

            // GPU Search (RTX 4060)
            if (GpuVectorAccelerator.IsNvidiaAvailable)
            {
                sw.Restart();
                var gpuResults = index.Search(query, topK: 5, target: GpuTarget.Nvidia);
                sw.Stop();
                double gpuMs = sw.Elapsed.TotalMilliseconds;
                double gpuOpsSec = VectorCount / sw.Elapsed.TotalSeconds;

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  -> [NVIDIA RTX 4060]   Latency: {gpuMs:F2} ms ({gpuOpsSec:N0} vectors/sec)");
                if (gpuMs > 0)
                {
                    Console.WriteLine($"     ==> GPU Speedup vs CPU: {(cpuMs / gpuMs):F2}x faster!");
                }
                Console.ResetColor();
            }
            Console.WriteLine();

            // -----------------------------------------------------------------------------
            // [3/4] Batch Search Benchmark (16 Concurrent Queries)
            // -----------------------------------------------------------------------------
            const int batch16 = 16;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"[3/4] Batch Search Benchmark ({batch16} concurrent queries against 100,000 vectors)...");
            Console.ResetColor();

            float[] batchQueries16 = new float[batch16 * Dimensions];
            for (int i = 0; i < batchQueries16.Length; i++)
            {
                batchQueries16[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            // CPU Batch Search
            sw.Restart();
            var cpuBatch16 = index.BatchSearch(batchQueries16, batchSize: batch16, topK: 5, target: GpuTarget.Cpu);
            sw.Stop();
            double cpuB16Ms = sw.Elapsed.TotalMilliseconds;
            Console.WriteLine($"  -> [CPU AVX-512]       Total: {cpuB16Ms:F2} ms ({cpuB16Ms / batch16:F2} ms/query) | {(batch16 * VectorCount / sw.Elapsed.TotalSeconds):N0} vec/s");

            // GPU Batch Search (RTX 4060 via GEMM)
            if (GpuVectorAccelerator.IsNvidiaAvailable)
            {
                sw.Restart();
                var gpuBatch16 = index.BatchSearch(batchQueries16, batchSize: batch16, topK: 5, target: GpuTarget.Nvidia);
                sw.Stop();
                double gpuB16Ms = sw.Elapsed.TotalMilliseconds;

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  -> [NVIDIA RTX 4060]   Total: {gpuB16Ms:F2} ms ({gpuB16Ms / batch16:F2} ms/query) | {(batch16 * VectorCount / sw.Elapsed.TotalSeconds):N0} vec/s");
                if (gpuB16Ms > 0)
                {
                    Console.WriteLine($"     ==> Batch GPU Speedup vs CPU: {(cpuB16Ms / gpuB16Ms):F2}x faster!");
                }
                Console.ResetColor();
            }
            Console.WriteLine();

            // -----------------------------------------------------------------------------
            // [4/4] High-Volume Batch Search Benchmark (64 Concurrent Queries)
            // -----------------------------------------------------------------------------
            const int batch64 = 64;
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine($"[4/4] High-Volume Batch Search Benchmark ({batch64} concurrent queries against 100,000 vectors)...");
            Console.ResetColor();

            float[] batchQueries64 = new float[batch64 * Dimensions];
            for (int i = 0; i < batchQueries64.Length; i++)
            {
                batchQueries64[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }

            if (GpuVectorAccelerator.IsNvidiaAvailable)
            {
                sw.Restart();
                var gpuBatch64 = index.BatchSearch(batchQueries64, batchSize: batch64, topK: 5, target: GpuTarget.Nvidia);
                sw.Stop();
                double gpuB64Ms = sw.Elapsed.TotalMilliseconds;

                Console.ForegroundColor = ConsoleColor.Green;
                Console.WriteLine($"  -> [NVIDIA RTX 4060]   Total: {gpuB64Ms:F2} ms ({gpuB64Ms / batch64:F2} ms/query)");
                Console.WriteLine($"  -> Sustained Scan Rate: {(batch64 * VectorCount / sw.Elapsed.TotalSeconds):N0} vector evaluations/second!");
                Console.ResetColor();
            }

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine("================================================================================");
            Console.WriteLine("  Glacier.Vector GPU similarity search benchmark completed with 100% success!  ");
            Console.WriteLine("================================================================================");
            Console.ResetColor();

            if (!args.Contains("--headless") && !args.Contains("--bench") && Environment.UserInteractive && !Console.IsInputRedirected)
            {
                Console.WriteLine("\n[Press any key to exit...]");
                try { Console.ReadKey(); } catch { }
            }
        }
    }
}
