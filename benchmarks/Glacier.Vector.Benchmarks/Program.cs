namespace Glacier.Vector.Benchmarks;

using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using Glacier.Vector.Compute;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

public static class Program
{
    public static void Main(string[] args)
    {
        if (args.Length > 0 && args[0].Equals("--bdn", StringComparison.OrdinalIgnoreCase))
        {
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
            return;
        }

        Console.WriteLine("================================================================================");
        Console.WriteLine("          GLACIER.VECTOR MICROBENCHMARK & VERIFICATION SUITE                    ");
        Console.WriteLine("================================================================================");

        RunDistanceKernelBenchmarks();
        RunHnswSearchBenchmarks();

        Console.WriteLine("================================================================================");
        Console.WriteLine("          ALL MICROBENCHMARKS COMPLETED SUCCESSFULLY                            ");
        Console.WriteLine("================================================================================");
    }

    private static void RunDistanceKernelBenchmarks()
    {
        Console.WriteLine("\n[1] SIMD Distance Kernels Throughput (DotProduct, Cosine, L2, L2Squared):");
        int[] dims = [64, 128, 768, 1536];
        var rand = new Random(42);

        foreach (int dim in dims)
        {
            float[] a = new float[dim];
            float[] b = new float[dim];
            for (int i = 0; i < dim; i++)
            {
                a[i] = (float)rand.NextDouble();
                b[i] = (float)rand.NextDouble();
            }

            const int warmup = 50_000;
            for (int i = 0; i < warmup; i++)
            {
                _ = DistanceKernels.DotProduct(a, b);
                _ = DistanceKernels.CosineDistance(a, b);
                _ = DistanceKernels.L2Distance(a, b);
                _ = DistanceKernels.L2DistanceSquared(a, b);
            }

            const int iterations = 1_000_000;
            var swDot = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) _ = DistanceKernels.DotProduct(a, b);
            swDot.Stop();

            var swCos = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) _ = DistanceKernels.CosineDistance(a, b);
            swCos.Stop();

            var swL2 = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) _ = DistanceKernels.L2Distance(a, b);
            swL2.Stop();

            var swL2Sq = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++) _ = DistanceKernels.L2DistanceSquared(a, b);
            swL2Sq.Stop();

            double dotMops = (iterations / swDot.Elapsed.TotalSeconds) / 1_000_000.0;
            double cosMops = (iterations / swCos.Elapsed.TotalSeconds) / 1_000_000.0;
            double l2Mops = (iterations / swL2.Elapsed.TotalSeconds) / 1_000_000.0;
            double l2SqMops = (iterations / swL2Sq.Elapsed.TotalSeconds) / 1_000_000.0;

            Console.WriteLine($"    Dim {dim,4}: Dot={dotMops:F1} MOps/s | Cosine={cosMops:F1} MOps/s | L2={l2Mops:F1} MOps/s | L2Sq={l2SqMops:F1} MOps/s");
        }
    }

    private static void RunHnswSearchBenchmarks()
    {
        Console.WriteLine("\n[2] HNSW Index Search Latency & Allocation (Allocating vs ZeroAlloc):");
        const int dims = 128;
        const int count = 2000;
        var rand = new Random(42);

        using var storage = new InMemoryVectorStorage(dims);
        using var index = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: count + 64);

        for (int i = 0; i < count; i++)
        {
            float[] vec = new float[dims];
            float sumSq = 0f;
            for (int d = 0; d < dims; d++)
            {
                vec[d] = (float)rand.NextDouble();
                sumSq += vec[d] * vec[d];
            }
            float norm = MathF.Sqrt(sumSq);
            for (int d = 0; d < dims; d++) vec[d] /= norm;
            index.Add(vec, $"doc_{i}");
        }

        float[] query = new float[dims];
        for (int d = 0; d < dims; d++) query[d] = 1.0f / MathF.Sqrt(dims);

        var destArray = new SearchResult[10];
        Span<SearchResult> destination = destArray.AsSpan();

        // Warmup
        for (int i = 0; i < 100; i++)
        {
            _ = index.Search(query, 10, efSearch: 64);
            _ = index.Search(query, destination, efSearch: 64);
        }

        const int queries = 10_000;

        // Baseline: Search with SearchResult[] allocation
        long allocBeforeAllocating = GC.GetAllocatedBytesForCurrentThread();
        var swAlloc = Stopwatch.StartNew();
        for (int i = 0; i < queries; i++)
        {
            _ = index.Search(query, 10, efSearch: 64);
        }
        swAlloc.Stop();
        long allocAfterAllocating = GC.GetAllocatedBytesForCurrentThread();
        long totalAllocatingBytes = allocAfterAllocating - allocBeforeAllocating;

        // Zero-Alloc: Search with Span<SearchResult> destination
        long allocBeforeZero = GC.GetAllocatedBytesForCurrentThread();
        var swZero = Stopwatch.StartNew();
        for (int i = 0; i < queries; i++)
        {
            _ = index.Search(query, destination, efSearch: 64);
        }
        swZero.Stop();
        long allocAfterZero = GC.GetAllocatedBytesForCurrentThread();
        long totalZeroAllocBytes = allocAfterZero - allocBeforeZero;

        double allocQps = queries / swAlloc.Elapsed.TotalSeconds;
        double zeroQps = queries / swZero.Elapsed.TotalSeconds;
        double allocLatencyUs = swAlloc.Elapsed.TotalMilliseconds / queries * 1000.0;
        double zeroLatencyUs = swZero.Elapsed.TotalMilliseconds / queries * 1000.0;

        Console.WriteLine($"    Search (Allocating): {allocQps:N0} QPS | Latency: {allocLatencyUs:F1} µs | Total Alloc: {totalAllocatingBytes:N0} bytes ({totalAllocatingBytes / (double)queries:F1} B/query)");
        Console.WriteLine($"    Search (Zero-Alloc): {zeroQps:N0} QPS | Latency: {zeroLatencyUs:F1} µs | Total Alloc: {totalZeroAllocBytes:N0} bytes ({totalZeroAllocBytes / (double)queries:F1} B/query)");
        if (totalAllocatingBytes > 0)
        {
            Console.WriteLine($"    Allocation Drop:     {((totalAllocatingBytes - totalZeroAllocBytes) / (double)totalAllocatingBytes) * 100.0:F1}% reduction");
        }
    }
}
