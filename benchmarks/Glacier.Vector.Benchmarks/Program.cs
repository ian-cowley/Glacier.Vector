namespace Glacier.Vector.Benchmarks;

using System;
using System.Diagnostics;
using BenchmarkDotNet.Running;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
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
        RunQuantizedAndHadamardBenchmarks();
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

    private static void RunQuantizedAndHadamardBenchmarks()
    {
        Console.WriteLine("\n[1b] Quantized Dot Products (FP32 vs INT8 vs Packed UINT4) & Hadamard/RaBitQ Kernels:");
        int[] dims = [64, 128, 768, 1536];
        var rand = new Random(123);

        foreach (int dim in dims)
        {
            float[] fa = new float[dim];
            float[] fb = new float[dim];
            sbyte[] i8a = new sbyte[dim];
            sbyte[] i8b = new sbyte[dim];
            byte[] nibA = new byte[dim];
            byte[] nibB = new byte[dim];
            for (int i = 0; i < dim; i++)
            {
                fa[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
                fb[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
                i8a[i] = (sbyte)rand.Next(-127, 128);
                i8b[i] = (sbyte)rand.Next(-127, 128);
                nibA[i] = (byte)rand.Next(0, 16);
                nibB[i] = (byte)rand.Next(0, 16);
            }
            byte[] u4a = new byte[(dim + 1) >> 1];
            byte[] u4b = new byte[(dim + 1) >> 1];
            DistanceKernels.PackUInt4(nibA, u4a);
            DistanceKernels.PackUInt4(nibB, u4b);

            for (int w = 0; w < 50_000; w++)
            {
                _ = DistanceKernels.DotProduct(fa, fb);
                _ = DistanceKernels.DotInt8(i8a, i8b);
                _ = DistanceKernels.DotUInt4Packed(u4a, u4b, dim);
            }

            const int iters = 1_000_000;
            var swFp32 = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) _ = DistanceKernels.DotProduct(fa, fb);
            swFp32.Stop();

            var swI8 = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) _ = DistanceKernels.DotInt8(i8a, i8b);
            swI8.Stop();

            var swU4 = Stopwatch.StartNew();
            for (int i = 0; i < iters; i++) _ = DistanceKernels.DotUInt4Packed(u4a, u4b, dim);
            swU4.Stop();

            double fp32Ns = swFp32.Elapsed.TotalNanoseconds / iters;
            double i8Ns = swI8.Elapsed.TotalNanoseconds / iters;
            double u4Ns = swU4.Elapsed.TotalNanoseconds / iters;
            Console.WriteLine(
                $"    Dim {dim,4}: FP32={fp32Ns:F1} ns ({1e3 / fp32Ns:F1} MOps/s) | INT8={i8Ns:F1} ns ({1e3 / i8Ns:F1} MOps/s) | UINT4={u4Ns:F1} ns ({1e3 / u4Ns:F1} MOps/s)");
        }

        Console.WriteLine("\n[1c] Hadamard FWHT & RaBitQ 1-Bit Rotation Throughput:");
        int[] rotDims = [128, 768, 1536, 4096];
        foreach (int dim in rotDims)
        {
            int padded = HadamardProjection.GetPaddedDimension(dim);
            float[] src = new float[dim];
            float[] work = new float[padded];
            float[] signs = new float[padded];
            ulong[] bits = new ulong[HadamardProjection.GetPackedWordCount(dim)];
            HadamardProjection.GenerateSigns(signs, seed: 42UL);
            for (int i = 0; i < dim; i++) src[i] = (float)(rand.NextDouble() * 2.0 - 1.0);

            for (int w = 0; w < 2_000; w++)
            {
                HadamardProjection.ApplyFwhtInPlace(work);
                _ = HadamardProjection.RotateAndBinarize(src, signs, bits);
            }

            const int rotIters = 50_000;
            var swFwht = Stopwatch.StartNew();
            for (int i = 0; i < rotIters; i++) HadamardProjection.ApplyFwhtInPlace(work);
            swFwht.Stop();

            var swRab = Stopwatch.StartNew();
            for (int i = 0; i < rotIters; i++) _ = HadamardProjection.RotateAndBinarize(src, signs, bits);
            swRab.Stop();

            double fwhtUs = swFwht.Elapsed.TotalMicroseconds / rotIters;
            double rabUs = swRab.Elapsed.TotalMicroseconds / rotIters;
            Console.WriteLine(
                $"    Dim {dim,4} (pad {padded,4}): FWHT={fwhtUs:F2} µs ({rotIters / swFwht.Elapsed.TotalSeconds / 1000.0:F1} kVec/s) | RotateAndBinarize={rabUs:F2} µs ({rotIters / swRab.Elapsed.TotalSeconds / 1000.0:F1} kVec/s)");
        }

        Console.WriteLine("\n[1d] Split-K FP32 & Zero-I2F dp4a INT8 GPU vs CPU Scan (100,000 vectors):");
        int[] scanDims = [128, 1536];
        const int vecCount = 100_000;
        float[] scores = new float[vecCount];
        foreach (int dim in scanDims)
        {
            float[] qFp32 = new float[dim];
            float[] dbFp32 = new float[vecCount * dim];
            sbyte[] qI8 = new sbyte[dim];
            sbyte[] dbI8 = new sbyte[vecCount * dim];
            for (int i = 0; i < dim; i++) { qFp32[i] = 0.25f; qI8[i] = (sbyte)((i % 17) - 8); }
            for (int i = 0; i < 1024; i++) { dbFp32[i] = 0.1f; dbI8[i] = (sbyte)(i & 15); }

            // Warmup CPU & GPU (storageId=999 keeps FP32 chunk warm in VRAM across queries)
            ComputeCpuFp32Scan(qFp32, dbFp32, scores, vecCount, dim);
            GpuVectorAccelerator.ScanChunkInt8Gpu(qI8, 0.01f, dbI8, 0.01f, scores, vecCount, dim, GpuTarget.Cpu);

            bool hasGpu = GpuVectorAccelerator.IsNvidiaAvailable;
            if (hasGpu)
            {
                GpuVectorAccelerator.ScanChunkGpu(qFp32, dbFp32, scores, vecCount, dim, isL2Distance: false, GpuTarget.Nvidia, storageId: 999, chunkIndex: 0, storageVersion: 1);
                GpuVectorAccelerator.ScanChunkInt8Gpu(qI8, 0.01f, dbI8, 0.01f, scores, vecCount, dim, GpuTarget.Nvidia);
            }

            const int trials = 20;
            var swCpuFp32 = Stopwatch.StartNew();
            for (int t = 0; t < trials; t++)
                ComputeCpuFp32Scan(qFp32, dbFp32, scores, vecCount, dim);
            swCpuFp32.Stop();

            var swCpuI8 = Stopwatch.StartNew();
            for (int t = 0; t < trials; t++)
                GpuVectorAccelerator.ScanChunkInt8Gpu(qI8, 0.01f, dbI8, 0.01f, scores, vecCount, dim, GpuTarget.Cpu);
            swCpuI8.Stop();

            double cpuFp32Ms = swCpuFp32.Elapsed.TotalMilliseconds / trials;
            double cpuI8Ms = swCpuI8.Elapsed.TotalMilliseconds / trials;
            Console.WriteLine($"    Dim {dim,4} CPU SIMD: FP32={cpuFp32Ms:F2} ms ({vecCount / cpuFp32Ms / 1000.0:F1} M vec/s) | INT8={cpuI8Ms:F2} ms ({vecCount / cpuI8Ms / 1000.0:F1} M vec/s)");

            if (hasGpu)
            {
                var swGpuFp32 = Stopwatch.StartNew();
                for (int t = 0; t < trials; t++)
                    GpuVectorAccelerator.ScanChunkGpu(qFp32, dbFp32, scores, vecCount, dim, isL2Distance: false, GpuTarget.Nvidia, storageId: 999, chunkIndex: 0, storageVersion: 1);
                swGpuFp32.Stop();

                var swGpuI8 = Stopwatch.StartNew();
                for (int t = 0; t < trials; t++)
                    GpuVectorAccelerator.ScanChunkInt8Gpu(qI8, 0.01f, dbI8, 0.01f, scores, vecCount, dim, GpuTarget.Nvidia);
                swGpuI8.Stop();

                double gpuFp32Ms = swGpuFp32.Elapsed.TotalMilliseconds / trials;
                double gpuI8Ms = swGpuI8.Elapsed.TotalMilliseconds / trials;
                Console.WriteLine($"    Dim {dim,4} NVIDIA GPU: SplitK-FP32(cached)={gpuFp32Ms:F2} ms ({vecCount / gpuFp32Ms / 1000.0:F1} M vec/s) | INT8-dp4a(incl. H2D)={gpuI8Ms:F2} ms ({vecCount / gpuI8Ms / 1000.0:F1} M vec/s)");
            }
        }
    }

    private static void ComputeCpuFp32Scan(float[] query, float[] db, float[] scores, int vectorCount, int dimensions)
    {
        const int blockSize = 1024;
        int blocks = (vectorCount + blockSize - 1) / blockSize;
        System.Threading.Tasks.Parallel.For(0, blocks, b =>
        {
            int start = b * blockSize;
            int end = Math.Min(start + blockSize, vectorCount);
            ReadOnlySpan<float> q = query.AsSpan(0, dimensions);
            for (int i = start; i < end; i++)
            {
                scores[i] = DistanceKernels.DotProduct(q, db.AsSpan(i * dimensions, dimensions));
            }
        });
    }
}
