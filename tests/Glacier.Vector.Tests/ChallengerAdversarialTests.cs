namespace Glacier.Vector.Tests;

using System;
using System.Collections.Concurrent;
using System.Linq;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

/// <summary>
/// Empirical Adversarial Test Suite for Milestone M4 Verification.
/// Stress-tests zero-allocation search, reader concurrency determinism, and SIMD kernel precision.
/// </summary>
public class ChallengerAdversarialTests
{
    private static float[] GenerateNormalizedVector(int dims, Random rng)
    {
        float[] v = new float[dims];
        float sumSq = 0f;
        for (int i = 0; i < dims; i++)
        {
            v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            sumSq += v[i] * v[i];
        }
        float invNorm = 1.0f / MathF.Sqrt(Math.Max(1e-9f, sumSq));
        for (int i = 0; i < dims; i++) v[i] *= invNorm;
        return v;
    }

    // =========================================================================
    // TASK 1: ZERO-ALLOCATION SEARCH ADVERSARIAL VERIFICATION
    // =========================================================================

    [Theory]
    [InlineData(32, 200, 10, 32)]
    [InlineData(64, 400, 10, 64)]
    [InlineData(128, 500, 20, 64)]
    public void Task1_ZeroAllocSearch_DestinationSpan_AllocatesStrictlyZeroBytes(
        int dims, int vectorCount, int topK, int efSearch)
    {
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: vectorCount + 64);

        var rng = new Random(42000 + dims);
        for (int i = 0; i < vectorCount; i++)
        {
            hnsw.Add(GenerateNormalizedVector(dims, rng), $"item_{i}");
        }

        float[] query = GenerateNormalizedVector(dims, rng);
        var destination = new SearchResult[topK];
        Span<SearchResult> destSpan = destination.AsSpan();

        // 1. Warmup phase: JIT compilation, thread-static scratch allocation & resizing
        for (int i = 0; i < 25; i++)
        {
            int found = hnsw.Search(query, destSpan, efSearch: efSearch);
            Assert.Equal(topK, found);
        }

        // 2. Measure cumulative allocations across 1,000 queries
        const int iterations = 1000;
        long allocBefore = GC.GetAllocatedBytesForCurrentThread();
        int totalWritten = 0;

        for (int i = 0; i < iterations; i++)
        {
            totalWritten += hnsw.Search(query, destSpan, efSearch: efSearch);
        }

        long allocAfter = GC.GetAllocatedBytesForCurrentThread();
        long bytesAllocated = allocAfter - allocBefore;

        Assert.Equal(iterations * topK, totalWritten);
        Assert.Equal(0, bytesAllocated);
    }

    [Fact]
    public void Task1_ZeroAllocSearch_DedicatedThread_ZeroAllocOver10000Queries()
    {
        const int dims = 64;
        const int vectorCount = 300;
        const int topK = 10;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: 512);

        var rng = new Random(777);
        for (int i = 0; i < vectorCount; i++)
        {
            hnsw.Add(GenerateNormalizedVector(dims, rng), $"node_{i}");
        }

        float[] query = GenerateNormalizedVector(dims, rng);
        long recordedAlloc = -1;
        int queryCount = 0;

        // Run on a clean, isolated thread to eliminate any ambient framework noise
        var thread = new Thread(() =>
        {
            var destination = new SearchResult[topK];
            Span<SearchResult> destSpan = destination.AsSpan();

            // Warmup
            for (int w = 0; w < 30; w++)
            {
                hnsw.Search(query, destSpan, efSearch: 32);
            }

            long before = GC.GetAllocatedBytesForCurrentThread();

            const int runs = 10_000;
            for (int i = 0; i < runs; i++)
            {
                queryCount += hnsw.Search(query, destSpan, efSearch: 32);
            }

            long after = GC.GetAllocatedBytesForCurrentThread();
            recordedAlloc = after - before;
        });

        thread.Start();
        thread.Join();

        Assert.Equal(10_000 * topK, queryCount);
        Assert.Equal(0, recordedAlloc);
    }

    [Fact]
    public void Task1_ZeroAllocSearch_PerQueryGranularCheck()
    {
        const int dims = 32;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: 256);

        var rng = new Random(999);
        for (int i = 0; i < 200; i++)
        {
            hnsw.Add(GenerateNormalizedVector(dims, rng), $"pt_{i}");
        }

        var destination = new SearchResult[5];
        Span<SearchResult> destSpan = destination.AsSpan();

        // Warmup with different queries
        for (int w = 0; w < 20; w++)
        {
            float[] qWarm = GenerateNormalizedVector(dims, rng);
            hnsw.Search(qWarm, destSpan, efSearch: 32);
        }

        // Check per-query delta across 100 distinct queries
        for (int i = 0; i < 100; i++)
        {
            float[] q = GenerateNormalizedVector(dims, rng);
            long before = GC.GetAllocatedBytesForCurrentThread();
            int written = hnsw.Search(q, destSpan, efSearch: 32);
            long after = GC.GetAllocatedBytesForCurrentThread();

            Assert.Equal(5, written);
            Assert.Equal(0, after - before);
        }
    }

    // =========================================================================
    // TASK 2: READER CONCURRENCY STRESS TEST (16+, 32, 64 THREADS)
    // =========================================================================

    [Theory]
    [InlineData(16, 100)] // 16 threads x 100 queries = 1,600 concurrent queries
    [InlineData(32, 100)] // 32 threads x 100 queries = 3,200 concurrent queries
    [InlineData(64, 50)]  // 64 threads x 50 queries = 3,200 concurrent queries
    public void Task2_ReaderConcurrency_HighThreadCount_100PercentDeterministic(int threadCount, int queriesPerThread)
    {
        const int dims = 48;
        const int vectorCount = 400;
        const int numTestQueries = 20;
        const int topK = 8;
        const int efSearch = 40;

        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: vectorCount + 64);

        var rng = new Random(1337);
        for (int i = 0; i < vectorCount; i++)
        {
            hnsw.Add(GenerateNormalizedVector(dims, rng), $"doc_{i}");
        }

        // Precompute baseline ground-truth single-threaded results
        float[][] testQueries = new float[numTestQueries][];
        SearchResult[][] baseline = new SearchResult[numTestQueries][];

        for (int q = 0; q < numTestQueries; q++)
        {
            testQueries[q] = GenerateNormalizedVector(dims, rng);
            baseline[q] = hnsw.Search(testQueries[q], topK, efSearch: efSearch);
            Assert.Equal(topK, baseline[q].Length);
        }

        var errors = new ConcurrentBag<string>();
        var startGun = new ManualResetEventSlim(false);
        int totalSuccessfulSearches = 0;

        var threads = Enumerable.Range(0, threadCount).Select(threadId => new Thread(() =>
        {
            startGun.Wait(); // Ensure all threads unleash simultaneously

            try
            {
                var destBuffer = new SearchResult[topK];
                Span<SearchResult> dest = destBuffer.AsSpan();

                for (int iter = 0; iter < queriesPerThread; iter++)
                {
                    int queryIdx = (threadId * 7 + iter) % numTestQueries;
                    int written = hnsw.Search(testQueries[queryIdx], dest, efSearch: efSearch);

                    if (written != topK)
                    {
                        errors.Add($"Thread {threadId} returned count {written} != {topK}");
                        continue;
                    }

                    var expected = baseline[queryIdx];
                    for (int k = 0; k < topK; k++)
                    {
                        if (dest[k].Id != expected[k].Id)
                        {
                            errors.Add($"Thread {threadId} query {queryIdx} rank {k} mismatch: expected ID {expected[k].Id}, got {dest[k].Id}");
                            break;
                        }

                        if (MathF.Abs(dest[k].Score - expected[k].Score) > 1e-5f)
                        {
                            errors.Add($"Thread {threadId} query {queryIdx} rank {k} score mismatch: expected {expected[k].Score}, got {dest[k].Score}");
                            break;
                        }
                    }

                    Interlocked.Increment(ref totalSuccessfulSearches);
                }
            }
            catch (Exception ex)
            {
                errors.Add($"Thread {threadId} exception: {ex}");
            }
        })).ToArray();

        foreach (var t in threads) t.Start();
        startGun.Set(); // BANG
        foreach (var t in threads) t.Join();

        Assert.Empty(errors);
        Assert.Equal(threadCount * queriesPerThread, totalSuccessfulSearches);
    }

    [Fact]
    public void Task2_ReaderConcurrency_VisitedTagWraparound_DeterministicAndSafe()
    {
        const int dims = 16;
        const int vectorCount = 100;
        const int topK = 5;

        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 8, efConstruction: 32, initialCapacity: 128);

        var rng = new Random(5555);
        for (int i = 0; i < vectorCount; i++)
        {
            hnsw.Add(GenerateNormalizedVector(dims, rng), $"item_{i}");
        }

        float[] query = GenerateNormalizedVector(dims, rng);
        var baseline = hnsw.Search(query, topK, efSearch: 24);

        // Access thread-static scratch via reflection to advance CurrentTag near uint.MaxValue
        var scratchField = typeof(HnswVectorIndex).GetField("t_scratch", BindingFlags.Static | BindingFlags.NonPublic);
        Assert.NotNull(scratchField);

        // Pre-warm thread scratch
        var dest = new SearchResult[topK];
        hnsw.Search(query, dest.AsSpan(), efSearch: 24);

        object? scratchObj = scratchField.GetValue(null);
        Assert.NotNull(scratchObj);

        var tagField = scratchObj.GetType().GetField("CurrentTag", BindingFlags.Public | BindingFlags.Instance);
        Assert.NotNull(tagField);

        // Set CurrentTag to uint.MaxValue - 2 so wraparound triggers on the 2nd/3rd call
        tagField.SetValue(scratchObj, uint.MaxValue - 2);

        for (int i = 0; i < 10; i++)
        {
            int written = hnsw.Search(query, dest.AsSpan(), efSearch: 24);
            Assert.Equal(topK, written);

            for (int k = 0; k < topK; k++)
            {
                Assert.Equal(baseline[k].Id, dest[k].Id);
                Assert.Equal(baseline[k].Score, dest[k].Score, precision: 5);
            }
        }
    }

    // =========================================================================
    // TASK 3: DISTANCE KERNELS PRECISION & SIMD BOUNDARY VERIFICATION
    // =========================================================================

    public static readonly int[] AllTestDimensions =
    [
        // Odd and small sizes
        1, 2, 3, 5, 7, 9, 11, 13, 14, 15,
        // AVX/SIMD Boundary 16
        16, 17, 19, 23, 31,
        // AVX/SIMD Boundary 32
        32, 33, 47, 63,
        // AVX/SIMD Boundary 64
        64, 65, 96, 127,
        // Boundary 128, 256, 768, 1536
        128, 129, 255, 256, 511, 512, 768, 1024, 1536
    ];

    [Theory]
    [MemberData(nameof(GetDimensionsTheoryData))]
    public void Task3_DistanceKernels_PrecisionAgainstDoubleReference_AllDimensions(int dim)
    {
        var rng = new Random(8888 + dim);

        // Test across 5 different randomized vector pairs per dimension
        for (int testCase = 0; testCase < 5; testCase++)
        {
            float[] a = new float[dim];
            float[] b = new float[dim];

            double refDot = 0.0;
            double refL2Sq = 0.0;

            for (int i = 0; i < dim; i++)
            {
                a[i] = (float)(rng.NextDouble() * 4.0 - 2.0);
                b[i] = (float)(rng.NextDouble() * 4.0 - 2.0);

                refDot += (double)a[i] * b[i];
                double diff = (double)a[i] - b[i];
                refL2Sq += diff * diff;
            }

            float actualDot = DistanceKernels.DotProduct(a, b);
            float actualCosine = DistanceKernels.CosineDistance(a, b);
            float actualL2Sq = DistanceKernels.L2DistanceSquared(a, b);
            float actualL2 = DistanceKernels.L2Distance(a, b);

            // Single-precision accumulation error bound scales with sqrt(N) to N * float.Epsilon
            float tolDot = MathF.Max(1e-4f, 2e-5f * dim);
            float tolL2Sq = MathF.Max(1e-4f, 2e-5f * dim);

            Assert.True(MathF.Abs(actualDot - (float)refDot) <= tolDot,
                $"DotProduct mismatch at dim {dim}: actual={actualDot}, ref={refDot}, tol={tolDot}");

            Assert.True(MathF.Abs(actualCosine - (1.0f - actualDot)) <= 1e-6f,
                $"CosineDistance definition mismatch at dim {dim}: actual={actualCosine}, expected={1.0f - actualDot}");

            Assert.True(MathF.Abs(actualL2Sq - (float)refL2Sq) <= tolL2Sq,
                $"L2DistanceSquared mismatch at dim {dim}: actual={actualL2Sq}, ref={refL2Sq}, tol={tolL2Sq}");

            Assert.True(MathF.Abs(actualL2 - MathF.Sqrt(actualL2Sq)) <= 1e-5f,
                $"L2Distance mismatch at dim {dim}: actual={actualL2}, expected={MathF.Sqrt(actualL2Sq)}");
        }
    }

    [Fact]
    public void Task3_DistanceKernels_ExtremeValuesAndEdgeCases()
    {
        int[] boundaryDims = [15, 16, 31, 32, 63, 64, 128, 768];

        foreach (int dim in boundaryDims)
        {
            // 1. Identical vectors -> Dot = |v|^2, Cosine = 1 - |v|^2, L2Sq = 0, L2 = 0
            float[] u = new float[dim];
            Array.Fill(u, 1.0f / MathF.Sqrt(dim));
            float dotSame = DistanceKernels.DotProduct(u, u);
            Assert.True(MathF.Abs(dotSame - 1.0f) < 1e-4f, $"Identical norm vector Dot should be 1.0, got {dotSame} (dim {dim})");
            Assert.True(MathF.Abs(DistanceKernels.CosineDistance(u, u)) < 1e-4f);
            Assert.Equal(0f, DistanceKernels.L2DistanceSquared(u, u));
            Assert.Equal(0f, DistanceKernels.L2Distance(u, u));

            // 2. Opposite vectors -> Dot = -1.0, Cosine = 2.0, L2Sq = 4.0, L2 = 2.0
            float[] negU = new float[dim];
            for (int i = 0; i < dim; i++) negU[i] = -u[i];
            float dotOpp = DistanceKernels.DotProduct(u, negU);
            Assert.True(MathF.Abs(dotOpp - (-1.0f)) < 1e-4f, $"Opposite vector Dot should be -1.0, got {dotOpp} (dim {dim})");
            Assert.True(MathF.Abs(DistanceKernels.CosineDistance(u, negU) - 2.0f) < 1e-4f);
            Assert.True(MathF.Abs(DistanceKernels.L2DistanceSquared(u, negU) - 4.0f) < 1e-4f);
            Assert.True(MathF.Abs(DistanceKernels.L2Distance(u, negU) - 2.0f) < 1e-4f);

            // 3. Orthogonal unit basis vectors e_0 and e_last
            float[] e0 = new float[dim];
            float[] eLast = new float[dim];
            e0[0] = 1.0f;
            eLast[dim - 1] = 1.0f;

            if (dim > 1)
            {
                Assert.Equal(0.0f, DistanceKernels.DotProduct(e0, eLast));
                Assert.Equal(1.0f, DistanceKernels.CosineDistance(e0, eLast));
                Assert.Equal(2.0f, DistanceKernels.L2DistanceSquared(e0, eLast));
                Assert.True(MathF.Abs(DistanceKernels.L2Distance(e0, eLast) - MathF.Sqrt(2.0f)) < 1e-5f);
            }

            // 4. Zero vector
            float[] zeros = new float[dim];
            Assert.Equal(0f, DistanceKernels.DotProduct(u, zeros));
            Assert.Equal(1.0f, DistanceKernels.CosineDistance(u, zeros));
            Assert.True(MathF.Abs(DistanceKernels.L2DistanceSquared(u, zeros) - 1.0f) < 1e-4f);
        }
    }

    public static TheoryData<int> GetDimensionsTheoryData()
    {
        var data = new TheoryData<int>();
        foreach (var dim in AllTestDimensions)
        {
            data.Add(dim);
        }
        return data;
    }
}
