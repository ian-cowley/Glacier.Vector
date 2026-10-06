namespace Glacier.Vector.Tests;

using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

public class HnswZeroAllocTests
{
    [Fact]
    public void HnswSearch_SpanOverload_PerformsZeroAllocations()
    {
        const int dims = 64;
        const int count = 200;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: 256);

        var rand = new Random(42);
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
            hnsw.Add(vec, $"vec_{i}");
        }

        float[] query = new float[dims];
        for (int d = 0; d < dims; d++) query[d] = 1.0f / MathF.Sqrt(dims);

        // Preallocate destination buffer
        var destinationArray = new SearchResult[10];
        Span<SearchResult> destination = destinationArray.AsSpan();

        // Warmup: JIT compilation, thread-static scratch allocation and sizing
        for (int i = 0; i < 10; i++)
        {
            int n = hnsw.Search(query, destination, efSearch: 32);
            Assert.Equal(10, n);
        }

        int totalFound = 0;
        long beforeAlloc = GC.GetAllocatedBytesForCurrentThread();

        const int iterations = 100;
        for (int i = 0; i < iterations; i++)
        {
            totalFound += hnsw.Search(query, destination, efSearch: 32);
        }

        long afterAlloc = GC.GetAllocatedBytesForCurrentThread();
        long allocatedBytes = afterAlloc - beforeAlloc;

        Assert.Equal(iterations * 10, totalFound);
        Assert.Equal(0, allocatedBytes);
    }

    [Fact]
    public void HnswSearch_ConcurrentReaders_ZeroTagCollisionInterference()
    {
        const int dims = 32;
        const int count = 300;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: 512);

        var rand = new Random(1234);
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
            hnsw.Add(vec, $"item_{i}");
        }

        // Generate 10 fixed query vectors and record baseline results (single-threaded)
        const int numQueries = 10;
        const int topK = 5;
        float[][] testQueries = new float[numQueries][];
        SearchResult[][] baselineResults = new SearchResult[numQueries][];

        for (int q = 0; q < numQueries; q++)
        {
            testQueries[q] = new float[dims];
            float sumSq = 0f;
            for (int d = 0; d < dims; d++)
            {
                testQueries[q][d] = (float)rand.NextDouble();
                sumSq += testQueries[q][d] * testQueries[q][d];
            }
            float norm = MathF.Sqrt(sumSq);
            for (int d = 0; d < dims; d++) testQueries[q][d] /= norm;

            baselineResults[q] = hnsw.Search(testQueries[q], topK, efSearch: 32);
            Assert.Equal(topK, baselineResults[q].Length);
        }

        // Launch 16 concurrent reader threads hammering search queries
        const int numThreads = 16;
        const int itersPerThread = 50;
        var errors = new ConcurrentBag<string>();

        Parallel.For(0, numThreads, threadId =>
        {
            try
            {
                var destArray = new SearchResult[topK];
                Span<SearchResult> dest = destArray.AsSpan();
                for (int iter = 0; iter < itersPerThread; iter++)
                {
                    int queryIdx = (threadId + iter) % numQueries;
                    int written = hnsw.Search(testQueries[queryIdx], dest, efSearch: 32);

                    if (written != topK)
                    {
                        errors.Add($"Thread {threadId} got {written} results instead of {topK}");
                        continue;
                    }

                    var expected = baselineResults[queryIdx];
                    for (int k = 0; k < topK; k++)
                    {
                        if (dest[k].Id != expected[k].Id)
                        {
                            errors.Add($"Thread {threadId} query {queryIdx} mismatch at rank {k}: expected {expected[k].Id}, got {dest[k].Id}");
                            break;
                        }
                    }
                }
            }
            catch (Exception ex)
            {
                errors.Add($"Thread {threadId} threw exception: {ex.Message}");
            }
        });

        Assert.Empty(errors);
    }

    [Fact]
    public void DistanceKernels_PrecisionAndAccuracy_AcrossDiverseDimensions()
    {
        int[] testDimensions = [1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 768, 1536];
        var rand = new Random(9876);

        foreach (int dim in testDimensions)
        {
            float[] a = new float[dim];
            float[] b = new float[dim];

            double expectedDot = 0.0;
            double expectedL2Sq = 0.0;

            for (int i = 0; i < dim; i++)
            {
                a[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
                b[i] = (float)(rand.NextDouble() * 2.0 - 1.0);

                expectedDot += (double)a[i] * b[i];
                double diff = (double)a[i] - b[i];
                expectedL2Sq += diff * diff;
            }

            float actualDot = DistanceKernels.DotProduct(a, b);
            float actualCosine = DistanceKernels.CosineDistance(a, b);
            float actualL2Sq = DistanceKernels.L2DistanceSquared(a, b);
            float actualL2 = DistanceKernels.L2Distance(a, b);

            // Floating point tolerance scales slightly with dimension
            float tolerance = MathF.Max(1e-4f, 1e-5f * dim);

            Assert.True(MathF.Abs(actualDot - (float)expectedDot) <= tolerance,
                $"DotProduct mismatch for dim {dim}: expected {expectedDot}, got {actualDot}");

            Assert.True(MathF.Abs(actualCosine - (1.0f - actualDot)) <= 1e-6f,
                $"CosineDistance definition mismatch for dim {dim}: expected {1.0f - actualDot}, got {actualCosine}");

            Assert.True(MathF.Abs(actualL2Sq - (float)expectedL2Sq) <= tolerance,
                $"L2DistanceSquared mismatch for dim {dim}: expected {expectedL2Sq}, got {actualL2Sq}");

            Assert.True(MathF.Abs(actualL2 - MathF.Sqrt(actualL2Sq)) <= 1e-5f,
                $"L2Distance mismatch for dim {dim}: expected {MathF.Sqrt(actualL2Sq)}, got {actualL2}");
        }
    }

    [Fact]
    public void HnswSearch_SpanOverload_MatchesArrayOverload()
    {
        const int dims = 16;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 8, efConstruction: 32, initialCapacity: 64);

        for (int i = 0; i < 50; i++)
        {
            float[] v = new float[dims];
            v[i % dims] = 1.0f;
            hnsw.Add(v, $"meta_{i}");
        }

        float[] query = new float[dims];
        query[2] = 1.0f;

        var arrayResults = hnsw.Search(query, topK: 5, efSearch: 32);

        var spanResultsArray = new SearchResult[5];
        Span<SearchResult> spanResults = spanResultsArray.AsSpan();
        int count = hnsw.Search(query, spanResults, efSearch: 32);

        Assert.Equal(arrayResults.Length, count);
        for (int i = 0; i < count; i++)
        {
            Assert.Equal(arrayResults[i].Id, spanResults[i].Id);
            Assert.Equal(arrayResults[i].Score, spanResults[i].Score, precision: 5);
            Assert.Equal(arrayResults[i].Metadata, spanResults[i].Metadata);
        }
    }
}
