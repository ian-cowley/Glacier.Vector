using System;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

namespace Glacier.Vector.Tests;

public class GpuVectorTests
{
    [Fact]
    public void VectorIndex_Search_GpuMatchesCpuTopResults()
    {
        const int dims = 64;
        const int count = 10_000;
        const int topK = 5;

        using var storage = new InMemoryVectorStorage(dims, vectorsPerChunk: count);
        using var index = new VectorIndex(storage);

        var rng = new Random(42);
        float[] temp = new float[dims];

        for (int i = 0; i < count; i++)
        {
            float sumSq = 0f;
            for (int d = 0; d < dims; d++)
            {
                float v = (float)(rng.NextDouble() * 2.0 - 1.0);
                temp[d] = v;
                sumSq += v * v;
            }
            float invLen = 1.0f / MathF.Sqrt(sumSq);
            for (int d = 0; d < dims; d++) temp[d] *= invLen;

            index.Add(temp, $"Doc_{i}");
        }

        // Target query
        float[] query = new float[dims];
        float qSumSq = 0f;
        for (int d = 0; d < dims; d++)
        {
            query[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
            qSumSq += query[d] * query[d];
        }
        float invQLen = 1.0f / MathF.Sqrt(qSumSq);
        for (int d = 0; d < dims; d++) query[d] *= invQLen;

        // 1. CPU search
        var cpuResults = index.Search(query, topK: topK, target: GpuTarget.Cpu);
        Assert.Equal(topK, cpuResults.Length);

        // 2. Auto search
        var autoResults = index.Search(query, topK: topK, target: GpuTarget.Auto);
        Assert.Equal(topK, autoResults.Length);

        // Verify top-1 ID matches between CPU and Auto
        Assert.Equal(cpuResults[0].Id, autoResults[0].Id);
        Assert.True(MathF.Abs(cpuResults[0].Score - autoResults[0].Score) < 1e-4f);

        // 3. Explicit NVIDIA GPU search if available
        if (GpuVectorAccelerator.IsNvidiaAvailable)
        {
            var nvidiaResults = index.Search(query, topK: topK, target: GpuTarget.Nvidia);
            Assert.Equal(topK, nvidiaResults.Length);
            Assert.Equal(cpuResults[0].Id, nvidiaResults[0].Id);
            Assert.True(MathF.Abs(cpuResults[0].Score - nvidiaResults[0].Score) < 1e-4f);
        }
    }

    [Fact]
    public void VectorIndex_BatchSearch_MatchesSingleQueries()
    {
        const int dims = 32;
        const int count = 2_000;
        const int batchSize = 4;
        const int topK = 3;

        using var storage = new InMemoryVectorStorage(dims, vectorsPerChunk: count);
        using var index = new VectorIndex(storage);

        var rng = new Random(123);
        float[] temp = new float[dims];

        for (int i = 0; i < count; i++)
        {
            for (int d = 0; d < dims; d++)
            {
                temp[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
            }
            index.Add(temp, $"Vec_{i}");
        }

        // Generate batch of queries
        float[] queries = new float[batchSize * dims];
        for (int i = 0; i < queries.Length; i++)
        {
            queries[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        var batchResults = index.BatchSearch(queries, batchSize: batchSize, topK: topK, target: GpuTarget.Auto);
        Assert.Equal(batchSize, batchResults.Length);

        for (int b = 0; b < batchSize; b++)
        {
            ReadOnlySpan<float> q = queries.AsSpan(b * dims, dims);
            var singleResult = index.Search(q, topK: topK, target: GpuTarget.Cpu);

            Assert.Equal(topK, batchResults[b].Length);
            Assert.Equal(singleResult[0].Id, batchResults[b][0].Id);
            Assert.True(MathF.Abs(singleResult[0].Score - batchResults[b][0].Score) < 1e-3f);
        }
    }
}
