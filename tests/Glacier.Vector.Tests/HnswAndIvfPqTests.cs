namespace Glacier.Vector.Tests;

using System;
using System.IO;
using System.Threading.Tasks;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

public class HnswAndIvfPqTests : IDisposable
{
    private readonly string _tempFile;

    public HnswAndIvfPqTests()
    {
        _tempFile = Path.Combine(Path.GetTempPath(), $"glacier_vec_test_{Guid.NewGuid():N}.bin");
    }

    public void Dispose()
    {
        if (File.Exists(_tempFile))
        {
            try { File.Delete(_tempFile); } catch { }
        }
    }

    [Fact]
    public void HnswVectorIndex_InsertAndSearch_ReturnsNearestNeighbors()
    {
        const int dims = 16;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 8, efConstruction: 50, initialCapacity: 128);

        // Add 3 orthogonal basis vectors
        float[] v0 = new float[dims]; v0[0] = 1.0f;
        float[] v1 = new float[dims]; v1[1] = 1.0f;
        float[] v2 = new float[dims]; v2[2] = 1.0f;

        hnsw.Add(v0, "Axis_X");
        hnsw.Add(v1, "Axis_Y");
        hnsw.Add(v2, "Axis_Z");

        Assert.Equal(3, hnsw.Count);

        // Query close to Axis_Y
        float[] query = new float[dims]; query[1] = 0.95f; query[0] = 0.05f;
        var results = hnsw.Search(query, topK: 2, efSearch: 32);

        Assert.Equal(2, results.Length);
        Assert.Equal("Axis_Y", results[0].Metadata);
        Assert.Equal(1, results[0].Id);
        Assert.True(results[0].Score > results[1].Score);
    }

    [Fact]
    public void HnswVectorIndex_ConcurrentSearchAndAdd_IsThreadSafe()
    {
        const int dims = 32;
        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 8, efConstruction: 32, initialCapacity: 512);

        // Pre-populate 50 vectors
        var rng = new Random(42);
        for (int i = 0; i < 50; i++)
        {
            float[] vec = new float[dims];
            for (int d = 0; d < dims; d++) vec[d] = (float)rng.NextDouble();
            hnsw.Add(vec, $"Doc_{i}");
        }

        // Run concurrent readers and writers
        Parallel.For(0, 16, taskIdx =>
        {
            if (taskIdx % 2 == 0)
            {
                // Writer
                for (int i = 0; i < 10; i++)
                {
                    float[] vec = new float[dims];
                    vec[0] = (float)taskIdx;
                    hnsw.Add(vec, $"Writer_{taskIdx}_{i}");
                }
            }
            else
            {
                // Reader
                float[] query = new float[dims];
                query[0] = 1.0f;
                for (int i = 0; i < 20; i++)
                {
                    var res = hnsw.Search(query, topK: 3);
                    Assert.NotEmpty(res);
                }
            }
        });

        Assert.True(hnsw.Count >= 50 + (8 * 10));
    }

    [Fact]
    public void IvfPqIndex_TrainAndSearch_ProducesTopResults()
    {
        const int dims = 32;
        const int count = 200;
        using var ivf = new IvfPqIndex(dims, centroidCount: 16, subVectorCount: 4);

        // Generate training data
        float[] trainingData = new float[count * dims];
        var rng = new Random(99);
        for (int i = 0; i < trainingData.Length; i++)
        {
            trainingData[i] = (float)rng.NextDouble();
        }

        ivf.Train(trainingData, count, iterations: 3);
        Assert.True(ivf.IsTrained);

        // Add vectors
        for (int i = 0; i < 50; i++)
        {
            var slice = trainingData.AsSpan(i * dims, dims);
            ivf.Add(slice, $"Item_{i}");
        }

        Assert.Equal(50, ivf.Count);

        // Query with the first training vector
        var query = trainingData.AsSpan(0, dims);
        var results = ivf.Search(query, topK: 5, nprobe: 8);

        Assert.NotEmpty(results);
        // Vector 0 was the exact query, so it should be among top results
        bool foundZero = false;
        for (int i = 0; i < results.Length; i++)
        {
            if (results[i].Id == 0) foundZero = true;
        }
        Assert.True(foundZero, "Exact match should be found in top results with nprobe=8");
    }

    [Fact]
    public void InMemoryVectorStorage_ConcurrentAppendAndRead_NoExceptions()
    {
        const int dims = 8;
        using var storage = new InMemoryVectorStorage(dims, vectorsPerChunk: 16);

        Parallel.For(0, 10, worker =>
        {
            for (int i = 0; i < 25; i++)
            {
                float[] vec = new float[dims];
                vec[0] = worker;
                storage.Append(vec);
                int count = storage.Count;
                if (count > 0)
                {
                    var span = storage.GetVector(count - 1);
                    Assert.Equal(dims, span.Length);
                }
            }
        });

        Assert.Equal(250, storage.Count);
        Assert.True(storage.ChunkCount > 1);
    }
}
