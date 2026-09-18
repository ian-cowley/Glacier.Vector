namespace Glacier.Vector.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

public class AdversarialStressTests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    private string CreateTempFile(long initialBytes = 0)
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_vec_stress_{Guid.NewGuid():N}.bin");
        _tempFiles.Add(path);
        if (initialBytes > 0)
        {
            using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
            fs.SetLength(initialBytes);
        }
        return path;
    }

    public void Dispose()
    {
        foreach (var file in _tempFiles)
        {
            if (File.Exists(file))
            {
                try { File.Delete(file); } catch { }
            }
        }
    }

    private static float[] GenerateNormalizedVector(int dims, Random rng)
    {
        float[] v = new float[dims];
        float sumSq = 0f;
        for (int i = 0; i < dims; i++)
        {
            v[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            sumSq += v[i] * v[i];
        }
        float invNorm = 1.0f / (float)Math.Sqrt(Math.Max(1e-8f, sumSq));
        for (int i = 0; i < dims; i++)
        {
            v[i] *= invNorm;
        }
        return v;
    }

    [Fact]
    public void MmfVectorStorage_ConcurrentRemapping_ZeroAccessViolations()
    {
        const int dims = 32;
        int vectorBytes = dims * sizeof(float);

        // Create initial file with exactly 1 vector (32 floats = 128 bytes)
        // so that the very next append forces ExpandMapping()
        string filePath = CreateTempFile(vectorBytes);

        using var storage = new MmfVectorStorage(filePath, dims);
        Assert.Equal(1, storage.Count);

        const int readerCount = 6;
        const int writerCount = 3;
        var startGate = new ManualResetEventSlim(false);
        var stopGate = new CancellationTokenSource();
        int readCount = 0;
        int writeCount = 0;
        Exception? readerException = null;

        var readers = Enumerable.Range(0, readerCount).Select(rId => new Thread(() =>
        {
            startGate.Wait();
            var rng = new Random(rId * 100);
            while (!stopGate.Token.IsCancellationRequested)
            {
                try
                {
                    int currentCount = storage.Count;
                    if (currentCount > 0)
                    {
                        int targetIdx = rng.Next(0, currentCount);
                        var span = storage.GetVector(targetIdx);
                        float sum = 0f;
                        for (int d = 0; d < span.Length; d++)
                        {
                            sum += span[d];
                        }
                        Interlocked.Increment(ref readCount);
                    }
                }
                catch (Exception ex)
                {
                    Volatile.Write(ref readerException, ex);
                    break;
                }
            }
        })).ToArray();

        var writers = Enumerable.Range(0, writerCount).Select(wId => new Thread(() =>
        {
            startGate.Wait();
            var rng = new Random(1000 + wId);
            for (int i = 0; i < 300; i++)
            {
                float[] vec = GenerateNormalizedVector(dims, rng);
                storage.Append(vec);
                Interlocked.Increment(ref writeCount);
                if (i % 20 == 0) Thread.Sleep(1);
            }
        })).ToArray();

        foreach (var r in readers) r.Start();
        foreach (var w in writers) w.Start();

        // Release start gate to start concurrent execution simultaneously
        startGate.Set();

        foreach (var w in writers) w.Join();

        // Allow readers to continue for 100ms after all writes complete
        Thread.Sleep(100);
        stopGate.Cancel();
        foreach (var r in readers) r.Join();

        Assert.Null(readerException);
        Assert.True(readCount > 100, $"Expected >100 successful reads, got {readCount}");
        Assert.True(storage.Count >= 1 + (writerCount * 300));
    }

    [Fact]
    public void VectorIndex_ConcurrentSearchesAndAdds_ZeroAccessViolations()
    {
        const int dims = 16;
        using var storage = new InMemoryVectorStorage(dims, vectorsPerChunk: 32);
        using var index = new VectorIndex(storage);

        // Pre-populate initial vectors
        var rng = new Random(42);
        for (int i = 0; i < 64; i++)
        {
            index.Add(GenerateNormalizedVector(dims, rng), $"Init_{i}");
        }

        const int readerCount = 6;
        const int writerCount = 3;
        var startGate = new ManualResetEventSlim(false);
        var stopGate = new CancellationTokenSource();
        int completedSearches = 0;
        Exception? searchEx = null;

        var readers = Enumerable.Range(0, readerCount).Select(rId => new Thread(() =>
        {
            startGate.Wait();
            var localRng = new Random(rId * 200);
            while (!stopGate.Token.IsCancellationRequested)
            {
                try
                {
                    float[] query = GenerateNormalizedVector(dims, localRng);
                    var results = index.Search(query, topK: 5);
                    Assert.NotEmpty(results);
                    Interlocked.Increment(ref completedSearches);
                }
                catch (Exception ex)
                {
                    Volatile.Write(ref searchEx, ex);
                    break;
                }
            }
        })).ToArray();

        var writers = Enumerable.Range(0, writerCount).Select(wId => new Thread(() =>
        {
            startGate.Wait();
            var localRng = new Random(3000 + wId);
            for (int i = 0; i < 100; i++)
            {
                float[] vec = GenerateNormalizedVector(dims, localRng);
                index.Add(vec, $"Worker_{wId}_{i}");
                Thread.Sleep(1);
            }
        })).ToArray();

        foreach (var r in readers) r.Start();
        foreach (var w in writers) w.Start();

        startGate.Set();

        foreach (var w in writers) w.Join();

        Thread.Sleep(150);
        stopGate.Cancel();
        foreach (var r in readers) r.Join();

        Assert.Null(searchEx);
        Assert.True(completedSearches > 10, $"Expected concurrent searches to execute safely, got {completedSearches}");
        Assert.True(index.Count >= 64 + (writerCount * 100));
    }

    [Fact]
    public void HnswVectorIndex_RecallAgainstFlatScan_ExceedsThreshold()
    {
        const int dims = 32;
        const int count = 250;
        const int topK = 5;
        const int queryCount = 20;

        var rng = new Random(12345);
        using var flatStorage = new InMemoryVectorStorage(dims);
        using var flatIndex = new VectorIndex(flatStorage);

        using var hnswStorage = new InMemoryVectorStorage(dims);
        using var hnswIndex = new HnswVectorIndex(hnswStorage, m: 16, efConstruction: 64, initialCapacity: count);

        float[][] dataset = new float[count][];
        for (int i = 0; i < count; i++)
        {
            dataset[i] = GenerateNormalizedVector(dims, rng);
            flatIndex.Add(dataset[i], $"Doc_{i}");
            hnswIndex.Add(dataset[i], $"Doc_{i}");
        }

        // Test recall over multiple queries
        int totalFound = 0;
        int totalExpected = queryCount * topK;

        for (int q = 0; q < queryCount; q++)
        {
            float[] query = GenerateNormalizedVector(dims, rng);

            // Ground truth from flat scan
            var groundTruth = flatIndex.Search(query, topK);
            var groundTruthIds = new HashSet<int>(groundTruth.Select(r => r.Id));

            // HNSW search
            var hnswResults = hnswIndex.Search(query, topK, efSearch: 48);

            foreach (var res in hnswResults)
            {
                if (groundTruthIds.Contains(res.Id))
                {
                    totalFound++;
                }
            }
        }

        double recall = (double)totalFound / totalExpected;
        Assert.True(recall >= 0.90, $"HNSW Recall@5 was {recall:P1}, expected >= 90%");
    }

    [Fact]
    public void IvfPqIndex_RecallAgainstFlatScan_ReturnsValidNearestNeighbors()
    {
        const int dims = 32;
        const int trainCount = 300;
        const int topK = 5;
        const int queryCount = 15;

        var rng = new Random(54321);
        float[] allData = new float[trainCount * dims];
        for (int i = 0; i < trainCount; i++)
        {
            float[] vec = GenerateNormalizedVector(dims, rng);
            Array.Copy(vec, 0, allData, i * dims, dims);
        }

        using var ivf = new IvfPqIndex(dims, centroidCount: 16, subVectorCount: 4);
        ivf.Train(allData, trainCount, iterations: 5);
        Assert.True(ivf.IsTrained);

        using var flatStorage = new InMemoryVectorStorage(dims);
        using var flatIndex = new VectorIndex(flatStorage);

        for (int i = 0; i < trainCount; i++)
        {
            var slice = allData.AsSpan(i * dims, dims);
            ivf.Add(slice, $"Item_{i}");
            flatIndex.Add(slice, $"Item_{i}");
        }

        int totalMatches = 0;
        int totalExpected = queryCount * topK;

        for (int q = 0; q < queryCount; q++)
        {
            float[] query = GenerateNormalizedVector(dims, rng);
            var groundTruth = flatIndex.Search(query, topK);
            var gtSet = new HashSet<int>(groundTruth.Select(r => r.Id));

            var ivfResults = ivf.Search(query, topK, nprobe: 12);
            Assert.NotEmpty(ivfResults);

            foreach (var r in ivfResults)
            {
                if (gtSet.Contains(r.Id))
                {
                    totalMatches++;
                }
            }
        }

        double recall = (double)totalMatches / totalExpected;
        Assert.True(recall >= 0.25, $"IVF-PQ Recall@5 was {recall:P1}, expected >= 25%");
    }

    [Fact]
    public void HnswVectorIndex_ConcurrentQueries_MaintainsRecallAndSafety()
    {
        const int dims = 16;
        const int count = 120;
        var rng = new Random(777);

        using var storage = new InMemoryVectorStorage(dims);
        using var hnsw = new HnswVectorIndex(storage, m: 8, efConstruction: 40);

        for (int i = 0; i < count; i++)
        {
            hnsw.Add(GenerateNormalizedVector(dims, rng), $"Vec_{i}");
        }

        Parallel.For(0, 16, workerId =>
        {
            var localRng = new Random(workerId * 100);
            for (int i = 0; i < 30; i++)
            {
                float[] query = GenerateNormalizedVector(dims, localRng);
                var results = hnsw.Search(query, topK: 3, efSearch: 20);
                Assert.Equal(3, results.Length);
                Assert.True(results[0].Score >= results[1].Score);
                Assert.True(results[1].Score >= results[2].Score);
            }
        });
    }
}
