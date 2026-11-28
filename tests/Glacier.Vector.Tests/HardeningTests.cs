using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

namespace Glacier.Vector.Tests;

public class HardeningTests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    private string NewTempFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"glacier_hardening_{Guid.NewGuid():N}.bin");
        _tempFiles.Add(path);
        _tempFiles.Add(path + ".meta");
        return path;
    }

    public void Dispose()
    {
        foreach (string f in _tempFiles)
        {
            if (File.Exists(f))
            {
                try { File.Delete(f); } catch { }
            }
        }
    }

    // =========================================================================
    // Feature #1: MmfVectorStorage Header Persistence, ReadVector & Safe GetVector
    // =========================================================================

    [Fact]
    public void MmfVectorStorage_CrashSnapshotReopen_PreservesCountAndPayloads()
    {
        const int dims = 16;
        const int vectorCount = 25;
        string livePath = NewTempFile();
        string crashPath = NewTempFile();

        float[][] expected = new float[vectorCount][];
        using (var liveStorage = new MmfVectorStorage(livePath, dims))
        {
            for (int i = 0; i < vectorCount; i++)
            {
                expected[i] = new float[dims];
                for (int d = 0; d < dims; d++) expected[i][d] = i * 100f + d;
                liveStorage.Append(expected[i]);
            }

            // Simulate crash before Dispose() by copying live file bytes (which still have 64MB pre-extended length)
            using (var src = new FileStream(livePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
            using (var dst = new FileStream(crashPath, FileMode.Create, FileAccess.Write, FileShare.None))
            {
                src.CopyTo(dst);
                dst.Flush(true);
            }
        }

        // Reopen the crash snapshot: MMF header must recover Count == 25 (not 0 and not 64MB/64)
        using var recovered = new MmfVectorStorage(crashPath, dims);
        Assert.Equal(vectorCount, recovered.Count);
        Assert.Equal(vectorCount * dims, recovered.GetChunkSpan(0).Length);

        float[] buf = new float[dims];
        for (int i = 0; i < vectorCount; i++)
        {
            recovered.ReadVector(i, buf);
            Assert.Equal(expected[i], buf);
            Assert.Equal(expected[i], recovered.GetVector(i).ToArray());
        }

        // Verify GetChunkSpan starts at vector 0, not header magic
        Assert.Equal(expected[0][0], recovered.GetChunkSpan(0)[0]);
    }

    [Fact]
    public void MmfVectorStorage_DimensionMismatch_AndPostDisposeSafety()
    {
        string path = NewTempFile();
        using (var storage = new MmfVectorStorage(path, 8))
        {
            storage.Append(new float[] { 1, 2, 3, 4, 5, 6, 7, 8 });
        }

        Assert.Throws<InvalidDataException>(() =>
        {
            using var wrongDims = new MmfVectorStorage(path, 16);
        });

        var s2 = new MmfVectorStorage(path, 8);
        ReadOnlySpan<float> safeCopy = s2.GetVector(0);
        s2.Dispose();
        s2.Dispose(); // idempotent

        // Safe copy remains readable even after Dispose() without AccessViolationException
        Assert.Equal(1f, safeCopy[0]);
        Assert.Equal(8f, safeCopy[7]);

        float[] dest = new float[8];
        Assert.Throws<ObjectDisposedException>(() => s2.ReadVector(0, dest));
        Assert.Throws<ObjectDisposedException>(() => s2.Append(dest));
        Assert.Throws<ObjectDisposedException>(() => s2.GetVector(0));
        Assert.Throws<ObjectDisposedException>(() => s2.GetChunkSpan(0));
    }

    [Fact]
    public void MmfVectorStorage_LegacyHeaderless_PreservesMetaOnDispose()
    {
        const int dims = 32;
        string path = NewTempFile();

        // Create raw headerless 128-byte file (1 zero vector)
        using (var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None))
        {
            fs.SetLength(dims * sizeof(float));
        }

        using (var storage = new MmfVectorStorage(path, dims))
        {
            Assert.Equal(1, storage.Count);
            float[] v1 = new float[dims];
            v1[0] = 42.5f;
            storage.Append(v1);
            storage.Append(v1);
            Assert.Equal(3, storage.Count);
        }

        // .meta must NOT be deleted on Dispose()
        Assert.True(File.Exists(path + ".meta"));

        using var reopened = new MmfVectorStorage(path, dims);
        Assert.Equal(3, reopened.Count);
        Assert.Equal(42.5f, reopened.GetVector(1)[0]);
        Assert.Equal(42.5f, reopened.GetVector(2)[0]);
    }

    // =========================================================================
    // Feature #4: PagedVectorStorage, BinaryVectorStorage & SIMD PackBinaryVector
    // =========================================================================

    [Fact]
    public void PagedVectorStorage_ManySimultaneousGetVectorSpans_NeverOverwrite()
    {
        const int dims = 4;
        string path = NewTempFile();
        using var storage = new PagedVectorStorage(path, dims, maxMemoryBytes: 8192);

        for (int i = 0; i < 12; i++)
        {
            storage.Append(new float[] { i, i + 0.1f, i + 0.2f, i + 0.3f });
        }

        // Hold 8 spans simultaneously on the same thread (previously >4 overwrote earlier spans)
        var s0 = storage.GetVector(0);
        var s1 = storage.GetVector(1);
        var s2 = storage.GetVector(2);
        var s3 = storage.GetVector(3);
        var s4 = storage.GetVector(4);
        var s5 = storage.GetVector(5);
        var s6 = storage.GetVector(6);
        var s7 = storage.GetVector(7);

        Assert.Equal(0f, s0[0]);
        Assert.Equal(1f, s1[0]);
        Assert.Equal(2f, s2[0]);
        Assert.Equal(3f, s3[0]);
        Assert.Equal(4f, s4[0]);
        Assert.Equal(5f, s5[0]);
        Assert.Equal(6f, s6[0]);
        Assert.Equal(7f, s7[0]);

        float[] dest = new float[dims];
        storage.ReadVector(11, dest);
        Assert.Equal(11f, dest[0]);

        storage.Dispose();
        storage.Dispose(); // idempotent
        Assert.Throws<ObjectDisposedException>(() => storage.GetVector(0));
        Assert.Throws<ObjectDisposedException>(() => storage.ReadVector(0, dest));
    }

    [Fact]
    public void BinaryVectorStorage_ConcurrentAppendRead_AndTailBitMasking()
    {
        const int dims = 67; // 2 words, 3 valid bits in word 1
        using var masked = new BinaryVectorStorage(dims, initialCapacity: 4);
        masked.AppendPacked(new ulong[] { ulong.MaxValue, ulong.MaxValue });
        Assert.Equal(0x7UL, masked.GetPackedVector(0)[1]);

        using var concurrent = new BinaryVectorStorage(128, initialCapacity: 4);
        var errors = new ConcurrentBag<Exception>();

        Parallel.For(0, 12, worker =>
        {
            try
            {
                float[] vec = new float[128];
                Array.Fill(vec, worker % 2 == 0 ? 1.0f : -1.0f);
                for (int i = 0; i < 100; i++)
                {
                    concurrent.Append(vec);
                    int c = concurrent.Count;
                    if (c > 0)
                    {
                        var packed = concurrent.GetPackedVector(c - 1);
                        Assert.Equal(2, packed.Length);
                    }
                }
            }
            catch (Exception ex)
            {
                errors.Add(ex);
            }
        });

        Assert.Empty(errors);
        Assert.Equal(1200, concurrent.Count);

        concurrent.Dispose();
        concurrent.Dispose();
        Assert.Throws<ObjectDisposedException>(() => concurrent.GetPackedVector(0));
    }

    [Fact]
    public void PackBinaryVector_SimdMatchesBitOracle_WithNaNAndSignedZero()
    {
        int[] lengths = [0, 1, 3, 4, 7, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 255, 256, 513];
        var rng = new Random(2026);

        foreach (int len in lengths)
        {
            float[] src = new float[len];
            for (int i = 0; i < len; i++)
            {
                src[i] = (i % 7) switch
                {
                    0 => 0.0f,
                    1 => -0.0f,
                    2 => float.NaN,
                    3 => float.PositiveInfinity,
                    4 => float.NegativeInfinity,
                    5 => (float)(rng.NextDouble() + 0.01),
                    _ => (float)(-rng.NextDouble() - 0.01)
                };
            }

            int u64Words = (len + 63) / 64;
            ulong[] u64Actual = new ulong[u64Words];
            DistanceKernels.PackBinaryVector(src, u64Actual);

            for (int i = 0; i < len; i++)
            {
                bool expected = src[i] >= 0.0f;
                bool actual = (u64Actual[i / 64] & (1UL << (i % 64))) != 0;
                Assert.Equal(expected, actual);
            }
            if (len % 64 != 0)
            {
                Assert.Equal(0UL, u64Actual[^1] & ~((1UL << (len % 64)) - 1UL));
            }

            int u32Words = (len + 31) / 32;
            uint[] u32Actual = new uint[u32Words];
            DistanceKernels.PackBinaryVector(src, u32Actual);

            for (int i = 0; i < len; i++)
            {
                bool expected = src[i] >= 0.0f;
                bool actual = (u32Actual[i / 32] & (1U << (i % 32))) != 0;
                Assert.Equal(expected, actual);
            }
            if (len % 32 != 0)
            {
                Assert.Equal(0U, u32Actual[^1] & ~((1U << (len % 32)) - 1U));
            }
        }
    }

    // =========================================================================
    // Feature #3: HnswVectorIndex & VectorIndex Hardening
    // =========================================================================

    [Fact]
    public void HnswVectorIndex_InsertIsInternal_ValidatesInputs_AndSupportsMmfAndLargeM()
    {
        // Insert must not be public; it is internal
        Assert.Null(typeof(HnswVectorIndex).GetMethod("Insert", BindingFlags.Instance | BindingFlags.Public));
        var internalInsert = typeof(HnswVectorIndex).GetMethod("Insert", BindingFlags.Instance | BindingFlags.NonPublic);
        Assert.NotNull(internalInsert);
        Assert.True(internalInsert!.IsAssembly);

        using var memStorage = new InMemoryVectorStorage(8);
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswVectorIndex(memStorage, m: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswVectorIndex(memStorage, efConstruction: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new HnswVectorIndex(memStorage, initialCapacity: 0));

        // Large m > 128 exercises ArrayPool fallback in SelectAndLinkNeighbors (count > 256)
        using (var largeMStorage = new InMemoryVectorStorage(8))
        using (var largeMHnsw = new HnswVectorIndex(largeMStorage, m: 200, efConstruction: 320, initialCapacity: 64))
        {
            var rng = new Random(7);
            for (int i = 0; i < 40; i++)
            {
                float[] v = new float[8];
                v[i % 8] = 1.0f;
                v[(i + 1) % 8] = (float)rng.NextDouble() * 0.1f;
                largeMHnsw.Add(v, $"n_{i}");
            }
            var res = largeMHnsw.Search(new float[] { 1, 0, 0, 0, 0, 0, 0, 0 }, 5);
            Assert.Equal(5, res.Length);
        }

        // HnswVectorIndex over MmfVectorStorage + internal Insert validation + Dispose checks
        string mmfPath = NewTempFile();
        using var mmfStorage = new MmfVectorStorage(mmfPath, 8);
        using var hnsw = new HnswVectorIndex(mmfStorage, m: 8, efConstruction: 32, initialCapacity: 32);

        float[] v0 = new float[8]; v0[0] = 1f;
        hnsw.Add(v0, "v0");

        Assert.Throws<ArgumentOutOfRangeException>(() => hnsw.Insert(-1, v0));
        Assert.Throws<ArgumentOutOfRangeException>(() => hnsw.Insert(1, v0));
        Assert.Throws<InvalidOperationException>(() => hnsw.Insert(0, v0));
        Assert.Throws<ArgumentException>(() => hnsw.Insert(0, new float[4]));

        mmfStorage.Append(new float[] { 0, 1, 0, 0, 0, 0, 0, 0 });
        hnsw.Insert(1, new float[] { 0, 1, 0, 0, 0, 0, 0, 0 });
        Assert.Equal(2, hnsw.Count);

        var mmfHits = hnsw.Search(new float[] { 0, 1, 0, 0, 0, 0, 0, 0 }, 2);
        Assert.Equal(1, mmfHits[0].Id);

        hnsw.Dispose();
        hnsw.Dispose(); // idempotent
        Assert.Throws<ObjectDisposedException>(() => hnsw.Add(v0));
        Assert.Throws<ObjectDisposedException>(() => hnsw.Insert(0, v0));
        Assert.Throws<ObjectDisposedException>(() => hnsw.Search(v0, 1));
    }

    [Fact]
    public void VectorIndex_HighDimSearchWithReRanking_UsesArrayPoolWithoutStackOverflow()
    {
        // 20,000 dims -> 313 ulong words (> 256 stackalloc limit)
        const int dims = 20_000;
        using var storage = new InMemoryVectorStorage(dims, vectorsPerChunk: 32);
        using var index = new VectorIndex(storage);

        for (int i = 0; i < 10; i++)
        {
            float[] v = new float[dims];
            Array.Fill(v, -0.01f);
            v.AsSpan(i * 1000, 1000).Fill(1.0f);
            index.Add(v, $"hi_{i}");
        }

        float[] query = new float[dims];
        Array.Fill(query, -0.01f);
        query.AsSpan(7 * 1000, 1000).Fill(1.0f);

        var res = index.SearchWithReRanking(query, k: 3, oversampleFactor: 2);
        Assert.Equal(3, res.Length);
        Assert.Equal("hi_7", res[0].Metadata);
    }

    // =========================================================================
    // Feature #5: IvfPqIndex Contiguous Arena, ValueMaxHeap & Parallel K-Means
    // =========================================================================

    [Fact]
    public void IvfPqIndex_ParallelTrain_LowAllocSearch_AndValidation()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new IvfPqIndex(0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IvfPqIndex(32, centroidCount: 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => new IvfPqIndex(32, centroidCount: 8, subVectorCount: 0));

        const int dims = 32;
        const int count = 512;
        var rng = new Random(999);
        float[] data = new float[count * dims];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        using var ivf = new IvfPqIndex(dims, centroidCount: 16, subVectorCount: 4);
        Assert.Throws<ArgumentOutOfRangeException>(() => ivf.Train(data, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ivf.Train(data, count, iterations: 0));

        ivf.Train(data, count, iterations: 4);
        Assert.True(ivf.IsTrained);

        for (int i = 0; i < count; i++)
        {
            ivf.Add(data.AsSpan(i * dims, dims), $"doc_{i}");
        }
        Assert.Equal(count, ivf.Count);

        float[] query = data.AsSpan(42 * dims, dims).ToArray();

        // Warmup JIT & ArrayPool
        _ = ivf.Search(query, topK: 5, nprobe: 8);

        long before = GC.GetAllocatedBytesForCurrentThread();
        var results = ivf.Search(query, topK: 5, nprobe: 8);
        long after = GC.GetAllocatedBytesForCurrentThread();

        Assert.Equal(5, results.Length);
        Assert.Contains(results, r => r.Id == 42);
        // Only SearchResult[5] array should be allocated (no PriorityQueue or byte[] per query)
        Assert.True(after - before <= 512, $"Expected <= 512 bytes allocated during Search, got {after - before}.");

        ivf.Dispose();
        ivf.Dispose();
        Assert.Throws<ObjectDisposedException>(() => ivf.Train(data, count));
        Assert.Throws<ObjectDisposedException>(() => ivf.Add(query));
        Assert.Throws<ObjectDisposedException>(() => ivf.Search(query, 5));
    }
}
