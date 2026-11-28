using System;
using System.Collections.Concurrent;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

namespace Glacier.Vector.Tests;

[CollectionDefinition("Gpu", DisableParallelization = true)]
public sealed class GpuCollectionDefinition
{
}

[Collection("Gpu")]
public class GpuHardeningTests
{
    [Fact]
    [Trait("Hardware", "Gpu")]
    public void GpuTarget_AmdAndDualGpu_ThrowNotSupportedException_AcrossAllApis()
    {
        Assert.False(GpuVectorAccelerator.IsAmdAvailable);

        float[] q = new float[16];
        float[] db = new float[64];
        float[] scores = new float[4];
        sbyte[] iq = new sbyte[16];
        sbyte[] idb = new sbyte[64];

        foreach (var unsupported in new[] { GpuTarget.Amd, GpuTarget.DualGpu })
        {
            Assert.Throws<NotSupportedException>(() =>
                GpuVectorAccelerator.ScanChunkGpu(q, db, scores, 4, 16, false, unsupported));

            Assert.Throws<NotSupportedException>(() =>
                GpuVectorAccelerator.ScanChunkInt8Gpu(iq, 1f, idb, 1f, scores, 4, 16, unsupported));

            Assert.Throws<NotSupportedException>(() =>
                GpuVectorAccelerator.BatchScanGpu(db, q, scores, 4, 16, 1, unsupported));

            using var storage = new InMemoryVectorStorage(16);
            using var index = new VectorIndex(storage);
            index.Add(q, "doc");

            Assert.Throws<NotSupportedException>(() => index.Search(q, 1, unsupported));
            Assert.Throws<NotSupportedException>(() => index.BatchSearch(q, 1, 1, unsupported));
        }
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void GpuVectorAccelerator_IntOverflowGuard_ReturnsFalseSafely()
    {
        float[] small = new float[16];
        float[] scores = new float[4];

        // 1_000_000 * 4096 overflows 32-bit int to negative; (long) guard must catch and return false
        Assert.False(GpuVectorAccelerator.ScanChunkGpu(small, small, scores, 1_000_000, 4096, false, GpuTarget.Cpu));
        Assert.False(GpuVectorAccelerator.BatchScanGpu(small, small, scores, 1_000_000, 4096, 1, GpuTarget.Cpu));
        Assert.False(GpuVectorAccelerator.BatchScanGpu(small, small, scores, 4, 4, 1_000_000_000, GpuTarget.Cpu));
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void GpuVectorAccelerator_MultiChunkLruCache_AndInt8CacheRetention_OnNvidia()
    {
        if (!GpuVectorAccelerator.IsNvidiaAvailable) return;

        GpuVectorAccelerator.ReleaseGpuResources();
        Assert.Equal(0, GpuVectorAccelerator.CachedEntryCount);

        const int dims = 128;
        const int count = 512;
        var rng = new Random(1337);

        float[] q = new float[dims];
        for (int d = 0; d < dims; d++) q[d] = (float)(rng.NextDouble() * 2.0 - 1.0);

        // Cache 3 distinct FP32 chunks
        float[][] chunks = new float[3][];
        for (int c = 0; c < 3; c++)
        {
            chunks[c] = new float[count * dims];
            for (int i = 0; i < chunks[c].Length; i++) chunks[c][i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        float[] scores = new float[count];
        long missesBefore = GpuVectorAccelerator.CacheMisses;
        long hitsBefore = GpuVectorAccelerator.CacheHits;

        for (int c = 0; c < 3; c++)
        {
            Assert.True(GpuVectorAccelerator.ScanChunkGpu(q, chunks[c], scores, count, dims, false, GpuTarget.Nvidia, storageId: 9001, chunkIndex: c, storageVersion: 1));
            Assert.True(GpuVectorAccelerator.IsEntryCached(9001, c, isInt8: false));
        }

        // Second pass over all 3 chunks should hit VRAM cache 3 times without re-uploading
        for (int c = 0; c < 3; c++)
        {
            Assert.True(GpuVectorAccelerator.ScanChunkGpu(q, chunks[c], scores, count, dims, false, GpuTarget.Nvidia, storageId: 9001, chunkIndex: c, storageVersion: 1));
        }

        Assert.True(GpuVectorAccelerator.CacheMisses - missesBefore >= 3);
        Assert.True(GpuVectorAccelerator.CacheHits - hitsBefore >= 3);

        // INT8 scan with storageId=9002 must cache INT8 entry AND NOT evict FP32 entries for 9001
        sbyte[] iq = new sbyte[dims];
        sbyte[] idb = new sbyte[count * dims];
        for (int d = 0; d < dims; d++) iq[d] = (sbyte)(d % 13 - 6);
        for (int i = 0; i < idb.Length; i++) idb[i] = (sbyte)(i % 19 - 9);

        long hitsBeforeInt8 = GpuVectorAccelerator.CacheHits;
        Assert.True(GpuVectorAccelerator.ScanChunkInt8Gpu(iq, 0.01f, idb, 0.02f, scores, count, dims, GpuTarget.Nvidia, storageId: 9002, chunkIndex: 0, storageVersion: 1));
        Assert.True(GpuVectorAccelerator.ScanChunkInt8Gpu(iq, 0.01f, idb, 0.02f, scores, count, dims, GpuTarget.Nvidia, storageId: 9002, chunkIndex: 0, storageVersion: 1));

        Assert.True(GpuVectorAccelerator.IsEntryCached(9001, 0, isInt8: false));
        Assert.True(GpuVectorAccelerator.IsEntryCached(9002, 0, isInt8: true));
        Assert.True(GpuVectorAccelerator.CacheHits - hitsBeforeInt8 >= 1);

        // Exceed MaxCacheSlots (16) -> LRU eviction keeps cache bounded <= 16
        for (int s = 0; s < 18; s++)
        {
            Assert.True(GpuVectorAccelerator.ScanChunkGpu(q, chunks[0], scores, count, dims, false, GpuTarget.Nvidia, storageId: 10_000 + s, chunkIndex: 0, storageVersion: 1));
        }
        Assert.True(GpuVectorAccelerator.CachedEntryCount <= 16);
    }

    [Fact]
    [Trait("Hardware", "Gpu")]
    public void GpuVectorAccelerator_ConcurrentMultiStreamScans_AndReleaseReinit_MatchCpu()
    {
        if (!GpuVectorAccelerator.IsNvidiaAvailable) return;

        const int dims = 128;
        const int count = 1024;
        var rng = new Random(4060);

        float[] q = new float[dims];
        float[] db = new float[count * dims];
        for (int d = 0; d < dims; d++) q[d] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < db.Length; i++) db[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] cpuExpected = new float[count];
        for (int i = 0; i < count; i++)
            cpuExpected[i] = DistanceKernels.DotProduct(q, db.AsSpan(i * dims, dims));

        var errors = new ConcurrentBag<string>();
        Parallel.For(0, 12, worker =>
        {
            try
            {
                float[] localScores = new float[count];
                // Mix cached (storageId > 0) and uncached ephemeral (storageId == 0) concurrently
                long sid = (worker % 2 == 0) ? 77700 + (worker % 3) : 0;
                bool ok = GpuVectorAccelerator.ScanChunkGpu(q, db, localScores, count, dims, false, GpuTarget.Nvidia, sid, 0, 1);
                if (!ok)
                {
                    errors.Add($"Worker {worker} ScanChunkGpu returned false.");
                    return;
                }
                for (int i = 0; i < count; i++)
                {
                    if (MathF.Abs(cpuExpected[i] - localScores[i]) > 1e-3f)
                    {
                        errors.Add($"Worker {worker} mismatch at {i}: cpu={cpuExpected[i]} gpu={localScores[i]}");
                        break;
                    }
                }
            }
            catch (Exception ex)
            {
                errors.Add($"Worker {worker} exception: {ex.Message}");
            }
        });

        Assert.Empty(errors);

        // Release all CUDA resources and verify clean re-initialization
        GpuVectorAccelerator.ReleaseGpuResources();
        Assert.Equal(0, GpuVectorAccelerator.CachedEntryCount);
        Assert.True(GpuVectorAccelerator.IsNvidiaAvailable);

        float[] postReleaseScores = new float[count];
        Assert.True(GpuVectorAccelerator.ScanChunkGpu(q, db, postReleaseScores, count, dims, false, GpuTarget.Nvidia, 8888, 0, 1));
        for (int i = 0; i < count; i++)
        {
            Assert.True(MathF.Abs(cpuExpected[i] - postReleaseScores[i]) <= 1e-3f);
        }
    }
}
