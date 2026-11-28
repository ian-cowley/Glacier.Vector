using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Gpu.Compilation;
using Glacier.Gpu.Drivers;
using Glacier.Vector.Core;

namespace Glacier.Vector.Compute;

/// <summary>
/// Hardware accelerator engine for Glacier.Vector.
/// Dispatches vector similarity scans and batch matrix operations to CUDA-capable NVIDIA GPUs
/// using pooled non-blocking CUDA streams, pinned host staging buffers, and a bounded multi-slot LRU VRAM cache,
/// with multi-core AVX-512 / AVX2 / Vector128 CPU fallback when GPU acceleration is unavailable.
/// </summary>
public static unsafe class GpuVectorAccelerator
{
    private const int MaxExecutionSlots = 4;
    private const int MaxCacheSlots = 16;
    private const nuint StagingChunkBytes = (nuint)(16 * 1024 * 1024);

    private static readonly Lock s_initLock = new();
    private static readonly object s_poolLock = new();
    private static readonly Lock s_cacheLock = new();
    private static readonly ReaderWriterLockSlim s_lifecycleLock = new(LockRecursionPolicy.SupportsRecursion);

    private static bool s_nvidiaInitialized;
    private static bool s_nvidiaAvailable;
    private static IntPtr s_cuContext;
    private static IntPtr s_cuModule;
    private static IntPtr s_fnDotScan;
    private static IntPtr s_fnDotScanSplitK;
    private static IntPtr s_fnDotScanInt8Dp4a;
    private static IntPtr s_fnBatchDotScan;
    private static IntPtr s_fnL2Scan;

    private static readonly List<CudaExecutionSlot> s_allSlots = new();
    private static readonly Stack<CudaExecutionSlot> s_freeSlots = new();
    private static int s_inUseSlots;

    private static readonly Dictionary<VramCacheKey, VramCacheEntry> s_vramCache = new();
    private static nuint s_cachedTotalBytes;
    private static nuint s_vramCacheBudgetBytes = (nuint)(512 * 1024 * 1024);
    private static long s_lruTick;
    private static long s_cacheHits;
    private static long s_cacheMisses;

    static GpuVectorAccelerator()
    {
        AppDomain.CurrentDomain.ProcessExit += static (_, _) =>
        {
            try { ReleaseGpuResources(); } catch { }
        };
    }

    private enum GpuDataKind : byte
    {
        Fp32 = 0,
        Int8 = 1
    }

    private readonly record struct VramCacheKey(long StorageId, int ChunkIndex, GpuDataKind Kind);

    private sealed class VramCacheEntry
    {
        public VramCacheKey Key;
        public long StorageVersion;
        public nuint ByteSize;
        public IntPtr DevicePtr;
        public long LastUsedTick;
        public int ActiveUsers;
        public bool PendingFree;
    }

    private sealed class CudaExecutionSlot
    {
        public IntPtr Stream;
        public IntPtr DQuery;
        public nuint CapQuery;
        public IntPtr DScores;
        public nuint CapScores;
        public IntPtr DEphemeralDb;
        public nuint CapEphemeralDb;
        public IntPtr HQueryPinned;
        public nuint CapHQueryPinned;
        public IntPtr HScoresPinned;
        public nuint CapHScoresPinned;
        public IntPtr HStagingPinned;
        public nuint CapHStagingPinned;
    }

    internal static int CachedEntryCount
    {
        get
        {
            lock (s_cacheLock) return s_vramCache.Count;
        }
    }

    internal static long CacheHits => Interlocked.Read(ref s_cacheHits);
    internal static long CacheMisses => Interlocked.Read(ref s_cacheMisses);

    internal static bool IsEntryCached(long storageId, int chunkIndex, bool isInt8 = false)
    {
        lock (s_cacheLock)
        {
            return s_vramCache.ContainsKey(new VramCacheKey(storageId, chunkIndex, isInt8 ? GpuDataKind.Int8 : GpuDataKind.Fp32));
        }
    }

    public static bool IsNvidiaAvailable => EnsureNvidiaInitialized();

    /// <summary>
    /// Gets whether an AMD GPU compute kernel backend is available for vector scans.
    /// Returns false until native HIP/Vulkan compute kernels are loaded, ensuring callers use the multi-core SIMD path honestly.
    /// </summary>
    public static bool IsAmdAvailable => false;

    public static bool IsGpuAvailable => IsNvidiaAvailable;

    #region Driver Initialization & Cleanup

    private static bool EnsureNvidiaInitialized()
    {
        if (Volatile.Read(ref s_nvidiaInitialized)) return Volatile.Read(ref s_nvidiaAvailable);
        lock (s_initLock)
        {
            if (s_nvidiaInitialized) return s_nvidiaAvailable;
            try
            {
                if (!CuDriver.IsAvailable())
                {
                    Volatile.Write(ref s_nvidiaAvailable, false);
                    return false;
                }

                if (CuDriver.Init(0) != 0)
                {
                    Volatile.Write(ref s_nvidiaAvailable, false);
                    return false;
                }

                if (CuDriver.DeviceGet(out int dev, 0) != 0)
                {
                    Volatile.Write(ref s_nvidiaAvailable, false);
                    return false;
                }

                string targetArch = CuDriver.GetComputeCapability(dev);

                if (CuDriver.CtxCreate(out s_cuContext, 0, dev) != 0)
                {
                    s_cuContext = IntPtr.Zero;
                    Volatile.Write(ref s_nvidiaAvailable, false);
                    return false;
                }

                string ptx = KernelCompiler.NormalizePtxDirectives(GpuVectorKernels.PtxSource, targetArch);
                byte[] ptxBytes = Encoding.UTF8.GetBytes(ptx + "\0");
                if (CuDriver.ModuleLoadData(out s_cuModule, ptxBytes) != 0)
                {
                    s_cuModule = IntPtr.Zero;
                    RollbackPartialInit();
                    Volatile.Write(ref s_nvidiaAvailable, false);
                    return false;
                }

                if (CuDriver.ModuleGetFunction(out s_fnDotScan, s_cuModule, "vector_dot_scan_fp32") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnDotScanSplitK, s_cuModule, "vector_dot_scan_splitk_fp32") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnDotScanInt8Dp4a, s_cuModule, "vector_dot_scan_int8_dp4a") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnBatchDotScan, s_cuModule, "vector_batch_dot_scan_fp32") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnL2Scan, s_cuModule, "vector_l2_scan_fp32") != 0)
                {
                    RollbackPartialInit();
                    Volatile.Write(ref s_nvidiaAvailable, false);
                    return false;
                }

                if (CuDriver.MemGetInfo(out _, out nuint totalVram) == 0 && totalVram > 0)
                {
                    nuint half = totalVram / 2;
                    nuint twoGb = (nuint)(2L * 1024 * 1024 * 1024);
                    nuint minBudget = (nuint)(64 * 1024 * 1024);
                    s_vramCacheBudgetBytes = Math.Max(minBudget, Math.Min(twoGb, half));
                }

                Volatile.Write(ref s_nvidiaAvailable, true);
            }
            catch
            {
                RollbackPartialInit();
                Volatile.Write(ref s_nvidiaAvailable, false);
            }
            finally
            {
                Volatile.Write(ref s_nvidiaInitialized, true);
            }

            return Volatile.Read(ref s_nvidiaAvailable);
        }
    }

    private static void RollbackPartialInit()
    {
        s_fnDotScan = IntPtr.Zero;
        s_fnDotScanSplitK = IntPtr.Zero;
        s_fnDotScanInt8Dp4a = IntPtr.Zero;
        s_fnBatchDotScan = IntPtr.Zero;
        s_fnL2Scan = IntPtr.Zero;

        if (s_cuModule != IntPtr.Zero)
        {
            try { CuDriver.ModuleUnload(s_cuModule); } catch { }
            s_cuModule = IntPtr.Zero;
        }
        if (s_cuContext != IntPtr.Zero)
        {
            try { CuDriver.CtxDestroy(s_cuContext); } catch { }
            s_cuContext = IntPtr.Zero;
        }
    }

    /// <summary>
    /// Releases all pooled CUDA streams, pinned host buffers, cached VRAM allocations,
    /// loaded PTX modules, and the active CUDA context. Safe to call multiple times;
    /// subsequent GPU calls will re-initialize CUDA cleanly.
    /// </summary>
    public static void ReleaseGpuResources()
    {
        s_lifecycleLock.EnterWriteLock();
        try
        {
            lock (s_initLock)
            {
                lock (s_poolLock)
                {
                    while (s_inUseSlots > 0)
                    {
                        Monitor.Wait(s_poolLock, 50);
                    }

                    if (s_cuContext != IntPtr.Zero)
                    {
                        try { CuDriver.CtxSetCurrent(s_cuContext); } catch { }
                    }

                    foreach (var slot in s_allSlots)
                    {
                        DestroySlot(slot);
                    }
                    s_allSlots.Clear();
                    s_freeSlots.Clear();
                }

                lock (s_cacheLock)
                {
                    foreach (var entry in s_vramCache.Values)
                    {
                        if (entry.DevicePtr != IntPtr.Zero)
                        {
                            try { CuDriver.MemFree(entry.DevicePtr); } catch { }
                            entry.DevicePtr = IntPtr.Zero;
                        }
                    }
                    s_vramCache.Clear();
                    s_cachedTotalBytes = 0;
                }

                RollbackPartialInit();
                Volatile.Write(ref s_nvidiaAvailable, false);
                Volatile.Write(ref s_nvidiaInitialized, false);
            }
        }
        finally
        {
            s_lifecycleLock.ExitWriteLock();
        }
    }

    private static void DestroySlot(CudaExecutionSlot slot)
    {
        if (slot.Stream != IntPtr.Zero)
        {
            try { CuDriver.StreamSynchronize(slot.Stream); } catch { }
            try { CuDriver.StreamDestroy(slot.Stream); } catch { }
            slot.Stream = IntPtr.Zero;
        }
        if (slot.DQuery != IntPtr.Zero) { try { CuDriver.MemFree(slot.DQuery); } catch { } slot.DQuery = IntPtr.Zero; slot.CapQuery = 0; }
        if (slot.DScores != IntPtr.Zero) { try { CuDriver.MemFree(slot.DScores); } catch { } slot.DScores = IntPtr.Zero; slot.CapScores = 0; }
        if (slot.DEphemeralDb != IntPtr.Zero) { try { CuDriver.MemFree(slot.DEphemeralDb); } catch { } slot.DEphemeralDb = IntPtr.Zero; slot.CapEphemeralDb = 0; }
        if (slot.HQueryPinned != IntPtr.Zero) { try { CuDriver.MemFreeHost(slot.HQueryPinned); } catch { } slot.HQueryPinned = IntPtr.Zero; slot.CapHQueryPinned = 0; }
        if (slot.HScoresPinned != IntPtr.Zero) { try { CuDriver.MemFreeHost(slot.HScoresPinned); } catch { } slot.HScoresPinned = IntPtr.Zero; slot.CapHScoresPinned = 0; }
        if (slot.HStagingPinned != IntPtr.Zero) { try { CuDriver.MemFreeHost(slot.HStagingPinned); } catch { } slot.HStagingPinned = IntPtr.Zero; slot.CapHStagingPinned = 0; }
    }

    private static CudaExecutionSlot? AcquireSlot()
    {
        CudaExecutionSlot? slot;
        lock (s_poolLock)
        {
            while (s_freeSlots.Count == 0 && s_allSlots.Count >= MaxExecutionSlots)
            {
                Monitor.Wait(s_poolLock);
            }

            if (s_freeSlots.Count > 0)
            {
                slot = s_freeSlots.Pop();
            }
            else
            {
                slot = new CudaExecutionSlot();
                s_allSlots.Add(slot);
            }

            s_inUseSlots++;
        }

        if (slot.Stream == IntPtr.Zero)
        {
            if (CuDriver.StreamCreate(out IntPtr stream, CuDriver.CU_STREAM_NON_BLOCKING) != 0)
            {
                ReleaseSlot(slot);
                return null;
            }
            slot.Stream = stream;
        }

        return slot;
    }

    private static void ReleaseSlot(CudaExecutionSlot slot)
    {
        lock (s_poolLock)
        {
            s_freeSlots.Push(slot);
            s_inUseSlots--;
            Monitor.PulseAll(s_poolLock);
        }
    }

    private static bool EnsureSlotQueryAndScores(CudaExecutionSlot slot, nuint bytesQuery, nuint bytesScores)
    {
        if (bytesQuery > slot.CapQuery)
        {
            if (slot.DQuery != IntPtr.Zero) CuDriver.MemFree(slot.DQuery);
            if (CuDriver.MemAlloc(out slot.DQuery, bytesQuery) != 0)
            {
                slot.DQuery = IntPtr.Zero;
                slot.CapQuery = 0;
                return false;
            }
            slot.CapQuery = bytesQuery;
        }

        if (bytesQuery > slot.CapHQueryPinned)
        {
            if (slot.HQueryPinned != IntPtr.Zero) CuDriver.MemFreeHost(slot.HQueryPinned);
            if (CuDriver.MemHostAlloc(out slot.HQueryPinned, bytesQuery, CuDriver.CU_MEMHOSTALLOC_PORTABLE) != 0)
            {
                slot.HQueryPinned = IntPtr.Zero;
                slot.CapHQueryPinned = 0;
                return false;
            }
            slot.CapHQueryPinned = bytesQuery;
        }

        if (bytesScores > slot.CapScores)
        {
            if (slot.DScores != IntPtr.Zero) CuDriver.MemFree(slot.DScores);
            if (CuDriver.MemAlloc(out slot.DScores, bytesScores) != 0)
            {
                slot.DScores = IntPtr.Zero;
                slot.CapScores = 0;
                return false;
            }
            slot.CapScores = bytesScores;
        }

        if (bytesScores > slot.CapHScoresPinned)
        {
            if (slot.HScoresPinned != IntPtr.Zero) CuDriver.MemFreeHost(slot.HScoresPinned);
            if (CuDriver.MemHostAlloc(out slot.HScoresPinned, bytesScores, CuDriver.CU_MEMHOSTALLOC_PORTABLE) != 0)
            {
                slot.HScoresPinned = IntPtr.Zero;
                slot.CapHScoresPinned = 0;
                return false;
            }
            slot.CapHScoresPinned = bytesScores;
        }

        return true;
    }

    private static bool UploadDbViaStagingAsync(CudaExecutionSlot slot, IntPtr dDst, void* hSrc, nuint totalBytes)
    {
        nuint desiredStaging = Math.Min(totalBytes, StagingChunkBytes);
        if (desiredStaging > slot.CapHStagingPinned)
        {
            if (slot.HStagingPinned != IntPtr.Zero) CuDriver.MemFreeHost(slot.HStagingPinned);
            if (CuDriver.MemHostAlloc(out slot.HStagingPinned, desiredStaging, CuDriver.CU_MEMHOSTALLOC_PORTABLE) != 0)
            {
                slot.HStagingPinned = IntPtr.Zero;
                slot.CapHStagingPinned = 0;
                return false;
            }
            slot.CapHStagingPinned = desiredStaging;
        }

        nuint offset = 0;
        while (offset < totalBytes)
        {
            nuint chunk = Math.Min(totalBytes - offset, slot.CapHStagingPinned);
            if (offset > 0 && CuDriver.StreamSynchronize(slot.Stream) != 0)
                return false;

            Buffer.MemoryCopy((byte*)hSrc + offset, (void*)slot.HStagingPinned, (long)slot.CapHStagingPinned, (long)chunk);
            if (CuDriver.MemcpyHtoDAsync((IntPtr)((byte*)dDst + offset), slot.HStagingPinned, chunk, slot.Stream) != 0)
                return false;

            offset += chunk;
        }

        return true;
    }

    private static bool AcquireOrUploadDbBuffer(
        CudaExecutionSlot slot,
        void* hDbPtr,
        nuint bytesDb,
        long storageId,
        int chunkIndex,
        long storageVersion,
        GpuDataKind kind,
        out IntPtr dDbPtr,
        out VramCacheEntry? cacheEntry)
    {
        dDbPtr = IntPtr.Zero;
        cacheEntry = null;

        if (storageId <= 0)
        {
            if (bytesDb > slot.CapEphemeralDb)
            {
                if (slot.DEphemeralDb != IntPtr.Zero) CuDriver.MemFree(slot.DEphemeralDb);
                if (CuDriver.MemAlloc(out slot.DEphemeralDb, bytesDb) != 0)
                {
                    slot.DEphemeralDb = IntPtr.Zero;
                    slot.CapEphemeralDb = 0;
                    return false;
                }
                slot.CapEphemeralDb = bytesDb;
            }

            if (!UploadDbViaStagingAsync(slot, slot.DEphemeralDb, hDbPtr, bytesDb))
                return false;

            dDbPtr = slot.DEphemeralDb;
            return true;
        }

        var key = new VramCacheKey(storageId, chunkIndex, kind);
        List<IntPtr>? toFree = null;

        lock (s_cacheLock)
        {
            if (s_vramCache.TryGetValue(key, out var existing))
            {
                if (existing.StorageVersion == storageVersion && existing.ByteSize == bytesDb && !existing.PendingFree)
                {
                    existing.ActiveUsers++;
                    existing.LastUsedTick = ++s_lruTick;
                    Interlocked.Increment(ref s_cacheHits);
                    dDbPtr = existing.DevicePtr;
                    cacheEntry = existing;
                    return true;
                }

                // Stale version or size: remove from map
                s_vramCache.Remove(key);
                s_cachedTotalBytes -= existing.ByteSize;
                if (existing.ActiveUsers == 0)
                {
                    toFree ??= new List<IntPtr>();
                    toFree.Add(existing.DevicePtr);
                    existing.DevicePtr = IntPtr.Zero;
                }
                else
                {
                    existing.PendingFree = true;
                }
            }

            Interlocked.Increment(ref s_cacheMisses);

            // Evict LRU entries with zero active users until within slot count and VRAM budget
            EvictLruLocked(bytesDb, ref toFree);
        }

        FreeDevicePointers(toFree);

        if (CuDriver.MemAlloc(out IntPtr newDPtr, bytesDb) != 0)
        {
            // Retry once after evicting all idle cached entries
            List<IntPtr>? allIdle = null;
            lock (s_cacheLock)
            {
                EvictAllIdleLocked(ref allIdle);
            }
            FreeDevicePointers(allIdle);

            if (CuDriver.MemAlloc(out newDPtr, bytesDb) != 0)
                return false;
        }

        // Upload and synchronize slot stream before publishing to cache so other streams see complete data
        if (!UploadDbViaStagingAsync(slot, newDPtr, hDbPtr, bytesDb) ||
            CuDriver.StreamSynchronize(slot.Stream) != 0)
        {
            try { CuDriver.MemFree(newDPtr); } catch { }
            return false;
        }

        IntPtr duplicateToFree = IntPtr.Zero;
        List<IntPtr>? replacedToFree = null;

        lock (s_cacheLock)
        {
            if (s_vramCache.TryGetValue(key, out var raceWinner) &&
                raceWinner.StorageVersion == storageVersion &&
                raceWinner.ByteSize == bytesDb &&
                !raceWinner.PendingFree)
            {
                raceWinner.ActiveUsers++;
                raceWinner.LastUsedTick = ++s_lruTick;
                duplicateToFree = newDPtr;
                dDbPtr = raceWinner.DevicePtr;
                cacheEntry = raceWinner;
            }
            else
            {
                if (s_vramCache.TryGetValue(key, out var stale))
                {
                    s_vramCache.Remove(key);
                    s_cachedTotalBytes -= stale.ByteSize;
                    if (stale.ActiveUsers == 0)
                    {
                        replacedToFree ??= new List<IntPtr>();
                        replacedToFree.Add(stale.DevicePtr);
                        stale.DevicePtr = IntPtr.Zero;
                    }
                    else
                    {
                        stale.PendingFree = true;
                    }
                }

                EvictLruLocked(bytesDb, ref replacedToFree);

                var newEntry = new VramCacheEntry
                {
                    Key = key,
                    StorageVersion = storageVersion,
                    ByteSize = bytesDb,
                    DevicePtr = newDPtr,
                    LastUsedTick = ++s_lruTick,
                    ActiveUsers = 1,
                    PendingFree = false
                };

                s_vramCache[key] = newEntry;
                s_cachedTotalBytes += bytesDb;
                dDbPtr = newDPtr;
                cacheEntry = newEntry;
            }
        }

        if (duplicateToFree != IntPtr.Zero)
        {
            try { CuDriver.MemFree(duplicateToFree); } catch { }
        }
        FreeDevicePointers(replacedToFree);

        return true;
    }

    private static void EvictLruLocked(nuint incomingBytes, ref List<IntPtr>? toFree)
    {
        while (s_vramCache.Count >= MaxCacheSlots ||
               (s_cachedTotalBytes + incomingBytes > s_vramCacheBudgetBytes && s_vramCache.Count > 0))
        {
            VramCacheEntry? lruVictim = null;
            long minTick = long.MaxValue;

            foreach (var candidate in s_vramCache.Values)
            {
                if (candidate.ActiveUsers == 0 && candidate.LastUsedTick < minTick)
                {
                    minTick = candidate.LastUsedTick;
                    lruVictim = candidate;
                }
            }

            if (lruVictim == null)
            {
                break; // All remaining entries have active users
            }

            s_vramCache.Remove(lruVictim.Key);
            s_cachedTotalBytes -= lruVictim.ByteSize;
            if (lruVictim.DevicePtr != IntPtr.Zero)
            {
                toFree ??= new List<IntPtr>();
                toFree.Add(lruVictim.DevicePtr);
                lruVictim.DevicePtr = IntPtr.Zero;
            }
        }
    }

    private static void EvictAllIdleLocked(ref List<IntPtr>? toFree)
    {
        List<VramCacheKey>? keysToRemove = null;
        foreach (var kvp in s_vramCache)
        {
            if (kvp.Value.ActiveUsers == 0)
            {
                keysToRemove ??= new List<VramCacheKey>();
                keysToRemove.Add(kvp.Key);
            }
        }

        if (keysToRemove == null) return;
        foreach (var k in keysToRemove)
        {
            var entry = s_vramCache[k];
            s_vramCache.Remove(k);
            s_cachedTotalBytes -= entry.ByteSize;
            if (entry.DevicePtr != IntPtr.Zero)
            {
                toFree ??= new List<IntPtr>();
                toFree.Add(entry.DevicePtr);
                entry.DevicePtr = IntPtr.Zero;
            }
        }
    }

    private static void FreeDevicePointers(List<IntPtr>? ptrs)
    {
        if (ptrs == null) return;
        foreach (var ptr in ptrs)
        {
            if (ptr != IntPtr.Zero)
            {
                try { CuDriver.MemFree(ptr); } catch { }
            }
        }
    }

    private static void ReleaseDbBuffer(VramCacheEntry? entry)
    {
        if (entry == null) return;
        IntPtr ptrToFree = IntPtr.Zero;

        lock (s_cacheLock)
        {
            entry.ActiveUsers--;
            if (entry.ActiveUsers == 0 && entry.PendingFree && entry.DevicePtr != IntPtr.Zero)
            {
                ptrToFree = entry.DevicePtr;
                entry.DevicePtr = IntPtr.Zero;
            }
        }

        if (ptrToFree != IntPtr.Zero)
        {
            try { CuDriver.MemFree(ptrToFree); } catch { }
        }
    }

    #endregion

    #region Vector Similarity Scan Offload

    /// <summary>
    /// Executes a brute-force similarity scan on GPU for a single query against a block of vectors.
    /// Returns true if executed on GPU, false if caller should use CPU fallback.
    /// </summary>
    public static bool ScanChunkGpu(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> chunkVectors,
        Span<float> scores,
        int vectorCount,
        int dimensions,
        bool isL2Distance = false,
        GpuTarget target = GpuTarget.Auto)
        => ScanChunkGpu(query, chunkVectors, scores, vectorCount, dimensions, isL2Distance, target, storageId: 0, chunkIndex: 0, storageVersion: 0);

    /// <summary>
    /// Executes a brute-force similarity scan on GPU with explicit storage identity and mutation version for safe VRAM caching.
    /// </summary>
    public static bool ScanChunkGpu(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> chunkVectors,
        Span<float> scores,
        int vectorCount,
        int dimensions,
        bool isL2Distance,
        GpuTarget target,
        long storageId,
        int chunkIndex,
        long storageVersion)
    {
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GPU target '{target}' is not supported. Use GpuTarget.Auto, GpuTarget.Nvidia, or GpuTarget.Cpu.");

        if (vectorCount <= 0 || dimensions <= 0) return false;
        if (query.Length < dimensions || (long)chunkVectors.Length < (long)vectorCount * dimensions || scores.Length < vectorCount)
            return false;

        switch (target)
        {
            case GpuTarget.Cpu:
                return false;

            case GpuTarget.Nvidia:
                return ExecuteNvidiaScan(query, chunkVectors, scores, vectorCount, dimensions, isL2Distance, storageId, chunkIndex, storageVersion);

            case GpuTarget.Auto:
            default:
                if (vectorCount < 5_000) return false;
                return EnsureNvidiaInitialized() &&
                       ExecuteNvidiaScan(query, chunkVectors, scores, vectorCount, dimensions, isL2Distance, storageId, chunkIndex, storageVersion);
        }
    }

    private static bool TryEnterNvidiaExecutionReadLock()
    {
        for (int attempt = 0; attempt < 2; attempt++)
        {
            if (!EnsureNvidiaInitialized()) return false;
            s_lifecycleLock.EnterReadLock();
            if (Volatile.Read(ref s_nvidiaAvailable) && s_cuContext != IntPtr.Zero)
            {
                return true;
            }
            s_lifecycleLock.ExitReadLock();
        }
        return false;
    }

    private static bool ExecuteNvidiaScan(
        ReadOnlySpan<float> query,
        ReadOnlySpan<float> chunkVectors,
        Span<float> scores,
        int vectorCount,
        int dimensions,
        bool isL2Distance,
        long storageId,
        int chunkIndex,
        long storageVersion)
    {
        if (!TryEnterNvidiaExecutionReadLock()) return false;
        try
        {
            if (CuDriver.CtxSetCurrent(s_cuContext) != 0) return false;

            nuint bytesQuery = (nuint)((long)dimensions * sizeof(float));
            nuint bytesDb = (nuint)((long)vectorCount * dimensions * sizeof(float));
            nuint bytesScores = (nuint)((long)vectorCount * sizeof(float));

            CudaExecutionSlot? slot = AcquireSlot();
            if (slot == null) return false;

            VramCacheEntry? cacheEntry = null;
            try
            {
                if (!EnsureSlotQueryAndScores(slot, bytesQuery, bytesScores))
                    return false;

                fixed (float* pQuery = query)
                fixed (float* pDb = chunkVectors)
                fixed (float* pScores = scores)
                {
                    Buffer.MemoryCopy(pQuery, (void*)slot.HQueryPinned, (long)slot.CapHQueryPinned, (long)bytesQuery);
                    if (CuDriver.MemcpyHtoDAsync(slot.DQuery, slot.HQueryPinned, bytesQuery, slot.Stream) != 0)
                        return false;

                    if (!AcquireOrUploadDbBuffer(slot, pDb, bytesDb, storageId, chunkIndex, storageVersion, GpuDataKind.Fp32, out IntPtr dDatabase, out cacheEntry))
                        return false;

                    IntPtr dQuery = slot.DQuery;
                    IntPtr dScores = slot.DScores;
                    int vecCount = vectorCount;
                    int dims = dimensions;

                    void** kernelParams = stackalloc void*[5];
                    kernelParams[0] = &dQuery;
                    kernelParams[1] = &dDatabase;
                    kernelParams[2] = &dScores;
                    kernelParams[3] = &vecCount;
                    kernelParams[4] = &dims;

                    bool useSplitK = !isL2Distance && dimensions >= 128 && s_fnDotScanSplitK != IntPtr.Zero;
                    IntPtr fn = isL2Distance ? s_fnL2Scan : (useSplitK ? s_fnDotScanSplitK : s_fnDotScan);
                    uint blockSize = 256;
                    uint gridSize = useSplitK
                        ? (uint)((vectorCount + 7) / 8)
                        : (uint)((vectorCount + 255) / 256);

                    int launchRes = CuDriver.LaunchKernel(
                        fn,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, slot.Stream,
                        (IntPtr)kernelParams,
                        IntPtr.Zero);

                    if (launchRes != 0) return false;

                    if (CuDriver.MemcpyDtoHAsync(slot.HScoresPinned, slot.DScores, bytesScores, slot.Stream) != 0)
                        return false;

                    if (CuDriver.StreamSynchronize(slot.Stream) != 0)
                        return false;

                    Buffer.MemoryCopy((void*)slot.HScoresPinned, pScores, (long)bytesScores, (long)bytesScores);
                    return true;
                }
            }
            finally
            {
                ReleaseDbBuffer(cacheEntry);
                ReleaseSlot(slot);
            }
        }
        finally
        {
            s_lifecycleLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Executes a zero-I2F contiguous warp Split-K INT8 quantized similarity scan (<c>dp4a.s32.s32</c>) on NVIDIA GPU,
    /// with multi-threaded AVX-512BW / AVX2 / SWAR CPU fallback.
    /// </summary>
    public static bool ScanChunkInt8Gpu(
        ReadOnlySpan<sbyte> query,
        float queryScale,
        ReadOnlySpan<sbyte> chunkVectors,
        float dbScale,
        Span<float> scores,
        int vectorCount,
        int dimensions,
        GpuTarget target = GpuTarget.Auto)
        => ScanChunkInt8Gpu(query, queryScale, chunkVectors, dbScale, scores, vectorCount, dimensions, target, storageId: 0, chunkIndex: 0, storageVersion: 0);

    /// <summary>
    /// Executes a zero-I2F contiguous warp Split-K INT8 quantized similarity scan with explicit storage identity and mutation version for safe VRAM caching.
    /// </summary>
    public static bool ScanChunkInt8Gpu(
        ReadOnlySpan<sbyte> query,
        float queryScale,
        ReadOnlySpan<sbyte> chunkVectors,
        float dbScale,
        Span<float> scores,
        int vectorCount,
        int dimensions,
        GpuTarget target,
        long storageId,
        int chunkIndex,
        long storageVersion)
    {
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GPU target '{target}' is not supported. Use GpuTarget.Auto, GpuTarget.Nvidia, or GpuTarget.Cpu.");

        if (vectorCount <= 0 || dimensions <= 0) return false;
        if (query.Length < dimensions ||
            (long)chunkVectors.Length < (long)vectorCount * dimensions ||
            scores.Length < vectorCount)
        {
            return false;
        }

        bool canUseDp4a = (dimensions & 3) == 0;
        switch (target)
        {
            case GpuTarget.Nvidia:
                return canUseDp4a && ExecuteNvidiaInt8Scan(query, queryScale, chunkVectors, dbScale, scores, vectorCount, dimensions, storageId, chunkIndex, storageVersion);

            case GpuTarget.Auto:
                if (canUseDp4a && vectorCount >= 2_000 && IsNvidiaAvailable &&
                    ExecuteNvidiaInt8Scan(query, queryScale, chunkVectors, dbScale, scores, vectorCount, dimensions, storageId, chunkIndex, storageVersion))
                {
                    return true;
                }
                break;

            case GpuTarget.Cpu:
                break;
        }

        ExecuteCpuInt8Scan(query, queryScale, chunkVectors, dbScale, scores, vectorCount, dimensions);
        return target == GpuTarget.Cpu || target == GpuTarget.Auto;
    }

    private static bool ExecuteNvidiaInt8Scan(
        ReadOnlySpan<sbyte> query,
        float queryScale,
        ReadOnlySpan<sbyte> chunkVectors,
        float dbScale,
        Span<float> scores,
        int vectorCount,
        int dimensions,
        long storageId,
        int chunkIndex,
        long storageVersion)
    {
        if (!TryEnterNvidiaExecutionReadLock()) return false;
        try
        {
            if (s_fnDotScanInt8Dp4a == IntPtr.Zero) return false;
            if (CuDriver.CtxSetCurrent(s_cuContext) != 0) return false;

            nuint bytesQuery = (nuint)dimensions;
            nuint bytesDb = (nuint)((long)vectorCount * dimensions);
            nuint bytesScores = (nuint)((long)vectorCount * sizeof(float));

            CudaExecutionSlot? slot = AcquireSlot();
            if (slot == null) return false;

            VramCacheEntry? cacheEntry = null;
            try
            {
                if (!EnsureSlotQueryAndScores(slot, bytesQuery, bytesScores))
                    return false;

                fixed (sbyte* pQuery = query)
                fixed (sbyte* pDb = chunkVectors)
                fixed (float* pScores = scores)
                {
                    Buffer.MemoryCopy(pQuery, (void*)slot.HQueryPinned, (long)slot.CapHQueryPinned, (long)bytesQuery);
                    if (CuDriver.MemcpyHtoDAsync(slot.DQuery, slot.HQueryPinned, bytesQuery, slot.Stream) != 0)
                        return false;

                    if (!AcquireOrUploadDbBuffer(slot, pDb, bytesDb, storageId, chunkIndex, storageVersion, GpuDataKind.Int8, out IntPtr dDatabase, out cacheEntry))
                        return false;

                    IntPtr dQuery = slot.DQuery;
                    IntPtr dScores = slot.DScores;
                    int vecCount = vectorCount;
                    int dimsDiv4 = dimensions >> 2;
                    float combinedScale = queryScale * dbScale;

                    void** kernelParams = stackalloc void*[6];
                    kernelParams[0] = &dQuery;
                    kernelParams[1] = &dDatabase;
                    kernelParams[2] = &dScores;
                    kernelParams[3] = &vecCount;
                    kernelParams[4] = &dimsDiv4;
                    kernelParams[5] = &combinedScale;

                    uint blockSize = 256; // 8 warps per CTA
                    uint gridSize = (uint)((vectorCount + 7) / 8);

                    int launchRes = CuDriver.LaunchKernel(
                        s_fnDotScanInt8Dp4a,
                        gridSize, 1, 1,
                        blockSize, 1, 1,
                        0, slot.Stream,
                        (IntPtr)kernelParams,
                        IntPtr.Zero);

                    if (launchRes != 0) return false;

                    if (CuDriver.MemcpyDtoHAsync(slot.HScoresPinned, slot.DScores, bytesScores, slot.Stream) != 0)
                        return false;

                    if (CuDriver.StreamSynchronize(slot.Stream) != 0)
                        return false;

                    Buffer.MemoryCopy((void*)slot.HScoresPinned, pScores, (long)bytesScores, (long)bytesScores);
                    return true;
                }
            }
            finally
            {
                ReleaseDbBuffer(cacheEntry);
                ReleaseSlot(slot);
            }
        }
        finally
        {
            s_lifecycleLock.ExitReadLock();
        }
    }

    private static void ExecuteCpuInt8Scan(
        ReadOnlySpan<sbyte> query,
        float queryScale,
        ReadOnlySpan<sbyte> chunkVectors,
        float dbScale,
        Span<float> scores,
        int vectorCount,
        int dimensions)
    {
        fixed (sbyte* pQ = query)
        fixed (sbyte* pDb = chunkVectors)
        fixed (float* pScores = scores)
        {
            sbyte* qPtr = pQ;
            sbyte* dbPtr = pDb;
            float* sPtr = pScores;
            float combined = queryScale * dbScale;

            int parallelism = VectorConcurrency.GetEffectiveParallelism();
            Parallel.For(0, vectorCount, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, i =>
            {
                var qSpan = new ReadOnlySpan<sbyte>(qPtr, dimensions);
                var vSpan = new ReadOnlySpan<sbyte>(dbPtr + (long)i * dimensions, dimensions);
                sPtr[i] = DistanceKernels.DotInt8(qSpan, vSpan) * combined;
            });
        }
    }

    #endregion

    #region Batch Search Acceleration (Kernel-based)

    private static bool ExecuteNvidiaBatchScan(
        ReadOnlySpan<float> queries,
        ReadOnlySpan<float> chunkVectors,
        Span<float> scoreMatrix,
        int vectorCount,
        int dimensions,
        int batchSize,
        long storageId,
        long storageVersion)
    {
        if (!TryEnterNvidiaExecutionReadLock()) return false;
        try
        {
            if (s_fnBatchDotScan == IntPtr.Zero) return false;
            if (CuDriver.CtxSetCurrent(s_cuContext) != 0) return false;

            nuint bytesQuery = (nuint)((long)batchSize * dimensions * sizeof(float));
            nuint bytesDb = (nuint)((long)vectorCount * dimensions * sizeof(float));
            nuint bytesScores = (nuint)((long)vectorCount * batchSize * sizeof(float));

            CudaExecutionSlot? slot = AcquireSlot();
            if (slot == null) return false;

            VramCacheEntry? cacheEntry = null;
            try
            {
                if (!EnsureSlotQueryAndScores(slot, bytesQuery, bytesScores))
                    return false;

                fixed (float* pQuery = queries)
                fixed (float* pDb = chunkVectors)
                fixed (float* pScores = scoreMatrix)
                {
                    Buffer.MemoryCopy(pQuery, (void*)slot.HQueryPinned, (long)slot.CapHQueryPinned, (long)bytesQuery);
                    if (CuDriver.MemcpyHtoDAsync(slot.DQuery, slot.HQueryPinned, bytesQuery, slot.Stream) != 0)
                        return false;

                    if (!AcquireOrUploadDbBuffer(slot, pDb, bytesDb, storageId, chunkIndex: 0, storageVersion, GpuDataKind.Fp32, out IntPtr dDatabase, out cacheEntry))
                        return false;

                    IntPtr dQuery = slot.DQuery;
                    IntPtr dScores = slot.DScores;
                    int vecCount = vectorCount;
                    int dims = dimensions;
                    int bSize = batchSize;

                    void** kernelParams = stackalloc void*[6];
                    kernelParams[0] = &dQuery;
                    kernelParams[1] = &dDatabase;
                    kernelParams[2] = &dScores;
                    kernelParams[3] = &vecCount;
                    kernelParams[4] = &dims;
                    kernelParams[5] = &bSize;

                    uint blockSizeX = 256;
                    uint gridDimX = (uint)((vectorCount + 255) / 256);
                    uint gridDimY = (uint)batchSize;

                    int launchRes = CuDriver.LaunchKernel(
                        s_fnBatchDotScan,
                        gridDimX, gridDimY, 1,
                        blockSizeX, 1, 1,
                        0, slot.Stream,
                        (IntPtr)kernelParams,
                        IntPtr.Zero);

                    if (launchRes != 0) return false;

                    if (CuDriver.MemcpyDtoHAsync(slot.HScoresPinned, slot.DScores, bytesScores, slot.Stream) != 0)
                        return false;

                    if (CuDriver.StreamSynchronize(slot.Stream) != 0)
                        return false;

                    Buffer.MemoryCopy((void*)slot.HScoresPinned, pScores, (long)bytesScores, (long)bytesScores);
                    return true;
                }
            }
            finally
            {
                ReleaseDbBuffer(cacheEntry);
                ReleaseSlot(slot);
            }
        }
        finally
        {
            s_lifecycleLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Executes high-throughput batch vector search: computes S = Database * Queries^T.
    /// Output scoreMatrix is shape [vectorCount, batchSize].
    /// Returns true if executed on GPU; when target is Cpu or Auto without GPU, computes via multi-threaded SIMD CPU kernels if target == Cpu, or returns false when an explicit GPU target is unavailable.
    /// </summary>
    public static bool BatchScanGpu(
        ReadOnlySpan<float> chunkVectors,
        ReadOnlySpan<float> queries,
        Span<float> scoreMatrix,
        int vectorCount,
        int dimensions,
        int batchSize,
        GpuTarget target = GpuTarget.Auto)
        => BatchScanGpu(chunkVectors, queries, scoreMatrix, vectorCount, dimensions, batchSize, target, storageId: 0, storageVersion: 0);

    /// <summary>
    /// Executes high-throughput batch vector search with explicit storage identity and mutation version for safe VRAM caching.
    /// </summary>
    public static bool BatchScanGpu(
        ReadOnlySpan<float> chunkVectors,
        ReadOnlySpan<float> queries,
        Span<float> scoreMatrix,
        int vectorCount,
        int dimensions,
        int batchSize,
        GpuTarget target,
        long storageId,
        long storageVersion)
    {
        if (target is GpuTarget.Amd or GpuTarget.DualGpu)
            throw new NotSupportedException($"GPU target '{target}' is not supported. Use GpuTarget.Auto, GpuTarget.Nvidia, or GpuTarget.Cpu.");

        if (vectorCount <= 0 || dimensions <= 0 || batchSize <= 0) return false;
        if ((long)chunkVectors.Length < (long)vectorCount * dimensions ||
            (long)queries.Length < (long)batchSize * dimensions ||
            (long)scoreMatrix.Length < (long)vectorCount * batchSize)
        {
            return false;
        }

        switch (target)
        {
            case GpuTarget.Nvidia:
                return ExecuteNvidiaBatchScan(queries, chunkVectors, scoreMatrix, vectorCount, dimensions, batchSize, storageId, storageVersion);

            case GpuTarget.Auto:
                if (IsNvidiaAvailable && ExecuteNvidiaBatchScan(queries, chunkVectors, scoreMatrix, vectorCount, dimensions, batchSize, storageId, storageVersion))
                {
                    return true;
                }
                break;

            case GpuTarget.Cpu:
                break;
        }

        // Multi-threaded SIMD CPU execution for GpuTarget.Cpu and GpuTarget.Auto fallback
        ExecuteCpuBatchScan(chunkVectors, queries, scoreMatrix, vectorCount, dimensions, batchSize);
        return target == GpuTarget.Cpu || target == GpuTarget.Auto;
    }

    private static void ExecuteCpuBatchScan(
        ReadOnlySpan<float> chunkVectors,
        ReadOnlySpan<float> queries,
        Span<float> scoreMatrix,
        int vectorCount,
        int dimensions,
        int batchSize)
    {
        fixed (float* pScores = scoreMatrix)
        fixed (float* pDb = chunkVectors)
        fixed (float* pQ = queries)
        {
            float* pS = pScores;
            float* pD = pDb;
            float* pQueries = pQ;

            int parallelism = VectorConcurrency.GetEffectiveParallelism();
            Parallel.For(0, vectorCount, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, i =>
            {
                float* vRow = pD + (long)i * dimensions;
                long sOffset = (long)i * batchSize;
                for (int b = 0; b < batchSize; b++)
                {
                    float* qRow = pQueries + (long)b * dimensions;
                    pS[sOffset + b] = DotProductSimd(vRow, qRow, dimensions);
                }
            });
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private static float DotProductSimd(float* a, float* b, int length)
    {
        float sum = 0f;
        int i = 0;

        if (Vector512.IsHardwareAccelerated && length >= Vector512<float>.Count)
        {
            var acc512 = Vector512<float>.Zero;
            int step = Vector512<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector512.Load(a + i);
                var vb = Vector512.Load(b + i);
                acc512 = Vector512.FusedMultiplyAdd(va, vb, acc512);
                i += step;
            }
            sum += Vector512.Sum(acc512);
        }
        else if (Vector256.IsHardwareAccelerated && length >= Vector256<float>.Count)
        {
            var acc256 = Vector256<float>.Zero;
            int step = Vector256<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector256.Load(a + i);
                var vb = Vector256.Load(b + i);
                acc256 = Vector256.FusedMultiplyAdd(va, vb, acc256);
                i += step;
            }
            sum += Vector256.Sum(acc256);
        }
        else if (Vector128.IsHardwareAccelerated && length >= Vector128<float>.Count)
        {
            var acc128 = Vector128<float>.Zero;
            int step = Vector128<float>.Count;
            int limit = length - step;
            while (i <= limit)
            {
                var va = Vector128.Load(a + i);
                var vb = Vector128.Load(b + i);
                acc128 = Vector128.FusedMultiplyAdd(va, vb, acc128);
                i += step;
            }
            sum += Vector128.Sum(acc128);
        }

        for (; i < length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    #endregion
}
