using System;
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
/// Dispatches vector similarity scans and batch matrix operations to CUDA-capable NVIDIA GPUs,
/// with multi-core AVX-512 / AVX2 / Vector128 CPU fallback when GPU acceleration is unavailable.
/// </summary>
public static unsafe class GpuVectorAccelerator
{
    private static readonly Lock s_initLock = new();
    private static bool s_nvidiaInitialized;
    private static bool s_nvidiaAvailable;
    private static IntPtr s_cuContext;
    private static IntPtr s_cuModule;
    private static IntPtr s_fnDotScan;
    private static IntPtr s_fnBatchDotScan;
    private static IntPtr s_fnL2Scan;

    // Persistent device buffer pool for high-throughput scans keyed by immutable storage identity + version
    private static IntPtr s_dQuery;
    private static IntPtr s_dDatabase;
    private static IntPtr s_dScores;
    private static nuint s_capQuery;
    private static nuint s_capDatabase;
    private static nuint s_capScores;
    private static long s_cachedStorageId;
    private static int s_cachedChunkIndex = -1;
    private static long s_cachedStorageVersion = -1;
    private static nuint s_lastUploadedDbBytes;

    public static bool IsNvidiaAvailable => EnsureNvidiaInitialized();

    /// <summary>
    /// Gets whether an AMD GPU compute kernel backend is available for vector scans.
    /// Returns false until native HIP/Vulkan compute kernels are loaded, ensuring callers use the multi-core SIMD path honestly.
    /// </summary>
    public static bool IsAmdAvailable => false;

    public static bool IsGpuAvailable => IsNvidiaAvailable;

    #region Driver Initialization

    private static bool EnsureNvidiaInitialized()
    {
        if (s_nvidiaInitialized) return s_nvidiaAvailable;
        lock (s_initLock)
        {
            if (s_nvidiaInitialized) return s_nvidiaAvailable;
            try
            {
                if (!CuDriver.IsAvailable())
                {
                    s_nvidiaAvailable = false;
                    return false;
                }

                if (CuDriver.Init(0) != 0)
                {
                    s_nvidiaAvailable = false;
                    return false;
                }

                if (CuDriver.DeviceGet(out int dev, 0) != 0)
                {
                    s_nvidiaAvailable = false;
                    return false;
                }

                string targetArch = CuDriver.GetComputeCapability(dev);

                if (CuDriver.CtxCreate(out s_cuContext, 0, dev) != 0)
                {
                    s_nvidiaAvailable = false;
                    return false;
                }

                string ptx = KernelCompiler.NormalizePtxDirectives(GpuVectorKernels.PtxSource, targetArch);
                byte[] ptxBytes = Encoding.UTF8.GetBytes(ptx + "\0");
                if (CuDriver.ModuleLoadData(out s_cuModule, ptxBytes) != 0)
                {
                    s_nvidiaAvailable = false;
                    return false;
                }

                if (CuDriver.ModuleGetFunction(out s_fnDotScan, s_cuModule, "vector_dot_scan_fp32") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnBatchDotScan, s_cuModule, "vector_batch_dot_scan_fp32") != 0 ||
                    CuDriver.ModuleGetFunction(out s_fnL2Scan, s_cuModule, "vector_l2_scan_fp32") != 0)
                {
                    s_nvidiaAvailable = false;
                    return false;
                }

                s_nvidiaAvailable = true;
            }
            catch
            {
                s_nvidiaAvailable = false;
            }
            finally
            {
                s_nvidiaInitialized = true;
            }

            return s_nvidiaAvailable;
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
        if (vectorCount <= 0 || dimensions <= 0) return false;
        if (query.Length < dimensions || chunkVectors.Length < vectorCount * dimensions || scores.Length < vectorCount)
            return false;

        switch (target)
        {
            case GpuTarget.Cpu:
            case GpuTarget.Amd:
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
        if (!EnsureNvidiaInitialized()) return false;

        nuint bytesQuery = (nuint)(dimensions * sizeof(float));
        nuint bytesDb = (nuint)((long)vectorCount * dimensions * sizeof(float));
        nuint bytesScores = (nuint)(vectorCount * sizeof(float));

        lock (s_initLock)
        {
            CuDriver.CtxSetCurrent(s_cuContext);

            if (bytesQuery > s_capQuery)
            {
                if (s_dQuery != IntPtr.Zero) CuDriver.MemFree(s_dQuery);
                if (CuDriver.MemAlloc(out s_dQuery, bytesQuery) != 0)
                {
                    s_capQuery = 0;
                    return false;
                }
                s_capQuery = bytesQuery;
            }

            if (bytesDb > s_capDatabase)
            {
                if (s_dDatabase != IntPtr.Zero) CuDriver.MemFree(s_dDatabase);
                if (CuDriver.MemAlloc(out s_dDatabase, bytesDb) != 0)
                {
                    s_capDatabase = 0;
                    InvalidateDbCache();
                    return false;
                }
                s_capDatabase = bytesDb;
                InvalidateDbCache();
            }

            if (bytesScores > s_capScores)
            {
                if (s_dScores != IntPtr.Zero) CuDriver.MemFree(s_dScores);
                if (CuDriver.MemAlloc(out s_dScores, bytesScores) != 0)
                {
                    s_capScores = 0;
                    return false;
                }
                s_capScores = bytesScores;
            }

            fixed (float* pQuery = query)
            fixed (float* pDb = chunkVectors)
            fixed (float* pScores = scores)
            {
                if (CuDriver.MemcpyHtoD(s_dQuery, (IntPtr)pQuery, bytesQuery) != 0)
                    return false;

                bool cacheHit = storageId > 0 &&
                                storageId == s_cachedStorageId &&
                                chunkIndex == s_cachedChunkIndex &&
                                storageVersion == s_cachedStorageVersion &&
                                bytesDb == s_lastUploadedDbBytes;

                if (!cacheHit)
                {
                    if (CuDriver.MemcpyHtoD(s_dDatabase, (IntPtr)pDb, bytesDb) != 0)
                        return false;

                    if (storageId > 0)
                    {
                        s_cachedStorageId = storageId;
                        s_cachedChunkIndex = chunkIndex;
                        s_cachedStorageVersion = storageVersion;
                        s_lastUploadedDbBytes = bytesDb;
                    }
                    else
                    {
                        InvalidateDbCache();
                    }
                }

                IntPtr dQuery = s_dQuery;
                IntPtr dDatabase = s_dDatabase;
                IntPtr dScores = s_dScores;
                int vecCount = vectorCount;
                int dims = dimensions;

                void** kernelParams = stackalloc void*[5];
                kernelParams[0] = &dQuery;
                kernelParams[1] = &dDatabase;
                kernelParams[2] = &dScores;
                kernelParams[3] = &vecCount;
                kernelParams[4] = &dims;

                IntPtr fn = isL2Distance ? s_fnL2Scan : s_fnDotScan;
                uint blockSize = 256;
                uint gridSize = (uint)((vectorCount + 255) / 256);

                int launchRes = CuDriver.LaunchKernel(
                    fn,
                    gridSize, 1, 1,
                    blockSize, 1, 1,
                    0, IntPtr.Zero,
                    (IntPtr)kernelParams,
                    IntPtr.Zero);

                if (launchRes != 0) return false;

                CuDriver.StreamSynchronize(IntPtr.Zero);
                return CuDriver.MemcpyDtoH((IntPtr)pScores, s_dScores, bytesScores) == 0;
            }
        }
    }

    private static void InvalidateDbCache()
    {
        s_cachedStorageId = 0;
        s_cachedChunkIndex = -1;
        s_cachedStorageVersion = -1;
        s_lastUploadedDbBytes = 0;
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
        if (!EnsureNvidiaInitialized() || s_fnBatchDotScan == IntPtr.Zero) return false;

        nuint bytesQuery = (nuint)((long)batchSize * dimensions * sizeof(float));
        nuint bytesDb = (nuint)((long)vectorCount * dimensions * sizeof(float));
        nuint bytesScores = (nuint)((long)vectorCount * batchSize * sizeof(float));

        lock (s_initLock)
        {
            CuDriver.CtxSetCurrent(s_cuContext);

            if (bytesQuery > s_capQuery)
            {
                if (s_dQuery != IntPtr.Zero) CuDriver.MemFree(s_dQuery);
                if (CuDriver.MemAlloc(out s_dQuery, bytesQuery) != 0)
                {
                    s_capQuery = 0;
                    return false;
                }
                s_capQuery = bytesQuery;
            }

            if (bytesDb > s_capDatabase)
            {
                if (s_dDatabase != IntPtr.Zero) CuDriver.MemFree(s_dDatabase);
                if (CuDriver.MemAlloc(out s_dDatabase, bytesDb) != 0)
                {
                    s_capDatabase = 0;
                    InvalidateDbCache();
                    return false;
                }
                s_capDatabase = bytesDb;
                InvalidateDbCache();
            }

            if (bytesScores > s_capScores)
            {
                if (s_dScores != IntPtr.Zero) CuDriver.MemFree(s_dScores);
                if (CuDriver.MemAlloc(out s_dScores, bytesScores) != 0)
                {
                    s_capScores = 0;
                    return false;
                }
                s_capScores = bytesScores;
            }

            fixed (float* pQuery = queries)
            fixed (float* pDb = chunkVectors)
            fixed (float* pScores = scoreMatrix)
            {
                if (CuDriver.MemcpyHtoD(s_dQuery, (IntPtr)pQuery, bytesQuery) != 0)
                    return false;

                bool cacheHit = storageId > 0 &&
                                storageId == s_cachedStorageId &&
                                s_cachedChunkIndex == 0 &&
                                storageVersion == s_cachedStorageVersion &&
                                bytesDb == s_lastUploadedDbBytes;

                if (!cacheHit)
                {
                    if (CuDriver.MemcpyHtoD(s_dDatabase, (IntPtr)pDb, bytesDb) != 0)
                        return false;

                    if (storageId > 0)
                    {
                        s_cachedStorageId = storageId;
                        s_cachedChunkIndex = 0;
                        s_cachedStorageVersion = storageVersion;
                        s_lastUploadedDbBytes = bytesDb;
                    }
                    else
                    {
                        InvalidateDbCache();
                    }
                }

                IntPtr dQuery = s_dQuery;
                IntPtr dDatabase = s_dDatabase;
                IntPtr dScores = s_dScores;
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
                    0, IntPtr.Zero,
                    (IntPtr)kernelParams,
                    IntPtr.Zero);

                if (launchRes != 0) return false;

                CuDriver.StreamSynchronize(IntPtr.Zero);
                return CuDriver.MemcpyDtoH((IntPtr)pScores, s_dScores, bytesScores) == 0;
            }
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
        if (vectorCount <= 0 || dimensions <= 0 || batchSize <= 0) return false;
        if (chunkVectors.Length < vectorCount * dimensions ||
            queries.Length < batchSize * dimensions ||
            scoreMatrix.Length < vectorCount * batchSize)
        {
            return false;
        }

        switch (target)
        {
            case GpuTarget.Nvidia:
                return ExecuteNvidiaBatchScan(queries, chunkVectors, scoreMatrix, vectorCount, dimensions, batchSize, storageId, storageVersion);

            case GpuTarget.Amd:
                return false;

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
                int sOffset = i * batchSize;
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
