namespace Glacier.Vector.Index;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;

/// <summary>
/// Inverted File with Product Quantization (IVF-PQ) index.
/// Features coarse centroid routing, 32x product quantization (8 floats per byte),
/// and Asymmetric Distance Computation (ADC) with precomputed SIMD lookup tables.
/// </summary>
public sealed unsafe class IvfPqIndex : IDisposable
{
    public int Dimensions { get; }
    public int CentroidCount { get; }      // K coarse centroids
    public int SubVectorCount { get; }     // M sub-vectors
    public int SubVectorDim { get; }       // d* = Dimensions / SubVectorCount (typically 8 for 32x compression)
    public int Count => Volatile.Read(ref _totalCount);
    public bool IsTrained => Volatile.Read(ref _isTrained);

    private readonly float[] _coarseCentroids; // K * Dimensions
    private readonly float[] _pqCodebooks;     // M * 256 * SubVectorDim

    // Inverted lists (contiguous byte arena per coarse centroid to avoid per-vector byte[] heap allocations)
    private readonly List<int>[] _invertedIds;
    private readonly List<byte>[] _invertedCodes;
    private readonly List<string> _metadata = new();
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);

    private int _totalCount = 0;
    private bool _isTrained = false;
    private bool _disposed = false;
    private int _disposeFlag = 0;

    public IvfPqIndex(int dimensions, int centroidCount = 256, int? subVectorCount = null)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimensions, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(centroidCount, 1);
        if (subVectorCount.HasValue)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(subVectorCount.Value, 1);
        }

        Dimensions = dimensions;
        CentroidCount = centroidCount;

        // Default to d* = 8 (8 floats = 32 bytes -> 1 byte code = 32x compression)
        if (subVectorCount.HasValue)
        {
            SubVectorCount = subVectorCount.Value;
        }
        else
        {
            SubVectorCount = dimensions % 8 == 0 ? dimensions / 8 : (dimensions % 4 == 0 ? dimensions / 4 : dimensions);
        }

        if (dimensions % SubVectorCount != 0)
            throw new ArgumentException($"Dimensions ({dimensions}) must be divisible by SubVectorCount ({SubVectorCount}).");

        SubVectorDim = dimensions / SubVectorCount;

        _coarseCentroids = new float[checked((int)((long)CentroidCount * Dimensions))];
        _pqCodebooks = new float[checked((int)((long)SubVectorCount * 256 * SubVectorDim))];

        _invertedIds = new List<int>[CentroidCount];
        _invertedCodes = new List<byte>[CentroidCount];
        for (int k = 0; k < CentroidCount; k++)
        {
            _invertedIds[k] = new List<int>();
            _invertedCodes[k] = new List<byte>();
        }
    }

    private void EnterReadLockSafe()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        try
        {
            _rwLock.EnterReadLock();
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(IvfPqIndex));
        }

        if (_disposed)
        {
            _rwLock.ExitReadLock();
            throw new ObjectDisposedException(nameof(IvfPqIndex));
        }
    }

    private void EnterWriteLockSafe()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        try
        {
            _rwLock.EnterWriteLock();
        }
        catch (ObjectDisposedException)
        {
            throw new ObjectDisposedException(nameof(IvfPqIndex));
        }

        if (_disposed)
        {
            _rwLock.ExitWriteLock();
            throw new ObjectDisposedException(nameof(IvfPqIndex));
        }
    }

    /// <summary>
    /// Trains coarse centroids and sub-vector codebooks on a sample of training vectors using K-Means.
    /// </summary>
    public void Train(ReadOnlySpan<float> trainingData, int vectorCount, int iterations = 5)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        ArgumentOutOfRangeException.ThrowIfLessThan(vectorCount, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(iterations, 1);
        if ((long)trainingData.Length < (long)vectorCount * Dimensions)
            throw new ArgumentException("trainingData length is smaller than vectorCount * Dimensions.", nameof(trainingData));

        EnterWriteLockSafe();
        try
        {
            TrainCoarseCentroids(trainingData, vectorCount, iterations);
            TrainPqCodebooks(trainingData, vectorCount, iterations);
            Volatile.Write(ref _isTrained, true);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    private void TrainCoarseCentroids(ReadOnlySpan<float> trainingData, int vectorCount, int iterations)
    {
        // Initialize coarse centroids from training data samples
        int step = Math.Max(1, vectorCount / CentroidCount);
        for (int k = 0; k < CentroidCount; k++)
        {
            int sampleIdx = (int)(((long)k * step) % vectorCount);
            ReadOnlySpan<float> sample = trainingData.Slice(checked((int)((long)sampleIdx * Dimensions)), Dimensions);
            sample.CopyTo(_coarseCentroids.AsSpan(k * Dimensions, Dimensions));
        }

        int centroidFloatLen = checked((int)((long)CentroidCount * Dimensions));
        float[] centroidSums = ArrayPool<float>.Shared.Rent(centroidFloatLen);
        int[] centroidCounts = ArrayPool<int>.Shared.Rent(CentroidCount);

        try
        {
            int parallelism = VectorConcurrency.GetEffectiveParallelism();
            bool useParallel = parallelism > 1 && vectorCount >= 256 && (long)vectorCount * CentroidCount * Dimensions >= 65_536;

            if (!useParallel)
            {
                for (int iter = 0; iter < iterations; iter++)
                {
                    Array.Clear(centroidSums, 0, centroidFloatLen);
                    Array.Clear(centroidCounts, 0, CentroidCount);

                    for (int i = 0; i < vectorCount; i++)
                    {
                        ReadOnlySpan<float> vec = trainingData.Slice(checked((int)((long)i * Dimensions)), Dimensions);
                        int nearestK = FindNearestCoarseCentroid(vec);

                        centroidCounts[nearestK]++;
                        int offset = nearestK * Dimensions;
                        for (int d = 0; d < Dimensions; d++)
                        {
                            centroidSums[offset + d] += vec[d];
                        }
                    }

                    UpdateCentroidsFromSums(centroidSums, centroidCounts);
                }
            }
            else
            {
                int numParts = Math.Min(parallelism, Math.Max(1, vectorCount / 64));
                int partSize = (vectorCount + numParts - 1) / numParts;
                float[][] partSums = new float[numParts][];
                int[][] partCounts = new int[numParts][];
                for (int p = 0; p < numParts; p++)
                {
                    partSums[p] = ArrayPool<float>.Shared.Rent(centroidFloatLen);
                    partCounts[p] = ArrayPool<int>.Shared.Rent(CentroidCount);
                }

                try
                {
                    fixed (float* pTrain = trainingData)
                    {
                        float* trainPtr = pTrain;
                        int dims = Dimensions;

                        for (int iter = 0; iter < iterations; iter++)
                        {
                            Parallel.For(0, numParts, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, p =>
                            {
                                float[] localSums = partSums[p];
                                int[] localCounts = partCounts[p];
                                Array.Clear(localSums, 0, centroidFloatLen);
                                Array.Clear(localCounts, 0, CentroidCount);

                                int start = p * partSize;
                                int end = Math.Min(start + partSize, vectorCount);
                                for (int i = start; i < end; i++)
                                {
                                    ReadOnlySpan<float> vec = new ReadOnlySpan<float>(trainPtr + (long)i * dims, dims);
                                    int nearestK = FindNearestCoarseCentroid(vec);

                                    localCounts[nearestK]++;
                                    int offset = nearestK * dims;
                                    for (int d = 0; d < dims; d++)
                                    {
                                        localSums[offset + d] += vec[d];
                                    }
                                }
                            });

                            Array.Clear(centroidSums, 0, centroidFloatLen);
                            Array.Clear(centroidCounts, 0, CentroidCount);
                            for (int p = 0; p < numParts; p++)
                            {
                                float[] localSums = partSums[p];
                                int[] localCounts = partCounts[p];
                                for (int k = 0; k < CentroidCount; k++)
                                {
                                    centroidCounts[k] += localCounts[k];
                                }
                                for (int idx = 0; idx < centroidFloatLen; idx++)
                                {
                                    centroidSums[idx] += localSums[idx];
                                }
                            }

                            UpdateCentroidsFromSums(centroidSums, centroidCounts);
                        }
                    }
                }
                finally
                {
                    for (int p = 0; p < numParts; p++)
                    {
                        ArrayPool<float>.Shared.Return(partSums[p]);
                        ArrayPool<int>.Shared.Return(partCounts[p]);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(centroidSums);
            ArrayPool<int>.Shared.Return(centroidCounts);
        }
    }

    private void UpdateCentroidsFromSums(float[] centroidSums, int[] centroidCounts)
    {
        for (int k = 0; k < CentroidCount; k++)
        {
            int count = centroidCounts[k];
            if (count > 0)
            {
                int offset = k * Dimensions;
                float invCount = 1.0f / count;
                for (int d = 0; d < Dimensions; d++)
                {
                    _coarseCentroids[offset + d] = centroidSums[offset + d] * invCount;
                }
            }
        }
    }

    private void TrainPqCodebooks(ReadOnlySpan<float> trainingData, int vectorCount, int iterations)
    {
        int parallelism = VectorConcurrency.GetEffectiveParallelism();
        bool useParallel = parallelism > 1 && SubVectorCount > 1 && (long)SubVectorCount * vectorCount >= 2048;

        fixed (float* pTrain = trainingData)
        {
            float* trainPtr = pTrain;
            if (!useParallel)
            {
                for (int m = 0; m < SubVectorCount; m++)
                {
                    TrainSingleSubCodebook(trainPtr, vectorCount, iterations, m);
                }
            }
            else
            {
                Parallel.For(0, SubVectorCount, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, m =>
                {
                    TrainSingleSubCodebook(trainPtr, vectorCount, iterations, m);
                });
            }
        }
    }

    private void TrainSingleSubCodebook(float* trainPtr, int vectorCount, int iterations, int m)
    {
        int codebookOffset = m * 256 * SubVectorDim;
        int step = Math.Max(1, vectorCount / 256);

        // Initialize centroids
        for (int c = 0; c < 256; c++)
        {
            int sampleIdx = (int)(((long)c * step) % vectorCount);
            var sub = new ReadOnlySpan<float>(trainPtr + (long)sampleIdx * Dimensions + (long)m * SubVectorDim, SubVectorDim);
            sub.CopyTo(_pqCodebooks.AsSpan(codebookOffset + c * SubVectorDim, SubVectorDim));
        }

        int subSumsLen = 256 * SubVectorDim;
        float[] subSums = ArrayPool<float>.Shared.Rent(subSumsLen);
        int[] subCounts = ArrayPool<int>.Shared.Rent(256);

        try
        {
            for (int iter = 0; iter < iterations; iter++)
            {
                Array.Clear(subSums, 0, subSumsLen);
                Array.Clear(subCounts, 0, 256);

                for (int i = 0; i < vectorCount; i++)
                {
                    var sub = new ReadOnlySpan<float>(trainPtr + (long)i * Dimensions + (long)m * SubVectorDim, SubVectorDim);
                    int nearestC = FindNearestSubCentroid(sub, codebookOffset);

                    subCounts[nearestC]++;
                    int offset = nearestC * SubVectorDim;
                    for (int d = 0; d < SubVectorDim; d++)
                    {
                        subSums[offset + d] += sub[d];
                    }
                }

                for (int c = 0; c < 256; c++)
                {
                    int count = subCounts[c];
                    if (count > 0)
                    {
                        int offset = c * SubVectorDim;
                        float invCount = 1.0f / count;
                        for (int d = 0; d < SubVectorDim; d++)
                        {
                            _pqCodebooks[codebookOffset + offset + d] = subSums[offset + d] * invCount;
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(subSums);
            ArrayPool<int>.Shared.Return(subCounts);
        }
    }

    /// <summary>
    /// Quantizes and adds a vector to the IVF-PQ inverted index.
    /// </summary>
    public void Add(ReadOnlySpan<float> vector, string metadata = "")
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        if (vector.Length != Dimensions)
            throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

        const int StackAllocByteLimit = 256;
        byte[]? rentedCodes = SubVectorCount > StackAllocByteLimit ? ArrayPool<byte>.Shared.Rent(SubVectorCount) : null;

        try
        {
            Span<byte> codes = rentedCodes != null ? rentedCodes.AsSpan(0, SubVectorCount) : stackalloc byte[SubVectorCount];

            EnterWriteLockSafe();
            try
            {
                if (!_isTrained)
                {
                    // Auto-initialize coarse centroids and codebooks from first vector if untrained
                    AutoInitUntrained(vector);
                }

                int id = _totalCount;
                Volatile.Write(ref _totalCount, id + 1);
                _metadata.Add(metadata ?? string.Empty);

                // 1. Assign to coarse centroid
                int nearestK = FindNearestCoarseCentroid(vector);

                // 2. Quantize sub-vectors to byte codes in stack/pooled buffer (zero per-vector byte[] alloc)
                for (int m = 0; m < SubVectorCount; m++)
                {
                    ReadOnlySpan<float> sub = vector.Slice(m * SubVectorDim, SubVectorDim);
                    int codebookOffset = m * 256 * SubVectorDim;
                    codes[m] = (byte)FindNearestSubCentroid(sub, codebookOffset);
                }

                _invertedIds[nearestK].Add(id);

                List<byte> codeList = _invertedCodes[nearestK];
                int oldLen = codeList.Count;
                CollectionsMarshal.SetCount(codeList, oldLen + SubVectorCount);
                codes.CopyTo(CollectionsMarshal.AsSpan(codeList).Slice(oldLen, SubVectorCount));
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }
        finally
        {
            if (rentedCodes != null) ArrayPool<byte>.Shared.Return(rentedCodes);
        }
    }

    private void AutoInitUntrained(ReadOnlySpan<float> vector)
    {
        // Perturb centroids deterministically so coarse Voronoi cells and PQ codes
        // retain discrimination even if Add() is called before Train().
        for (int k = 0; k < CentroidCount; k++)
        {
            var dst = _coarseCentroids.AsSpan(k * Dimensions, Dimensions);
            vector.CopyTo(dst);
            if (k > 0)
            {
                uint h = (uint)k * 2654435761u;
                int d1 = (int)(h % (uint)Dimensions);
                int d2 = (int)(((h >> 8) ^ (uint)k) % (uint)Dimensions);
                float delta = 0.05f * (1 + (k % 7)) * ((k & 1) == 0 ? 1f : -1f);
                dst[d1] += delta;
                dst[d2] -= delta * 0.5f;
            }
        }
        for (int m = 0; m < SubVectorCount; m++)
        {
            ReadOnlySpan<float> sub = vector.Slice(m * SubVectorDim, SubVectorDim);
            int codebookOffset = m * 256 * SubVectorDim;
            for (int c = 0; c < 256; c++)
            {
                var dst = _pqCodebooks.AsSpan(codebookOffset + c * SubVectorDim, SubVectorDim);
                sub.CopyTo(dst);
                if (c > 0)
                {
                    uint hc = ((uint)c * 2246822519u) ^ ((uint)m * 3266489917u);
                    int d = (int)(hc % (uint)SubVectorDim);
                    float delta = 0.02f * (((c + 1) / 2) * ((c & 1) == 0 ? 1f : -1f));
                    dst[d] += delta;
                }
            }
        }
        Volatile.Write(ref _isTrained, true);
    }

    /// <summary>
    /// Performs Asymmetric Distance Computation (ADC) search over top-nprobe coarse inverted lists.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public SearchResult[] Search(ReadOnlySpan<float> query, int topK, int nprobe = 16)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        if (query.Length != Dimensions)
            throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

        EnterReadLockSafe();
        try
        {
            if (_totalCount == 0 || topK <= 0) return Array.Empty<SearchResult>();

            int actualProbe = Math.Clamp(nprobe, 1, CentroidCount);

            var centroidHeapBuf = ArrayPool<(int Id, float Dist)>.Shared.Rent(actualProbe + 1);
            float[] rentedLut = ArrayPool<float>.Shared.Rent(SubVectorCount * 256);
            var globalHeapBuf = ArrayPool<(int Id, float Dist)>.Shared.Rent(topK + 1);

            try
            {
                // 1. Route to top-nprobe coarse centroids using minimum L2DistanceSquared via zero-alloc ValueMaxHeap.
                var topCentroids = new ValueMaxHeap(centroidHeapBuf);
                for (int k = 0; k < CentroidCount; k++)
                {
                    ReadOnlySpan<float> centroid = _coarseCentroids.AsSpan(k * Dimensions, Dimensions);
                    float dist = DistanceKernels.L2DistanceSquared(query, centroid);
                    if (topCentroids.Count < actualProbe)
                    {
                        topCentroids.Enqueue(k, dist);
                    }
                    else if (topCentroids.Peek(out _, out float worstCentroidDist) && dist < worstCentroidDist)
                    {
                        topCentroids.TryDequeue(out _, out _);
                        topCentroids.Enqueue(k, dist);
                    }
                }

                // 2. Precompute Asymmetric Distance Lookup Table (LUT): M x 256 floats
                Span<float> lut = rentedLut.AsSpan(0, SubVectorCount * 256);
                PrecomputeLut(query, lut);

                // 3. Scan quantized vectors in top-nprobe inverted lists using LUT and ValueMaxHeap
                var globalQueue = new ValueMaxHeap(globalHeapBuf);
                int mCount = SubVectorCount;

                while (topCentroids.TryDequeue(out int listId, out _))
                {
                    ReadOnlySpan<int> ids = CollectionsMarshal.AsSpan(_invertedIds[listId]);
                    ReadOnlySpan<byte> codes = CollectionsMarshal.AsSpan(_invertedCodes[listId]);
                    int listCount = ids.Length;

                    for (int i = 0; i < listCount; i++)
                    {
                        int id = ids[i];
                        ReadOnlySpan<byte> code = codes.Slice(i * mCount, mCount);

                        float dist = 0f;
                        for (int m = 0; m < mCount; m++)
                        {
                            dist += lut[m * 256 + code[m]];
                        }

                        if (globalQueue.Count < topK)
                        {
                            globalQueue.Enqueue(id, dist);
                        }
                        else if (globalQueue.Peek(out _, out float worstDist) && dist < worstDist)
                        {
                            globalQueue.TryDequeue(out _, out _);
                            globalQueue.Enqueue(id, dist);
                        }
                    }
                }

                var results = new SearchResult[globalQueue.Count];
                int metaCount = _metadata.Count;
                for (int i = results.Length - 1; i >= 0; i--)
                {
                    globalQueue.TryDequeue(out int id, out float dist);
                    string meta = (id < metaCount) ? _metadata[id] : string.Empty;
                    results[i] = new SearchResult(id, -dist, meta);
                }
                return results;
            }
            finally
            {
                ArrayPool<(int Id, float Dist)>.Shared.Return(centroidHeapBuf);
                ArrayPool<float>.Shared.Return(rentedLut);
                ArrayPool<(int Id, float Dist)>.Shared.Return(globalHeapBuf);
            }
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private void PrecomputeLut(ReadOnlySpan<float> query, Span<float> lut)
    {
        for (int m = 0; m < SubVectorCount; m++)
        {
            ReadOnlySpan<float> qSub = query.Slice(m * SubVectorDim, SubVectorDim);
            int codebookOffset = m * 256 * SubVectorDim;

            for (int c = 0; c < 256; c++)
            {
                ReadOnlySpan<float> centroid = _pqCodebooks.AsSpan(codebookOffset + c * SubVectorDim, SubVectorDim);
                lut[m * 256 + c] = DistanceKernels.L2DistanceSquared(qSub, centroid);
            }
        }
    }

    private int FindNearestCoarseCentroid(ReadOnlySpan<float> vector)
    {
        int bestK = 0;
        float bestDist = float.MaxValue;
        for (int k = 0; k < CentroidCount; k++)
        {
            ReadOnlySpan<float> centroid = _coarseCentroids.AsSpan(k * Dimensions, Dimensions);
            float d = DistanceKernels.L2DistanceSquared(vector, centroid);
            if (d < bestDist)
            {
                bestDist = d;
                bestK = k;
            }
        }
        return bestK;
    }

    private int FindNearestSubCentroid(ReadOnlySpan<float> subVector, int codebookOffset)
    {
        int bestC = 0;
        float bestDist = float.MaxValue;
        for (int c = 0; c < 256; c++)
        {
            ReadOnlySpan<float> centroid = _pqCodebooks.AsSpan(codebookOffset + c * SubVectorDim, SubVectorDim);
            float d = DistanceKernels.L2DistanceSquared(subVector, centroid);
            if (d < bestDist)
            {
                bestDist = d;
                bestC = c;
            }
        }
        return bestC;
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeFlag, 1) != 0) return;

        try
        {
            _rwLock.EnterWriteLock();
        }
        catch (ObjectDisposedException)
        {
            return;
        }

        try
        {
            if (_disposed) return;
            Volatile.Write(ref _disposed, true);
        }
        finally
        {
            _rwLock.ExitWriteLock();
            _rwLock.Dispose();
        }
    }
}
