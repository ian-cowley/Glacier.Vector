namespace Glacier.Vector.Index;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using Glacier.Vector.Compute;

/// <summary>
/// Inverted File with Product Quantization (IVF-PQ) index.
/// Features coarse centroid routing, 32x product quantization (8 floats per byte),
/// and Asymmetric Distance Computation (ADC) with precomputed SIMD lookup tables.
/// </summary>
public sealed class IvfPqIndex : IDisposable
{
    public int Dimensions { get; }
    public int CentroidCount { get; }      // K coarse centroids
    public int SubVectorCount { get; }     // M sub-vectors
    public int SubVectorDim { get; }       // d* = Dimensions / SubVectorCount (typically 8 for 32x compression)
    public int Count => _totalCount;
    public bool IsTrained => _isTrained;

    private readonly float[] _coarseCentroids; // K * Dimensions
    private readonly float[] _pqCodebooks;     // M * 256 * SubVectorDim

    // Inverted lists
    private readonly List<int>[] _invertedIds;
    private readonly List<byte[]>[] _invertedCodes;
    private readonly List<string> _metadata = new();
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);

    private int _totalCount = 0;
    private bool _isTrained = false;

    public IvfPqIndex(int dimensions, int centroidCount = 256, int? subVectorCount = null)
    {
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

        _coarseCentroids = new float[CentroidCount * Dimensions];
        _pqCodebooks = new float[SubVectorCount * 256 * SubVectorDim];

        _invertedIds = new List<int>[CentroidCount];
        _invertedCodes = new List<byte[]>[CentroidCount];
        for (int k = 0; k < CentroidCount; k++)
        {
            _invertedIds[k] = new List<int>();
            _invertedCodes[k] = new List<byte[]>();
        }
    }

    /// <summary>
    /// Trains coarse centroids and sub-vector codebooks on a sample of training vectors using K-Means.
    /// </summary>
    public void Train(ReadOnlySpan<float> trainingData, int vectorCount, int iterations = 5)
    {
        if (vectorCount <= 0) throw new ArgumentException("vectorCount must be positive.", nameof(vectorCount));
        if (trainingData.Length < vectorCount * Dimensions)
            throw new ArgumentException("trainingData length is smaller than vectorCount * Dimensions.", nameof(trainingData));

        _rwLock.EnterWriteLock();
        try
        {
            TrainCoarseCentroids(trainingData, vectorCount, iterations);
            TrainPqCodebooks(trainingData, vectorCount, iterations);
            _isTrained = true;
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
            int sampleIdx = (k * step) % vectorCount;
            ReadOnlySpan<float> sample = trainingData.Slice(sampleIdx * Dimensions, Dimensions);
            sample.CopyTo(_coarseCentroids.AsSpan(k * Dimensions, Dimensions));
        }

        // K-Means iterations
        float[] centroidSums = new float[CentroidCount * Dimensions];
        int[] centroidCounts = new int[CentroidCount];

        for (int iter = 0; iter < iterations; iter++)
        {
            Array.Clear(centroidSums, 0, centroidSums.Length);
            Array.Clear(centroidCounts, 0, centroidCounts.Length);

            for (int i = 0; i < vectorCount; i++)
            {
                ReadOnlySpan<float> vec = trainingData.Slice(i * Dimensions, Dimensions);
                int nearestK = FindNearestCoarseCentroid(vec);

                centroidCounts[nearestK]++;
                int offset = nearestK * Dimensions;
                for (int d = 0; d < Dimensions; d++)
                {
                    centroidSums[offset + d] += vec[d];
                }
            }

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
    }

    private void TrainPqCodebooks(ReadOnlySpan<float> trainingData, int vectorCount, int iterations)
    {
        // Train codebook of 256 centroids for each sub-vector space
        for (int m = 0; m < SubVectorCount; m++)
        {
            int codebookOffset = m * 256 * SubVectorDim;
            int step = Math.Max(1, vectorCount / 256);

            // Initialize centroids
            for (int c = 0; c < 256; c++)
            {
                int sampleIdx = (c * step) % vectorCount;
                ReadOnlySpan<float> sub = trainingData.Slice(sampleIdx * Dimensions + m * SubVectorDim, SubVectorDim);
                sub.CopyTo(_pqCodebooks.AsSpan(codebookOffset + c * SubVectorDim, SubVectorDim));
            }

            // K-Means for sub-vectors
            float[] subSums = new float[256 * SubVectorDim];
            int[] subCounts = new int[256];

            for (int iter = 0; iter < iterations; iter++)
            {
                Array.Clear(subSums, 0, subSums.Length);
                Array.Clear(subCounts, 0, subCounts.Length);

                for (int i = 0; i < vectorCount; i++)
                {
                    ReadOnlySpan<float> sub = trainingData.Slice(i * Dimensions + m * SubVectorDim, SubVectorDim);
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
    }

    /// <summary>
    /// Quantizes and adds a vector to the IVF-PQ inverted index.
    /// </summary>
    public void Add(ReadOnlySpan<float> vector, string metadata = "")
    {
        if (vector.Length != Dimensions)
            throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

        _rwLock.EnterWriteLock();
        try
        {
            if (!_isTrained)
            {
                // Auto-initialize coarse centroids and codebooks from first vector if untyped
                AutoInitUntrained(vector);
            }

            int id = _totalCount++;
            _metadata.Add(metadata);

            // 1. Assign to coarse centroid
            int nearestK = FindNearestCoarseCentroid(vector);

            // 2. Quantize sub-vectors to byte codes
            byte[] codes = new byte[SubVectorCount];
            for (int m = 0; m < SubVectorCount; m++)
            {
                ReadOnlySpan<float> sub = vector.Slice(m * SubVectorDim, SubVectorDim);
                int codebookOffset = m * 256 * SubVectorDim;
                codes[m] = (byte)FindNearestSubCentroid(sub, codebookOffset);
            }

            _invertedIds[nearestK].Add(id);
            _invertedCodes[nearestK].Add(codes);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    private void AutoInitUntrained(ReadOnlySpan<float> vector)
    {
        for (int k = 0; k < CentroidCount; k++)
        {
            vector.CopyTo(_coarseCentroids.AsSpan(k * Dimensions, Dimensions));
        }
        for (int m = 0; m < SubVectorCount; m++)
        {
            ReadOnlySpan<float> sub = vector.Slice(m * SubVectorDim, SubVectorDim);
            int codebookOffset = m * 256 * SubVectorDim;
            for (int c = 0; c < 256; c++)
            {
                sub.CopyTo(_pqCodebooks.AsSpan(codebookOffset + c * SubVectorDim, SubVectorDim));
            }
        }
        _isTrained = true;
    }

    /// <summary>
    /// Performs Asymmetric Distance Computation (ADC) search over top-nprobe coarse inverted lists.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public SearchResult[] Search(ReadOnlySpan<float> query, int topK, int nprobe = 16)
    {
        if (query.Length != Dimensions)
            throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

        _rwLock.EnterReadLock();
        try
        {
            if (_totalCount == 0 || topK <= 0) return Array.Empty<SearchResult>();

            int actualProbe = Math.Clamp(nprobe, 1, CentroidCount);

            // 1. Route to top-nprobe coarse centroids using SIMD Dot Product
            Span<float> centroidScores = stackalloc float[CentroidCount];
            for (int k = 0; k < CentroidCount; k++)
            {
                ReadOnlySpan<float> centroid = _coarseCentroids.AsSpan(k * Dimensions, Dimensions);
                centroidScores[k] = DistanceKernels.DotProduct(query, centroid);
            }

            // Min-heap to find top nprobe centroids with highest dot products
            var topCentroids = new PriorityQueue<int, float>(actualProbe + 1);
            for (int k = 0; k < CentroidCount; k++)
            {
                topCentroids.Enqueue(k, centroidScores[k]);
                if (topCentroids.Count > actualProbe)
                {
                    topCentroids.Dequeue();
                }
            }

            // 2. Precompute Asymmetric Distance Lookup Table (LUT): M x 256 floats
            float[] rentedLut = ArrayPool<float>.Shared.Rent(SubVectorCount * 256);
            try
            {
                Span<float> lut = rentedLut.AsSpan(0, SubVectorCount * 256);
                PrecomputeLut(query, lut);

                // 3. Scan quantized vectors in top-nprobe inverted lists using LUT
                var globalQueue = new PriorityQueue<int, float>(topK + 1);
                float minScore = float.MinValue;

                while (topCentroids.TryDequeue(out int listId, out _))
                {
                    var ids = _invertedIds[listId];
                    var codes = _invertedCodes[listId];
                    int listCount = ids.Count;

                    for (int i = 0; i < listCount; i++)
                    {
                        int id = ids[i];
                        byte[] code = codes[i];

                        float dist = 0f;
                        for (int m = 0; m < SubVectorCount; m++)
                        {
                            byte codeByte = code[m];
                            dist += lut[m * 256 + codeByte];
                        }

                        // Convert distance to descending score (similarity proxy)
                        float score = -dist;
                        if (globalQueue.Count < topK || score > minScore)
                        {
                            globalQueue.Enqueue(id, score);
                            if (globalQueue.Count > topK)
                            {
                                globalQueue.Dequeue();
                                globalQueue.TryPeek(out _, out minScore);
                            }
                        }
                    }
                }

                var results = new SearchResult[globalQueue.Count];
                int metaCount = _metadata.Count;
                for (int i = results.Length - 1; i >= 0; i--)
                {
                    globalQueue.TryDequeue(out int id, out float score);
                    string meta = (id < metaCount) ? _metadata[id] : string.Empty;
                    results[i] = new SearchResult(id, score, meta);
                }
                return results;
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rentedLut);
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
        _rwLock.Dispose();
    }
}
