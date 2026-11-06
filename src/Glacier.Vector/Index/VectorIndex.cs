using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
using Glacier.Vector.Storage;

namespace Glacier.Vector.Index
{
    /// <summary>
    /// Represents a single matched document from a vector search.
    /// </summary>
    public readonly struct SearchResult : IComparable<SearchResult>
    {
        public int Id { get; }
        public float Score { get; }
        public string Metadata { get; }

        public SearchResult(int id, float score, string metadata)
        {
            Id = id;
            Score = score;
            Metadata = metadata;
        }

        // Default sort by score descending
        public int CompareTo(SearchResult other)
        {
            return other.Score.CompareTo(Score);
        }
    }

    /// <summary>
    /// High-performance, thread-safe multi-threaded vector search orchestrator.
    /// Combines SIMD compute kernels and CUDA GPU offloading with chunked Task parallelism.
    /// </summary>
    public class VectorIndex : IDisposable
    {
        private readonly IVectorStorage _storage;
        private readonly bool _leaveStorageOpen;
        private readonly List<string> _metadata;
        private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);
        private bool _disposed;

        public int Dimensions => _storage.Dimensions;
        public int Count => _storage.Count;

        public VectorIndex(IVectorStorage storage)
            : this(storage, leaveStorageOpen: false)
        {
        }

        public VectorIndex(IVectorStorage storage, bool leaveStorageOpen)
        {
            _storage = storage ?? throw new ArgumentNullException(nameof(storage));
            _leaveStorageOpen = leaveStorageOpen;
            _metadata = new List<string>(storage.Count);
            for (int i = 0; i < storage.Count; i++)
            {
                _metadata.Add(string.Empty);
            }
        }

        /// <summary>
        /// Appends a new vector and its associated metadata to the index atomically.
        /// Thread-safe for concurrent ingestion and search.
        /// </summary>
        public void Add(ReadOnlySpan<float> vector, string metadata)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            _rwLock.EnterWriteLock();
            try
            {
                _storage.Append(vector);
                _metadata.Add(metadata ?? string.Empty);
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }

        /// <summary>
        /// Performs nearest neighbor search on the CPU/GPU with default options.
        /// Preserved for binary backward compatibility with assemblies compiled against prior versions.
        /// </summary>
        public unsafe SearchResult[] Search(ReadOnlySpan<float> query, int topK)
        {
            return Search(query, topK, GpuTarget.Auto, 0);
        }

        /// <summary>
        /// Performs nearest neighbor search on the CPU/GPU with default topK and options.
        /// Preserved for binary backward compatibility with assemblies compiled against prior versions.
        /// </summary>
        public unsafe SearchResult[] Search(ReadOnlySpan<float> query)
        {
            return Search(query, 5, GpuTarget.Auto, 0);
        }

        /// <summary>
        /// Performs nearest neighbor search on the specified hardware target (Auto, Nvidia, Amd, or Cpu).
        /// </summary>
        public unsafe SearchResult[] Search(
            ReadOnlySpan<float> query,
            int topK = 5,
            GpuTarget target = GpuTarget.Auto,
            int maxDegreeOfParallelism = 0)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (query.Length != Dimensions)
                throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

            if (topK <= 0) return Array.Empty<SearchResult>();

            if (target != GpuTarget.Cpu && _storage.Count >= 5_000 && GpuVectorAccelerator.IsGpuAvailable)
            {
                var gpuRes = TryGpuSearch(query, topK, target);
                if (gpuRes != null) return gpuRes;
            }

            return SearchCpu(query, topK, maxDegreeOfParallelism);
        }

        private unsafe SearchResult[]? TryGpuSearch(ReadOnlySpan<float> query, int topK, GpuTarget target)
        {
            _rwLock.EnterReadLock();
            try
            {
                int totalCount = _storage.Count;
                if (totalCount == 0) return Array.Empty<SearchResult>();

                int dims = Dimensions;
                int chunkCount = _storage.ChunkCount;
                if (chunkCount <= 0) return null;

                long storageId = _storage.StorageId;
                long storageVersion = _storage.Version;
                var globalQueue = new PriorityQueue<int, float>(topK + 1);

                int baseVectorIndex = 0;
                for (int c = 0; c < chunkCount; c++)
                {
                    ReadOnlySpan<float> chunkSpan = _storage.GetChunkSpan(c);
                    int chunkVectors = _storage.GetChunkVectorCount(c);
                    if (chunkSpan.IsEmpty || chunkVectors == 0) return null;

                    float[] pooledScores = ArrayPool<float>.Shared.Rent(chunkVectors);
                    try
                    {
                        Span<float> scores = pooledScores.AsSpan(0, chunkVectors);
                        if (!GpuVectorAccelerator.ScanChunkGpu(
                            query,
                            chunkSpan,
                            scores,
                            chunkVectors,
                            dims,
                            isL2Distance: false,
                            target: target,
                            storageId: storageId,
                            chunkIndex: c,
                            storageVersion: storageVersion))
                        {
                            return null;
                        }

                        float minQueueScore = float.MinValue;
                        for (int i = 0; i < chunkVectors; i++)
                        {
                            float score = scores[i];
                            int globalId = baseVectorIndex + i;

                            if (globalQueue.Count < topK || score > minQueueScore)
                            {
                                globalQueue.Enqueue(globalId, score);
                                if (globalQueue.Count > topK)
                                {
                                    globalQueue.Dequeue();
                                    globalQueue.TryPeek(out _, out minQueueScore);
                                }
                            }
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(pooledScores);
                    }

                    baseVectorIndex += chunkVectors;
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
                _rwLock.ExitReadLock();
            }
        }

        /// <summary>
        /// High-throughput batch vector search: searches multiple query vectors in parallel.
        /// Accelerated on GPU via matrix multiplication (Database * Queries^T).
        /// </summary>
        public unsafe SearchResult[][] BatchSearch(
            ReadOnlySpan<float> queries,
            int batchSize,
            int topK = 5,
            GpuTarget target = GpuTarget.Auto)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (batchSize <= 0 || topK <= 0)
                return Array.Empty<SearchResult[]>();

            if (queries.Length < batchSize * Dimensions)
                throw new ArgumentException("Queries span length does not match batchSize * Dimensions.");

            var batchResults = new SearchResult[batchSize][];

            _rwLock.EnterReadLock();
            try
            {
                int totalCount = _storage.Count;
                if (totalCount == 0)
                    return Array.Empty<SearchResult[]>();

                if (target != GpuTarget.Cpu && totalCount >= 1024 && _storage.ChunkCount == 1 && GpuVectorAccelerator.IsGpuAvailable)
                {
                    ReadOnlySpan<float> dbSpan = _storage.GetChunkSpan(0);
                    if (!dbSpan.IsEmpty)
                    {
                        int matrixLen = totalCount * batchSize;
                        float[] pooledScoreMatrix = ArrayPool<float>.Shared.Rent(matrixLen);

                        try
                        {
                            Span<float> scoreMatrix = pooledScoreMatrix.AsSpan(0, matrixLen);

                            if (GpuVectorAccelerator.BatchScanGpu(
                                dbSpan,
                                queries,
                                scoreMatrix,
                                totalCount,
                                Dimensions,
                                batchSize,
                                target: target == GpuTarget.Auto ? GpuTarget.Nvidia : target,
                                storageId: _storage.StorageId,
                                storageVersion: _storage.Version))
                            {
                                int metaCount = _metadata.Count;
                                int parallelism = VectorConcurrency.GetEffectiveParallelism();

                                Parallel.For(0, batchSize, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, b =>
                                {
                                    var queue = new PriorityQueue<int, float>(topK + 1);
                                    float minScore = float.MinValue;
                                    for (int i = 0; i < totalCount; i++)
                                    {
                                        float score = pooledScoreMatrix[i * batchSize + b];
                                        if (queue.Count < topK || score > minScore)
                                        {
                                            queue.Enqueue(i, score);
                                            if (queue.Count > topK)
                                            {
                                                queue.Dequeue();
                                                queue.TryPeek(out _, out minScore);
                                            }
                                        }
                                    }

                                    var res = new SearchResult[queue.Count];
                                    for (int i = res.Length - 1; i >= 0; i--)
                                    {
                                        queue.TryDequeue(out int id, out float score);
                                        string meta = (id < metaCount) ? _metadata[id] : string.Empty;
                                        res[i] = new SearchResult(id, score, meta);
                                    }
                                    batchResults[b] = res;
                                });

                                return batchResults;
                            }
                        }
                        finally
                        {
                            ArrayPool<float>.Shared.Return(pooledScoreMatrix);
                        }
                    }
                }

                // Fallback: search each query on CPU under read lock
                fixed (float* pQueries = queries)
                {
                    IntPtr queriesPtr = (IntPtr)pQueries;
                    int dims = Dimensions;
                    int parallelism = VectorConcurrency.GetEffectiveParallelism();
                    Parallel.For(0, batchSize, new ParallelOptions { MaxDegreeOfParallelism = parallelism }, b =>
                    {
                        ReadOnlySpan<float> q = new ReadOnlySpan<float>((float*)queriesPtr + ((long)b * dims), dims);
                        batchResults[b] = SearchCpuLocked(q, totalCount, topK, maxDegreeOfParallelism: 1);
                    });
                }

                return batchResults;
            }
            finally
            {
                _rwLock.ExitReadLock();
            }
        }

        private unsafe SearchResult[] SearchCpu(ReadOnlySpan<float> query, int topK, int maxDegreeOfParallelism)
        {
            if (query.Length != Dimensions)
                throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

            _rwLock.EnterReadLock();
            try
            {
                int totalCount = _storage.Count;
                if (totalCount == 0 || topK <= 0) return Array.Empty<SearchResult>();
                return SearchCpuLocked(query, totalCount, topK, maxDegreeOfParallelism);
            }
            finally
            {
                _rwLock.ExitReadLock();
            }
        }

        private unsafe SearchResult[] SearchCpuLocked(ReadOnlySpan<float> query, int totalCount, int topK, int maxDegreeOfParallelism)
        {
            int parallelism = VectorConcurrency.GetEffectiveParallelism(maxDegreeOfParallelism);
            long totalOps = (long)totalCount * Dimensions;

            if (parallelism <= 1 || totalOps < 32_768)
            {
                return SearchChunk(query, 0, totalCount, topK);
            }

            int numChunks = Math.Min(parallelism, Math.Max(1, (totalCount + 255) / 256));
            int chunkSize = (totalCount + numChunks - 1) / numChunks;
            var chunkResults = new SearchResult[numChunks][];

            fixed (float* pQuery = query)
            {
                IntPtr queryPtr = (IntPtr)pQuery;
                int dims = Dimensions;

                var parallelOptions = new ParallelOptions
                {
                    MaxDegreeOfParallelism = parallelism
                };

                Parallel.For(0, numChunks, parallelOptions, t =>
                {
                    int start = t * chunkSize;
                    int end = Math.Min(start + chunkSize, totalCount);

                    if (start >= end)
                    {
                        chunkResults[t] = Array.Empty<SearchResult>();
                        return;
                    }

                    ReadOnlySpan<float> localQuery = new ReadOnlySpan<float>((float*)queryPtr, dims);
                    chunkResults[t] = SearchChunk(localQuery, start, end, topK);
                });
            }

            var globalQueue = new PriorityQueue<SearchResult, float>(topK + 1);

            for (int t = 0; t < numChunks; t++)
            {
                var localResults = chunkResults[t];
                if (localResults == null) continue;
                for (int j = 0; j < localResults.Length; j++)
                {
                    var res = localResults[j];
                    globalQueue.Enqueue(res, res.Score);
                    if (globalQueue.Count > topK)
                    {
                        globalQueue.Dequeue();
                    }
                }
            }

            var finalResults = new SearchResult[globalQueue.Count];
            for (int i = finalResults.Length - 1; i >= 0; i--)
            {
                finalResults[i] = globalQueue.Dequeue();
            }

            return finalResults;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private SearchResult[] SearchChunk(ReadOnlySpan<float> querySpan, int start, int end, int topK)
        {
            var localQueue = new PriorityQueue<int, float>(topK + 1);
            float localMinScore = float.MinValue;

            for (int i = start; i < end; i++)
            {
                ReadOnlySpan<float> dbVector = _storage.GetVector(i);
                float score = DistanceKernels.DotProduct(querySpan, dbVector);

                if (score > localMinScore || localQueue.Count < topK)
                {
                    localQueue.Enqueue(i, score);

                    if (localQueue.Count > topK)
                    {
                        localQueue.Dequeue();
                        localQueue.TryPeek(out _, out localMinScore);
                    }
                }
            }

            var results = new SearchResult[localQueue.Count];
            int idx = results.Length - 1;
            int metaCount = _metadata.Count;

            while (localQueue.TryDequeue(out int id, out float score))
            {
                string meta = (id < metaCount) ? _metadata[id] : string.Empty;
                results[idx--] = new SearchResult(id, score, meta);
            }

            return results;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            if (!_leaveStorageOpen)
            {
                _storage?.Dispose();
            }
            _rwLock.Dispose();
        }
    }
}
