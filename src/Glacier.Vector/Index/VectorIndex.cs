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
    /// The high-performance, multi-threaded search orchestrator.
    /// Combines SIMD compute kernels with chunked Task parallelism.
    /// </summary>
    public class VectorIndex : IDisposable
    {
        private readonly IVectorStorage _storage;

        // Parallel list to hold metadata. Since vector IDs are sequential (0 to N-1),
        // a simple List<string> is significantly faster and uses less memory than a Dictionary.
        private readonly List<string> _metadata;
        private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);
        private readonly object _writerLock = new();

        public int Dimensions => _storage.Dimensions;
        public int Count => _storage.Count;

        public VectorIndex(IVectorStorage storage)
        {
            _storage = storage ?? throw new ArgumentNullException(nameof(storage));
            _metadata = new List<string>(storage.Count);
        }

        /// <summary>
        /// Appends a new vector and its associated metadata to the index.
        /// Thread-safe for concurrent ingestion.
        /// </summary>
        public void Add(ReadOnlySpan<float> vector, string metadata)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            _storage.Append(vector);

            lock (_metadata)
            {
                _metadata.Add(metadata);
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
            if (query.Length != Dimensions)
                throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

            int totalCount = _storage.Count;
            if (totalCount == 0 || topK <= 0) return Array.Empty<SearchResult>();

            if (target != GpuTarget.Cpu && totalCount >= 5_000 && GpuVectorAccelerator.IsGpuAvailable)
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
                int dims = Dimensions;
                int chunkCount = _storage.ChunkCount;

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
                        if (!GpuVectorAccelerator.ScanChunkGpu(query, chunkSpan, scores, chunkVectors, dims, isL2Distance: false, target))
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
            if (queries.Length < batchSize * Dimensions)
                throw new ArgumentException("Queries span length does not match batchSize * Dimensions.");

            int totalCount = _storage.Count;
            if (totalCount == 0 || topK <= 0 || batchSize <= 0)
                return Array.Empty<SearchResult[]>();

            var batchResults = new SearchResult[batchSize][];

            if (target != GpuTarget.Cpu && totalCount >= 1024 && _storage.ChunkCount == 1 && GpuVectorAccelerator.IsGpuAvailable)
            {
                _rwLock.EnterReadLock();
                try
                {
                    ReadOnlySpan<float> dbSpan = _storage.GetChunkSpan(0);
                    if (!dbSpan.IsEmpty)
                    {
                        int matrixLen = totalCount * batchSize;
                        float[] pooledScoreMatrix = ArrayPool<float>.Shared.Rent(matrixLen);

                        try
                        {
                            Span<float> scoreMatrix = pooledScoreMatrix.AsSpan(0, matrixLen);

                            if (GpuVectorAccelerator.BatchScanGpu(dbSpan, queries, scoreMatrix, totalCount, Dimensions, batchSize, target))
                            {
                                int metaCount = _metadata.Count;

                                Parallel.For(0, batchSize, b =>
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
                finally
                {
                    _rwLock.ExitReadLock();
                }
            }

            // Fallback: search each query sequentially / parallel on CPU
            fixed (float* pQueries = queries)
            {
                IntPtr queriesPtr = (IntPtr)pQueries;
                int dims = Dimensions;
                Parallel.For(0, batchSize, b =>
                {
                    ReadOnlySpan<float> q = new ReadOnlySpan<float>((float*)queriesPtr + (b * dims), dims);
                    batchResults[b] = Search(q, topK, GpuTarget.Cpu, maxDegreeOfParallelism: 1);
                });
            }

            return batchResults;
        }

        private unsafe SearchResult[] SearchCpu(ReadOnlySpan<float> query, int topK, int maxDegreeOfParallelism)
        {
            if (query.Length != Dimensions)
                throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

            int totalCount = _storage.Count;
            if (totalCount == 0 || topK <= 0) return Array.Empty<SearchResult>();

            int parallelism = VectorConcurrency.GetEffectiveParallelism(maxDegreeOfParallelism);
            long totalOps = (long)totalCount * Dimensions;

            // For small datasets or single-thread configuration, run sequentially with zero thread overhead
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

                var parallelOptions = new System.Threading.Tasks.ParallelOptions
                {
                    MaxDegreeOfParallelism = parallelism
                };

                System.Threading.Tasks.Parallel.For(0, numChunks, parallelOptions, t =>
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

            // Final Merge Phase: Combine the Top-K from all chunks
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

            // Extract from queue and sort descending
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
            // .NET 6+ PriorityQueue is a Min-Heap. By prioritizing by Score, 
            // the SMALLEST score stays at the top of the queue, making it perfectly
            // positioned to be popped off when we exceed our topK capacity.
            var localQueue = new PriorityQueue<int, float>(topK + 1);

            float localMinScore = float.MinValue;

            for (int i = start; i < end; i++)
            {
                ReadOnlySpan<float> dbVector = _storage.GetVector(i);

                // Call our blazing-fast SIMD hardware intrinsics
                float score = DistanceKernels.DotProduct(querySpan, dbVector);

                // Hot-path optimization: bypass the PriorityQueue entirely 99.9% of the time
                if (score > localMinScore || localQueue.Count < topK)
                {
                    localQueue.Enqueue(i, score);

                    if (localQueue.Count > topK)
                    {
                        localQueue.Dequeue(); // Remove the lowest score
                        // Update the gatekeeper threshold
                        localQueue.TryPeek(out _, out localMinScore);
                    }
                }
            }

            // Materialize the local results, fetching the metadata strings
            var results = new SearchResult[localQueue.Count];
            int idx = results.Length - 1;

            lock (_metadata)
            {
                int metaCount = _metadata.Count;
                while (localQueue.TryDequeue(out int id, out float score))
                {
                    string meta = (id < metaCount) ? _metadata[id] : string.Empty;
                    results[idx--] = new SearchResult(id, score, meta);
                }
            }

            return results;
        }

        public void Dispose()
        {
            _storage?.Dispose();
            _rwLock?.Dispose();
        }
    }
}