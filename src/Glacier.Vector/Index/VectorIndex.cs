using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
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
        private readonly object _writeLock = new object();

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

            lock (_writeLock)
            {
                _storage.Append(vector);
                _metadata.Add(metadata);
            }
        }

        /// <summary>
        /// Performs a brute-force (Flat) Exact Nearest Neighbor search using 
        /// AVX2/AVX-512 SIMD instructions and dynamic thread-isolated chunking.
        /// </summary>
        public unsafe SearchResult[] Search(ReadOnlySpan<float> query, int topK = 5, int maxDegreeOfParallelism = 0)
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

            while (localQueue.TryDequeue(out int id, out float score))
            {
                string meta = string.Empty;
                lock (_writeLock) // Safe read
                {
                    if (id < _metadata.Count) meta = _metadata[id];
                }

                results[idx--] = new SearchResult(id, score, meta);
            }

            return results;
        }

        public void Dispose()
        {
            _storage?.Dispose();
        }
    }
}