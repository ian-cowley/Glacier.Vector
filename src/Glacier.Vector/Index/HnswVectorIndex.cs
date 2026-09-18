namespace Glacier.Vector.Index;

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using Glacier.Vector.Compute;
using Glacier.Vector.Storage;

/// <summary>
/// High-performance, zero-allocation Hierarchical Navigable Small World (HNSW) index.
/// Employs contiguous flat adjacency arrays, generation-tagged visited tracking,
/// and AVX-512 / AVX2 hardware intrinsic distance kernels.
/// </summary>
public sealed class HnswVectorIndex : IDisposable
{
    private readonly IVectorStorage _storage;
    private readonly int _m;
    private readonly int _m0;
    private readonly int _efConstruction;
    private readonly double _mL;
    private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.SupportsRecursion);
    private readonly List<string> _metadata = new();

    // Contiguous graph adjacency storage:
    // Node ID * M0 -> list of neighbor IDs (terminated by -1)
    private int[] _layer0Edges;
    // Per upper layer (layer - 1): Node ID * M -> neighbor IDs (terminated by -1)
    private readonly List<int[]> _upperLayers = new();
    private int[] _nodeMaxLayers;
    private int _enterNodeId = -1;
    private int _maxLayer = -1;

    // Zero-allocation visited state tracker
    private uint[] _visitedTags;
    private uint _currentTag = 0;

    public int Dimensions => _storage.Dimensions;
    public int Count => _storage.Count;
    public int M => _m;
    public int EfConstruction => _efConstruction;

    public HnswVectorIndex(
        IVectorStorage storage,
        int m = 16,
        int efConstruction = 100,
        int initialCapacity = 65536)
    {
        _storage = storage ?? throw new ArgumentNullException(nameof(storage));
        _m = m;
        _m0 = 2 * m;
        _efConstruction = efConstruction;
        _mL = 1.0 / Math.Log(m > 1 ? m : 2);

        int cap = Math.Max(initialCapacity, storage.Count + 16);
        _layer0Edges = new int[cap * _m0];
        Array.Fill(_layer0Edges, -1);
        _nodeMaxLayers = new int[cap];
        Array.Fill(_nodeMaxLayers, -1);
        _visitedTags = new uint[cap];

        // If storage already contains vectors, index them
        for (int i = 0; i < storage.Count; i++)
        {
            _metadata.Add(string.Empty);
            InsertInternal(i, storage.GetVector(i));
        }
    }

    /// <summary>
    /// Appends a new vector to storage and indexes it in the HNSW graph.
    /// </summary>
    public void Add(ReadOnlySpan<float> vector, string metadata = "")
    {
        if (vector.Length != Dimensions)
            throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

        _rwLock.EnterWriteLock();
        try
        {
            int id = _storage.Count;
            _storage.Append(vector);
            _metadata.Add(metadata);
            InsertInternal(id, vector);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    /// <summary>
    /// Inserts an existing vector from storage by ID into the HNSW graph.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public void Insert(int id, ReadOnlySpan<float> vector)
    {
        _rwLock.EnterWriteLock();
        try
        {
            InsertInternal(id, vector);
        }
        finally
        {
            _rwLock.ExitWriteLock();
        }
    }

    private void InsertInternal(int id, ReadOnlySpan<float> vector)
    {
        EnsureCapacity(id + 1);

        int nodeLayer = SelectRandomLayer();
        _nodeMaxLayers[id] = nodeLayer;

        if (_enterNodeId == -1)
        {
            _enterNodeId = id;
            _maxLayer = nodeLayer;
            return;
        }

        int currObj = _enterNodeId;
        float currDist = ComputeDistance(vector, _storage.GetVector(currObj));

        // Phase 1: Greedy search from top down to nodeLayer + 1
        for (int l = _maxLayer; l > nodeLayer; l--)
        {
            bool changed = true;
            while (changed)
            {
                changed = false;
                var neighbors = GetNeighbors(currObj, l);
                for (int n = 0; n < neighbors.Length; n++)
                {
                    int neighbor = neighbors[n];
                    if (neighbor == -1) break;
                    float d = ComputeDistance(vector, _storage.GetVector(neighbor));
                    if (d < currDist)
                    {
                        currDist = d;
                        currObj = neighbor;
                        changed = true;
                    }
                }
            }
        }

        // Phase 2: From min(maxLayer, nodeLayer) down to layer 0: search efConstruction and connect
        for (int l = Math.Min(_maxLayer, nodeLayer); l >= 0; l--)
        {
            var candidates = SearchLayer(vector, currObj, _efConstruction, l);
            SelectAndLinkNeighbors(id, candidates, l);
            if (candidates.Count > 0)
            {
                candidates.TryPeek(out currObj, out _);
            }
        }

        if (nodeLayer > _maxLayer)
        {
            _maxLayer = nodeLayer;
            _enterNodeId = id;
        }
    }

    /// <summary>
    /// Searches the HNSW graph for the top-K nearest neighbors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public SearchResult[] Search(ReadOnlySpan<float> query, int topK, int efSearch = 64)
    {
        if (query.Length != Dimensions)
            throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

        _rwLock.EnterReadLock();
        try
        {
            if (_enterNodeId == -1 || topK <= 0) return Array.Empty<SearchResult>();

            int currObj = _enterNodeId;
            float currDist = ComputeDistance(query, _storage.GetVector(currObj));

            // Greedy routing in upper layers
            for (int l = _maxLayer; l > 0; l--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    var neighbors = GetNeighbors(currObj, l);
                    for (int n = 0; n < neighbors.Length; n++)
                    {
                        int neighbor = neighbors[n];
                        if (neighbor == -1) break;
                        float d = ComputeDistance(query, _storage.GetVector(neighbor));
                        if (d < currDist)
                        {
                            currDist = d;
                            currObj = neighbor;
                            changed = true;
                        }
                    }
                }
            }

            // Beam search at layer 0
            int ef = Math.Max(efSearch, topK);
            var w = SearchLayer(query, currObj, ef, 0);

            int resultCount = Math.Min(topK, w.Count);
            // Trim w down to topK best results
            while (w.Count > resultCount)
            {
                w.Dequeue();
            }

            var results = new SearchResult[resultCount];
            int metaCount = _metadata.Count;

            // w is a min-heap on -distance, so Dequeue pops the farthest among the topK first.
            // Populating backwards fills indices [resultCount-1 ... 0], placing highest score at index 0!
            for (int i = resultCount - 1; i >= 0; i--)
            {
                w.TryDequeue(out int id, out _);
                float score = DistanceKernels.DotProduct(query, _storage.GetVector(id));
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

    private PriorityQueue<int, float> SearchLayer(
        ReadOnlySpan<float> query,
        int enterNode,
        int ef,
        int layer)
    {
        uint tag = Interlocked.Increment(ref _currentTag);
        if (tag == 0)
        {
            Array.Clear(_visitedTags, 0, _visitedTags.Length);
            tag = Interlocked.Increment(ref _currentTag);
        }

        // candidates: Min-heap of positive distances (closest node popped first for greedy walk)
        var candidates = new PriorityQueue<int, float>(ef + 1);
        // results: Min-heap of negative distances (farthest node among best results at root for eviction)
        var results = new PriorityQueue<int, float>(ef + 1);

        float dEnter = ComputeDistance(query, _storage.GetVector(enterNode));
        candidates.Enqueue(enterNode, dEnter);
        results.Enqueue(enterNode, -dEnter);
        _visitedTags[enterNode] = tag;

        while (candidates.Count > 0)
        {
            candidates.TryDequeue(out int c, out float dC);
            results.TryPeek(out _, out float negWorstDist);
            float worstDist = -negWorstDist;

            if (results.Count >= ef && dC > worstDist) break;

            var neighbors = GetNeighbors(c, layer);
            for (int n = 0; n < neighbors.Length; n++)
            {
                int e = neighbors[n];
                if (e == -1) break;

                if (_visitedTags[e] != tag)
                {
                    _visitedTags[e] = tag;
                    float dE = ComputeDistance(query, _storage.GetVector(e));

                    results.TryPeek(out _, out negWorstDist);
                    worstDist = -negWorstDist;

                    if (dE < worstDist || results.Count < ef)
                    {
                        candidates.Enqueue(e, dE);
                        results.Enqueue(e, -dE);
                        if (results.Count > ef)
                        {
                            results.Dequeue();
                        }
                    }
                }
            }
        }

        return results;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        // Cosine distance = 1.0f - DotProduct for normalized vectors
        float dot = DistanceKernels.DotProduct(a, b);
        return 1.0f - dot;
    }

    private ReadOnlySpan<int> GetNeighbors(int id, int layer)
    {
        if (layer == 0)
        {
            int offset = id * _m0;
            return _layer0Edges.AsSpan(offset, _m0);
        }
        if (layer - 1 < _upperLayers.Count)
        {
            var layerArr = _upperLayers[layer - 1];
            int offset = id * _m;
            if (offset + _m <= layerArr.Length)
                return layerArr.AsSpan(offset, _m);
        }
        return ReadOnlySpan<int>.Empty;
    }

    private Span<int> GetNeighborsSpan(int id, int layer)
    {
        if (layer == 0)
        {
            int offset = id * _m0;
            return _layer0Edges.AsSpan(offset, _m0);
        }
        while (_upperLayers.Count < layer)
        {
            var newL = new int[_nodeMaxLayers.Length * _m];
            Array.Fill(newL, -1);
            _upperLayers.Add(newL);
        }
        var arr = _upperLayers[layer - 1];
        int upperOffset = id * _m;
        return arr.AsSpan(upperOffset, _m);
    }

    private void SelectAndLinkNeighbors(int id, PriorityQueue<int, float> candidates, int layer)
    {
        int maxConn = layer == 0 ? _m0 : _m;

        // candidates contains elements with priority = -distance.
        // Truncate to keep the maxConn closest items (those with largest negative distance, i.e. smallest positive distance).
        while (candidates.Count > maxConn)
        {
            candidates.Dequeue();
        }

        int count = candidates.Count;
        Span<int> selected = stackalloc int[count];
        for (int i = 0; i < count; i++)
        {
            candidates.TryDequeue(out int neighbor, out _);
            selected[i] = neighbor;
        }

        var myNeighbors = GetNeighborsSpan(id, layer);
        selected.CopyTo(myNeighbors);
        if (selected.Length < maxConn)
        {
            myNeighbors.Slice(selected.Length).Fill(-1);
        }

        for (int i = 0; i < selected.Length; i++)
        {
            AddBidirectionalEdge(selected[i], id, layer, maxConn);
        }
    }

    private void AddBidirectionalEdge(int from, int to, int layer, int maxConn)
    {
        var existing = GetNeighborsSpan(from, layer);
        int freeSlot = -1;
        int worstSlot = -1;
        float worstDist = float.MinValue;
        float distToTo = ComputeDistance(_storage.GetVector(from), _storage.GetVector(to));

        for (int i = 0; i < maxConn; i++)
        {
            int neighbor = existing[i];
            if (neighbor == to) return; // already linked
            if (neighbor == -1)
            {
                freeSlot = i;
                break;
            }
            float d = ComputeDistance(_storage.GetVector(from), _storage.GetVector(neighbor));
            if (d > worstDist)
            {
                worstDist = d;
                worstSlot = i;
            }
        }

        if (freeSlot != -1)
        {
            existing[freeSlot] = to;
        }
        else if (worstSlot != -1 && distToTo < worstDist)
        {
            existing[worstSlot] = to;
        }
    }

    private int SelectRandomLayer()
    {
        double r = Random.Shared.NextDouble();
        if (r <= 0.0) r = 0.0000001;
        return (int)(-Math.Log(r) * _mL);
    }

    private void EnsureCapacity(int requiredCount)
    {
        if (requiredCount <= _nodeMaxLayers.Length) return;
        int newCap = Math.Max(_nodeMaxLayers.Length * 2, requiredCount);

        Array.Resize(ref _nodeMaxLayers, newCap);
        Array.Resize(ref _visitedTags, newCap);

        int oldL0Len = _layer0Edges.Length;
        Array.Resize(ref _layer0Edges, newCap * _m0);
        _layer0Edges.AsSpan(oldL0Len).Fill(-1);

        for (int l = 0; l < _upperLayers.Count; l++)
        {
            var arr = _upperLayers[l];
            int oldLen = arr.Length;
            Array.Resize(ref arr, newCap * _m);
            arr.AsSpan(oldLen).Fill(-1);
            _upperLayers[l] = arr;
        }
    }

    public void Dispose()
    {
        _rwLock.Dispose();
    }
}
