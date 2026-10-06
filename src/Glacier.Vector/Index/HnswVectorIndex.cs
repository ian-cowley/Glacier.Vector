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

    // Thread-static scratch context for zero-allocation, thread-safe search traversal
    [ThreadStatic]
    private static HnswSearchScratch? t_scratch;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static HnswSearchScratch GetScratch() => t_scratch ??= new HnswSearchScratch();

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
        var scratch = GetScratch();
        for (int l = Math.Min(_maxLayer, nodeLayer); l >= 0; l--)
        {
            SearchLayer(vector, currObj, _efConstruction, l, scratch);
            int bestCandidate = scratch.Results.FindMin().Id;
            SelectAndLinkNeighbors(id, ref scratch.Results, l);
            if (bestCandidate != -1)
            {
                currObj = bestCandidate;
            }
        }

        if (nodeLayer > _maxLayer)
        {
            _maxLayer = nodeLayer;
            _enterNodeId = id;
        }
    }

    /// <summary>
    /// Searches the HNSW graph for the top-K nearest neighbors, writing results directly into the destination span.
    /// Performs zero GC heap allocations. Returns the number of results written.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public int Search(ReadOnlySpan<float> query, Span<SearchResult> destination, int efSearch = 64)
    {
        if (query.Length != Dimensions)
            throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

        if (destination.Length == 0) return 0;

        _rwLock.EnterReadLock();
        try
        {
            if (_enterNodeId == -1) return 0;

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
            int topK = destination.Length;
            int ef = Math.Max(efSearch, topK);
            var scratch = GetScratch();
            SearchLayer(query, currObj, ef, 0, scratch);

            ref var w = ref scratch.Results;
            int resultCount = Math.Min(topK, w.Count);

            // Trim w down to topK best results (dequeue the worst results)
            while (w.Count > resultCount)
            {
                w.TryDequeue(out _, out _);
            }

            int metaCount = _metadata.Count;

            // w is a max-heap on distance, so TryDequeue pops the farthest among the topK first.
            // Populating backwards fills destination[resultCount-1 ... 0], placing highest score (closest) at index 0!
            for (int i = resultCount - 1; i >= 0; i--)
            {
                w.TryDequeue(out int id, out _);
                float score = DistanceKernels.DotProduct(query, _storage.GetVector(id));
                string meta = (id < metaCount) ? _metadata[id] : string.Empty;
                destination[i] = new SearchResult(id, score, meta);
            }

            return resultCount;
        }
        finally
        {
            _rwLock.ExitReadLock();
        }
    }

    /// <summary>
    /// Searches the HNSW graph for the top-K nearest neighbors.
    /// Allocates the result array once and delegates to the zero-allocation overload.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public SearchResult[] Search(ReadOnlySpan<float> query, int topK, int efSearch = 64)
    {
        if (query.Length != Dimensions)
            throw new ArgumentException($"Expected query dimension {Dimensions}, got {query.Length}");

        if (_enterNodeId == -1 || topK <= 0) return Array.Empty<SearchResult>();

        int count = Math.Min(topK, _storage.Count);
        if (count <= 0) return Array.Empty<SearchResult>();

        var results = new SearchResult[count];
        int written = Search(query, results.AsSpan(), efSearch);
        if (written < count)
        {
            Array.Resize(ref results, written);
        }
        return results;
    }

    private void SearchLayer(
        ReadOnlySpan<float> query,
        int enterNode,
        int ef,
        int layer,
        HnswSearchScratch scratch)
    {
        scratch.EnsureCapacity(_storage.Count, ef);

        uint tag = ++scratch.CurrentTag;
        if (tag == 0)
        {
            Array.Clear(scratch.VisitedTags, 0, scratch.VisitedTags.Length);
            tag = ++scratch.CurrentTag;
        }

        ref var candidates = ref scratch.Candidates;
        ref var results = ref scratch.Results;

        candidates.Clear();
        results.Clear();

        float dEnter = ComputeDistance(query, _storage.GetVector(enterNode));
        candidates.Enqueue(enterNode, dEnter);
        results.Enqueue(enterNode, dEnter);
        scratch.VisitedTags[enterNode] = tag;

        while (candidates.Count > 0)
        {
            candidates.TryDequeue(out int c, out float dC);
            results.Peek(out _, out float worstDist);

            if (results.Count >= ef && dC > worstDist) break;

            var neighbors = GetNeighbors(c, layer);
            for (int n = 0; n < neighbors.Length; n++)
            {
                int e = neighbors[n];
                if (e == -1) break;

                if (scratch.VisitedTags[e] != tag)
                {
                    scratch.VisitedTags[e] = tag;
                    float dE = ComputeDistance(query, _storage.GetVector(e));

                    results.Peek(out _, out worstDist);

                    if (dE < worstDist || results.Count < ef)
                    {
                        candidates.Enqueue(e, dE);
                        results.Enqueue(e, dE);
                        if (results.Count > ef)
                        {
                            results.TryDequeue(out _, out _);
                        }
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ComputeDistance(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        return DistanceKernels.CosineDistance(a, b);
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

    private void SelectAndLinkNeighbors(int id, ref ValueMaxHeap results, int layer)
    {
        int maxConn = layer == 0 ? _m0 : _m;

        // results is a max-heap on distance (worst node at root).
        // Truncate to keep the maxConn closest items (those with smallest distance).
        while (results.Count > maxConn)
        {
            results.TryDequeue(out _, out _);
        }

        int count = results.Count;
        Span<int> selected = stackalloc int[count];
        for (int i = count - 1; i >= 0; i--)
        {
            results.TryDequeue(out int neighbor, out _);
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

/// <summary>
/// Thread-static scratch workspace for zero-allocation HNSW graph search.
/// </summary>
internal sealed class HnswSearchScratch
{
    public uint CurrentTag;
    public uint[] VisitedTags;
    public ValueMinHeap Candidates;
    public ValueMaxHeap Results;

    public HnswSearchScratch(int initialNodes = 1024, int initialCapacity = 256)
    {
        CurrentTag = 0;
        VisitedTags = new uint[initialNodes];
        Candidates = new ValueMinHeap(new (int, float)[initialCapacity]);
        Results = new ValueMaxHeap(new (int, float)[initialCapacity]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EnsureCapacity(int nodeCount, int ef)
    {
        if (VisitedTags.Length < nodeCount)
        {
            int newCap = Math.Max(nodeCount + 256, VisitedTags.Length * 2);
            Array.Resize(ref VisitedTags, newCap);
        }
        int minHeapCap = Math.Max(ef + 1, 256);
        Candidates.EnsureCapacity(minHeapCap);
        Results.EnsureCapacity(minHeapCap);
    }
}

/// <summary>
/// Zero-allocation, struct-based binary min-heap for (nodeId, distance) pairs.
/// Root contains the item with the minimum distance.
/// </summary>
internal struct ValueMinHeap
{
    private (int Id, float Dist)[] _buffer;
    private int _count;

    public ValueMinHeap((int Id, float Dist)[] buffer)
    {
        _buffer = buffer;
        _count = 0;
    }

    public readonly int Count => _count;
    public readonly bool IsEmpty => _count == 0;
    public readonly int Capacity => _buffer.Length;
    internal (int Id, float Dist)[] Buffer => _buffer;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clear() => _count = 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EnsureCapacity(int minCapacity)
    {
        if (_buffer.Length < minCapacity)
        {
            Array.Resize(ref _buffer, Math.Max(minCapacity, _buffer.Length * 2));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Enqueue(int id, float dist)
    {
        if (_count == _buffer.Length)
        {
            Array.Resize(ref _buffer, Math.Max(4, _buffer.Length * 2));
        }

        int i = _count++;
        while (i > 0)
        {
            int p = (i - 1) >> 1;
            if (dist >= _buffer[p].Dist) break;
            _buffer[i] = _buffer[p];
            i = p;
        }
        _buffer[i] = (id, dist);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryDequeue(out int id, out float dist)
    {
        if (_count == 0)
        {
            id = -1;
            dist = float.MaxValue;
            return false;
        }

        id = _buffer[0].Id;
        dist = _buffer[0].Dist;
        _count--;

        if (_count > 0)
        {
            var target = _buffer[_count];
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Dist < _buffer[child].Dist)
                {
                    child = right;
                }
                if (target.Dist <= _buffer[child].Dist) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = target;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Peek(out int id, out float dist)
    {
        if (_count == 0)
        {
            id = -1;
            dist = float.MaxValue;
            return false;
        }
        id = _buffer[0].Id;
        dist = _buffer[0].Dist;
        return true;
    }
}

/// <summary>
/// Zero-allocation, struct-based binary max-heap for (nodeId, distance) pairs.
/// Root contains the item with the maximum distance (worst candidate in top results).
/// </summary>
internal struct ValueMaxHeap
{
    private (int Id, float Dist)[] _buffer;
    private int _count;

    public ValueMaxHeap((int Id, float Dist)[] buffer)
    {
        _buffer = buffer;
        _count = 0;
    }

    public readonly int Count => _count;
    public readonly bool IsEmpty => _count == 0;
    public readonly int Capacity => _buffer.Length;
    internal (int Id, float Dist)[] Buffer => _buffer;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Clear() => _count = 0;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void EnsureCapacity(int minCapacity)
    {
        if (_buffer.Length < minCapacity)
        {
            Array.Resize(ref _buffer, Math.Max(minCapacity, _buffer.Length * 2));
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Enqueue(int id, float dist)
    {
        if (_count == _buffer.Length)
        {
            Array.Resize(ref _buffer, Math.Max(4, _buffer.Length * 2));
        }

        int i = _count++;
        while (i > 0)
        {
            int p = (i - 1) >> 1;
            if (dist <= _buffer[p].Dist) break;
            _buffer[i] = _buffer[p];
            i = p;
        }
        _buffer[i] = (id, dist);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool TryDequeue(out int id, out float dist)
    {
        if (_count == 0)
        {
            id = -1;
            dist = float.MinValue;
            return false;
        }

        id = _buffer[0].Id;
        dist = _buffer[0].Dist;
        _count--;

        if (_count > 0)
        {
            var target = _buffer[_count];
            int i = 0;
            int half = _count >> 1;
            while (i < half)
            {
                int child = (i << 1) + 1;
                int right = child + 1;
                if (right < _count && _buffer[right].Dist > _buffer[child].Dist)
                {
                    child = right;
                }
                if (target.Dist >= _buffer[child].Dist) break;
                _buffer[i] = _buffer[child];
                i = child;
            }
            _buffer[i] = target;
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public bool Peek(out int id, out float dist)
    {
        if (_count == 0)
        {
            id = -1;
            dist = float.MinValue;
            return false;
        }
        id = _buffer[0].Id;
        dist = _buffer[0].Dist;
        return true;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly (int Id, float Dist) FindMin()
    {
        if (_count == 0) return (-1, float.MaxValue);
        int bestId = _buffer[0].Id;
        float bestDist = _buffer[0].Dist;
        for (int i = 1; i < _count; i++)
        {
            if (_buffer[i].Dist < bestDist)
            {
                bestDist = _buffer[i].Dist;
                bestId = _buffer[i].Id;
            }
        }
        return (bestId, bestDist);
    }
}
