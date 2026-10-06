namespace Glacier.Vector.Benchmarks;

using System;
using BenchmarkDotNet.Attributes;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

[MemoryDiagnoser]
public class HnswSearchBenchmarks
{
    private const int Dimensions = 128;
    private const int VectorCount = 2000;

    private InMemoryVectorStorage _storage = null!;
    private HnswVectorIndex _index = null!;
    private float[] _query = null!;
    private SearchResult[] _destination = null!;

    [Params(10, 50)]
    public int TopK { get; set; }

    [GlobalSetup]
    public void Setup()
    {
        var rand = new Random(42);
        _storage = new InMemoryVectorStorage(Dimensions);
        _index = new HnswVectorIndex(_storage, m: 16, efConstruction: 64, initialCapacity: VectorCount + 64);

        for (int i = 0; i < VectorCount; i++)
        {
            float[] vec = new float[Dimensions];
            float sumSq = 0f;
            for (int d = 0; d < Dimensions; d++)
            {
                vec[d] = (float)rand.NextDouble();
                sumSq += vec[d] * vec[d];
            }
            float norm = MathF.Sqrt(sumSq);
            for (int d = 0; d < Dimensions; d++) vec[d] /= norm;
            _index.Add(vec, $"doc_{i}");
        }

        _query = new float[Dimensions];
        for (int d = 0; d < Dimensions; d++) _query[d] = 1.0f / MathF.Sqrt(Dimensions);

        _destination = new SearchResult[TopK];
    }

    [GlobalCleanup]
    public void Cleanup()
    {
        _index.Dispose();
        _storage.Dispose();
    }

    [Benchmark(Baseline = true)]
    public SearchResult[] Search_AllocatingArray()
    {
        return _index.Search(_query, TopK, efSearch: 64);
    }

    [Benchmark]
    public int Search_ZeroAllocSpan()
    {
        return _index.Search(_query, _destination.AsSpan(0, TopK), efSearch: 64);
    }
}
