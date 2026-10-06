namespace Glacier.Vector.Benchmarks;

using System;
using BenchmarkDotNet.Attributes;
using Glacier.Vector.Compute;

[MemoryDiagnoser]
public class DistanceKernelBenchmarks
{
    [Params(64, 128, 768, 1536)]
    public int Dim { get; set; }

    private float[] _vecA = null!;
    private float[] _vecB = null!;

    [GlobalSetup]
    public void Setup()
    {
        var rand = new Random(42);
        _vecA = new float[Dim];
        _vecB = new float[Dim];
        for (int i = 0; i < Dim; i++)
        {
            _vecA[i] = (float)rand.NextDouble();
            _vecB[i] = (float)rand.NextDouble();
        }
    }

    [Benchmark]
    public float DotProduct()
    {
        return DistanceKernels.DotProduct(_vecA, _vecB);
    }

    [Benchmark]
    public float CosineDistance()
    {
        return DistanceKernels.CosineDistance(_vecA, _vecB);
    }

    [Benchmark]
    public float L2Distance()
    {
        return DistanceKernels.L2Distance(_vecA, _vecB);
    }

    [Benchmark]
    public float L2DistanceSquared()
    {
        return DistanceKernels.L2DistanceSquared(_vecA, _vecB);
    }
}
