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
    private sbyte[] _i8A = null!;
    private sbyte[] _i8B = null!;
    private byte[] _u4A = null!;
    private byte[] _u4B = null!;
    private float[] _rotSrc = null!;
    private float[] _rotWork = null!;
    private float[] _signs = null!;
    private ulong[] _packedBits = null!;
    private int _paddedDim;

    [GlobalSetup]
    public void Setup()
    {
        var rand = new Random(42);
        _vecA = new float[Dim];
        _vecB = new float[Dim];
        _i8A = new sbyte[Dim];
        _i8B = new sbyte[Dim];
        byte[] nibA = new byte[Dim];
        byte[] nibB = new byte[Dim];
        for (int i = 0; i < Dim; i++)
        {
            _vecA[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
            _vecB[i] = (float)(rand.NextDouble() * 2.0 - 1.0);
            _i8A[i] = (sbyte)rand.Next(-127, 128);
            _i8B[i] = (sbyte)rand.Next(-127, 128);
            nibA[i] = (byte)rand.Next(0, 16);
            nibB[i] = (byte)rand.Next(0, 16);
        }

        _u4A = new byte[(Dim + 1) >> 1];
        _u4B = new byte[(Dim + 1) >> 1];
        DistanceKernels.PackUInt4(nibA, _u4A);
        DistanceKernels.PackUInt4(nibB, _u4B);

        _paddedDim = HadamardProjection.GetPaddedDimension(Dim);
        _rotSrc = new float[Dim];
        _vecA.CopyTo(_rotSrc, 0);
        _rotWork = new float[_paddedDim];
        _signs = new float[_paddedDim];
        HadamardProjection.GenerateSigns(_signs, seed: 0x9E3779B97F4A7C15UL);
        _packedBits = new ulong[HadamardProjection.GetPackedWordCount(Dim)];
    }

    [Benchmark]
    public float DotProduct() => DistanceKernels.DotProduct(_vecA, _vecB);

    [Benchmark]
    public int DotInt8() => DistanceKernels.DotInt8(_i8A, _i8B);

    [Benchmark]
    public int DotUInt4Packed() => DistanceKernels.DotUInt4Packed(_u4A, _u4B, Dim);

    [Benchmark]
    public void FwhtInPlace() => HadamardProjection.ApplyFwhtInPlace(_rotWork);

    [Benchmark]
    public BinaryRotationMetadata RotateAndBinarize() =>
        HadamardProjection.RotateAndBinarize(_rotSrc, _signs, _packedBits);

    [Benchmark]
    public float CosineDistance() => DistanceKernels.CosineDistance(_vecA, _vecB);

    [Benchmark]
    public float L2Distance() => DistanceKernels.L2Distance(_vecA, _vecB);

    [Benchmark]
    public float L2DistanceSquared() => DistanceKernels.L2DistanceSquared(_vecA, _vecB);
}
