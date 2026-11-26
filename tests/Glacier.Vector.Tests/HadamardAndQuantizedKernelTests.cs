using System;
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
using Xunit;

namespace Glacier.Vector.Tests;

public class HadamardAndQuantizedKernelTests
{
    // =========================================================================
    // 1. HADAMARD PROJECTION — PARSEVAL'S THEOREM & ORTHOGONALITY
    // =========================================================================

    [Theory]
    [InlineData(64)]
    [InlineData(512)]
    [InlineData(1024)]
    [InlineData(4096)]
    public void HadamardProjection_Fwht_PreservesInnerProductAndNorm_Parseval(int dim)
    {
        var rng = new Random(1000 + dim);
        float[] x = new float[dim];
        float[] y = new float[dim];
        float[] hx = new float[dim];
        float[] hy = new float[dim];

        for (int i = 0; i < dim; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            y[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            hx[i] = x[i];
            hy[i] = y[i];
        }

        float dotBefore = DistanceKernels.DotProduct(x, y);
        float normXBefore = DistanceKernels.DotProduct(x, x);

        HadamardProjection.ApplyFwhtInPlace(hx);
        HadamardProjection.ApplyFwhtInPlace(hy);

        float dotAfter = DistanceKernels.DotProduct(hx, hy);
        float normXAfter = DistanceKernels.DotProduct(hx, hx);

        Assert.True(MathF.Abs(dotBefore - dotAfter) < 1e-2f,
            $"Parseval <Hx, Hy> == <x, y> violated for dim={dim}: before={dotBefore}, after={dotAfter}");
        Assert.True(MathF.Abs(normXBefore - normXAfter) / MathF.Max(1f, normXBefore) < 1e-5f,
            $"Parseval ||Hx||^2 == ||x||^2 violated for dim={dim}: before={normXBefore}, after={normXAfter}");

        // Self-inverse check: H(H(x)) == x
        HadamardProjection.ApplyFwhtInPlace(hx);
        float maxErr = 0f;
        for (int i = 0; i < dim; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(hx[i] - x[i]));
        Assert.True(maxErr < 1e-4f, $"FWHT self-inverse maxErr={maxErr}");
    }

    [Theory]
    [InlineData(384, 512)]
    [InlineData(768, 1024)]
    [InlineData(1536, 2048)]
    [InlineData(4096, 4096)]
    public void HadamardProjection_RandomRotation_PreservesInnerProductAndRoundtrips(int dim, int expectedPadded)
    {
        Assert.Equal(expectedPadded, HadamardProjection.GetPaddedDimension(dim));

        float[] signs = new float[expectedPadded];
        HadamardProjection.GenerateSigns(signs, seed: 0xDEADBEEF_CAFEBABEUL);

        var rng = new Random(2026 + dim);
        float[] x = new float[dim];
        float[] y = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            y[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        }

        float dotExact = DistanceKernels.DotProduct(x, y);

        float[] rx = new float[expectedPadded];
        float[] ry = new float[expectedPadded];
        HadamardProjection.ApplyRandomRotation(x, rx, signs);
        HadamardProjection.ApplyRandomRotation(y, ry, signs);

        float dotRot = DistanceKernels.DotProduct(rx, ry);
        Assert.True(MathF.Abs(dotExact - dotRot) < 1e-2f,
            $"Random rotation inner product drift for dim={dim}: exact={dotExact}, rot={dotRot}");

        // Exact inverse recovers original unpadded vector
        float[] recovered = new float[dim];
        HadamardProjection.ApplyRandomRotationInverse(rx, recovered, signs);
        float maxErr = 0f;
        for (int i = 0; i < dim; i++)
            maxErr = MathF.Max(maxErr, MathF.Abs(recovered[i] - x[i]));
        Assert.True(maxErr < 1e-4f, $"Random rotation inverse maxErr={maxErr} for dim={dim}");
    }

    // =========================================================================
    // 2. RABITQ BINARY ROTATION & HAMMING COSINE CORRELATION
    // =========================================================================

    [Fact]
    public void EstimateCosineFromHamming_ExactBoundaryValues()
    {
        Assert.Equal(1.0f, HadamardProjection.EstimateCosineFromHamming(0, 1024), 5);
        Assert.Equal(0.0f, HadamardProjection.EstimateCosineFromHamming(512, 1024), 5);
        Assert.Equal(-1.0f, HadamardProjection.EstimateCosineFromHamming(1024, 1024), 5);
        Assert.Equal(0.5f, HadamardProjection.EstimateCosineFromHamming(100, 300), 5);
    }

    [Theory]
    [InlineData(1024)]
    [InlineData(4096)]
    public void RotateAndBinarize_HammingAndRaBitQAsymmetric_StronglyCorrelateWithExactCosine(int dim)
    {
        int padded = HadamardProjection.GetPaddedDimension(dim);
        int words = HadamardProjection.GetPackedWordCount(dim);
        float[] signs = new float[padded];
        HadamardProjection.GenerateSigns(signs, seed: 0x1234_5678_9ABC_DEF0UL);

        var rng = new Random(42 + dim);
        float[] query = CreateRandomUnitVector(rng, dim);
        float[] qRot = new float[padded];
        HadamardProjection.ApplyRandomRotation(query, qRot, signs);

        ulong[] qBits = new ulong[words];
        BinaryRotationMetadata qMeta = HadamardProjection.RotateAndBinarize(query, signs, qBits);
        Assert.Equal(padded, qMeta.PaddedDimensions);
        Assert.InRange(qMeta.Norm, 0.999f, 1.001f);
        // For high-D Gaussian-like rotated coordinates, E[|z|]/sqrt(E[z^2]) = sqrt(2/pi) ~= 0.7979
        Assert.InRange(qMeta.CentroidInnerProduct, 0.70f, 0.90f);

        const int numPairs = 80;
        double[] exactCos = new double[numPairs];
        double[] hammingCos = new double[numPairs];
        double[] asymCos = new double[numPairs];

        ulong[] dbBits = new ulong[words];
        float[] scratch = new float[padded];

        for (int p = 0; p < numPairs; p++)
        {
            // Sweep target cosine across [-0.9, +0.95]
            float alpha = -0.9f + (1.85f * p) / (numPairs - 1);
            float[] noise = CreateRandomUnitVector(rng, dim);
            float[] candidate = new float[dim];
            for (int i = 0; i < dim; i++)
                candidate[i] = alpha * query[i] + MathF.Sqrt(MathF.Max(0f, 1f - alpha * alpha)) * noise[i];
            NormalizeInPlace(candidate);

            exactCos[p] = DistanceKernels.DotProduct(query, candidate);

            BinaryRotationMetadata dbMeta = HadamardProjection.RotateAndBinarize(candidate, signs, dbBits, scratch);
            hammingCos[p] = HadamardProjection.EstimateCosineFromHamming(qBits, dbBits, padded);
            asymCos[p] = HadamardProjection.EstimateAsymmetricCosine(qRot, dbBits, in dbMeta);
        }

        double corrHamming = PearsonCorrelation(exactCos, hammingCos);
        double corrAsym = PearsonCorrelation(exactCos, asymCos);

        Assert.True(corrHamming > 0.985, $"Binary Hamming cosine correlation too low ({corrHamming:F4}) at dim={dim}");
        Assert.True(corrAsym > 0.990, $"RaBitQ asymmetric cosine correlation too low ({corrAsym:F4}) at dim={dim}");
    }

    // =========================================================================
    // 3. QUANTIZED INT8 & 4-BIT PACKED DOT-PRODUCT EXACTNESS (SIMD vs SWAR vs SCALAR)
    // =========================================================================

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(7)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(17)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(127)]
    [InlineData(512)]
    [InlineData(1536)]
    public void DotInt8_And_SumInt8_SimdMatchesSwarAndScalarExactly(int length)
    {
        var rng = new Random(7000 + length);
        sbyte[] a = new sbyte[length];
        sbyte[] b = new sbyte[length];
        int expectedDot = 0;
        int expectedSumA = 0;

        for (int i = 0; i < length; i++)
        {
            a[i] = (sbyte)rng.Next(-128, 128);
            b[i] = (sbyte)rng.Next(-128, 128);
            expectedDot += a[i] * b[i];
            expectedSumA += a[i];
        }

        int simdDot = DistanceKernels.DotInt8(a, b);
        int swarDot = DistanceKernels.DotInt8Swar(a, b);
        int simdSum = DistanceKernels.SumInt8(a);

        Assert.Equal(expectedDot, simdDot);
        Assert.Equal(expectedDot, swarDot);
        Assert.Equal(expectedSumA, simdSum);
    }

    [Fact]
    public void DotInt8_ExtremeMinValues_NoInt16Overflow()
    {
        // (-128) * (-128) = +16384 per lane; pairwise sum = +32768 (exceeds short.MaxValue = 32767!).
        // Verify vpmaddwd widens directly to int32 with zero saturation/overflow.
        const int length = 1024;
        sbyte[] a = new sbyte[length];
        sbyte[] b = new sbyte[length];
        Array.Fill(a, sbyte.MinValue);
        Array.Fill(b, sbyte.MinValue);

        int expected = length * 128 * 128;
        Assert.Equal(expected, DistanceKernels.DotInt8(a, b));
        Assert.Equal(expected, DistanceKernels.DotInt8Swar(a, b));
        Assert.Equal(-128 * length, DistanceKernels.SumInt8(a));
    }

    [Fact]
    public void DotInt8Quantized_SymmetricAffineAndFloatInt8_MatchDequantizedFloatDot()
    {
        const int length = 259; // non-multiple of 16/32
        var rng = new Random(8888);
        sbyte[] a = new sbyte[length];
        sbyte[] b = new sbyte[length];
        float[] fa = new float[length];
        float[] fb = new float[length];
        float[] query = new float[length];

        const float scaleA = 0.015f;
        const float scaleB = 0.025f;
        const int zpA = -13;
        const int zpB = 27;

        double expectedSym = 0.0;
        double expectedAffine = 0.0;
        double expectedFloatInt8 = 0.0;

        for (int i = 0; i < length; i++)
        {
            a[i] = (sbyte)rng.Next(-128, 128);
            b[i] = (sbyte)rng.Next(-128, 128);
            query[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

            fa[i] = (a[i] - zpA) * scaleA;
            fb[i] = (b[i] - zpB) * scaleB;

            expectedSym += (a[i] * scaleA) * (b[i] * scaleB);
            expectedAffine += (double)fa[i] * fb[i];
            expectedFloatInt8 += (double)query[i] * (b[i] * scaleB);
        }

        float actualSym = DistanceKernels.DotInt8Quantized(a, scaleA, b, scaleB);
        float actualAffine = DistanceKernels.DotInt8Quantized(a, scaleA, zpA, b, scaleB, zpB);
        float actualFloatInt8 = DistanceKernels.DotFloatInt8(query, b, scaleB);

        Assert.True(Math.Abs(expectedSym - actualSym) < 1e-2, $"Sym mismatch: {expectedSym} vs {actualSym}");
        Assert.True(Math.Abs(expectedAffine - actualAffine) < 1e-2, $"Affine mismatch: {expectedAffine} vs {actualAffine}");
        Assert.True(Math.Abs(expectedFloatInt8 - actualFloatInt8) < 1e-3, $"FloatInt8 mismatch: {expectedFloatInt8} vs {actualFloatInt8}");
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(8)]
    [InlineData(9)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(129)]
    [InlineData(255)]
    [InlineData(512)]
    [InlineData(1536)]
    public void DotUInt4Packed_And_SumUInt4Packed_SimdMatchesSwarAndScalarExactly(int dimensions)
    {
        var rng = new Random(9000 + dimensions);
        byte[] aNibbles = new byte[dimensions];
        byte[] bNibbles = new byte[dimensions];

        int expectedDot = 0;
        int expectedSumA = 0;
        for (int i = 0; i < dimensions; i++)
        {
            aNibbles[i] = (byte)rng.Next(0, 16);
            bNibbles[i] = (byte)rng.Next(0, 16);
            expectedDot += aNibbles[i] * bNibbles[i];
            expectedSumA += aNibbles[i];
        }

        int packedLen = (dimensions + 1) / 2;
        byte[] aPacked = new byte[packedLen];
        byte[] bPacked = new byte[packedLen];
        DistanceKernels.PackUInt4(aNibbles, aPacked);
        DistanceKernels.PackUInt4(bNibbles, bPacked);

        // Deliberately poison high nibble of last byte when dimensions is odd to verify masking!
        if ((dimensions & 1) != 0)
        {
            aPacked[packedLen - 1] |= 0xF0;
            bPacked[packedLen - 1] |= 0xF0;
        }

        int simdDot = DistanceKernels.DotUInt4Packed(aPacked, bPacked, dimensions);
        int swarDot = DistanceKernels.DotUInt4PackedSwar(aPacked, bPacked, dimensions);
        int simdSum = DistanceKernels.SumUInt4Packed(aPacked, dimensions);

        Assert.Equal(expectedDot, simdDot);
        Assert.Equal(expectedDot, swarDot);
        Assert.Equal(expectedSumA, simdSum);
    }

    [Fact]
    public void DotUInt4Packed_MaxNibblesAndAffineQuantized_NoOverflowAndExact()
    {
        const int dims = 513;
        byte[] aPacked = new byte[(dims + 1) / 2];
        byte[] bPacked = new byte[(dims + 1) / 2];
        Array.Fill(aPacked, (byte)0xFF);
        Array.Fill(bPacked, (byte)0xFF);

        // All 513 nibbles are 15 -> dot = 513 * 225, sum = 513 * 15
        Assert.Equal(dims * 225, DistanceKernels.DotUInt4Packed(aPacked, bPacked, dims));
        Assert.Equal(dims * 15, DistanceKernels.SumUInt4Packed(aPacked, dims));

        const float scaleA = 0.125f, minA = -1.0f;
        const float scaleB = 0.0625f, minB = -0.5f;
        // Every element of A = -1.0 + 15*0.125 = 0.875; every element of B = -0.5 + 15*0.0625 = 0.4375
        float expected = dims * 0.875f * 0.4375f;
        float actual = DistanceKernels.DotUInt4PackedQuantized(aPacked, scaleA, minA, bPacked, scaleB, minB, dims);
        Assert.Equal(expected, actual, 3);
    }

    // =========================================================================
    // 4. GPU CONTIGUOUS SPLIT-K FP32 & ZERO-I2F DP4A INT8 PARITY TESTS
    // =========================================================================

    [Fact]
    public void GpuVectorAccelerator_SplitK_FP32_HighDim_MatchesCpuExactly()
    {
        const int dims = 512; // >= 128 triggers vector_dot_scan_splitk_fp32 on NVIDIA GPU
        const int count = 256;
        var rng = new Random(12345);

        float[] query = new float[dims];
        float[] db = new float[count * dims];
        for (int i = 0; i < dims; i++) query[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
        for (int i = 0; i < db.Length; i++) db[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        float[] cpuScores = new float[count];
        for (int v = 0; v < count; v++)
            cpuScores[v] = DistanceKernels.DotProduct(query, db.AsSpan(v * dims, dims));

        if (GpuVectorAccelerator.IsNvidiaAvailable)
        {
            float[] gpuScores = new float[count];
            bool ok = GpuVectorAccelerator.ScanChunkGpu(
                query, db, gpuScores, count, dims, isL2Distance: false, target: GpuTarget.Nvidia);
            Assert.True(ok, "NVIDIA Split-K FP32 scan failed to execute.");

            for (int v = 0; v < count; v++)
            {
                Assert.True(MathF.Abs(cpuScores[v] - gpuScores[v]) < 1e-3f,
                    $"Split-K FP32 GPU mismatch at vec {v}: cpu={cpuScores[v]}, gpu={gpuScores[v]}");
            }
        }
    }

    [Fact]
    public void GpuVectorAccelerator_Int8Dp4a_MatchesCpuQuantizedDot()
    {
        const int dims = 256;
        const int count = 512;
        var rng = new Random(54321);

        sbyte[] query = new sbyte[dims];
        sbyte[] db = new sbyte[count * dims];
        for (int i = 0; i < dims; i++) query[i] = (sbyte)rng.Next(-128, 128);
        for (int i = 0; i < db.Length; i++) db[i] = (sbyte)rng.Next(-128, 128);

        const float scaleQ = 0.01f;
        const float scaleDb = 0.02f;

        float[] cpuScores = new float[count];
        bool cpuOk = GpuVectorAccelerator.ScanChunkInt8Gpu(
            query, scaleQ, db, scaleDb, cpuScores, count, dims, GpuTarget.Cpu);
        Assert.True(cpuOk);

        for (int v = 0; v < count; v++)
        {
            float expected = DistanceKernels.DotInt8Quantized(query, scaleQ, db.AsSpan(v * dims, dims), scaleDb);
            Assert.Equal(expected, cpuScores[v], 4);
        }

        if (GpuVectorAccelerator.IsNvidiaAvailable)
        {
            float[] gpuScores = new float[count];
            bool gpuOk = GpuVectorAccelerator.ScanChunkInt8Gpu(
                query, scaleQ, db, scaleDb, gpuScores, count, dims, GpuTarget.Nvidia);
            Assert.True(gpuOk, "NVIDIA dp4a INT8 scan failed to execute.");

            for (int v = 0; v < count; v++)
            {
                Assert.True(MathF.Abs(cpuScores[v] - gpuScores[v]) < 1e-3f,
                    $"dp4a INT8 GPU mismatch at vec {v}: cpu={cpuScores[v]}, gpu={gpuScores[v]}");
            }
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    private static float[] CreateRandomUnitVector(Random rng, int dim)
    {
        float[] v = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            // Box-Muller Gaussian for isotropic random directions
            double u1 = Math.Max(1e-12, rng.NextDouble());
            double u2 = rng.NextDouble();
            v[i] = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        NormalizeInPlace(v);
        return v;
    }

    private static void NormalizeInPlace(float[] v)
    {
        float norm = MathF.Sqrt(DistanceKernels.DotProduct(v, v));
        if (norm > 0f)
        {
            float inv = 1f / norm;
            for (int i = 0; i < v.Length; i++) v[i] *= inv;
        }
    }

    private static double PearsonCorrelation(double[] xs, double[] ys)
    {
        double meanX = 0, meanY = 0;
        for (int i = 0; i < xs.Length; i++) { meanX += xs[i]; meanY += ys[i]; }
        meanX /= xs.Length;
        meanY /= ys.Length;

        double num = 0, denX = 0, denY = 0;
        for (int i = 0; i < xs.Length; i++)
        {
            double dx = xs[i] - meanX;
            double dy = ys[i] - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
        }
        return num / Math.Sqrt(denX * denY);
    }
}
