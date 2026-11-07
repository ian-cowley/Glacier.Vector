namespace Glacier.Vector.Tests;

using System;
using System.Collections.Generic;
using System.IO;
using System.Numerics;
using Glacier.Vector.Compute;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

/// <summary>
/// Tier-5 White-Box Adversarial Coverage Hardening for Glacier.Vector.
/// Stress-tests 1-bit binary quantization sign packing across arbitrary dimension boundaries,
/// AVX-512 vs scalar Hamming distance oracle verification, and two-tier FP32 re-ranking top-k recall dynamics.
/// </summary>
public class Tier5VectorQuantizationAdversarialTests : IDisposable
{
    private readonly string _tempFile;

    public Tier5VectorQuantizationAdversarialTests()
    {
        _tempFile = Path.Combine(Path.GetTempPath(), $"tier5_vec_{Guid.NewGuid():N}.bin");
    }

    public void Dispose()
    {
        if (File.Exists(_tempFile))
        {
            try { File.Delete(_tempFile); } catch { }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(15)]
    [InlineData(16)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(63)]
    [InlineData(64)]
    [InlineData(65)]
    [InlineData(127)]
    [InlineData(128)]
    [InlineData(255)]
    [InlineData(256)]
    [InlineData(384)]
    [InlineData(512)]
    public void BinaryQuantization_ArbitraryDimensions_MatchesBitOracle(int dims)
    {
        var rng = new Random(42 + dims);
        float[] source = new float[dims];
        for (int i = 0; i < dims; i++)
        {
            // Mix positive, negative, zero, subnormals
            int mode = i % 5;
            source[i] = mode switch
            {
                0 => 0.0f,
                1 => -0.0f,
                2 => (float)(rng.NextDouble() * 10.0 + 0.1),
                3 => (float)(-(rng.NextDouble() * 10.0 + 0.1)),
                _ => 1e-45f
            };
        }

        // 1. Test ulong packing
        int ulongWords = (dims + 63) / 64;
        Span<ulong> packedUlong = stackalloc ulong[ulongWords];
        DistanceKernels.PackBinaryVector(source, packedUlong);

        for (int i = 0; i < dims; i++)
        {
            int w = i / 64;
            int b = i % 64;
            bool expectedBit = source[i] >= 0.0f;
            bool actualBit = (packedUlong[w] & (1UL << b)) != 0;
            Assert.Equal(expectedBit, actualBit);
        }

        // Verify remaining padding bits in the last word are strictly zero
        int remUlong = dims % 64;
        if (remUlong > 0)
        {
            ulong tailPaddingMask = ~((1UL << remUlong) - 1UL);
            Assert.Equal(0UL, packedUlong[^1] & tailPaddingMask);
        }

        // 2. Test uint packing
        int uintWords = (dims + 31) / 32;
        Span<uint> packedUint = stackalloc uint[uintWords];
        DistanceKernels.PackBinaryVector(source, packedUint);

        for (int i = 0; i < dims; i++)
        {
            int w = i / 32;
            int b = i % 32;
            bool expectedBit = source[i] >= 0.0f;
            bool actualBit = (packedUint[w] & (1U << b)) != 0;
            Assert.Equal(expectedBit, actualBit);
        }

        int remUint = dims % 32;
        if (remUint > 0)
        {
            uint tailPaddingMask = ~((1U << remUint) - 1U);
            Assert.Equal(0U, packedUint[^1] & tailPaddingMask);
        }
    }

    [Fact]
    public void HammingDistance_MetricAxiomsAndSoftwareReferenceOracle()
    {
        int[] wordLengths = [1, 2, 3, 4, 5, 7, 8, 9, 12, 16, 24];
        var rng = new Random(10101);

        foreach (int words in wordLengths)
        {
            ulong[] a = new ulong[words];
            ulong[] b = new ulong[words];
            ulong[] c = new ulong[words];

            for (int i = 0; i < words; i++)
            {
                a[i] = (ulong)rng.NextInt64();
                b[i] = (ulong)rng.NextInt64();
                c[i] = (ulong)rng.NextInt64();
            }

            // Software Reference Oracle (bit-by-bit software counting)
            int expectedDistAB = 0;
            for (int w = 0; w < words; w++)
            {
                ulong xor = a[w] ^ b[w];
                expectedDistAB += BitOperations.PopCount(xor);
            }

            int actualDistAB = DistanceKernels.HammingDistance(a, b);
            Assert.Equal(expectedDistAB, actualDistAB);

            // Metric Axiom 1: Identity of Indiscernibles d(a, a) == 0
            Assert.Equal(0, DistanceKernels.HammingDistance(a, a));

            // Metric Axiom 2: Symmetry d(a, b) == d(b, a)
            int actualDistBA = DistanceKernels.HammingDistance(b, a);
            Assert.Equal(actualDistAB, actualDistBA);

            // Metric Axiom 3: Triangle Inequality d(a, c) <= d(a, b) + d(b, c)
            int actualDistAC = DistanceKernels.HammingDistance(a, c);
            int actualDistBC = DistanceKernels.HammingDistance(b, c);
            Assert.True(actualDistAC <= actualDistAB + actualDistBC,
                $"Triangle inequality violation: d(a,c)={actualDistAC} > d(a,b)={actualDistAB} + d(b,c)={actualDistBC}");
        }
    }

    [Fact]
    public void TwoTierFp32ReRanking_RecallDynamics_OversamplingSweep()
    {
        const int dims = 64;
        const int numDocs = 500;
        const int numQueries = 20;
        const int k = 10;
        var rng = new Random(777);

        using var storage = new PagedVectorStorage(_tempFile, dims, maxMemoryBytes: 128 * 1024);
        using var index = new VectorIndex(storage);

        float[][] docs = new float[numDocs][];
        for (int i = 0; i < numDocs; i++)
        {
            docs[i] = GenerateUnitVector(dims, rng);
            index.Add(docs[i], $"doc_{i}");
        }

        float[][] queries = new float[numQueries][];
        for (int q = 0; q < numQueries; q++)
        {
            queries[q] = GenerateUnitVector(dims, rng);
        }

        // Test Recall@10 across oversampling factors: 1, 2, 4, 8, 16
        int[] oversampleFactors = [1, 2, 4, 8, 16];
        double prevAvgRecall = 0.0;

        foreach (int factor in oversampleFactors)
        {
            double totalRecall = 0.0;

            for (int q = 0; q < numQueries; q++)
            {
                var query = queries[q];

                // Exact ground truth brute force
                var exact = index.Search(query, topK: k);
                var exactSet = new HashSet<int>(exact.Select(r => r.Id));

                // Two-tier re-ranking
                var approx = index.SearchWithReRanking(query, k: k, oversampleFactor: factor);
                Assert.Equal(k, approx.Length);

                int overlap = approx.Count(r => exactSet.Contains(r.Id));
                totalRecall += overlap / (double)k;
            }

            double avgRecall = totalRecall / numQueries;

            // Recall must monotonically improve or maintain with higher oversampling factors
            Assert.True(avgRecall >= prevAvgRecall - 0.02,
                $"Recall dropped with higher oversampling: factor {factor} recall {avgRecall:P1} < prev {prevAvgRecall:P1}");
            prevAvgRecall = avgRecall;

            // Factor 16 should achieve >= 85% Recall@10 on 64-d random embeddings
            if (factor >= 16)
            {
                Assert.True(avgRecall >= 0.85, $"Expected Recall@10 >= 85% for factor 16, got {avgRecall:P1}");
            }
        }
    }

    [Fact]
    public void TwoTierFp32ReRanking_CornerCases_SmallDatabaseAndOrthogonalVectors()
    {
        const int dims = 32;
        using var storage = new PagedVectorStorage(_tempFile, dims);
        using var index = new VectorIndex(storage);

        // 1. Database smaller than K (N=3, K=5)
        float[] v1 = new float[dims]; v1[0] = 1.0f;
        float[] v2 = new float[dims]; v2[1] = 1.0f;
        float[] v3 = new float[dims]; v3[2] = 1.0f;
        index.Add(v1, "v1");
        index.Add(v2, "v2");
        index.Add(v3, "v3");

        var query = new float[dims];
        query[0] = 1.0f; // Exact match with v1

        var results = index.SearchWithReRanking(query, k: 5, oversampleFactor: 4);
        Assert.Equal(3, results.Length); // Returns all 3 available
        Assert.Equal(0, results[0].Id);   // v1 is top match
        Assert.Equal(1.0f, results[0].Score, precision: 5);

        // 2. K = 1
        var top1 = index.SearchWithReRanking(query, k: 1, oversampleFactor: 2);
        Assert.Single(top1);
        Assert.Equal(0, top1[0].Id);

        // 3. Diametrically opposed vectors (-query)
        float[] oppQuery = new float[dims];
        oppQuery[0] = -1.0f;
        var oppResults = index.SearchWithReRanking(oppQuery, k: 3, oversampleFactor: 2);
        Assert.Equal(3, oppResults.Length);
        // v1 has dot product -1.0f, while orthogonal v2 and v3 have dot product 0.0f
        // So orthogonal v2 and v3 should rank above opposed v1!
        Assert.True(oppResults[0].Score >= 0.0f);
        Assert.Equal(0, oppResults[^1].Id); // v1 is worst match (-1.0f)
        Assert.Equal(-1.0f, oppResults[^1].Score, precision: 5);
    }

    private static float[] GenerateUnitVector(int dims, Random rng)
    {
        float[] vec = new float[dims];
        float sumSq = 0f;
        for (int i = 0; i < dims; i++)
        {
            vec[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
            sumSq += vec[i] * vec[i];
        }
        float invNorm = 1.0f / MathF.Sqrt(Math.Max(1e-9f, sumSq));
        for (int i = 0; i < dims; i++) vec[i] *= invNorm;
        return vec;
    }
}
