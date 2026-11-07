using System;
using System.Collections.Generic;
using System.IO;
using Glacier.Vector.Compute;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Xunit;

namespace Glacier.Vector.Tests
{
    public class BinaryQuantizationTests : IDisposable
    {
        private readonly string _tempFile;

        public BinaryQuantizationTests()
        {
            _tempFile = Path.Combine(Path.GetTempPath(), $"glacier_bin_test_{Guid.NewGuid():N}.bin");
        }

        public void Dispose()
        {
            if (File.Exists(_tempFile))
            {
                try { File.Delete(_tempFile); } catch { }
            }
        }

        [Fact]
        public void PackBinaryVector_ProducesCorrectSignBits_ForUlongAndUint()
        {
            // 64 dimensions: even indices positive (+1.5f), odd indices negative (-2.0f)
            float[] vector = new float[64];
            for (int i = 0; i < 64; i++)
            {
                vector[i] = (i % 2 == 0) ? 1.5f : -2.0f;
            }

            Span<ulong> packedUlong = stackalloc ulong[1];
            DistanceKernels.PackBinaryVector(vector, packedUlong);

            // Bit i should be 1 if vector[i] >= 0, else 0
            ulong expected = 0;
            for (int i = 0; i < 64; i++)
            {
                if (vector[i] >= 0f)
                {
                    expected |= (1UL << i);
                }
            }
            Assert.Equal(expected, packedUlong[0]);

            // Test uint overload (2 uints)
            Span<uint> packedUint = stackalloc uint[2];
            DistanceKernels.PackBinaryVector(vector, packedUint);

            uint expectedLow = (uint)(expected & 0xFFFFFFFF);
            uint expectedHigh = (uint)(expected >> 32);
            Assert.Equal(expectedLow, packedUint[0]);
            Assert.Equal(expectedHigh, packedUint[1]);
        }

        [Fact]
        public void PackBinaryVector_HandlesArbitraryDimensionsWithPadding()
        {
            // 75 dimensions -> requires 2 ulongs
            float[] vector = new float[75];
            for (int i = 0; i < 75; i++)
            {
                vector[i] = (i < 65) ? 1.0f : -1.0f;
            }

            Span<ulong> packed = stackalloc ulong[2];
            DistanceKernels.PackBinaryVector(vector, packed);

            // First word: all 64 bits are 1
            Assert.Equal(ulong.MaxValue, packed[0]);

            // Second word: bit 0 is 1 (index 64 is >= 0), bits 1..10 (indices 65..74) are 0
            Assert.Equal(1UL, packed[1]);
        }

        [Fact]
        public void HammingDistance_SelfIsZero_OppositeIsFullDimension()
        {
            const int dims = 128;
            float[] posVec = new float[dims];
            float[] negVec = new float[dims];
            Array.Fill(posVec, 1.0f);
            Array.Fill(negVec, -1.0f);

            Span<ulong> posPacked = stackalloc ulong[2];
            Span<ulong> negPacked = stackalloc ulong[2];
            DistanceKernels.PackBinaryVector(posVec, posPacked);
            DistanceKernels.PackBinaryVector(negVec, negPacked);

            // Self distance must be 0
            int selfDist = DistanceKernels.HammingDistance(posPacked, posPacked);
            Assert.Equal(0, selfDist);

            // Opposite distance must be 128
            int oppDist = DistanceKernels.HammingDistance(posPacked, negPacked);
            Assert.Equal(128, oppDist);

            // Check uint overload
            Span<uint> posPackedUint = stackalloc uint[4];
            Span<uint> negPackedUint = stackalloc uint[4];
            DistanceKernels.PackBinaryVector(posVec, posPackedUint);
            DistanceKernels.PackBinaryVector(negVec, negPackedUint);

            Assert.Equal(0, DistanceKernels.HammingDistance(posPackedUint, posPackedUint));
            Assert.Equal(128, DistanceKernels.HammingDistance(posPackedUint, negPackedUint));
        }

        [Fact]
        public void HammingDistance_MultiWordUnrolling_ComputesExactDistance()
        {
            // Test 8 ulongs (512 dimensions) to exercise the 4-word unrolled loop + remainder
            ulong[] a = new ulong[] { 0x01, 0x03, 0x07, 0x0F, 0x1F, 0x3F, 0x7F, 0xFF };
            ulong[] b = new ulong[] { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

            // Popcounts: 1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 = 36
            int dist = DistanceKernels.HammingDistance(a, b);
            Assert.Equal(36, dist);
        }

        [Fact]
        public void BinaryVectorStorage_StoresAndRetrievesPackedVectors()
        {
            const int dims = 128;
            using var storage = new BinaryVectorStorage(dims, initialCapacity: 4);

            Assert.Equal(dims, storage.Dimensions);
            Assert.Equal(2, storage.WordsPerVector);
            Assert.Equal(0, storage.Count);

            float[] v1 = new float[dims];
            float[] v2 = new float[dims];
            Array.Fill(v1, 1.0f);
            Array.Fill(v2, -1.0f);

            storage.Append(v1);
            storage.Append(v2);

            Assert.Equal(2, storage.Count);

            ReadOnlySpan<ulong> p1 = storage.GetPackedVector(0);
            ReadOnlySpan<ulong> p2 = storage.GetPackedVector(1);

            Assert.Equal(ulong.MaxValue, p1[0]);
            Assert.Equal(ulong.MaxValue, p1[1]);
            Assert.Equal(0UL, p2[0]);
            Assert.Equal(0UL, p2[1]);

            // AppendPacked direct
            ulong[] manualPacked = new ulong[] { 0xAAAAAAAAAAAAAAAAUL, 0x5555555555555555UL };
            storage.AppendPacked(manualPacked);
            Assert.Equal(3, storage.Count);

            ReadOnlySpan<ulong> p3 = storage.GetPackedVector(2);
            Assert.Equal(0xAAAAAAAAAAAAAAAAUL, p3[0]);
            Assert.Equal(0x5555555555555555UL, p3[1]);
        }

        [Fact]
        public void SearchWithReRanking_FindsNearestNeighborWithHighAccuracy()
        {
            const int dims = 64;
            const int numDocs = 200;
            var rand = new Random(42);

            using var pagedStorage = new PagedVectorStorage(_tempFile, dims, maxMemoryBytes: 65536);
            using var index = new VectorIndex(pagedStorage);

            float[][] docs = new float[numDocs][];
            for (int i = 0; i < numDocs; i++)
            {
                docs[i] = new float[dims];
                for (int d = 0; d < dims; d++)
                {
                    docs[i][d] = (float)rand.NextDouble() * 2f - 1f;
                }
                // Normalize for cosine similarity
                float norm = MathF.Sqrt(DistanceKernels.DotProduct(docs[i], docs[i]));
                for (int d = 0; d < dims; d++)
                {
                    docs[i][d] /= norm;
                }
                index.Add(docs[i], $"doc_{i}");
            }

            // Query is close to doc 42
            float[] query = new float[dims];
            for (int d = 0; d < dims; d++)
            {
                query[d] = docs[42][d] + (float)(rand.NextDouble() - 0.5) * 0.05f;
            }
            float qNorm = MathF.Sqrt(DistanceKernels.DotProduct(query, query));
            for (int d = 0; d < dims; d++)
            {
                query[d] /= qNorm;
            }

            // Exact search
            var exactResults = index.Search(query, topK: 5);
            Assert.NotEmpty(exactResults);
            Assert.Equal(42, exactResults[0].Id);

            // Two-tier re-ranking search
            var reRankResults = index.SearchWithReRanking(query, k: 5, oversampleFactor: 8);
            Assert.NotEmpty(reRankResults);
            Assert.Equal(5, reRankResults.Length);

            // The nearest neighbor (doc 42) must match
            Assert.Equal(42, reRankResults[0].Id);
            Assert.Equal(exactResults[0].Score, reRankResults[0].Score, 1e-4);
        }

        [Fact]
        public void SearchWithReRanking_EdgeCases_EmptyOrInvalidK()
        {
            const int dims = 32;
            using var pagedStorage = new PagedVectorStorage(_tempFile, dims);
            using var index = new VectorIndex(pagedStorage);

            float[] query = new float[dims];

            // Empty index
            var resEmpty = index.SearchWithReRanking(query, k: 5);
            Assert.Empty(resEmpty);

            // k <= 0
            index.Add(query, "meta");
            var resZero = index.SearchWithReRanking(query, k: 0);
            Assert.Empty(resZero);
        }
    }
}
