using System;
using System.IO;
using Xunit;
using Glacier.Vector.Storage;
using Glacier.Vector.Index;

namespace Glacier.Vector.Tests
{
    public class UnitTest1 : IDisposable
    {
        private readonly string _tempFile;

        public UnitTest1()
        {
            _tempFile = Path.Combine(Path.GetTempPath(), $"glacier_vector_test_{Guid.NewGuid():N}.bin");
        }

        public void Dispose()
        {
            if (File.Exists(_tempFile))
            {
                try { File.Delete(_tempFile); } catch { }
            }
        }

        [Fact]
        public void TestPagedVectorStorageBasicOps()
        {
            const int dims = 4;
            // Align page size to multiple of 4 * 4 = 16 bytes. Max(4096, 4096) = 4096.
            // 4096 / 16 = 256 vectors per page.
            // maxMemoryBytes = 8192, so we have exactly 2 pages.
            using (var storage = new PagedVectorStorage(_tempFile, dims, maxMemoryBytes: 8192))
            {
                Assert.Equal(0, storage.Count);
                Assert.Equal(dims, storage.Dimensions);

                // Append some vectors
                storage.Append(new float[] { 1f, 2f, 3f, 4f });
                storage.Append(new float[] { 5f, 6f, 7f, 8f });
                storage.Append(new float[] { 9f, 10f, 11f, 12f });

                Assert.Equal(3, storage.Count);

                // Get and check
                var v0 = storage.GetVector(0);
                Assert.Equal(new float[] { 1f, 2f, 3f, 4f }, v0.ToArray());

                var v1 = storage.GetVector(1);
                Assert.Equal(new float[] { 5f, 6f, 7f, 8f }, v1.ToArray());

                var v2 = storage.GetVector(2);
                Assert.Equal(new float[] { 9f, 10f, 11f, 12f }, v2.ToArray());
            }

            // Re-open and check persistence
            using (var storage = new PagedVectorStorage(_tempFile, dims, maxMemoryBytes: 8192))
            {
                Assert.Equal(3, storage.Count);
                Assert.Equal(new float[] { 5f, 6f, 7f, 8f }, storage.GetVector(1).ToArray());

                // Append more
                storage.Append(new float[] { 13f, 14f, 15f, 16f });
                Assert.Equal(4, storage.Count);
                Assert.Equal(new float[] { 13f, 14f, 15f, 16f }, storage.GetVector(3).ToArray());
            }
        }

        [Fact]
        public void TestVectorIndexSearchWithPagedStorage()
        {
            const int dims = 3;
            using (var storage = new PagedVectorStorage(_tempFile, dims, maxMemoryBytes: 4096))
            using (var index = new VectorIndex(storage))
            {
                index.Add(new float[] { 1f, 0f, 0f }, "Doc A");
                index.Add(new float[] { 0f, 1f, 0f }, "Doc B");
                index.Add(new float[] { 0f, 0f, 1f }, "Doc C");

                Assert.Equal(3, index.Count);

                // Query closest to Doc B
                var query = new float[] { 0.1f, 0.9f, 0f };
                var results = index.Search(query, topK: 2);

                Assert.Equal(2, results.Length);
                Assert.Equal("Doc B", results[0].Metadata);
                Assert.Equal("Doc A", results[1].Metadata);
            }
        }

        [Fact]
        public void TestDistanceKernelsAccuracy()
        {
            const int dims = 135; // Non-multiple of 16/32/64 to test unrolling + tail
            var rng = new Random(42);
            float[] a = new float[dims];
            float[] b = new float[dims];
            double expectedDot = 0;
            double expectedL2 = 0;

            for (int i = 0; i < dims; i++)
            {
                a[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                b[i] = (float)(rng.NextDouble() * 2.0 - 1.0);
                expectedDot += a[i] * b[i];
                float diff = a[i] - b[i];
                expectedL2 += diff * diff;
            }

            float actualDot = Glacier.Vector.Compute.DistanceKernels.DotProduct(a, b);
            float actualL2 = Glacier.Vector.Compute.DistanceKernels.L2DistanceSquared(a, b);

            Assert.True(Math.Abs(actualDot - (float)expectedDot) < 1e-4f,
                $"DotProduct mismatch: expected {expectedDot:F6}, got {actualDot:F6}");
            Assert.True(Math.Abs(actualL2 - (float)expectedL2) < 1e-4f,
                $"L2DistanceSquared mismatch: expected {expectedL2:F6}, got {actualL2:F6}");
        }

        [Fact]
        public void TestVectorConcurrency()
        {
            int initial = Glacier.Vector.Core.VectorConcurrency.MaxDegreeOfParallelism;
            try
            {
                Glacier.Vector.Core.VectorConcurrency.MaxDegreeOfParallelism = 0;
                Assert.True(Glacier.Vector.Core.VectorConcurrency.GetEffectiveParallelism() >= 1);

                Glacier.Vector.Core.VectorConcurrency.MaxDegreeOfParallelism = 4;
                Assert.Equal(Math.Min(4, Environment.ProcessorCount), Glacier.Vector.Core.VectorConcurrency.GetEffectiveParallelism());

                Assert.Equal(2, Glacier.Vector.Core.VectorConcurrency.GetEffectiveParallelism(requested: 2));

                Glacier.Vector.Core.VectorConcurrency.MaxDegreeOfParallelism = -1;
                Assert.Equal(Math.Max(1, Environment.ProcessorCount - 2), Glacier.Vector.Core.VectorConcurrency.GetEffectiveParallelism());
            }
            finally
            {
                Glacier.Vector.Core.VectorConcurrency.MaxDegreeOfParallelism = initial;
            }
        }

        [Fact]
        public void TestVectorIndexParallelSearchConsistency()
        {
            const int dims = 64;
            const int count = 500;
            var rng = new Random(123);

            using (var storage = new InMemoryVectorStorage(dims))
            using (var index = new VectorIndex(storage))
            {
                for (int i = 0; i < count; i++)
                {
                    float[] vec = new float[dims];
                    for (int d = 0; d < dims; d++)
                        vec[d] = (float)rng.NextDouble();
                    index.Add(vec, $"Doc_{i}");
                }

                float[] query = new float[dims];
                for (int d = 0; d < dims; d++) query[d] = (float)rng.NextDouble();

                var seqResults = index.Search(query, topK: 5, maxDegreeOfParallelism: 1);
                var parResults = index.Search(query, topK: 5, maxDegreeOfParallelism: 4);

                Assert.Equal(seqResults.Length, parResults.Length);
                for (int i = 0; i < seqResults.Length; i++)
                {
                    Assert.Equal(seqResults[i].Id, parResults[i].Id);
                    Assert.True(Math.Abs(seqResults[i].Score - parResults[i].Score) < 1e-5f);
                }
            }
        }
    }
}
