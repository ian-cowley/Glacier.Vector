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
    }
}
