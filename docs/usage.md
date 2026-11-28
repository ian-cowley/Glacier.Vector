# Glacier.Vector Usage Guide

This guide covers choosing a storage backend (`IVectorStorage`), running approximate nearest-neighbor indexes (`HnswVectorIndex` and `IvfPqIndex`), using quantized distance kernels and Hadamard 1-bit compression, and building the solution.

## Storage Backends

All exact and graph indexes operate against `IVectorStorage` (or `BinaryVectorStorage` for 1-bit Hamming filtering):

| Type | Backing | Thread Safety | When to Use |
| :--- | :--- | :--- | :--- |
| `InMemoryVectorStorage` | Chunked managed `float[]` arenas (default `65,536` vectors/chunk) | Concurrent chunk reads, locked append | Maximum query speed when vectors fit in RAM |
| `MmfVectorStorage` | OS `MemoryMappedFile` (`float*`) with atomic `Version` tracking | `ReaderWriterLockSlim` | Datasets larger than RAM when OS virtual memory is available |
| `PagedVectorStorage` | Bounded LRU page cache over `RandomAccess` file reads | Internal lock per cache lookup | Tight RAM budgets (`maxMemoryBytes`, e.g. `256 * 1024`) or restricted virtual memory |
| `BinaryVectorStorage` | 1-bit `ulong[]` packed vectors via `DistanceKernels.PackBinaryVector` | Lock-protected append, lock-free reads | Fast Hamming candidate pre-filtering (`32x` compression vs FP32) |

### Out-of-Core Search With `PagedVectorStorage` or `MmfVectorStorage`

```csharp
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

using var storage = new PagedVectorStorage("vectors.db", dimensions: 1536, maxMemoryBytes: 512 * 1024);
using var index = new VectorIndex(storage);

var vecA = new float[1536];
vecA[0] = 1.0f;
index.Add(vecA, "Document A");

var query = new float[1536];
query[0] = 1.0f;
SearchResult[] results = index.Search(query, topK: 5);

foreach (SearchResult hit in results)
{
    Console.WriteLine($"{hit.Metadata}: {hit.Score:F4}");
}
```

## Approximate Nearest Neighbors (`HnswVectorIndex` & `IvfPqIndex`)

### HNSW With Zero-Allocation Search

```csharp
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

using var storage = new InMemoryVectorStorage(dimensions: 128);
using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64);

// Populate vectors via hnsw.Add(...)
Span<SearchResult> results = stackalloc SearchResult[10];
int count = hnsw.Search(new float[128], results, efSearch: 64);
```

### IVF-PQ (32x Product Quantization)

```csharp
using Glacier.Vector.Index;

const int dims = 128;
const int numVectors = 2_000;
float[] trainingVectors = new float[numVectors * dims]; // Fill with representative data

using var ivf = new IvfPqIndex(dimensions: dims, centroidCount: 32);
ivf.Train(trainingVectors, vectorCount: numVectors, iterations: 5);

for (int i = 0; i < numVectors; i++)
{
    ivf.Add(trainingVectors.AsSpan(i * dims, dims), $"doc_{i}");
}

SearchResult[] hits = ivf.Search(new float[dims], topK: 10, nprobe: 8);
```

## Quantized Kernels & Hadamard 1-Bit Compression

`DistanceKernels` and `HadamardProjection` expose low-level kernels for INT8, packed UINT4 (two nibbles per byte), and RaBitQ-style 1-bit binary projection:

```csharp
using Glacier.Vector.Compute;

const int dim = 1536;
int padded = HadamardProjection.GetPaddedDimension(dim); // Next power of 2 (2048)
float[] signs = new float[padded];
HadamardProjection.GenerateSigns(signs, seed: 42UL);

float[] vector = new float[dim];
float[] query = new float[dim];
ulong[] packed = new ulong[HadamardProjection.GetPackedWordCount(dim)];

float norm = HadamardProjection.RotateAndBinarize(vector, signs, packed);
```

For INT8 GPU vs. CPU chunk scans, use `GpuVectorAccelerator.ScanChunkGpu` (cached FP32 VRAM chunks via `StorageId` and `Version`) or `GpuVectorAccelerator.ScanChunkInt8Gpu` (single-shot `dp4a` or CPU AVX-512 INT8).

## Build, Test & Demo

```bash
dotnet build -c Release
dotnet test -c Release --no-build
dotnet run --project samples/Glacier.Vector.Demo/Glacier.Vector.Demo.csproj -c Release
```

Note: if a live MCP host process (`Glacier.Vector.Host`) holds a file lock on `src/Glacier.Vector.Host/bin/Release/net10.0/`, run `dotnet test -c Debug`.
