# Glacier.Vector User Manual

## Contents

1. [Installation](#1-installation)
2. [Storage Backends (`IVectorStorage` & `BinaryVectorStorage`)](#2-storage-backends-ivectorstorage--binaryvectorstorage)
3. [Exact Search (`VectorIndex`)](#3-exact-search-vectorindex)
4. [HNSW Graph Search (`HnswVectorIndex`)](#4-hnsw-graph-search-hnswvectorindex)
5. [IVF-PQ Compressed Search (`IvfPqIndex`)](#5-ivf-pq-compressed-search-ivfpqindex)
6. [Distance Kernels & Quantized Dot Products](#6-distance-kernels--quantized-dot-products)
7. [Hadamard / RaBitQ 1-Bit Compression](#7-hadamard--rabitq-1-bit-compression)
8. [NVIDIA CUDA GPU Acceleration](#8-nvidia-cuda-gpu-acceleration)
9. [Building, Testing & Demo](#9-building--testing--demo)

---

## 1. Installation

```bash
dotnet add package Glacier.Vector
```

For stdio Model Context Protocol integration (`add_vector`, `search_vectors`, `ping`), see the [MCP Server Setup](mcp.md).

---

## 2. Storage Backends (`IVectorStorage` & `BinaryVectorStorage`)

| Type | Constructor | Backing | Best Fit |
| :--- | :--- | :--- | :--- |
| `InMemoryVectorStorage` | `(int dimensions, int vectorsPerChunk = 65536)` | Fixed-size managed `float[]` chunks | Lowest latency when dataset fits in RAM |
| `MmfVectorStorage` | `(string filePath, int dimensions)` | OS `MemoryMappedFile` (`float*`) + sidecar metadata | Datasets larger than RAM on OSes with virtual memory |
| `PagedVectorStorage` | `(string filePath, int dimensions, int maxMemoryBytes = 256 * 1024)` | Bounded LRU cache over `RandomAccess` disk I/O | Strict RAM caps (`maxMemoryBytes`) or containers without large mmap headroom |
| `BinaryVectorStorage` | `(int dimensions, int initialCapacity = 1024)` | Packed 1-bit `ulong[]` arrays (`32x` smaller than FP32) | Coarse Hamming candidate filtering |

`PagedVectorStorage` aligns page size to exact multiples of `dimensions` so no vector splits across two pages. `MmfVectorStorage` flushes and truncates the backing file cleanly on `Dispose()`.

```csharp
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

using var storage = new PagedVectorStorage("vectors.db", dimensions: 1536, maxMemoryBytes: 512 * 1024);
using var index = new VectorIndex(storage);

var vec = new float[1536];
vec[0] = 1.0f;
index.Add(vec, "Document A");

SearchResult[] hits = index.Search(vec, topK: 5);
```

---

## 3. Exact Search (`VectorIndex`)

`VectorIndex(IVectorStorage storage, bool leaveStorageOpen = false)` coordinates concurrent readers/writers (`ReaderWriterLockSlim`), splits large scans across thread-pool blocks, and merges thread-local top-K bounded heaps.

- Call `index.Add(ReadOnlySpan<float> vector, string metadata)` safely from concurrent threads.
- Pass `leaveStorageOpen: true` if multiple indexes share one `IVectorStorage` lifetime.
- Dispose `VectorIndex` when done to release locks (and underlying storage when `leaveStorageOpen` is `false`).

---

## 4. HNSW Graph Search (`HnswVectorIndex`)

`HnswVectorIndex` builds a multi-layer Hierarchical Navigable Small World graph over any `IVectorStorage`:

```csharp
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

using var storage = new InMemoryVectorStorage(dimensions: 128);
using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64, initialCapacity: 4096);

// Populate vectors via hnsw.Add(vector, metadata)
Span<SearchResult> destination = stackalloc SearchResult[10];
int count = hnsw.Search(new float[128], destination, efSearch: 64);
```

### Parameter Tuning
- `m` (default `16`): Max bi-directional links per node. Raising `m` improves recall at higher memory and build cost.
- `efConstruction` (default `64`): Candidate list size during graph insertion.
- `efSearch` (default `64`): Candidate list size during queries. Prefer `Search(ReadOnlySpan<float> query, Span<SearchResult> destination, int efSearch)` on hot paths: it reuses struct heaps and thread-local scratch buffers so query allocations stay at `0 B` per query.

---

## 5. IVF-PQ Compressed Search (`IvfPqIndex`)

`IvfPqIndex` partitions vectors across `centroidCount` coarse Voronoi clusters (routed by `L2DistanceSquared`) and encodes each vector into compact sub-vector codebook IDs (`SubVectorDim = 8` by default, yielding 32x compression vs FP32):

```csharp
using Glacier.Vector.Index;

const int dims = 128;
const int count = 2_000;
float[] vectors = new float[count * dims]; // Populate with dataset vectors

using var ivf = new IvfPqIndex(dimensions: dims, centroidCount: 32, subVectorCount: null);
ivf.Train(vectors, vectorCount: count, iterations: 5);

for (int i = 0; i < count; i++)
{
    ivf.Add(vectors.AsSpan(i * dims, dims), $"doc_{i}");
}

SearchResult[] hits = ivf.Search(new float[dims], topK: 10, nprobe: 8);
```

### Parameter Tuning
- `centroidCount` (default `256`): Number of coarse clusters; must be <= training vector count in typical workloads.
- `subVectorCount`: Defaults to `dimensions / 8` when divisible by `8` (`dimensions / 4` or `dimensions` otherwise). `dimensions % subVectorCount == 0` is required.
- `nprobe` (default `16`): Number of nearest coarse centroids inspected per query using precomputed Asymmetric Distance Computation (ADC) lookup tables.

---

## 6. Distance Kernels & Quantized Dot Products

All static helpers in `Glacier.Vector.Compute.DistanceKernels` automatically select AVX-512, AVX2/FMA (`vpmaddubsw` + `vpmaddwd`), `Vector128`, or portable 32-bit SWAR paths:

- **FP32:** `DotProduct`, `CosineDistance`, `CosineDistanceUnnormalized`, `L2Distance`, `L2DistanceSquared`
- **Binary (64-bit / 32-bit packed):** `PackBinaryVector`, `HammingDistance`
- **Signed INT8:** `DotInt8`, `SumInt8`, `DotInt8Quantized` (supports scaled `(scaleA, scaleB)` inputs), `DotFloatInt8`
- **Packed UINT4 (2 nibbles/byte):** `PackUInt4`, `DotUInt4Packed(aPacked, bPacked, dimensions)`, `SumUInt4Packed`, `DotUInt4PackedQuantized`

---

## 7. Hadamard / RaBitQ 1-Bit Compression

`HadamardProjection` orthogonalizes coordinate outliers via Fast Walsh-Hadamard Transform before 1-bit binarization:

```csharp
using Glacier.Vector.Compute;

const int dim = 1536;
int padded = HadamardProjection.GetPaddedDimension(dim);       // Next power of 2 (2048)
int words = HadamardProjection.GetPackedWordCount(dim);        // Packed 64-bit word count
float[] signs = new float[padded];
HadamardProjection.GenerateSigns(signs, seed: 42UL);

float[] vec = new float[dim];
float[] query = new float[dim];
ulong[] dbBits = new ulong[words];
ulong[] qBits = new ulong[words];

BinaryRotationMetadata dbMeta = HadamardProjection.RotateAndBinarize(vec, signs, dbBits);
HadamardProjection.RotateAndBinarize(query, signs, qBits);

float estCosine = HadamardProjection.EstimateCosineFromHamming(qBits, dbBits, dim);
```

Additional helpers include `ApplyFwhtInPlace`, `ApplyRandomRotation`, `ApplyRandomRotationInverse`, `EstimateAsymmetricDotProduct`, and `EstimateAsymmetricCosine`.

---

## 8. NVIDIA CUDA GPU Acceleration

`GpuVectorAccelerator` binds directly to `nvcuda.dll` / `libcuda.so.1`:

- **`ScanChunkGpu`:** Scores a single query against a contiguous FP32 chunk, automatically picking Split-K warp reductions for `dimensions >= 128` and keying VRAM cache entries by `(storageId, chunkIndex, storageVersion)` (`storageId = 0` disables caching).
- **`ScanChunkInt8Gpu`:** Scores quantized INT8 chunks via `dp4a.s32.s32` (or CPU SIMD INT8 when `GpuTarget.Cpu` / no NVIDIA GPU). Uncached 100k-vector INT8 scans pay host-to-device copy time per call, making CPU SIMD INT8 faster for single uncached passes.
- **`BatchScanGpu`:** Computes `Database * Queries^T` score matrices across batched queries.
- **`ReleaseGpuResources()`:** Releases cached VRAM allocations and CUDA execution slots explicitly (also registered on `AppDomain.CurrentDomain.ProcessExit`).

---

## 9. Building, Testing & Demo

```bash
dotnet build -c Release
dotnet test -c Release --no-build
dotnet run --project samples/Glacier.Vector.Demo/Glacier.Vector.Demo.csproj -c Release
```

*(Note: if a live MCP host process locks `src/Glacier.Vector.Host/bin/Release/net10.0/`, run `dotnet test -c Debug`.)*
