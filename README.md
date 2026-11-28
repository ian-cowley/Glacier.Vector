![Glacier.Vector Banner](assets/banner.jpg)

# Glacier.Vector

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![.NET 10](https://img.shields.io/badge/.NET-10.0-purple.svg)](https://dotnet.microsoft.com/)
[![Native AOT](https://img.shields.io/badge/Native%20AOT-Ready-brightgreen.svg)](https://learn.microsoft.com/dotnet/core/deploying/native-aot/)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.Vector.svg)](https://www.nuget.org/packages/Glacier.Vector/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Glacier.Vector.svg)](https://www.nuget.org/packages/Glacier.Vector/)

`Glacier.Vector` is an embedded vector search engine and similarity toolkit for C# and .NET 10.

If you are building retrieval-augmented generation (RAG), semantic search, clustering, or local agent memory inside a .NET application, you often have to run an external Python/C++ vector service or marshal large float arrays across HTTP. `Glacier.Vector` keeps indexing, storage, quantization, and search in-process inside a single Native AOT-compatible .NET library—complete with exact scans, HNSW and IVF-PQ indexes, memory-mapped storage for datasets larger than RAM, and an optional stdio MCP server.

Read the background write-up: [I built a zero-dependency C# vector database that saturates DDR5 RAM bandwidth](https://dev.to/iancowley/i-built-a-zero-dependency-c-vector-database-that-saturates-ddr5-ram-bandwidth-5f9a).

## Install

```bash
dotnet add package Glacier.Vector
```

## Quick Start

### Add and Search Embeddings In Memory

```csharp
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

using var storage = new InMemoryVectorStorage(dimensions: 1536);
using var index = new VectorIndex(storage);

float[] docVector = new float[1536];
docVector[0] = 1.0f;
index.Add(docVector, metadata: "Greeting message");

float[] query = new float[1536];
query[0] = 1.0f;
SearchResult[] hits = index.Search(query, topK: 5);

foreach (SearchResult hit in hits)
{
    Console.WriteLine($"ID={hit.Id}, Score={hit.Score:F4}, Metadata={hit.Metadata}");
}
```

### Zero-Allocation HNSW Search & Batch GPU Scan

```csharp
using Glacier.Vector.Compute;
using Glacier.Vector.Core;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

using var storage = new InMemoryVectorStorage(dimensions: 128);
using var hnsw = new HnswVectorIndex(storage, m: 16, efConstruction: 64);

Span<SearchResult> topK = stackalloc SearchResult[10];
int found = hnsw.Search(new float[128], topK, efSearch: 64);

// Optional 2D batch scan on NVIDIA CUDA (falls back automatically when unavailable)
float[] db = new float[50_000 * 128];
float[] queries = new float[32 * 128];
float[] scores = new float[50_000 * 32];
GpuVectorAccelerator.BatchScanGpu(db, queries, scores, vectorCount: 50_000, dimensions: 128, batchSize: 32, target: GpuTarget.Auto);
```

## What You Can Do

- **Pick exact or approximate search:** Run parallel exact cosine/dot/L2 search (`VectorIndex`), zero-allocation HNSW graph search (`HnswVectorIndex`), or 32x-compressed IVF-PQ search (`IvfPqIndex`).
- **Fit your memory budget:** Store vectors in RAM (`InMemoryVectorStorage`), on disk via OS virtual memory (`MmfVectorStorage`), behind a strict byte-capped LRU cache (`PagedVectorStorage`), or as 1-bit packed binary codes (`BinaryVectorStorage`).
- **Compress embeddings without losing speed:** Compute INT8, packed 4-bit UINT4, and Hadamard-rotated 1-bit RaBitQ similarities using vectorized AVX-512 / AVX2 kernels or portable fallbacks.
- **Connect AI coding agents directly:** Run `Glacier.Vector.Host` over stdio JSON-RPC to expose `add_vector`, `search_vectors`, and `ping` via the Model Context Protocol.

## Platform & Compatibility

- Runs on Windows, Linux, and macOS (.NET 10, Native AOT compatible).
- Optional GPU scanning uses NVIDIA CUDA driver bindings (`nvcuda.dll` / `libcuda.so.1`) and falls back automatically to multi-core CPU SIMD when unavailable.
- Caching FP32 database chunks in VRAM beats CPU FP32 scans, whereas one-shot uncached INT8 scans pay host-to-device PCIe copy overhead that makes multi-core CPU SIMD INT8 faster.

## Benchmarks

Measured in Release on **AMD Ryzen AI 9 HX 370 (12C/24T Zen 5, AVX-512) + NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), 31 GB RAM, Windows 11, .NET 10.0.15 (.NET SDK 11.0.100)**:

| Workload | Scope | Measured Result | Notes |
| :--- | :--- | ---: | :--- |
| `DistanceKernels` (`Dim=1536`, FP32 / INT8 / UINT4) | Single pair | **63.7 ns** / **24.9 ns** / **4.9 ns** | 15.7 / 40.1 / **206.1 MOps/s** |
| `HadamardProjection.RotateAndBinarize` | `Dim=1536` (`pad 2048`) | **0.97 us** | 1,028.8 kVec/s |
| Chunk Scan (`Dim=128`, 100k vecs) | CPU FP32 / CPU INT8 / GPU SplitK (cached) | **1.02 ms** / **0.35 ms** / **0.39 ms** | GPU INT8 `dp4a` incl. H2D: **1.55 ms** |
| Chunk Scan (`Dim=1536`, 100k vecs) | CPU FP32 / CPU INT8 / GPU SplitK (cached) | **7.92 ms** / **2.27 ms** / **2.96 ms** | GPU INT8 `dp4a` incl. H2D: **17.77 ms** |
| `HnswVectorIndex.Search` (Alloc vs Zero-Alloc) | 10k queries (`2k x 128D`) | **35.5 us** (184 B/q) vs **23.8 us** (**0 B/q**) | **28,136** vs **42,015 QPS** |
| `IvfPqIndex.Search` (`K=32`, `nprobe=8`) | `2k x 128D` | **23.8 us/q** | **42,069 QPS** |

Reproduce locally:

```bash
dotnet run --project benchmarks/Glacier.Vector.Benchmarks/Glacier.Vector.Benchmarks.csproj -c Release
```

## Documentation

- [User Manual](docs/manual.md) — storage choices, exact/HNSW/IVF-PQ tuning, distance kernels, Hadamard 1-bit compression, and GPU scans
- [Architecture](docs/architecture.md) — chunk layout, SIMD intrinsics dispatch, zero-alloc HNSW heaps, and CUDA VRAM cache design
- [Benchmarks](docs/benchmarks.md) — complete kernel, scan, HNSW, IVF-PQ, and storage measurements
- [MCP Server Setup](docs/mcp.md) — building and configuring `Glacier.Vector.Host` for MCP clients
- [Changelog](CHANGELOG.md) — release notes

## License & Credits

Licensed under the [MIT License](LICENSE).

Developed by Ian Cowley, Antigravity (Google DeepMind) and Claude.
