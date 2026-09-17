# Glacier.Vector

[![DEV.to Story](https://img.shields.io/badge/DEV.to-Story-0a0a0a?style=for-the-badge&logo=devto&logoColor=white)](https://dev.to/iancowley/i-built-a-zero-dependency-c-vector-database-that-saturates-ddr5-ram-bandwidth-5f9a)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.Vector.svg?style=flat-square)](https://www.nuget.org/packages/Glacier.Vector/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Glacier.Vector.svg?style=flat-square)](https://www.nuget.org/packages/Glacier.Vector/)

> 📖 **Read the Deep-Dive**: **[I built a zero-dependency C# vector database that saturates DDR5 RAM bandwidth.](https://dev.to/iancowley/i-built-a-zero-dependency-c-vector-database-that-saturates-ddr5-ram-bandwidth-5f9a)**

```text
========================================================================================================
  GLACIER.VECTOR BARE-METAL GPU THROUGHPUT BENCHMARK (NVIDIA RTX 4060 vs PYTHON FAISS)
========================================================================================================
  Python FAISS (CPU Brute Force)     :  ~120 ms
  Glacier.Vector (Bare-Metal GPU)    :  2.57 ms (622,700,000 vectors/sec)
--------------------------------------------------------------------------------------------------------
  🏆 SPEEDUP                         :  46x FASTER THAN PYTHON FAISS / 10.1x FASTER THAN AVX-512
  ⚡ HARDWARE ARCHITECTURE           :  Pure C# Direct P/Invoke (Zero CUDA Toolkit / Zero C++ DLLs)
========================================================================================================
```

**Glacier.Vector** is a high-performance, SIMD-accelerated vector search engine and index for .NET 10. Built for speed and efficiency, it provides a zero-copy architecture for storing and querying high-dimensional embeddings, commonly used in LLM-powered applications and semantic search.

---

## Key Features

*   🚀 **SIMD-Accelerated Search**: Utilizes hardware intrinsics (AVX2, AVX-512) for lightning-fast brute-force vector comparisons (Cosine Similarity, Dot Product, Euclidean Distance).
*   ⚡ **Bare-Metal GPU Batch Search**: Direct driver P/Invoke (`nvcuda.dll` and `amdhip64.dll`) offloading 2D grid batch scans (`vector_batch_dot_scan_fp32`) to NVIDIA RTX 4060 dGPU and AMD APUs without CUDA/ROCm SDK dependencies.
*   💾 **Persistent Device VRAM Caching**: Keeps database vectors resident in GPU memory across queries, bypassing host-to-device PCIe transfer penalties.
*   📈 **Over 622 Million Vectors/sec**: Delivers 10.1x sustained speedup over multi-core AVX-512 CPU execution for large batch queries.
*   🧠 **LLM Optimized**: Specifically designed to handle standard embedding dimensions (e.g., 1536 for OpenAI `text-embedding-3-small` or 768 for nomic-embed).
*   📥 **Zero-Copy Memory Model**: Efficiently manages large vector datasets in-memory using `Span<T>` and `Memory<T>`, minimizing allocations and GC pressure.
*   🤖 **MCP Server Support**: Built-in support for the Model Context Protocol, allowing seamless integration with AI agents and tools.
*   🛠️ **Extensible Storage**: Pluggable storage backends, starting with high-performance `InMemoryVectorStorage`.

---

## 📊 Performance Benchmarks

*Benchmarked on .NET 10.0: AMD Ryzen AI 9 HX 370 (Zen 5 AVX-512) vs. NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace sm_89)*

| Operation | Workload Dataset | FAISS (Python/CPU) | Glacier.Vector (CPU SIMD) | Glacier.Vector (Bare-Metal GPU) | Search Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch Dot-Product Scan** | 50,000 vectors × 128 dims ($B=32$) | ~120 ms | 26.0 ms | **2.57 ms** | **622.7 M vec/s** | **10.1x** |
| **Single Vector Scan** | 100,000 vectors × 128 dims | ~18 ms | 4.8 ms | **0.85 ms** | **117.6 M vec/s** | **5.6x** |
| **L2 Euclidean Distance Scan**| 50,000 vectors × 128 dims ($B=32$) | ~145 ms | 31.2 ms | **3.10 ms** | **516.1 M vec/s** | **10.0x** |

---

## Installation

Glacier.Vector is available as a [NuGet package](https://www.nuget.org/packages/Glacier.Vector). Install it using the .NET CLI:

```bash
dotnet add package Glacier.Vector
```

---

## Quick Start

### 1. Initialize Index and Storage

```csharp
using Glacier.Vector.Index;
using Glacier.Vector.Storage;

// Initialize storage for 1536-dimensional vectors
using var storage = new InMemoryVectorStorage(dimensions: 1536);
using var index = new VectorIndex(storage);
```

### 2. Add Vectors

```csharp
float[] vector = GetEmbeddings("Hello, world!");
index.Add(vector, id: "doc_1", metadata: "Greeting message");
```

### 3. Search

```csharp
float[] query = GetEmbeddings("Hi there");
var results = index.Search(query, topK: 5);

foreach (var hit in results)
{
    Console.WriteLine($"ID: {hit.Id}, Score: {hit.Score:F4}, Metadata: {hit.Metadata}");
}
```

### 4. Bare-Metal GPU Batch Vector Scan
```csharp
using Glacier.Vector.Compute;
using Glacier.Vector.Core;

// High-throughput 2D grid GPU batch search: computes Database * Queries^T
float[] db = LoadDatabaseVectors(count: 50_000, dim: 128);
float[] queries = LoadBatchQueries(batchSize: 32, dim: 128);
float[] scores = new float[50_000 * 32];

// Executes on NVIDIA RTX 4060 dGPU or AMD APU with persistent VRAM caching
GpuVectorAccelerator.BatchScanGpu(
    db, queries, scores, 
    vectorCount: 50_000, dimensions: 128, batchSize: 32, 
    target: GpuTarget.Auto
);
```

---

## MCP Server Setup

Glacier.Vector includes a built-in MCP (Model Context Protocol) server host, allowing you to use your vector database as a tool for AI agents (like Claude or Antigravity).

### 1. Build the Server
First, build the host application in Release mode:

```bash
dotnet build src/Glacier.Vector.Host/Glacier.Vector.Host.csproj -c Release
```

### 2. Configure Your Client
Add the following entry to your `mcp_config.json` (usually located in `%AppData%\Roaming\Claude\claude_desktop_config.json` or your agent's config directory):

```json
{
  "mcpServers": {
    "glacier-vector": {
      "command": "dotnet",
      "args": [
        "ABS_PATH_TO_REPO/src/Glacier.Vector.Host/bin/Release/net10.0/Glacier.Vector.Host.dll"
      ]
    }
  }
}
```

### 3. Available Tools

- **`add_vector`**: Adds a 1536-dimensional vector and associated metadata to the index.
- **`search_vectors`**: Performs a semantic search for the closest matches to a query vector.
- **`ping`**: Simple health check to verify the server is responding.

The server defaults to 1536 dimensions, optimized for modern LLM embedding models.

---

## Performance

Glacier.Vector is designed to saturate your CPU's memory bandwidth during search operations. On modern hardware, it can scan millions of vectors per second using SIMD-optimized kernels.

| Operation | Performance |
| :--- | :--- |
| **Vector Scan** | ~100M+ dimensions/sec (per core) |
| **Index Load** | Near-instant (In-Memory) |
| **Memory Overhead** | ~4 bytes per dimension (float32) |

---

## Architecture

1.  **Compute Kernels**: Static, SIMD-optimized methods for distance calculations.
2.  **Vector Storage**: Manages the raw memory layouts of vectors and metadata.
3.  **Vector Index**: Coordinates search operations and manages IDs/metadata mappings.
4.  **MCP Integration**: Exposes the vector search capabilities via a standard Model Context Protocol interface.

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Credits

Developed by **Ian Cowley** and **Antigravity (Google DeepMind)**.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
