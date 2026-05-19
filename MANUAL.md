# Glacier.Vector Manual

Glacier.Vector is a high-performance, SIMD-accelerated vector search and index engine for .NET 10. It is designed to run close to the metal with zero-copy semantics, multi-threaded querying, and multiple backend storage layouts suitable for both pure in-memory and low-memory/out-of-core situations.

---

## 1. Storage Backends (`IVectorStorage`)

Glacier.Vector abstracts vector storage via the `IVectorStorage` interface. This allows you to choose the most efficient storage engine based on your hardware constraints.

### A. `InMemoryVectorStorage`
- **Purpose**: Pure in-memory caching.
- **Design**: Stores vectors in a chunked list layout (`List<float[]>`), avoiding the huge garbage collection pauses and large allocations associated with resizing giant flat arrays.
- **Use Case**: Medium-sized indices where maximum search speed is required and RAM is plentiful.

### B. `MmfVectorStorage` (Out-of-Core / OS Paged)
- **Purpose**: Disk-backed storage mapping directly into virtual memory.
- **Design**: Uses C# `MemoryMappedFile` and `MemoryMappedViewAccessor` to acquire an unmanaged pointer (`float*`) to the mapped memory. It allows querying vector databases larger than physical RAM by relying on the OS virtual memory paging system.
- **Use Case**: Datasets that exceed RAM, running on standard operating systems (Windows, Linux, macOS) with virtual memory.

### C. `PagedVectorStorage` (Out-of-Core / Paged Cache)
- **Purpose**: Disk-backed storage with strict memory constraints.
- **Design**: Uses a software-controlled Least Recently Used (LRU) page cache over direct file reads (`RandomAccess`). It enforces a strict memory ceiling (e.g. 256KB) via its constructor parameter `maxMemoryBytes`.
- **Page Alignment Optimization**: The page size is automatically aligned to be an exact multiple of the vector dimension size. This guarantees that no vector is ever split across two pages, eliminating boundary-split reading overhead and keeping access allocation-free and thread-safe.
- **Use Case**: Embedded systems, resource-constrained containers, or platforms where virtual memory mapping is limited or prohibited.

---

## 2. Distance Kernels & SIMD acceleration

Glacier.Vector uses optimized SIMD kernels for distance computations (e.g., `DistanceKernels.DotProduct`).
- Automatically leverages **AVX2 / AVX-512** instructions when running on supported processors.
- Aggressively optimized with `.NET` `MethodImplOptions.AggressiveOptimization` and `AggressiveInlining` annotations.
- Achieves peak floating-point calculation speeds by processing vectors in SIMD-register-sized chunks.

---

## 3. High-Performance Indexing (`VectorIndex`)

The `VectorIndex` class orchestrates ingestion and search:
- **Thread-Safe Ingestion**: Ingest vectors concurrently using `index.Add(vector, metadata)` with isolated lock-free write operations.
- **Task Parallelism**: For large datasets, query searches are split into chunks and processed in parallel across the ThreadPool using the **Parallel Block Pattern**. Small searches bypass thread pool overhead entirely.
- **Top-K Merge Phase**: Each thread isolates its results using a min-heap `.NET` `PriorityQueue` with a threshold gatekeeper. A final merge phase combines the thread-local results into the final top-K nearest neighbors.

### Search Example

```csharp
using Glacier.Vector.Storage;
using Glacier.Vector.Index;

// 1. Initialize Paged Vector Storage (limit page cache to 512KB)
using var storage = new PagedVectorStorage("vectors.db", dimensions: 1536, maxMemoryBytes: 512 * 1024);

// 2. Wrap in the Index Orchestrator
using var index = new VectorIndex(storage);

// 3. Add vectors
index.Add(new float[1536] { /*...*/ }, "Document A");
index.Add(new float[1536] { /*...*/ }, "Document B");

// 4. Perform Search
ReadOnlySpan<float> query = new float[1536] { /*...*/ };
SearchResult[] results = index.Search(query, topK: 5);

foreach (var hit in results)
{
    Console.WriteLine($"Matched: {hit.Metadata} (Score: {hit.Score})");
}
```

---

## 4. MCP Integration

Glacier.Vector includes built-in Model Context Protocol (MCP) server support, enabling LLMs or agents to interact with the database via standard tools:
- `add_vector`: Add an embedding and metadata string to the database.
- `search_vectors`: Search for the top-K closest vectors matching a query embedding.
- `ping`: Health check.
