# Glacier.Vector Architecture

`Glacier.Vector` compiles under `.NET 10.0` with `<IsAotCompatible>true</IsAotCompatible>`, `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>`, and unsafe pointer blocks for SIMD intrinsics, memory-mapped I/O, and direct CUDA driver P/Invoke (`nvcuda.dll` / `libcuda.so.1`).

## Subsystem Overview

1. **Compute Kernels (`DistanceKernels`, `HadamardProjection`, `GpuVectorAccelerator`):**
   - **CPU SIMD:** Dispatches `Vector512`, `Vector256`/FMA, `Vector128`, or scalar/SWAR loops at JIT/AOT runtime using `IsHardwareAccelerated` checks.
   - **Quantized Integer Paths:** `DotInt8` / `DotInt8Quantized` / `DotUInt4Packed` use `AvxVnni` / `Avx512BW` / `Avx2` (`vpmaddubsw` + `vpmaddwd`) on x86-64 and portable 32-bit SWAR multiplication elsewhere.
   - **Hadamard / RaBitQ Compression:** `HadamardProjection` pads vectors to the next power of two, applies pseudo-random diagonal sign flips (`GenerateSigns`), runs an in-place Fast Walsh-Hadamard Transform (`ApplyFwhtInPlace`), and extracts sign bits into 64-bit `ulong[]` words.
   - **GPU Compute (`GpuVectorAccelerator`):** Loads PTX kernels directly through `nvcuda.dll` / `libcuda.so.1` with stack-allocated parameter pointers (`stackalloc void*[]`), preallocated `CudaExecutionSlot` pinned host buffers (`MemFreeHost`), and deterministic `ProcessExit` + `TryCudaCleanup` teardown. When no NVIDIA driver is present, GPU entry points return `false` or execute the multi-threaded CPU fallback when `GpuTarget.Auto` is passed.

2. **Storage Backends (`IVectorStorage`, `BinaryVectorStorage`):**
   - `InMemoryVectorStorage` allocates fixed-size vector chunks so growth does not trigger large-object array copies, exposing contiguous chunk spans for batch GPU/SIMD scans and incrementing `Version` on writes.
   - `MmfVectorStorage` maps a persistent binary file into virtual address space (`float*`), grows via `Remap` with rollback on partial failure, maintains a crash-recovery sidecar metadata file, and truncates to exact size on shutdown.
   - `PagedVectorStorage` aligns page size to whole vector boundaries so a vector never straddles two LRU cache pages.

3. **Indexing (`VectorIndex`, `HnswVectorIndex`, `IvfPqIndex`):**
   - `VectorIndex` coordinates `ReaderWriterLockSlim` synchronization and partitions large exact scans across thread-pool workers (`Parallel Block Pattern`) before merging thread-local bounded heaps.
   - `HnswVectorIndex` replaces per-query `PriorityQueue` heap allocations in `SearchLayer` with bounded struct heaps (`ValueMinHeap`, `ValueMaxHeap`) and `[ThreadStatic]` scratch buffers (`HnswSearchScratch`), delivering 0 bytes allocated per query on `Search(ReadOnlySpan<float>, Span<SearchResult>, int)`.
   - `IvfPqIndex` partitions space into `CentroidCount` Voronoi clusters via K-Means `L2DistanceSquared` routing, compresses residuals into sub-vector PQ codes (`SubVectorDim = 8` by default, 32x smaller than FP32), and scores queries with precomputed Asymmetric Distance Computation (ADC) lookup tables.

4. **MCP Server Host (`Glacier.Vector.Host`):**
   - `McpServer` implements JSON-RPC 2.0 over stdio using `Utf8JsonWriter` and `JsonDocument` with zero reflection-based JSON serialization, protected by `ReaderWriterLockSlim` state access and `GLACIER_VECTOR_MCP_LOG` diagnostic gating.
