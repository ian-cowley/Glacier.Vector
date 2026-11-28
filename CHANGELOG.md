# Changelog

## Unreleased (Documentation & Exception Hygiene)

- Replaced all remaining bare `catch {}` blocks with `TryCudaCleanup`, narrowed cleanup catches, and `Trace.TraceWarning` diagnostics across `GpuVectorAccelerator`, `VectorStorage`, and `McpServer`.
- Extended `Glacier.Vector.Benchmarks` to cover `MmfVectorStorage`, `PagedVectorStorage`, `BinaryVectorStorage`, and `IvfPqIndex`.
- Split detailed architecture, MCP configuration, usage, and benchmark documentation into `docs/` and simplified `README.md`.

## v1.3.0

- **Hadamard & RaBitQ 1-Bit Compression (`HadamardProjection`):** AVX-512F / AVX2 normalized FWHT random rotation (`ApplyFwhtInPlace`, `ApplyRandomRotation`, `RotateAndBinarize`) packing 32x compressed 1-bit codes with asymmetric inner-product/cosine estimation (`EstimateCosineFromHamming`, `EstimateAsymmetricCosine`).
- **INT8 & Packed UINT4 Kernels (`DistanceKernels`):** AVX-512BW / AVX2 (`vpmaddubsw` + `vpmaddwd`) and 32-bit SWAR kernels (`DotInt8`, `SumInt8`, `DotInt8Quantized`, `DotFloatInt8`, `PackUInt4`, `DotUInt4Packed`, `SumUInt4Packed`, `DotUInt4PackedQuantized`).
- **GPU Split-K FP32 & Zero-`I2F` `dp4a` INT8 Scans (`GpuVectorAccelerator`):** Warp-cooperative Split-K FP32 (`dimensions >= 128`) and `ScanChunkInt8Gpu` (`dp4a.s32.s32`) with multi-threaded SIMD CPU fallback.

## v1.1.0

- **Native AOT & Strict Build Hygiene:** Enabled `<IsAotCompatible>true</IsAotCompatible>` and `<TreatWarningsAsErrors>true</TreatWarningsAsErrors>`, replacing reflection JSON serialization in `McpServer` with `Utf8JsonWriter` / `JsonDocument` over stdio.
- **L2 Voronoi Routing in `IvfPqIndex`:** Routed coarse centroid probes via minimum `L2DistanceSquared` matching K-Means Voronoi assignment.
- **Versioned VRAM Cache & `stackalloc` CUDA Launches:** Keyed GPU chunk caching by `(StorageId, ChunkIndex, Version)` and removed `GCHandle` boxing on kernel launches.
- **Storage Bounds & Concurrency Hardening:** Added `ReaderWriterLockSlim` coordination in `VectorIndex` (with `leaveStorageOpen` support), bounds checks across all `IVectorStorage` backends, and 64-bit LRU counters in `PagedVectorStorage`.

## v1.0.8

- **Zero-Allocation HNSW Traversal:** Replaced per-query `PriorityQueue` heap allocations in `HnswVectorIndex.SearchLayer` with bounded struct heaps (`ValueMinHeap`, `ValueMaxHeap`) and `[ThreadStatic]` scratch buffers (`HnswSearchScratch`).
- **Zero-Allocation Search API:** Added `Search(ReadOnlySpan<float> query, Span<SearchResult> destination, int efSearch = 64)` overload.
- **`Glacier.Vector.Benchmarks` Suite:** Added BenchmarkDotNet and Stopwatch suite profiling SIMD kernels and HNSW throughput.

## v1.0.7

- Removed heap allocations from hot HNSW and IVF-PQ traversal loops.
- Used `Volatile.Write` atomic pointer swap in `MmfVectorStorage` for concurrent segment remapping.
