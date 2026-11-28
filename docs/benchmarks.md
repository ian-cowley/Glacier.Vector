# Glacier.Vector Benchmarks

All measurements below were captured directly from `Glacier.Vector.Benchmarks` in Release mode without external or approximated competitor numbers.

## Test Environment

- **CPU:** AMD Ryzen AI 9 HX 370 w/ Radeon 890M (12 physical cores / 24 logical threads, Zen 5, AVX-512)
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (8 GB GDDR6)
- **Memory & OS:** 31 GB RAM, Microsoft Windows NT `10.0.26300.0`
- **Runtime:** `.NET 10.0.15` (`.NET SDK 11.0.100`)
- **Command:**

```bash
dotnet run --project benchmarks/Glacier.Vector.Benchmarks/Glacier.Vector.Benchmarks.csproj -c Release
```

## 1. SIMD FP32 Distance Kernels (`1,000,000` iterations)

| Dimension | `DotProduct` | `CosineDistance` | `L2Distance` | `L2DistanceSquared` |
| ---: | ---: | ---: | ---: | ---: |
| 64 | 244.5 MOps/s | 244.2 MOps/s | 227.0 MOps/s | 234.2 MOps/s |
| 128 | 175.4 MOps/s | 181.1 MOps/s | 174.5 MOps/s | 172.8 MOps/s |
| 768 | 31.1 MOps/s | 32.4 MOps/s | 31.3 MOps/s | 32.4 MOps/s |
| 1536 | 15.6 MOps/s | 13.9 MOps/s | 14.4 MOps/s | 16.0 MOps/s |

## 2. Quantized Dot Products: FP32 vs. INT8 vs. Packed UINT4 (`1,000,000` iterations)

| Dimension | `DotProduct` (FP32) | `DotInt8` | `DotUInt4Packed` |
| ---: | ---: | ---: | ---: |
| 64 | 6.4 ns (155.8 MOps/s) | 2.5 ns (401.3 MOps/s) | 2.3 ns (432.2 MOps/s) |
| 128 | 7.8 ns (128.3 MOps/s) | 2.1 ns (477.0 MOps/s) | 1.8 ns (564.7 MOps/s) |
| 768 | 36.3 ns (27.6 MOps/s) | 15.7 ns (63.7 MOps/s) | 2.9 ns (350.5 MOps/s) |
| 1536 | 63.7 ns (15.7 MOps/s) | 24.9 ns (40.1 MOps/s) | 4.9 ns (206.1 MOps/s) |

## 3. Hadamard FWHT & RaBitQ 1-Bit Rotation (`50,000` iterations)

| Dimension (Padded) | `ApplyFwhtInPlace` | `RotateAndBinarize` |
| ---: | ---: | ---: |
| 128 (128) | 0.03 us (29,238.1 kVec/s) | 0.07 us (13,435.1 kVec/s) |
| 768 (1024) | 0.28 us (3,631.9 kVec/s) | 0.61 us (1,647.3 kVec/s) |
| 1536 (2048) | 0.58 us (1,725.2 kVec/s) | 0.97 us (1,028.8 kVec/s) |
| 4096 (4096) | 1.24 us (806.9 kVec/s) | 2.02 us (494.6 kVec/s) |

## 4. `100,000`-Vector Chunk Scans: Multi-Core CPU SIMD vs. NVIDIA RTX 4060

| Dimension | CPU SIMD FP32 | CPU SIMD INT8 | NVIDIA GPU SplitK-FP32 (Cached VRAM) | NVIDIA GPU `dp4a` INT8 (Uncached incl. H2D) |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 1.02 ms (98.1 M vec/s) | **0.35 ms** (285.9 M vec/s) | **0.39 ms** (256.0 M vec/s) | 1.55 ms (64.6 M vec/s) |
| 1536 | 7.92 ms (12.6 M vec/s) | **2.27 ms** (44.1 M vec/s) | **2.96 ms** (33.8 M vec/s) | 17.77 ms (5.6 M vec/s) |

Because `ScanChunkInt8Gpu` transfers the INT8 chunk over PCIe on every uncached call (`12.8 MB` at `128D`, `153.6 MB` at `1536D`), multi-core CPU SIMD INT8 is faster on uncached scans. Cached VRAM `SplitK-FP32` avoids repeated H2D copies and beats CPU FP32 by `2.62x` at `128D` and `2.68x` at `1536D`.

## 5. HNSW Search & Allocations (`10,000` queries, `2,000 x 128D`, `topK=10, efSearch=64`)

| Mode | Throughput | Latency | Allocated per Query |
| :--- | ---: | ---: | ---: |
| `Search(query, 10, efSearch: 64)` | 28,136 QPS | 35.5 us | 184.0 B (`1,840,000 B` total) |
| `Search(query, Span<SearchResult>, efSearch: 64)` | **42,015 QPS** | **23.8 us** | **0.0 B** (`0 B` total) |

## 6. Storage Backends & IVF-PQ Index (`2,000 x 128D`, `topK=10`)

| Engine | Setup / Ingest | Search / Scan Latency | Throughput |
| :--- | ---: | ---: | ---: |
| `InMemoryVectorStorage` + `VectorIndex` | 0.28 ms ingest | 65.1 us/q | 15,356 QPS |
| `MmfVectorStorage` + `VectorIndex` | 1.14 ms ingest | 241.8 us/q | 4,136 QPS |
| `PagedVectorStorage` (`256 KB` LRU) + `VectorIndex` | 5.80 ms ingest | 1,414.7 us/q | 707 QPS |
| `BinaryVectorStorage` (`HammingDistance` scan) | 0.08 ms ingest | 31.7 us/q | 31,497 QPS |
| `IvfPqIndex` (`K=32`, `nprobe=8`) | 63.7 ms train / 199.3 ms add | 23.8 us/q | 42,069 QPS |
