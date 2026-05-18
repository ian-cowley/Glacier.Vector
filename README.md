# Glacier.Vector

[![DEV.to Story](https://img.shields.io/badge/DEV.to-Story-0a0a0a?style=for-the-badge&logo=devto&logoColor=white)](https://dev.to/iancowley/i-built-a-zero-dependency-c-vector-database-that-saturates-ddr5-ram-bandwidth-5f9a)
[![NuGet Version](https://img.shields.io/nuget/v/Glacier.Vector.svg?style=flat-square)](https://www.nuget.org/packages/Glacier.Vector/)
[![NuGet Downloads](https://img.shields.io/nuget/dt/Glacier.Vector.svg?style=flat-square)](https://www.nuget.org/packages/Glacier.Vector/)

> 📖 **Read the Deep-Dive**: **[I built a zero-dependency C# vector database that saturates DDR5 RAM bandwidth.](https://dev.to/iancowley/i-built-a-zero-dependency-c-vector-database-that-saturates-ddr5-ram-bandwidth-5f9a)**


**Glacier.Vector** is a high-performance, SIMD-accelerated vector search engine and index for .NET 10. Built for speed and efficiency, it provides a zero-copy architecture for storing and querying high-dimensional embeddings, commonly used in LLM-powered applications and semantic search.

---

## Key Features

*   🚀 **SIMD-Accelerated Search**: Utilizes hardware intrinsics (AVX2, AVX-512) for lightning-fast brute-force vector comparisons (Cosine Similarity, Dot Product, Euclidean Distance).
*   🧠 **LLM Optimized**: Specifically designed to handle standard embedding dimensions (e.g., 1536 for OpenAI `text-embedding-3-small`).
*   📥 **Zero-Copy Memory Model**: Efficiently manages large vector datasets in-memory using `Span<T>` and `Memory<T>`, minimizing allocations and GC pressure.
*   🤖 **MCP Server Support**: Built-in support for the Model Context Protocol, allowing seamless integration with AI agents and tools.
*   🛠️ **Extensible Storage**: Pluggable storage backends, starting with high-performance `InMemoryVectorStorage`.

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
