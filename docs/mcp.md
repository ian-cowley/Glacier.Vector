# Glacier.Vector MCP Server Setup

`Glacier.Vector.Host` exposes an in-process stdio Model Context Protocol (MCP) server for AI assistants and agents using Native AOT-compatible `Utf8JsonWriter` / `JsonDocument` serialization.

## Build the Host Binary

```bash
dotnet build src/Glacier.Vector.Host/Glacier.Vector.Host.csproj -c Release
```

## Configure an MCP Client

Add the built host assembly to your client's `mcp_config.json` (for example `%AppData%\Roaming\Claude\claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "glacier-vector": {
      "command": "dotnet",
      "args": [
        "C:/path/to/Glacier.Vector/src/Glacier.Vector.Host/bin/Release/net10.0/Glacier.Vector.Host.dll"
      ]
    }
  }
}
```

Set `GLACIER_VECTOR_MCP_LOG=1` in the server environment if you need diagnostic messages written to `stderr`.

## Exposed Tools

| Tool Name | Purpose |
| :--- | :--- |
| `add_vector` | Appends a 1536-dimensional embedding vector and metadata payload to the index (default configured dimension). |
| `search_vectors` | Searches for the top-K nearest vectors to a query embedding. |
| `ping` | Health check returning server readiness. |
