using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using Glacier.Vector.Index;

namespace Glacier.Vector.Mcp
{
    /// <summary>
    /// A zero-dependency Model Context Protocol (MCP) server over Stdio.
    /// Allows AI Agents to natively interact with the Vector Database.
    /// </summary>
    public class McpServer
    {
        private readonly VectorIndex _index;
        private readonly StreamReader _reader;
        private readonly StreamWriter _writer;

        // Use a static JsonSerializerOptions to prevent allocation overhead
        private static readonly JsonSerializerOptions JsonOpts = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        public McpServer(VectorIndex index)
        {
            _index = index ?? throw new ArgumentNullException(nameof(index));

            // Setup standard IO for MCP communication
            Stream openStandardInput = Console.OpenStandardInput();
            _reader = new StreamReader(openStandardInput);

            Stream openStandardOutput = Console.OpenStandardOutput();
            // AutoFlush is critical for MCP, otherwise the AI host will hang waiting for the response
            _writer = new StreamWriter(openStandardOutput) { AutoFlush = true };
        }

        public async Task RunAsync()
        {
            // The main JSON-RPC listening loop
            while (await _reader.ReadLineAsync() is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                try
                {
                    var request = JsonSerializer.Deserialize<RpcRequest>(line, JsonOpts);
                    if (request == null) continue;

                    // Route the MCP Methods
                    switch (request.Method)
                    {
                        case "initialize":
                            await SendResponseAsync(request.Id, HandleInitialize());
                            break;

                        case "notifications/initialized":
                            // Client acknowledges initialization; no response needed
                            break;

                        case "tools/list":
                            await SendResponseAsync(request.Id, HandleToolsList());
                            break;

                        case "tools/call":
                            var result = HandleToolCall(request.Params);
                            await SendResponseAsync(request.Id, result);
                            break;

                        default:
                            // Method not found
                            await SendErrorAsync(request.Id, -32601, $"Method '{request.Method}' not found.");
                            break;
                    }
                }
                catch (Exception ex)
                {
                    // Log to stderr so we don't break the stdout JSON protocol
                    Console.Error.WriteLine($"[MCP Error] {ex.Message}");
                }
            }
        }

        private object HandleInitialize()
        {
            return new
            {
                protocolVersion = "2024-11-05",
                capabilities = new { tools = new { listChanged = false } },
                serverInfo = new { name = "Glacier.Vector", version = "1.0.0" }
            };
        }

        private object HandleToolsList()
        {
            return new
            {
                tools = new object[]
                {
                    new
                    {
                        name = "add_vector",
                        description = "Adds a new vector and its associated text/metadata to the database.",
                        inputSchema = new
                        {
                            type = "object",
                            properties = new
                            {
                                vector = new { type = "array", items = new { type = "number" }, description = "The embedding vector (e.g., 1536 floats)." },
                                metadata = new { type = "string", description = "The text or JSON metadata associated with this vector." }
                            },
                            required = new[] { "vector", "metadata" }
                        }
                    },
                    new
                    {
                        name = "search_vectors",
                        description = "Searches the vector database for the closest semantic matches to a query vector.",
                        inputSchema = new
                        {
                            type = "object",
                            properties = new
                            {
                                query = new { type = "array", items = new { type = "number" }, description = "The query embedding vector." },
                                top_k = new { type = "integer", description = "The number of results to return (default 5)." }
                            },
                            required = new[] { "query" }
                        }
                    }
                }
            };
        }

        private object HandleToolCall(JsonElement parameters)
        {
            string name = parameters.GetProperty("name").GetString() ?? "";
            JsonElement args = parameters.GetProperty("arguments");

            if (name == "add_vector")
            {
                float[] vector = JsonSerializer.Deserialize<float[]>(args.GetProperty("vector").GetRawText())!;
                string metadata = args.GetProperty("metadata").GetString() ?? "";

                _index.Add(vector, metadata);

                return CreateToolResponse($"Successfully added vector with metadata: '{metadata}' to index. Total vectors: {_index.Count}");
            }
            else if (name == "search_vectors")
            {
                float[] query = JsonSerializer.Deserialize<float[]>(args.GetProperty("query").GetRawText())!;
                int topK = args.TryGetProperty("top_k", out JsonElement topKElem) ? topKElem.GetInt32() : 5;

                // Call our blistering fast SIMD multi-threaded search engine
                SearchResult[] results = _index.Search(query, topK);

                var formattedResults = new List<string>();
                for (int i = 0; i < results.Length; i++)
                {
                    formattedResults.Add($"[Rank {i + 1} | Score: {results[i].Score:F4}] {results[i].Metadata}");
                }

                string content = results.Length > 0
                    ? string.Join("\n", formattedResults)
                    : "No results found.";

                return CreateToolResponse(content);
            }

            throw new Exception($"Tool '{name}' is not supported.");
        }

        private static object CreateToolResponse(string text)
        {
            // MCP requires tools to return an array of content objects
            return new
            {
                content = new[]
                {
                    new { type = "text", text = text }
                }
            };
        }

        private async Task SendResponseAsync(JsonElement id, object result)
        {
            var response = new RpcResponse { Id = id, Result = result };
            string json = JsonSerializer.Serialize(response, JsonOpts);
            await _writer.WriteLineAsync(json);
        }

        private async Task SendErrorAsync(JsonElement id, int code, string message)
        {
            var response = new RpcResponse
            {
                Id = id,
                Error = new { code, message }
            };
            string json = JsonSerializer.Serialize(response, JsonOpts);
            await _writer.WriteLineAsync(json);
        }

        // --- Lightweight JSON-RPC Models ---

        private class RpcRequest
        {
            [JsonPropertyName("jsonrpc")] public string JsonRpc { get; set; } = "2.0";
            [JsonPropertyName("id")] public JsonElement Id { get; set; }
            [JsonPropertyName("method")] public string Method { get; set; } = "";
            [JsonPropertyName("params")] public JsonElement Params { get; set; }
        }

        private class RpcResponse
        {
            [JsonPropertyName("jsonrpc")] public string JsonRpc { get; set; } = "2.0";
            [JsonPropertyName("id")] public JsonElement Id { get; set; }
            [JsonPropertyName("result")] public object? Result { get; set; }
            [JsonPropertyName("error")] public object? Error { get; set; }
        }
    }
}