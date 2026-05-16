using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
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

        private static readonly JsonSerializerOptions JsonOpts = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull
        };

        private readonly string _logPath;

        private void Log(string message)
        {
            try
            {
                File.AppendAllText(_logPath, $"[{DateTime.Now:T}] {message}\n");
            }
            catch { /* Ignore logging errors */ }
        }

        public McpServer(VectorIndex index)
        {
            _index = index ?? throw new ArgumentNullException(nameof(index));
            _logPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "mcp_log.txt");

            Log("Server instance created.");

            // Setup standard IO for MCP communication
            Stream openStandardInput = Console.OpenStandardInput();
            _reader = new StreamReader(openStandardInput);

            Stream openStandardOutput = Console.OpenStandardOutput();
            // Use UTF8 without BOM (important for some MCP hosts)
            _writer = new StreamWriter(openStandardOutput, new UTF8Encoding(false)) { AutoFlush = true };
            _writer.NewLine = "\n";
            
            Log("IO setup complete.");
        }

        public async Task RunAsync()
        {
            // The main JSON-RPC listening loop
            while (await _reader.ReadLineAsync() is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                JsonElement requestId = default;
                try
                {
                    Log($"Received: {line}");
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    if (!root.TryGetProperty("method", out JsonElement methodElem))
                    {
                        Log("Message missing 'method' property.");
                        continue;
                    }

                    string method = methodElem.GetString() ?? "";
                    root.TryGetProperty("id", out requestId);

                    Log($"Method: {method}");

                    // Route the MCP Methods
                    switch (method)
                    {
                        case "initialize":
                            await SendResponseAsync(requestId, HandleInitialize());
                            break;

                        case "notifications/initialized":
                            Log("Initialized notification received.");
                            break;

                        case "notifications/cancelled":
                            Log("Client cancelled a request.");
                            break;

                        case "tools/list":
                            await SendResponseAsync(requestId, HandleToolsList());
                            break;

                        case "tools/call":
                            try
                            {
                                if (root.TryGetProperty("params", out JsonElement parameters))
                                {
                                    var result = await HandleToolCallAsync(parameters);
                                    await SendResponseAsync(requestId, result);
                                }
                                else
                                {
                                    await SendErrorAsync(requestId, -32602, "Missing parameters for tools/call");
                                }
                            }
                            catch (Exception toolEx)
                            {
                                Log($"Tool execution error: {toolEx.Message}");
                                await SendErrorAsync(requestId, -32000, toolEx.Message);
                            }
                            break;

                        default:
                            Log($"Method '{method}' not handled.");
                            if (requestId.ValueKind != JsonValueKind.Undefined && requestId.ValueKind != JsonValueKind.Null)
                            {
                                await SendErrorAsync(requestId, -32601, $"Method '{method}' not found.");
                            }
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Log($"CRITICAL ERROR: {ex.Message}\n{ex.StackTrace}");
                    if (requestId.ValueKind != JsonValueKind.Undefined && requestId.ValueKind != JsonValueKind.Null)
                    {
                        try { await SendErrorAsync(requestId, -32603, "Internal server error."); } catch { }
                    }
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
                tools = new[]
                {
                    new Dictionary<string, object>
                    {
                        { "name", "add_vector" },
                        { "description", "Adds a new vector and its associated text/metadata to the database." },
                        { "inputSchema", new Dictionary<string, object>
                            {
                                { "type", "object" },
                                { "properties", new Dictionary<string, object>
                                    {
                                        { "vector", new Dictionary<string, object>
                                            {
                                                { "type", "array" },
                                                { "items", new Dictionary<string, object> { { "type", "number" } } },
                                                { "description", "The embedding vector (e.g., 1536 floats)." }
                                            }
                                        },
                                        { "metadata", new Dictionary<string, object>
                                            {
                                                { "type", "string" },
                                                { "description", "The text or JSON metadata associated with this vector." }
                                            }
                                        }
                                    }
                                },
                                { "required", new[] { "vector", "metadata" } }
                            }
                        }
                    },
                    new Dictionary<string, object>
                    {
                        { "name", "search_vectors" },
                        { "description", "Searches the vector database for the closest semantic matches to a query vector." },
                        { "inputSchema", new Dictionary<string, object>
                            {
                                { "type", "object" },
                                { "properties", new Dictionary<string, object>
                                    {
                                        { "query", new Dictionary<string, object>
                                            {
                                                { "type", "array" },
                                                { "items", new Dictionary<string, object> { { "type", "number" } } },
                                                { "description", "The query embedding vector." }
                                            }
                                        },
                                        { "top_k", new Dictionary<string, object>
                                            {
                                                { "type", "integer" },
                                                { "description", "The number of results to return (default 5)." }
                                            }
                                        }
                                    }
                                },
                                { "required", new[] { "query" } }
                            }
                        }
                    },
                    new Dictionary<string, object>
                    {
                        { "name", "ping" },
                        { "description", "A simple ping tool to verify the server is responding." },
                        { "inputSchema", new Dictionary<string, object>
                            {
                                { "type", "object" },
                                { "properties", new Dictionary<string, object>() }
                            }
                        }
                    }
                }
            };
        }

        private async Task<object> HandleToolCallAsync(JsonElement parameters)
        {
            string name = parameters.GetProperty("name").GetString() ?? "";
            JsonElement args = parameters.GetProperty("arguments");

            if (name == "ping")
            {
                return CreateToolResponse("pong");
            }
            else if (name == "add_vector")
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
            var response = new Dictionary<string, object>
            {
                ["jsonrpc"] = "2.0",
                ["result"] = result
            };

            if (id.ValueKind != JsonValueKind.Undefined)
            {
                response["id"] = id;
            }

            string json = JsonSerializer.Serialize(response, JsonOpts);
            Log($"Sending response: {json}");
            await _writer.WriteLineAsync(json);
        }

        private async Task SendErrorAsync(JsonElement id, int code, string message)
        {
            var response = new Dictionary<string, object>
            {
                ["jsonrpc"] = "2.0",
                ["error"] = new { code, message }
            };

            if (id.ValueKind != JsonValueKind.Undefined)
            {
                response["id"] = id;
            }

            string json = JsonSerializer.Serialize(response, JsonOpts);
            Log($"Sending error: {json}");
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