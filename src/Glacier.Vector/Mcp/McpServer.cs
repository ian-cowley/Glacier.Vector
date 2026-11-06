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
    /// Native AOT-compatible Model Context Protocol (MCP) server over Stdio.
    /// Allows AI Agents to natively interact with the Vector Database without reflection serialization.
    /// </summary>
    public class McpServer
    {
        private readonly VectorIndex _index;
        private readonly StreamReader _reader;
        private readonly StreamWriter _writer;
        private readonly bool _debugLog;

        public McpServer(VectorIndex index)
            : this(index, Console.OpenStandardInput(), Console.OpenStandardOutput())
        {
        }

        public McpServer(VectorIndex index, Stream inputStream, Stream outputStream)
        {
            _index = index ?? throw new ArgumentNullException(nameof(index));
            ArgumentNullException.ThrowIfNull(inputStream);
            ArgumentNullException.ThrowIfNull(outputStream);

            _debugLog = string.Equals(Environment.GetEnvironmentVariable("GLACIER_MCP_DEBUG"), "1", StringComparison.Ordinal);
            _reader = new StreamReader(inputStream, Encoding.UTF8);
            _writer = new StreamWriter(outputStream, new UTF8Encoding(encoderShouldEmitUTF8Identifier: false)) { AutoFlush = true, NewLine = "\n" };
        }

        private void Log(string message)
        {
            if (_debugLog)
            {
                try { Console.Error.WriteLine($"[Glacier.Vector.Mcp] {message}"); } catch { }
            }
        }

        public async Task RunAsync()
        {
            while (await _reader.ReadLineAsync() is { } line)
            {
                if (string.IsNullOrWhiteSpace(line)) continue;

                JsonElement requestId = default;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    if (!root.TryGetProperty("method", out JsonElement methodElem))
                    {
                        continue;
                    }

                    string method = methodElem.GetString() ?? string.Empty;
                    root.TryGetProperty("id", out requestId);

                    switch (method)
                    {
                        case "initialize":
                            await WriteJsonResponseAsync(requestId, w => WriteInitializeResult(w));
                            break;

                        case "notifications/initialized":
                        case "notifications/cancelled":
                            break;

                        case "tools/list":
                            await WriteJsonResponseAsync(requestId, w => WriteToolsListResult(w));
                            break;

                        case "tools/call":
                            try
                            {
                                if (root.TryGetProperty("params", out JsonElement parameters))
                                {
                                    string text = HandleToolCall(parameters);
                                    await WriteJsonResponseAsync(requestId, w => WriteToolTextResult(w, text));
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
                            if (requestId.ValueKind != JsonValueKind.Undefined && requestId.ValueKind != JsonValueKind.Null)
                            {
                                await SendErrorAsync(requestId, -32601, $"Method '{method}' not found.");
                            }
                            break;
                    }
                }
                catch (Exception ex)
                {
                    Log($"Protocol error: {ex.Message}");
                    if (requestId.ValueKind != JsonValueKind.Undefined && requestId.ValueKind != JsonValueKind.Null)
                    {
                        try { await SendErrorAsync(requestId, -32603, "Internal server error."); } catch { }
                    }
                }
            }
        }

        private string HandleToolCall(JsonElement parameters)
        {
            string name = parameters.GetProperty("name").GetString() ?? string.Empty;
            JsonElement args = parameters.TryGetProperty("arguments", out JsonElement a) ? a : default;

            if (name == "ping")
            {
                return "pong";
            }

            if (name == "add_vector")
            {
                float[] vector = ParseFloatArray(args.GetProperty("vector"));
                string metadata = args.TryGetProperty("metadata", out JsonElement m) ? (m.GetString() ?? string.Empty) : string.Empty;
                _index.Add(vector, metadata);
                return $"Successfully added vector with metadata: '{metadata}' to index. Total vectors: {_index.Count}";
            }

            if (name == "search_vectors")
            {
                float[] query = ParseFloatArray(args.GetProperty("query"));
                int topK = args.TryGetProperty("top_k", out JsonElement topKElem) ? topKElem.GetInt32() : 5;

                SearchResult[] results = _index.Search(query, topK);
                if (results.Length == 0)
                    return "No results found.";

                var sb = new StringBuilder();
                for (int i = 0; i < results.Length; i++)
                {
                    if (i > 0) sb.Append('\n');
                    sb.Append($"[Rank {i + 1} | Score: {results[i].Score:F4}] {results[i].Metadata}");
                }
                return sb.ToString();
            }

            throw new InvalidOperationException($"Tool '{name}' is not supported.");
        }

        private static float[] ParseFloatArray(JsonElement element)
        {
            if (element.ValueKind != JsonValueKind.Array)
                throw new ArgumentException("Expected JSON array of numbers.");

            int len = element.GetArrayLength();
            float[] values = new float[len];
            int idx = 0;
            foreach (JsonElement item in element.EnumerateArray())
            {
                values[idx++] = item.GetSingle();
            }
            return values;
        }

        private async Task WriteJsonResponseAsync(JsonElement id, Action<Utf8JsonWriter> writeResult)
        {
            using var ms = new MemoryStream();
            using (var writer = new Utf8JsonWriter(ms))
            {
                writer.WriteStartObject();
                writer.WriteString("jsonrpc", "2.0");
                if (id.ValueKind != JsonValueKind.Undefined)
                {
                    writer.WritePropertyName("id");
                    id.WriteTo(writer);
                }
                writer.WritePropertyName("result");
                writeResult(writer);
                writer.WriteEndObject();
            }

            string json = Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
            await _writer.WriteLineAsync(json);
        }

        private async Task SendErrorAsync(JsonElement id, int code, string message)
        {
            using var ms = new MemoryStream();
            using (var writer = new Utf8JsonWriter(ms))
            {
                writer.WriteStartObject();
                writer.WriteString("jsonrpc", "2.0");
                if (id.ValueKind != JsonValueKind.Undefined)
                {
                    writer.WritePropertyName("id");
                    id.WriteTo(writer);
                }
                writer.WriteStartObject("error");
                writer.WriteNumber("code", code);
                writer.WriteString("message", message);
                writer.WriteEndObject();
                writer.WriteEndObject();
            }

            string json = Encoding.UTF8.GetString(ms.GetBuffer(), 0, (int)ms.Length);
            await _writer.WriteLineAsync(json);
        }

        private static void WriteInitializeResult(Utf8JsonWriter writer)
        {
            writer.WriteStartObject();
            writer.WriteString("protocolVersion", "2024-11-05");
            writer.WriteStartObject("capabilities");
            writer.WriteStartObject("tools");
            writer.WriteBoolean("listChanged", false);
            writer.WriteEndObject();
            writer.WriteEndObject();
            writer.WriteStartObject("serverInfo");
            writer.WriteString("name", "Glacier.Vector");
            writer.WriteString("version", "1.1.0");
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        private static void WriteToolTextResult(Utf8JsonWriter writer, string text)
        {
            writer.WriteStartObject();
            writer.WriteStartArray("content");
            writer.WriteStartObject();
            writer.WriteString("type", "text");
            writer.WriteString("text", text);
            writer.WriteEndObject();
            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        private static void WriteToolsListResult(Utf8JsonWriter writer)
        {
            writer.WriteStartObject();
            writer.WriteStartArray("tools");

            // 1. add_vector
            writer.WriteStartObject();
            writer.WriteString("name", "add_vector");
            writer.WriteString("description", "Adds a new vector and its associated text/metadata to the database.");
            writer.WriteStartObject("inputSchema");
            writer.WriteString("type", "object");
            writer.WriteStartObject("properties");
            writer.WriteStartObject("vector");
            writer.WriteString("type", "array");
            writer.WriteStartObject("items");
            writer.WriteString("type", "number");
            writer.WriteEndObject();
            writer.WriteString("description", "The embedding vector (e.g., 1536 floats).");
            writer.WriteEndObject();
            writer.WriteStartObject("metadata");
            writer.WriteString("type", "string");
            writer.WriteString("description", "The text or JSON metadata associated with this vector.");
            writer.WriteEndObject();
            writer.WriteEndObject();
            writer.WriteStartArray("required");
            writer.WriteStringValue("vector");
            writer.WriteStringValue("metadata");
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.WriteEndObject();

            // 2. search_vectors
            writer.WriteStartObject();
            writer.WriteString("name", "search_vectors");
            writer.WriteString("description", "Searches the vector database for the closest semantic matches to a query vector.");
            writer.WriteStartObject("inputSchema");
            writer.WriteString("type", "object");
            writer.WriteStartObject("properties");
            writer.WriteStartObject("query");
            writer.WriteString("type", "array");
            writer.WriteStartObject("items");
            writer.WriteString("type", "number");
            writer.WriteEndObject();
            writer.WriteString("description", "The query embedding vector.");
            writer.WriteEndObject();
            writer.WriteStartObject("top_k");
            writer.WriteString("type", "integer");
            writer.WriteString("description", "The number of results to return (default 5).");
            writer.WriteEndObject();
            writer.WriteEndObject();
            writer.WriteStartArray("required");
            writer.WriteStringValue("query");
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.WriteEndObject();

            // 3. ping
            writer.WriteStartObject();
            writer.WriteString("name", "ping");
            writer.WriteString("description", "A simple ping tool to verify the server is responding.");
            writer.WriteStartObject("inputSchema");
            writer.WriteString("type", "object");
            writer.WriteStartObject("properties");
            writer.WriteEndObject();
            writer.WriteEndObject();
            writer.WriteEndObject();

            writer.WriteEndArray();
            writer.WriteEndObject();
        }
    }
}
