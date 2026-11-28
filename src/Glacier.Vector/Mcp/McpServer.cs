using System;
using System.Buffers;
using System.IO;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Text.Json.Serialization.Metadata;
using System.Threading;
using System.Threading.Tasks;
using Glacier.Vector.Index;

namespace Glacier.Vector.Mcp
{
    /// <summary>
    /// Native AOT-compatible Model Context Protocol (MCP) server over Stdio.
    /// Uses source-generated System.Text.Json DTOs and a reusable <see cref="ArrayBufferWriter{T}"/>
    /// written directly to the output stream with zero intermediate <see cref="MemoryStream"/> or string allocations.
    /// </summary>
    public sealed class McpServer : IDisposable
    {
        private static readonly string s_serverVersion = ResolveServerVersion();
        private static readonly McpToolsListResult s_cachedToolsList = BuildToolsListResult();

        private readonly VectorIndex _index;
        private readonly Stream _outputStream;
        private readonly StreamReader _reader;
        private readonly ArrayBufferWriter<byte> _writeBuffer = new(4096);
        private readonly bool _ownsStreams;
        private readonly bool _debugLog;
        private bool _disposed;
        private int _disposeFlag;

        public McpServer(VectorIndex index)
            : this(index, Console.OpenStandardInput(), Console.OpenStandardOutput(), ownsStreams: true)
        {
        }

        public McpServer(VectorIndex index, Stream inputStream, Stream outputStream)
            : this(index, inputStream, outputStream, ownsStreams: false)
        {
        }

        private McpServer(VectorIndex index, Stream inputStream, Stream outputStream, bool ownsStreams)
        {
            _index = index ?? throw new ArgumentNullException(nameof(index));
            ArgumentNullException.ThrowIfNull(inputStream);
            ArgumentNullException.ThrowIfNull(outputStream);

            _outputStream = outputStream;
            _ownsStreams = ownsStreams;
            _debugLog = string.Equals(Environment.GetEnvironmentVariable("GLACIER_MCP_DEBUG"), "1", StringComparison.Ordinal);
            _reader = new StreamReader(inputStream, Encoding.UTF8, detectEncodingFromByteOrderMarks: false, bufferSize: 4096, leaveOpen: !ownsStreams);
        }

        private static string ResolveServerVersion()
        {
            Version? v = typeof(McpServer).Assembly.GetName().Version;
            if (v != null && v.Major > 0)
            {
                return v.Build >= 0 ? $"{v.Major}.{v.Minor}.{v.Build}" : $"{v.Major}.{v.Minor}.0";
            }
            return "1.3.1";
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
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);

            while (await _reader.ReadLineAsync().ConfigureAwait(false) is { } line)
            {
                ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
                if (string.IsNullOrWhiteSpace(line)) continue;

                JsonElement requestId = default;
                try
                {
                    using var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;

                    if (root.ValueKind != JsonValueKind.Object)
                    {
                        await SendErrorAsync(default, -32600, "Invalid Request.").ConfigureAwait(false);
                        continue;
                    }

                    root.TryGetProperty("id", out requestId);

                    if (!root.TryGetProperty("method", out JsonElement methodElem) ||
                        methodElem.ValueKind != JsonValueKind.String ||
                        string.IsNullOrWhiteSpace(methodElem.GetString()))
                    {
                        await SendErrorAsync(requestId, -32600, "Invalid Request: missing method.").ConfigureAwait(false);
                        continue;
                    }

                    string method = methodElem.GetString()!;

                    switch (method)
                    {
                        case "initialize":
                            var initResult = new McpInitializeResult(
                                "2024-11-05",
                                new McpCapabilities(new McpToolsCapability(false)),
                                new McpServerInfo("Glacier.Vector", s_serverVersion));
                            await WriteJsonResponseAsync(requestId, initResult, VectorMcpJsonContext.Default.McpInitializeResult).ConfigureAwait(false);
                            break;

                        case "ping":
                            await WriteJsonResponseAsync(requestId, McpEmptyResult.Instance, VectorMcpJsonContext.Default.McpEmptyResult).ConfigureAwait(false);
                            break;

                        case "notifications/initialized":
                        case "notifications/cancelled":
                            break;

                        case "tools/list":
                            await WriteJsonResponseAsync(requestId, s_cachedToolsList, VectorMcpJsonContext.Default.McpToolsListResult).ConfigureAwait(false);
                            break;

                        case "tools/call":
                            if (!root.TryGetProperty("params", out JsonElement parameters) || parameters.ValueKind != JsonValueKind.Object)
                            {
                                await SendErrorAsync(requestId, -32602, "Missing parameters for tools/call").ConfigureAwait(false);
                                break;
                            }

                            if (!parameters.TryGetProperty("name", out JsonElement nameElem) ||
                                nameElem.ValueKind != JsonValueKind.String ||
                                string.IsNullOrWhiteSpace(nameElem.GetString()))
                            {
                                await SendErrorAsync(requestId, -32602, "Missing tool name for tools/call").ConfigureAwait(false);
                                break;
                            }

                            try
                            {
                                string text = HandleToolCall(nameElem.GetString()!, parameters);
                                var okResult = new McpToolCallResult(new[] { new McpToolTextContent("text", text) }, IsError: null);
                                await WriteJsonResponseAsync(requestId, okResult, VectorMcpJsonContext.Default.McpToolCallResult).ConfigureAwait(false);
                            }
                            catch (Exception toolEx)
                            {
                                Log($"Tool execution error: {toolEx.Message}");
                                var errResult = new McpToolCallResult(new[] { new McpToolTextContent("text", toolEx.Message) }, IsError: true);
                                await WriteJsonResponseAsync(requestId, errResult, VectorMcpJsonContext.Default.McpToolCallResult).ConfigureAwait(false);
                            }
                            break;

                        default:
                            if (requestId.ValueKind != JsonValueKind.Undefined && requestId.ValueKind != JsonValueKind.Null)
                            {
                                await SendErrorAsync(requestId, -32601, $"Method '{method}' not found.").ConfigureAwait(false);
                            }
                            break;
                    }
                }
                catch (JsonException jsonEx)
                {
                    Log($"JSON parse error: {jsonEx.Message}");
                    await SendErrorAsync(default, -32700, "Invalid JSON was received by the server.").ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    Log($"Protocol error: {ex.Message}");
                    if (requestId.ValueKind != JsonValueKind.Undefined && requestId.ValueKind != JsonValueKind.Null)
                    {
                        try { await SendErrorAsync(requestId, -32603, "Internal server error.").ConfigureAwait(false); } catch { }
                    }
                }
            }
        }

        private string HandleToolCall(string name, JsonElement parameters)
        {
            JsonElement args = parameters.TryGetProperty("arguments", out JsonElement a) ? a : default;

            if (name == "ping")
            {
                return "pong";
            }

            if (name == "add_vector")
            {
                if (args.ValueKind != JsonValueKind.Object || !args.TryGetProperty("vector", out JsonElement vecElem))
                    throw new ArgumentException("Missing required argument 'vector'.");

                float[] vector = ParseFloatArray(vecElem);
                string metadata = args.TryGetProperty("metadata", out JsonElement m) && m.ValueKind == JsonValueKind.String
                    ? (m.GetString() ?? string.Empty)
                    : string.Empty;
                _index.Add(vector, metadata);
                return $"Successfully added vector with metadata: '{metadata}' to index. Total vectors: {_index.Count}";
            }

            if (name == "search_vectors")
            {
                if (args.ValueKind != JsonValueKind.Object || !args.TryGetProperty("query", out JsonElement queryElem))
                    throw new ArgumentException("Missing required argument 'query'.");

                float[] query = ParseFloatArray(queryElem);
                int topK = 5;
                if (args.TryGetProperty("top_k", out JsonElement topKElem))
                {
                    if (topKElem.ValueKind != JsonValueKind.Number || !topKElem.TryGetInt32(out topK))
                        throw new ArgumentException("Argument 'top_k' must be a valid integer.");
                    if (topK <= 0)
                        throw new ArgumentOutOfRangeException(nameof(topK), "Argument 'top_k' must be greater than zero.");
                }

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
                if (item.ValueKind != JsonValueKind.Number)
                    throw new ArgumentException("Expected numeric elements in vector array.");

                float value = item.GetSingle();
                if (!float.IsFinite(value))
                    throw new ArgumentException("Vector contains non-finite value (NaN or Infinity).");

                values[idx++] = value;
            }
            return values;
        }

        private async Task WriteJsonResponseAsync<T>(JsonElement id, T result, JsonTypeInfo<T> typeInfo)
        {
            _writeBuffer.Clear();
            using (var writer = new Utf8JsonWriter(_writeBuffer))
            {
                writer.WriteStartObject();
                writer.WriteString("jsonrpc", "2.0");
                writer.WritePropertyName("id");
                if (id.ValueKind != JsonValueKind.Undefined)
                {
                    id.WriteTo(writer);
                }
                else
                {
                    writer.WriteNullValue();
                }
                writer.WritePropertyName("result");
                JsonSerializer.Serialize(writer, result, typeInfo);
                writer.WriteEndObject();
            }

            Span<byte> newline = _writeBuffer.GetSpan(1);
            newline[0] = (byte)'\n';
            _writeBuffer.Advance(1);

            await _outputStream.WriteAsync(_writeBuffer.WrittenMemory).ConfigureAwait(false);
            await _outputStream.FlushAsync().ConfigureAwait(false);
        }

        private async Task SendErrorAsync(JsonElement id, int code, string message)
        {
            _writeBuffer.Clear();
            using (var writer = new Utf8JsonWriter(_writeBuffer))
            {
                writer.WriteStartObject();
                writer.WriteString("jsonrpc", "2.0");
                writer.WritePropertyName("id");
                if (id.ValueKind != JsonValueKind.Undefined)
                {
                    id.WriteTo(writer);
                }
                else
                {
                    writer.WriteNullValue();
                }
                writer.WritePropertyName("error");
                JsonSerializer.Serialize(writer, new McpJsonRpcError(code, message), VectorMcpJsonContext.Default.McpJsonRpcError);
                writer.WriteEndObject();
            }

            Span<byte> newline = _writeBuffer.GetSpan(1);
            newline[0] = (byte)'\n';
            _writeBuffer.Advance(1);

            await _outputStream.WriteAsync(_writeBuffer.WrittenMemory).ConfigureAwait(false);
            await _outputStream.FlushAsync().ConfigureAwait(false);
        }

        private static McpToolsListResult BuildToolsListResult()
        {
            return new McpToolsListResult(new[]
            {
                new McpToolDefinition(
                    "add_vector",
                    "Adds a new vector and its associated text/metadata to the database.",
                    new McpToolInputSchema(
                        "object",
                        new McpToolProperties(
                            Vector: new McpToolProperty("array", "The embedding vector (e.g., 1536 floats).", new McpToolItemSchema("number")),
                            Metadata: new McpToolProperty("string", "The text or JSON metadata associated with this vector.", null),
                            Query: null,
                            TopK: null),
                        new[] { "vector", "metadata" })),
                new McpToolDefinition(
                    "search_vectors",
                    "Searches the vector database for the closest semantic matches to a query vector.",
                    new McpToolInputSchema(
                        "object",
                        new McpToolProperties(
                            Vector: null,
                            Metadata: null,
                            Query: new McpToolProperty("array", "The query embedding vector.", new McpToolItemSchema("number")),
                            TopK: new McpToolProperty("integer", "The number of results to return (default 5).", null)),
                        new[] { "query" })),
                new McpToolDefinition(
                    "ping",
                    "A simple ping tool to verify the server is responding.",
                    new McpToolInputSchema(
                        "object",
                        new McpToolProperties(null, null, null, null),
                        null))
            });
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposeFlag, 1) != 0) return;
            Volatile.Write(ref _disposed, true);
            _reader.Dispose();
            if (_ownsStreams)
            {
                try { _outputStream.Dispose(); } catch { }
            }
        }
    }

    internal sealed record McpEmptyResult
    {
        public static readonly McpEmptyResult Instance = new();
    }

    internal sealed record McpInitializeResult(
        [property: JsonPropertyName("protocolVersion")] string ProtocolVersion,
        [property: JsonPropertyName("capabilities")] McpCapabilities Capabilities,
        [property: JsonPropertyName("serverInfo")] McpServerInfo ServerInfo);

    internal sealed record McpCapabilities(
        [property: JsonPropertyName("tools")] McpToolsCapability Tools);

    internal sealed record McpToolsCapability(
        [property: JsonPropertyName("listChanged")] bool ListChanged);

    internal sealed record McpServerInfo(
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("version")] string Version);

    internal sealed record McpToolCallResult(
        [property: JsonPropertyName("content")] McpToolTextContent[] Content,
        [property: JsonPropertyName("isError")] bool? IsError);

    internal sealed record McpToolTextContent(
        [property: JsonPropertyName("type")] string Type,
        [property: JsonPropertyName("text")] string Text);

    internal sealed record McpJsonRpcError(
        [property: JsonPropertyName("code")] int Code,
        [property: JsonPropertyName("message")] string Message);

    internal sealed record McpToolsListResult(
        [property: JsonPropertyName("tools")] McpToolDefinition[] Tools);

    internal sealed record McpToolDefinition(
        [property: JsonPropertyName("name")] string Name,
        [property: JsonPropertyName("description")] string Description,
        [property: JsonPropertyName("inputSchema")] McpToolInputSchema InputSchema);

    internal sealed record McpToolInputSchema(
        [property: JsonPropertyName("type")] string Type,
        [property: JsonPropertyName("properties")] McpToolProperties Properties,
        [property: JsonPropertyName("required")] string[]? Required);

    internal sealed record McpToolProperties(
        [property: JsonPropertyName("vector")] McpToolProperty? Vector,
        [property: JsonPropertyName("metadata")] McpToolProperty? Metadata,
        [property: JsonPropertyName("query")] McpToolProperty? Query,
        [property: JsonPropertyName("top_k")] McpToolProperty? TopK);

    internal sealed record McpToolProperty(
        [property: JsonPropertyName("type")] string Type,
        [property: JsonPropertyName("description")] string Description,
        [property: JsonPropertyName("items")] McpToolItemSchema? Items);

    internal sealed record McpToolItemSchema(
        [property: JsonPropertyName("type")] string Type);

    [JsonSourceGenerationOptions(DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull)]
    [JsonSerializable(typeof(McpEmptyResult))]
    [JsonSerializable(typeof(McpInitializeResult))]
    [JsonSerializable(typeof(McpToolCallResult))]
    [JsonSerializable(typeof(McpJsonRpcError))]
    [JsonSerializable(typeof(McpToolsListResult))]
    internal partial class VectorMcpJsonContext : JsonSerializerContext
    {
    }
}
