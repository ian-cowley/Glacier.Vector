using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Glacier.Vector.Index;
using Glacier.Vector.Mcp;
using Glacier.Vector.Storage;
using Xunit;

namespace Glacier.Vector.Tests
{
    public class VectorHostTests
    {
        [Fact]
        public void ResolveDimensions_UsesDefaultAndValidCustomArgs()
        {
            Assert.Equal(1536, Glacier.Vector.Host.Program.ResolveDimensions(Array.Empty<string>()));
            Assert.Equal(1536, Glacier.Vector.Host.Program.ResolveDimensions(new[] { "not-a-number" }));
            Assert.Equal(1536, Glacier.Vector.Host.Program.ResolveDimensions(new[] { "0" }));
            Assert.Equal(1536, Glacier.Vector.Host.Program.ResolveDimensions(new[] { "-128" }));
            Assert.Equal(4, Glacier.Vector.Host.Program.ResolveDimensions(new[] { "4" }));
        }

        [Fact]
        public void McpServer_Constructor_RejectsNullArguments()
        {
            using var storage = new InMemoryVectorStorage(4);
            using var index = new VectorIndex(storage);

            Assert.Throws<ArgumentNullException>(() => new McpServer(null!, new MemoryStream(), new MemoryStream()));
            Assert.Throws<ArgumentNullException>(() => new McpServer(index, null!, new MemoryStream()));
            Assert.Throws<ArgumentNullException>(() => new McpServer(index, new MemoryStream(), null!));
        }

        [Fact]
        public async Task HostRunAsync_SupportsInitializeToolsListAndVectorWorkflow()
        {
            string requests = string.Join("\n", new[]
            {
                """{"jsonrpc":"2.0","id":1,"method":"initialize"}""",
                """{"jsonrpc":"2.0","method":"notifications/initialized"}""",
                """{"jsonrpc":"2.0","method":"notifications/cancelled"}""",
                """{"jsonrpc":"2.0","id":2,"method":"tools/list"}""",
                """{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"ping","arguments":{}}}""",
                """{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"search_vectors","arguments":{"query":[1.0,0.0,0.0,0.0],"top_k":3}}}""",
                """{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"add_vector","arguments":{"vector":[1.0,0.0,0.0,0.0],"metadata":"doc-alpha"}}}""",
                """{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"add_vector","arguments":{"vector":[0.0,1.0,0.0,0.0],"metadata":"doc-beta"}}}""",
                """{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"search_vectors","arguments":{"query":[0.95,0.05,0.0,0.0],"top_k":1}}}"""
            }) + "\n";

            using var input = new MemoryStream(Encoding.UTF8.GetBytes(requests));
            using var output = new MemoryStream();

            await Glacier.Vector.Host.Program.RunAsync(new[] { "4" }, input, output);

            var responses = ParseLines(Encoding.UTF8.GetString(output.ToArray()));
            Assert.Equal(7, responses.Count);

            // 1: initialize
            Assert.Equal("2024-11-05", responses[0].RootElement.GetProperty("result").GetProperty("protocolVersion").GetString());
            Assert.Equal("Glacier.Vector", responses[0].RootElement.GetProperty("result").GetProperty("serverInfo").GetProperty("name").GetString());

            // 2: tools/list
            var toolNames = responses[1].RootElement.GetProperty("result").GetProperty("tools")
                .EnumerateArray()
                .Select(t => t.GetProperty("name").GetString())
                .ToHashSet();
            Assert.Equal(new[] { "add_vector", "search_vectors", "ping" }.ToHashSet(), toolNames);

            // 3: ping
            Assert.Equal("pong", GetToolText(responses[2]));

            // 4: empty search
            Assert.Equal("No results found.", GetToolText(responses[3]));

            // 5..6: add_vector
            Assert.Contains("Total vectors: 1", GetToolText(responses[4]));
            Assert.Contains("Total vectors: 2", GetToolText(responses[5]));

            // 7: search_vectors
            string searchResult = GetToolText(responses[6]);
            Assert.Contains("[Rank 1 | Score:", searchResult);
            Assert.Contains("doc-alpha", searchResult);
            Assert.DoesNotContain("doc-beta", searchResult);
        }

        [Fact]
        public async Task HostRunAsync_ReturnsJsonRpcErrors_ForInvalidRequestsAndArguments()
        {
            string requests = string.Join("\n", new[]
            {
                "",
                "{invalid-json",
                """{"jsonrpc":"2.0","id":1}""",
                """{"jsonrpc":"2.0","id":2,"method":"unknown/method"}""",
                """{"jsonrpc":"2.0","method":"unknown/notification"}""",
                """{"jsonrpc":"2.0","id":3,"method":"tools/call"}""",
                """{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"unsupported_tool","arguments":{}}}""",
                """{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"add_vector","arguments":{"vector":"not-an-array","metadata":"bad"}}}""",
                """{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"add_vector","arguments":{"vector":[1.0,2.0],"metadata":"wrong-dim"}}}""",
                """{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"search_vectors","arguments":{}}}"""
            }) + "\n";

            using var input = new MemoryStream(Encoding.UTF8.GetBytes(requests));
            using var output = new MemoryStream();

            await Glacier.Vector.Host.Program.RunAsync(new[] { "4" }, input, output);

            var responses = ParseLines(Encoding.UTF8.GetString(output.ToArray()));
            Assert.Equal(6, responses.Count);

            Assert.Equal(-32601, GetErrorCode(responses[0])); // unknown/method
            Assert.Equal(-32602, GetErrorCode(responses[1])); // missing params
            Assert.Equal(-32000, GetErrorCode(responses[2])); // unsupported tool
            Assert.Equal(-32000, GetErrorCode(responses[3])); // non-array vector
            Assert.Equal(-32000, GetErrorCode(responses[4])); // dimension mismatch
            Assert.Equal(-32000, GetErrorCode(responses[5])); // missing query
        }

        private static List<JsonDocument> ParseLines(string output)
        {
            return output
                .Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries)
                .Select(line => JsonDocument.Parse(line))
                .ToList();
        }

        private static string GetToolText(JsonDocument doc)
        {
            return doc.RootElement
                .GetProperty("result")
                .GetProperty("content")[0]
                .GetProperty("text")
                .GetString() ?? string.Empty;
        }

        private static int GetErrorCode(JsonDocument doc)
        {
            return doc.RootElement
                .GetProperty("error")
                .GetProperty("code")
                .GetInt32();
        }
    }
}
