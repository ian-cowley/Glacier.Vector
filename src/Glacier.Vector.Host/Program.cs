using System;
using System.Threading.Tasks;
using Glacier.Vector.Index;
using Glacier.Vector.Storage;
using Glacier.Vector.Mcp;

namespace Glacier.Vector.Host
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            await RunAsync(args);
        }

        internal static int ResolveDimensions(string[] args)
        {
            // For the MCP server, we'll use a 1536-dimensional index by default (LLM standard)
            int dimensions = 1536;

            if (args != null && args.Length > 0 && int.TryParse(args[0], out int customDimensions) && customDimensions > 0)
            {
                dimensions = customDimensions;
            }

            return dimensions;
        }

        internal static async Task RunAsync(string[] args, Stream? inputStream = null, Stream? outputStream = null)
        {
            int dimensions = ResolveDimensions(args);

            // Setup the vector stack
            using var storage = new InMemoryVectorStorage(dimensions);
            using var index = new VectorIndex(storage);

            // Create and run the MCP server
            var server = inputStream != null && outputStream != null
                ? new McpServer(index, inputStream, outputStream)
                : new McpServer(index);

            // Log startup info to stderr (so it doesn't interfere with the Stdio protocol on stdout)
            Console.Error.WriteLine($"Glacier.Vector MCP Server started.");
            Console.Error.WriteLine($"Dimensions: {dimensions}");
            Console.Error.WriteLine($"Listening for JSON-RPC on stdin...");

            await server.RunAsync();
        }
    }
}
