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
            // For the MCP server, we'll use a 1536-dimensional index by default (LLM standard)
            int dimensions = 1536;
            
            if (args.Length > 0 && int.TryParse(args[0], out int customDimensions))
            {
                dimensions = customDimensions;
            }

            // Setup the vector stack
            using var storage = new InMemoryVectorStorage(dimensions);
            using var index = new VectorIndex(storage);
            
            // Create and run the MCP server
            var server = new McpServer(index);
            
            // Log startup info to stderr (so it doesn't interfere with the Stdio protocol on stdout)
            Console.Error.WriteLine($"Glacier.Vector MCP Server started.");
            Console.Error.WriteLine($"Dimensions: {dimensions}");
            Console.Error.WriteLine($"Listening for JSON-RPC on stdin...");

            await server.RunAsync();
        }
    }
}
