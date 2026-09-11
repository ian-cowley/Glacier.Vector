using System;

namespace Glacier.Vector.Core;

/// <summary>
/// Provides global and ambient concurrency configuration for Glacier.Vector operations,
/// enabling dynamic scaling or throttling across CPU hardware cores.
/// </summary>
public static class VectorConcurrency
{
    private static int _maxDegreeOfParallelism = 0;

    /// <summary>
    /// Gets or sets the global maximum degree of parallelism for Glacier.Vector index and search operations.
    /// <list type="bullet">
    ///   <item><description><c>0</c> (default): Automatically scales to all available hardware cores or reads <c>GLACIER_NUM_THREADS</c>.</description></item>
    ///   <item><description><c>-1</c>: Conservative mode (leaves 2 cores free for OS/UI responsiveness).</description></item>
    ///   <item><description><c>&gt; 0</c>: Explicit cap on the number of concurrent worker threads.</description></item>
    /// </list>
    /// </summary>
    public static int MaxDegreeOfParallelism
    {
        get => _maxDegreeOfParallelism;
        set => _maxDegreeOfParallelism = value;
    }

    /// <summary>
    /// Resolves the effective degree of parallelism for an operation based on:
    /// 1. Explicit argument (if &gt; 0).
    /// 2. Ambient <see cref="MaxDegreeOfParallelism"/> setting.
    /// 3. Environment variable <c>GLACIER_NUM_THREADS</c>.
    /// 4. System hardware thread count (<see cref="Environment.ProcessorCount"/>).
    /// </summary>
    public static int GetEffectiveParallelism(int requested = 0)
    {
        if (requested > 0)
            return Math.Min(requested, Environment.ProcessorCount);

        if (_maxDegreeOfParallelism > 0)
            return Math.Min(_maxDegreeOfParallelism, Environment.ProcessorCount);

        if (_maxDegreeOfParallelism == -1)
            return Math.Max(1, Environment.ProcessorCount - 2);

        string? env = Environment.GetEnvironmentVariable("GLACIER_NUM_THREADS");
        if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int envThreads) && envThreads > 0)
            return Math.Min(envThreads, Environment.ProcessorCount);

        return Math.Max(1, Environment.ProcessorCount);
    }
}
