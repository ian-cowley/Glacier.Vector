namespace Glacier.Vector.Core;

/// <summary>
/// Specifies the hardware execution target for Glacier.Vector accelerated search operations.
/// </summary>
public enum GpuTarget
{
    /// <summary>
    /// Automatically selects the best available hardware (NVIDIA RTX 4060 dGPU, AMD Radeon 890M APU, or SIMD CPU).
    /// </summary>
    Auto = 0,

    /// <summary>
    /// Targets bare-metal NVIDIA CUDA acceleration (RTX 4060 / Ada Lovelace).
    /// </summary>
    Nvidia = 1,

    /// <summary>
    /// Targets bare-metal AMD ROCm/HIP acceleration (Radeon 890M / RDNA 3.5).
    /// </summary>
    Amd = 2,

    /// <summary>
    /// Distributes workload dynamically across both GPUs if available.
    /// </summary>
    DualGpu = 3,

    /// <summary>
    /// Forces CPU-only execution with AVX-512 / AVX2 SIMD vectorization.
    /// </summary>
    Cpu = 4
}
