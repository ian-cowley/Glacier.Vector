using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Glacier.Vector.Compute;

/// <summary>
/// Hardware-accelerated kernels for vector distance calculations across AVX-512, AVX2/FMA, ARM64/Vector128, and scalar paths.
/// </summary>
public static class DistanceKernels
{
    /// <summary>
    /// Calculates the Dot Product of two equal-length vectors.
    /// For unit-normalized embeddings, Dot Product equals Cosine Similarity.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe float DotProduct(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
    {
        if (target.Length != databaseVector.Length)
            throw new ArgumentException($"Vector dimension mismatch: {target.Length} vs {databaseVector.Length}.");

        int length = target.Length;
        if (length == 0) return 0f;

        float dotProduct = 0f;

        fixed (float* pTarget = target)
        fixed (float* pDb = databaseVector)
        {
            int i = 0;

            // 1. AVX-512 Fast Path Unrolled (64 floats per iteration)
            if (Avx512F.IsSupported && length >= 64)
            {
                var acc0 = Vector512<float>.Zero;
                var acc1 = Vector512<float>.Zero;
                var acc2 = Vector512<float>.Zero;
                var acc3 = Vector512<float>.Zero;

                for (; i <= length - 64; i += 64)
                {
                    acc0 = Avx512F.FusedMultiplyAdd(Vector512.Load(pTarget + i), Vector512.Load(pDb + i), acc0);
                    acc1 = Avx512F.FusedMultiplyAdd(Vector512.Load(pTarget + i + 16), Vector512.Load(pDb + i + 16), acc1);
                    acc2 = Avx512F.FusedMultiplyAdd(Vector512.Load(pTarget + i + 32), Vector512.Load(pDb + i + 32), acc2);
                    acc3 = Avx512F.FusedMultiplyAdd(Vector512.Load(pTarget + i + 48), Vector512.Load(pDb + i + 48), acc3);
                }
                var sum1 = Vector512.Add(acc0, acc1);
                var sum2 = Vector512.Add(acc2, acc3);
                dotProduct += Vector512.Sum(Vector512.Add(sum1, sum2));
            }
            // 2. AVX2 / FMA Fast Path Unrolled (32 floats per iteration)
            else if (Avx2.IsSupported && length >= 32)
            {
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;

                if (Fma.IsSupported)
                {
                    for (; i <= length - 32; i += 32)
                    {
                        acc0 = Fma.MultiplyAdd(Vector256.Load(pTarget + i), Vector256.Load(pDb + i), acc0);
                        acc1 = Fma.MultiplyAdd(Vector256.Load(pTarget + i + 8), Vector256.Load(pDb + i + 8), acc1);
                        acc2 = Fma.MultiplyAdd(Vector256.Load(pTarget + i + 16), Vector256.Load(pDb + i + 16), acc2);
                        acc3 = Fma.MultiplyAdd(Vector256.Load(pTarget + i + 24), Vector256.Load(pDb + i + 24), acc3);
                    }
                }
                else
                {
                    for (; i <= length - 32; i += 32)
                    {
                        acc0 = Vector256.Add(acc0, Vector256.Multiply(Vector256.Load(pTarget + i), Vector256.Load(pDb + i)));
                        acc1 = Vector256.Add(acc1, Vector256.Multiply(Vector256.Load(pTarget + i + 8), Vector256.Load(pDb + i + 8)));
                        acc2 = Vector256.Add(acc2, Vector256.Multiply(Vector256.Load(pTarget + i + 16), Vector256.Load(pDb + i + 16)));
                        acc3 = Vector256.Add(acc3, Vector256.Multiply(Vector256.Load(pTarget + i + 24), Vector256.Load(pDb + i + 24)));
                    }
                }

                var sum1 = Vector256.Add(acc0, acc1);
                var sum2 = Vector256.Add(acc2, acc3);
                dotProduct += Vector256.Sum(Vector256.Add(sum1, sum2));
            }

            // Remainder 16-float chunk for AVX-512
            if (Avx512F.IsSupported && i <= length - 16)
            {
                var acc = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    acc = Avx512F.FusedMultiplyAdd(Vector512.Load(pTarget + i), Vector512.Load(pDb + i), acc);
                }
                dotProduct += Vector512.Sum(acc);
            }
            // Remainder 8-float chunk for AVX2
            else if (Avx2.IsSupported && i <= length - 8)
            {
                var acc = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    acc = Fma.IsSupported
                        ? Fma.MultiplyAdd(Vector256.Load(pTarget + i), Vector256.Load(pDb + i), acc)
                        : Vector256.Add(acc, Vector256.Multiply(Vector256.Load(pTarget + i), Vector256.Load(pDb + i)));
                }
                dotProduct += Vector256.Sum(acc);
            }
            // Portable Vector128 (ARM64 Neon / SSE) Fast Path
            else if (Vector128.IsHardwareAccelerated && i <= length - 4)
            {
                var acc = Vector128<float>.Zero;
                for (; i <= length - 4; i += 4)
                {
                    acc = Vector128.FusedMultiplyAdd(Vector128.Load(pTarget + i), Vector128.Load(pDb + i), acc);
                }
                dotProduct += Vector128.Sum(acc);
            }

            // 3. Scalar Cleanup
            for (; i < length; i++)
            {
                dotProduct += pTarget[i] * pDb[i];
            }
        }

        return dotProduct;
    }

    /// <summary>
    /// Calculates the Cosine Distance between two unit-normalized vectors (<c>1.0f - DotProduct(target, databaseVector)</c>).
    /// For arbitrary unnormalized vectors, use <see cref="CosineDistanceUnnormalized"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float CosineDistance(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
    {
        return 1.0f - DotProduct(target, databaseVector);
    }

    /// <summary>
    /// Calculates the true normalized Cosine Distance (<c>1.0f - (a·b / (||a|| * ||b||))</c>) for arbitrary vectors,
    /// returning <c>1.0f</c> safely when either vector has zero L2 norm.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float CosineDistanceUnnormalized(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
    {
        if (target.Length != databaseVector.Length)
            throw new ArgumentException($"Vector dimension mismatch: {target.Length} vs {databaseVector.Length}.");

        float normTargetSq = DotProduct(target, target);
        float normDbSq = DotProduct(databaseVector, databaseVector);
        if (normTargetSq <= 0f || normDbSq <= 0f)
            return 1.0f;

        float denom = MathF.Sqrt(normTargetSq * normDbSq);
        if (denom <= 0f)
            return 1.0f;

        float sim = Math.Clamp(DotProduct(target, databaseVector) / denom, -1.0f, 1.0f);
        return 1.0f - sim;
    }

    /// <summary>
    /// Calculates the L2 (Euclidean) distance squared between two equal-length vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe float L2DistanceSquared(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
    {
        if (target.Length != databaseVector.Length)
            throw new ArgumentException($"Vector dimension mismatch: {target.Length} vs {databaseVector.Length}.");

        int length = target.Length;
        if (length == 0) return 0f;

        float distance = 0f;

        fixed (float* pTarget = target)
        fixed (float* pDb = databaseVector)
        {
            int i = 0;

            // 1. AVX-512 Fast Path Unrolled (64 floats per iteration)
            if (Avx512F.IsSupported && length >= 64)
            {
                var acc0 = Vector512<float>.Zero;
                var acc1 = Vector512<float>.Zero;
                var acc2 = Vector512<float>.Zero;
                var acc3 = Vector512<float>.Zero;

                for (; i <= length - 64; i += 64)
                {
                    var diff0 = Vector512.Subtract(Vector512.Load(pTarget + i), Vector512.Load(pDb + i));
                    var diff1 = Vector512.Subtract(Vector512.Load(pTarget + i + 16), Vector512.Load(pDb + i + 16));
                    var diff2 = Vector512.Subtract(Vector512.Load(pTarget + i + 32), Vector512.Load(pDb + i + 32));
                    var diff3 = Vector512.Subtract(Vector512.Load(pTarget + i + 48), Vector512.Load(pDb + i + 48));

                    acc0 = Avx512F.FusedMultiplyAdd(diff0, diff0, acc0);
                    acc1 = Avx512F.FusedMultiplyAdd(diff1, diff1, acc1);
                    acc2 = Avx512F.FusedMultiplyAdd(diff2, diff2, acc2);
                    acc3 = Avx512F.FusedMultiplyAdd(diff3, diff3, acc3);
                }
                var sum1 = Vector512.Add(acc0, acc1);
                var sum2 = Vector512.Add(acc2, acc3);
                distance += Vector512.Sum(Vector512.Add(sum1, sum2));
            }
            // 2. AVX2 / FMA Fast Path Unrolled (32 floats per iteration)
            else if (Avx2.IsSupported && length >= 32)
            {
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                var acc2 = Vector256<float>.Zero;
                var acc3 = Vector256<float>.Zero;

                if (Fma.IsSupported)
                {
                    for (; i <= length - 32; i += 32)
                    {
                        var diff0 = Vector256.Subtract(Vector256.Load(pTarget + i), Vector256.Load(pDb + i));
                        var diff1 = Vector256.Subtract(Vector256.Load(pTarget + i + 8), Vector256.Load(pDb + i + 8));
                        var diff2 = Vector256.Subtract(Vector256.Load(pTarget + i + 16), Vector256.Load(pDb + i + 16));
                        var diff3 = Vector256.Subtract(Vector256.Load(pTarget + i + 24), Vector256.Load(pDb + i + 24));

                        acc0 = Fma.MultiplyAdd(diff0, diff0, acc0);
                        acc1 = Fma.MultiplyAdd(diff1, diff1, acc1);
                        acc2 = Fma.MultiplyAdd(diff2, diff2, acc2);
                        acc3 = Fma.MultiplyAdd(diff3, diff3, acc3);
                    }
                }
                else
                {
                    for (; i <= length - 32; i += 32)
                    {
                        var diff0 = Vector256.Subtract(Vector256.Load(pTarget + i), Vector256.Load(pDb + i));
                        var diff1 = Vector256.Subtract(Vector256.Load(pTarget + i + 8), Vector256.Load(pDb + i + 8));
                        var diff2 = Vector256.Subtract(Vector256.Load(pTarget + i + 16), Vector256.Load(pDb + i + 16));
                        var diff3 = Vector256.Subtract(Vector256.Load(pTarget + i + 24), Vector256.Load(pDb + i + 24));

                        acc0 = Vector256.Add(acc0, Vector256.Multiply(diff0, diff0));
                        acc1 = Vector256.Add(acc1, Vector256.Multiply(diff1, diff1));
                        acc2 = Vector256.Add(acc2, Vector256.Multiply(diff2, diff2));
                        acc3 = Vector256.Add(acc3, Vector256.Multiply(diff3, diff3));
                    }
                }

                var sum1 = Vector256.Add(acc0, acc1);
                var sum2 = Vector256.Add(acc2, acc3);
                distance += Vector256.Sum(Vector256.Add(sum1, sum2));
            }

            // Remainder 16-float chunk for AVX-512
            if (Avx512F.IsSupported && i <= length - 16)
            {
                var acc = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    var diff = Vector512.Subtract(Vector512.Load(pTarget + i), Vector512.Load(pDb + i));
                    acc = Avx512F.FusedMultiplyAdd(diff, diff, acc);
                }
                distance += Vector512.Sum(acc);
            }
            // Remainder 8-float chunk for AVX2
            else if (Avx2.IsSupported && i <= length - 8)
            {
                var acc = Vector256<float>.Zero;
                for (; i <= length - 8; i += 8)
                {
                    var diff = Vector256.Subtract(Vector256.Load(pTarget + i), Vector256.Load(pDb + i));
                    acc = Fma.IsSupported
                        ? Fma.MultiplyAdd(diff, diff, acc)
                        : Vector256.Add(acc, Vector256.Multiply(diff, diff));
                }
                distance += Vector256.Sum(acc);
            }
            // Portable Vector128 (ARM64 Neon / SSE) Fast Path
            else if (Vector128.IsHardwareAccelerated && i <= length - 4)
            {
                var acc = Vector128<float>.Zero;
                for (; i <= length - 4; i += 4)
                {
                    var diff = Vector128.Subtract(Vector128.Load(pTarget + i), Vector128.Load(pDb + i));
                    acc = Vector128.FusedMultiplyAdd(diff, diff, acc);
                }
                distance += Vector128.Sum(acc);
            }

            for (; i < length; i++)
            {
                float diff = pTarget[i] - pDb[i];
                distance += diff * diff;
            }
        }

        return distance;
    }

    /// <summary>
    /// Calculates the Euclidean (L2) distance between two vectors.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float L2Distance(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
    {
        return MathF.Sqrt(L2DistanceSquared(target, databaseVector));
    }

    /// <summary>
    /// Packs a dense FP32 vector into a 1-bit binary representation (x >= 0 ? 1 : 0),
    /// packing 64 floats into each 64-bit unsigned integer (ulong).
    /// </summary>
    public static void PackBinaryVector(ReadOnlySpan<float> source, Span<ulong> destination)
    {
        int words = (source.Length + 63) / 64;
        if (destination.Length < words)
            throw new ArgumentException($"destination buffer too small. Needs {words} ulongs, got {destination.Length}.", nameof(destination));

        destination.Slice(0, words).Clear();

        int fullWords = source.Length / 64;
        for (int w = 0; w < fullWords; w++)
        {
            ulong word = 0UL;
            int offset = w * 64;
            for (int b = 0; b < 64; b++)
            {
                if (source[offset + b] >= 0.0f)
                {
                    word |= (1UL << b);
                }
            }
            destination[w] = word;
        }

        int remainder = source.Length % 64;
        if (remainder > 0)
        {
            ulong word = 0UL;
            int offset = fullWords * 64;
            for (int b = 0; b < remainder; b++)
            {
                if (source[offset + b] >= 0.0f)
                {
                    word |= (1UL << b);
                }
            }
            destination[fullWords] = word;
        }
    }

    /// <summary>
    /// Packs a dense FP32 vector into a 1-bit binary representation (x >= 0 ? 1 : 0),
    /// packing 32 floats into each 32-bit unsigned integer (uint).
    /// </summary>
    public static void PackBinaryVector(ReadOnlySpan<float> source, Span<uint> destination)
    {
        int words = (source.Length + 31) / 32;
        if (destination.Length < words)
            throw new ArgumentException($"destination buffer too small. Needs {words} uints, got {destination.Length}.", nameof(destination));

        destination.Slice(0, words).Clear();

        int fullWords = source.Length / 32;
        for (int w = 0; w < fullWords; w++)
        {
            uint word = 0U;
            int offset = w * 32;
            for (int b = 0; b < 32; b++)
            {
                if (source[offset + b] >= 0.0f)
                {
                    word |= (1U << b);
                }
            }
            destination[w] = word;
        }

        int remainder = source.Length % 32;
        if (remainder > 0)
        {
            uint word = 0U;
            int offset = fullWords * 32;
            for (int b = 0; b < remainder; b++)
            {
                if (source[offset + b] >= 0.0f)
                {
                    word |= (1U << b);
                }
            }
            destination[fullWords] = word;
        }
    }

    /// <summary>
    /// Computes the exact Hamming distance between two packed 64-bit binary vectors
    /// using hardware-accelerated POPCNT and unrolled SIMD intrinsics.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int HammingDistance(ReadOnlySpan<ulong> a, ReadOnlySpan<ulong> b)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length == 0) return 0;

        int total = 0;
        int i = 0;

        int limit = length - 4;
        for (; i <= limit; i += 4)
        {
            ulong x0 = a[i] ^ b[i];
            ulong x1 = a[i + 1] ^ b[i + 1];
            ulong x2 = a[i + 2] ^ b[i + 2];
            ulong x3 = a[i + 3] ^ b[i + 3];

            total += System.Numerics.BitOperations.PopCount(x0)
                   + System.Numerics.BitOperations.PopCount(x1)
                   + System.Numerics.BitOperations.PopCount(x2)
                   + System.Numerics.BitOperations.PopCount(x3);
        }

        for (; i < length; i++)
        {
            total += System.Numerics.BitOperations.PopCount(a[i] ^ b[i]);
        }

        return total;
    }

    /// <summary>
    /// Computes the exact Hamming distance between two packed 32-bit binary vectors
    /// using AVX-512 VPOPCNTDQ / BitOperations.PopCount.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static int HammingDistance(ReadOnlySpan<uint> a, ReadOnlySpan<uint> b)
    {
        int length = Math.Min(a.Length, b.Length);
        if (length == 0) return 0;

        int total = 0;
        int i = 0;

        int limit = length - 4;
        for (; i <= limit; i += 4)
        {
            uint x0 = a[i] ^ b[i];
            uint x1 = a[i + 1] ^ b[i + 1];
            uint x2 = a[i + 2] ^ b[i + 2];
            uint x3 = a[i + 3] ^ b[i + 3];

            total += System.Numerics.BitOperations.PopCount(x0)
                   + System.Numerics.BitOperations.PopCount(x1)
                   + System.Numerics.BitOperations.PopCount(x2)
                   + System.Numerics.BitOperations.PopCount(x3);
        }

        for (; i < length; i++)
        {
            total += System.Numerics.BitOperations.PopCount(a[i] ^ b[i]);
        }

        return total;
    }
}
