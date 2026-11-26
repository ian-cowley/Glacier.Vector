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

    // =========================================================================
    // QUANTIZED INT8 & 4-BIT PACKED DOT-PRODUCT KERNELS (AVX-512 / AVX2 / SWAR)
    // =========================================================================

    /// <summary>
    /// Computes the exact integer dot product of two signed 8-bit quantized vectors (<c>sum(a[i] * b[i])</c>)
    /// using AVX-512BW, AVX2 (<c>vpmaddwd</c>), or a portable SWAR fallback.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe int DotInt8(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector dimension mismatch: {a.Length} vs {b.Length}.");

        int length = a.Length;
        if (length == 0) return 0;

        int sum = 0;
        fixed (sbyte* pA = a)
        fixed (sbyte* pB = b)
        {
            int i = 0;

            if (Avx512BW.IsSupported && length >= 32)
            {
                var acc = Vector512<int>.Zero;
                for (; i <= length - 32; i += 32)
                {
                    var va = Avx512BW.ConvertToVector512Int16(Vector256.Load(pA + i));
                    var vb = Avx512BW.ConvertToVector512Int16(Vector256.Load(pB + i));
                    acc = Avx512F.Add(acc, Avx512BW.MultiplyAddAdjacent(va, vb));
                }
                sum += Vector512.Sum(acc);
            }
            else if (Avx2.IsSupported && length >= 32)
            {
                var acc0 = Vector256<int>.Zero;
                var acc1 = Vector256<int>.Zero;
                for (; i <= length - 32; i += 32)
                {
                    var va0 = Avx2.ConvertToVector256Int16(Vector128.Load(pA + i));
                    var vb0 = Avx2.ConvertToVector256Int16(Vector128.Load(pB + i));
                    var va1 = Avx2.ConvertToVector256Int16(Vector128.Load(pA + i + 16));
                    var vb1 = Avx2.ConvertToVector256Int16(Vector128.Load(pB + i + 16));
                    acc0 = Avx2.Add(acc0, Avx2.MultiplyAddAdjacent(va0, vb0));
                    acc1 = Avx2.Add(acc1, Avx2.MultiplyAddAdjacent(va1, vb1));
                }
                sum += Vector256.Sum(Avx2.Add(acc0, acc1));
            }

            if (Avx2.IsSupported && i <= length - 16)
            {
                var acc = Vector256<int>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    var va = Avx2.ConvertToVector256Int16(Vector128.Load(pA + i));
                    var vb = Avx2.ConvertToVector256Int16(Vector128.Load(pB + i));
                    acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(va, vb));
                }
                sum += Vector256.Sum(acc);
            }

            if (i < length)
            {
                sum += DotInt8SwarPtr(pA + i, pB + i, length - i);
            }
        }

        return sum;
    }

    /// <summary>
    /// Portable 32-bit SWAR fallback for signed int8 dot product (exposed internally for parity tests).
    /// </summary>
    internal static unsafe int DotInt8Swar(ReadOnlySpan<sbyte> a, ReadOnlySpan<sbyte> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector dimension mismatch: {a.Length} vs {b.Length}.");
        if (a.IsEmpty) return 0;

        fixed (sbyte* pA = a)
        fixed (sbyte* pB = b)
        {
            return DotInt8SwarPtr(pA, pB, a.Length);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe int DotInt8SwarPtr(sbyte* pA, sbyte* pB, int length)
    {
        int sum = 0;
        int i = 0;
        int limit = length - 4;
        for (; i <= limit; i += 4)
        {
            uint wa = *(uint*)(pA + i);
            uint wb = *(uint*)(pB + i);

            int a0 = (sbyte)wa;
            int a1 = (sbyte)(wa >> 8);
            int a2 = (sbyte)(wa >> 16);
            int a3 = (sbyte)(wa >> 24);

            int b0 = (sbyte)wb;
            int b1 = (sbyte)(wb >> 8);
            int b2 = (sbyte)(wb >> 16);
            int b3 = (sbyte)(wb >> 24);

            sum += a0 * b0 + a1 * b1 + a2 * b2 + a3 * b3;
        }
        for (; i < length; i++)
        {
            sum += pA[i] * pB[i];
        }
        return sum;
    }

    /// <summary>
    /// Computes the horizontal sum of a signed int8 vector using AVX-512BW, AVX2, or portable 32-bit SWAR byte summation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe int SumInt8(ReadOnlySpan<sbyte> v)
    {
        int length = v.Length;
        if (length == 0) return 0;

        int sum = 0;
        fixed (sbyte* p = v)
        {
            int i = 0;
            if (Avx512BW.IsSupported && length >= 32)
            {
                var ones = Vector512.Create((short)1);
                var acc = Vector512<int>.Zero;
                for (; i <= length - 32; i += 32)
                {
                    var v16 = Avx512BW.ConvertToVector512Int16(Vector256.Load(p + i));
                    acc = Avx512F.Add(acc, Avx512BW.MultiplyAddAdjacent(v16, ones));
                }
                sum += Vector512.Sum(acc);
            }
            else if (Avx2.IsSupported && length >= 16)
            {
                var ones = Vector256.Create((short)1);
                var acc = Vector256<int>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    var v16 = Avx2.ConvertToVector256Int16(Vector128.Load(p + i));
                    acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(v16, ones));
                }
                sum += Vector256.Sum(acc);
            }

            // Portable 32-bit SWAR horizontal byte sum: flip MSB to map [-128,127] -> [0,255], sum 4 bytes at once
            int limit = length - 4;
            for (; i <= limit; i += 4)
            {
                uint u = (*(uint*)(p + i)) ^ 0x80808080u;
                uint pairSum = (u & 0x00FF00FFu) + ((u >> 8) & 0x00FF00FFu);
                int blockSum = (int)((pairSum * 0x00010001u) >> 16) - 512;
                sum += blockSum;
            }
            for (; i < length; i++)
            {
                sum += p[i];
            }
        }
        return sum;
    }

    /// <summary>
    /// Computes the symmetric int8 quantized dot product <c>scaleA * scaleB * sum(a[i] * b[i])</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float DotInt8Quantized(ReadOnlySpan<sbyte> a, float scaleA, ReadOnlySpan<sbyte> b, float scaleB)
    {
        return DotInt8(a, b) * (scaleA * scaleB);
    }

    /// <summary>
    /// Computes the asymmetric affine int8 quantized dot product
    /// <c>scaleA * scaleB * sum((a[i] - zeroPointA) * (b[i] - zeroPointB))</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotInt8Quantized(
        ReadOnlySpan<sbyte> a,
        float scaleA,
        int zeroPointA,
        ReadOnlySpan<sbyte> b,
        float scaleB,
        int zeroPointB)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector dimension mismatch: {a.Length} vs {b.Length}.");
        if (a.IsEmpty) return 0f;

        long dot = DotInt8(a, b);
        if (zeroPointB != 0) dot -= (long)zeroPointB * SumInt8(a);
        if (zeroPointA != 0) dot -= (long)zeroPointA * SumInt8(b);
        if (zeroPointA != 0 && zeroPointB != 0) dot += (long)a.Length * zeroPointA * zeroPointB;

        return dot * (scaleA * scaleB);
    }

    /// <summary>
    /// Computes the asymmetric dot product between an FP32 query vector and a signed int8 quantized vector
    /// (<c>scale * sum(query[i] * quantized[i])</c>) using AVX-512F / AVX2+FMA.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe float DotFloatInt8(ReadOnlySpan<float> query, ReadOnlySpan<sbyte> quantized, float scale)
    {
        if (query.Length != quantized.Length)
            throw new ArgumentException($"Vector dimension mismatch: {query.Length} vs {quantized.Length}.");

        int length = query.Length;
        if (length == 0) return 0f;

        float dot = 0f;
        fixed (float* pQ = query)
        fixed (sbyte* pI = quantized)
        {
            int i = 0;
            if (Avx512F.IsSupported && length >= 16)
            {
                var acc = Vector512<float>.Zero;
                for (; i <= length - 16; i += 16)
                {
                    var vf = Avx512F.ConvertToVector512Single(Avx512F.ConvertToVector512Int32(Vector128.Load(pI + i)));
                    acc = Avx512F.FusedMultiplyAdd(Vector512.Load(pQ + i), vf, acc);
                }
                dot += Vector512.Sum(acc);
            }
            else if (Avx2.IsSupported && length >= 16)
            {
                var acc0 = Vector256<float>.Zero;
                var acc1 = Vector256<float>.Zero;
                if (Fma.IsSupported)
                {
                    for (; i <= length - 16; i += 16)
                    {
                        var vf0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pI + i));
                        var vf1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pI + i + 8));
                        acc0 = Fma.MultiplyAdd(Vector256.Load(pQ + i), vf0, acc0);
                        acc1 = Fma.MultiplyAdd(Vector256.Load(pQ + i + 8), vf1, acc1);
                    }
                }
                else
                {
                    for (; i <= length - 16; i += 16)
                    {
                        var vf0 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pI + i));
                        var vf1 = Avx.ConvertToVector256Single(Avx2.ConvertToVector256Int32(pI + i + 8));
                        acc0 = Avx.Add(acc0, Avx.Multiply(Vector256.Load(pQ + i), vf0));
                        acc1 = Avx.Add(acc1, Avx.Multiply(Vector256.Load(pQ + i + 8), vf1));
                    }
                }
                dot += Vector256.Sum(Avx.Add(acc0, acc1));
            }

            for (; i < length; i++)
            {
                dot += pQ[i] * pI[i];
            }
        }

        return dot * scale;
    }

    /// <summary>
    /// Packs unsigned 4-bit values (<c>0..15</c>) into dense bytes (2 nibbles per byte:
    /// low nibble = even index, high nibble = odd index).
    /// </summary>
    public static void PackUInt4(ReadOnlySpan<byte> nibbles, Span<byte> destination)
    {
        int packedLen = (nibbles.Length + 1) >> 1;
        if (destination.Length < packedLen)
            throw new ArgumentException($"Destination length ({destination.Length}) must be at least {packedLen}.", nameof(destination));

        int fullBytes = nibbles.Length >> 1;
        for (int i = 0; i < fullBytes; i++)
        {
            destination[i] = (byte)((nibbles[i << 1] & 0x0F) | ((nibbles[(i << 1) + 1] & 0x0F) << 4));
        }

        if ((nibbles.Length & 1) != 0)
        {
            destination[fullBytes] = (byte)(nibbles[nibbles.Length - 1] & 0x0F);
        }
    }

    /// <summary>
    /// Computes the exact integer dot product of two 4-bit packed unsigned vectors (2 nibbles/byte:
    /// low nibble = even index, high nibble = odd index) across <paramref name="dimensions"/> elements
    /// using AVX-512BW, AVX2 (<c>vpmaddubsw</c> + <c>vpmaddwd</c>), or a portable 32-bit SWAR fallback.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe int DotUInt4Packed(ReadOnlySpan<byte> aPacked, ReadOnlySpan<byte> bPacked, int dimensions)
    {
        if (dimensions < 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Dimensions must be non-negative.");
        if (dimensions == 0) return 0;

        int requiredBytes = (dimensions + 1) >> 1;
        if (aPacked.Length < requiredBytes || bPacked.Length < requiredBytes)
            throw new ArgumentException($"Packed spans must contain at least {requiredBytes} bytes for {dimensions} dimensions.");

        int fullBytes = dimensions >> 1;
        int sum = 0;

        fixed (byte* pA = aPacked)
        fixed (byte* pB = bPacked)
        {
            int i = 0;

            if (Avx512BW.IsSupported && fullBytes >= 64)
            {
                var mask0F = Vector512.Create((byte)0x0F);
                var ones16 = Vector512.Create((short)1);
                var acc = Vector512<int>.Zero;

                for (; i <= fullBytes - 64; i += 64)
                {
                    var va = Vector512.Load(pA + i);
                    var vb = Vector512.Load(pB + i);

                    var aLo = Vector512.BitwiseAnd(va, mask0F);
                    var bLo = Vector512.BitwiseAnd(vb, mask0F);
                    var aHi = Vector512.BitwiseAnd(Avx512BW.ShiftRightLogical(va.AsUInt16(), 4).AsByte(), mask0F);
                    var bHi = Vector512.BitwiseAnd(Avx512BW.ShiftRightLogical(vb.AsUInt16(), 4).AsByte(), mask0F);

                    var pLo = Avx512BW.MultiplyAddAdjacent(aLo, bLo.AsSByte());
                    var pHi = Avx512BW.MultiplyAddAdjacent(aHi, bHi.AsSByte());
                    var pSum16 = Avx512BW.Add(pLo, pHi);
                    acc = Avx512F.Add(acc, Avx512BW.MultiplyAddAdjacent(pSum16, ones16));
                }
                sum += Vector512.Sum(acc);
            }

            if (Avx2.IsSupported && i <= fullBytes - 32)
            {
                var mask0F = Vector256.Create((byte)0x0F);
                var ones16 = Vector256.Create((short)1);
                var acc = Vector256<int>.Zero;

                for (; i <= fullBytes - 32; i += 32)
                {
                    var va = Vector256.Load(pA + i);
                    var vb = Vector256.Load(pB + i);

                    var aLo = Avx2.And(va, mask0F);
                    var bLo = Avx2.And(vb, mask0F);
                    var aHi = Avx2.And(Avx2.ShiftRightLogical(va.AsUInt16(), 4).AsByte(), mask0F);
                    var bHi = Avx2.And(Avx2.ShiftRightLogical(vb.AsUInt16(), 4).AsByte(), mask0F);

                    var pLo = Avx2.MultiplyAddAdjacent(aLo, bLo.AsSByte());
                    var pHi = Avx2.MultiplyAddAdjacent(aHi, bHi.AsSByte());
                    var pSum16 = Avx2.Add(pLo, pHi);
                    acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(pSum16, ones16));
                }
                sum += Vector256.Sum(acc);
            }

            if (i < fullBytes)
            {
                sum += DotUInt4FullBytesSwarPtr(pA + i, pB + i, fullBytes - i);
            }

            if ((dimensions & 1) != 0)
            {
                sum += (pA[fullBytes] & 0x0F) * (pB[fullBytes] & 0x0F);
            }
        }

        return sum;
    }

    /// <summary>
    /// Portable 32-bit SWAR fallback for 4-bit packed dot product (exposed internally for parity tests).
    /// </summary>
    internal static unsafe int DotUInt4PackedSwar(ReadOnlySpan<byte> aPacked, ReadOnlySpan<byte> bPacked, int dimensions)
    {
        if (dimensions < 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Dimensions must be non-negative.");
        if (dimensions == 0) return 0;

        int requiredBytes = (dimensions + 1) >> 1;
        if (aPacked.Length < requiredBytes || bPacked.Length < requiredBytes)
            throw new ArgumentException($"Packed spans must contain at least {requiredBytes} bytes for {dimensions} dimensions.");

        int fullBytes = dimensions >> 1;
        fixed (byte* pA = aPacked)
        fixed (byte* pB = bPacked)
        {
            int sum = DotUInt4FullBytesSwarPtr(pA, pB, fullBytes);
            if ((dimensions & 1) != 0)
            {
                sum += (pA[fullBytes] & 0x0F) * (pB[fullBytes] & 0x0F);
            }
            return sum;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe int DotUInt4FullBytesSwarPtr(byte* pA, byte* pB, int fullBytes)
    {
        int sum = 0;
        int i = 0;
        int limit = fullBytes - 4;
        for (; i <= limit; i += 4)
        {
            uint wa = *(uint*)(pA + i);
            uint wb = *(uint*)(pB + i);

            uint aLo = wa & 0x0F0F0F0Fu;
            uint bLo = wb & 0x0F0F0F0Fu;
            uint aHi = (wa >> 4) & 0x0F0F0F0Fu;
            uint bHi = (wb >> 4) & 0x0F0F0F0Fu;

            sum += (int)(
                (aLo & 0xFFu) * (bLo & 0xFFu) +
                ((aLo >> 8) & 0xFFu) * ((bLo >> 8) & 0xFFu) +
                ((aLo >> 16) & 0xFFu) * ((bLo >> 16) & 0xFFu) +
                (aLo >> 24) * (bLo >> 24) +
                (aHi & 0xFFu) * (bHi & 0xFFu) +
                ((aHi >> 8) & 0xFFu) * ((bHi >> 8) & 0xFFu) +
                ((aHi >> 16) & 0xFFu) * ((bHi >> 16) & 0xFFu) +
                (aHi >> 24) * (bHi >> 24));
        }
        for (; i < fullBytes; i++)
        {
            byte ba = pA[i];
            byte bb = pB[i];
            sum += (ba & 0x0F) * (bb & 0x0F) + (ba >> 4) * (bb >> 4);
        }
        return sum;
    }

    /// <summary>
    /// Computes the horizontal sum of all 4-bit unsigned nibbles in a packed byte span
    /// using AVX2 or portable 32-bit SWAR (<c>(((w &amp; 0x0F0F0F0Fu) + ((w &gt;&gt; 4) &amp; 0x0F0F0F0Fu)) * 0x01010101u) &gt;&gt; 24</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static unsafe int SumUInt4Packed(ReadOnlySpan<byte> packed, int dimensions)
    {
        if (dimensions < 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Dimensions must be non-negative.");
        if (dimensions == 0) return 0;

        int requiredBytes = (dimensions + 1) >> 1;
        if (packed.Length < requiredBytes)
            throw new ArgumentException($"Packed span must contain at least {requiredBytes} bytes for {dimensions} dimensions.", nameof(packed));

        int fullBytes = dimensions >> 1;
        int sum = 0;

        fixed (byte* p = packed)
        {
            int i = 0;
            if (Avx2.IsSupported && fullBytes >= 32)
            {
                var mask0F = Vector256.Create((byte)0x0F);
                var zero = Vector256<byte>.Zero;
                var acc = Vector256<ulong>.Zero;

                for (; i <= fullBytes - 32; i += 32)
                {
                    var v = Vector256.Load(p + i);
                    var lo = Avx2.And(v, mask0F);
                    var hi = Avx2.And(Avx2.ShiftRightLogical(v.AsUInt16(), 4).AsByte(), mask0F);
                    var byteSums = Avx2.Add(lo, hi); // max 30 per byte, zero overflow
                    acc = Avx2.Add(acc, Avx2.SumAbsoluteDifferences(byteSums, zero).AsUInt64());
                }
                sum += (int)Vector256.Sum(acc);
            }

            // Portable 32-bit SWAR horizontal nibble sum (8 nibbles = 4 bytes per iter, max byte sum = 120)
            int limit = fullBytes - 4;
            for (; i <= limit; i += 4)
            {
                uint w = *(uint*)(p + i);
                uint nibblePairs = (w & 0x0F0F0F0Fu) + ((w >> 4) & 0x0F0F0F0Fu);
                sum += (int)((nibblePairs * 0x01010101u) >> 24);
            }
            for (; i < fullBytes; i++)
            {
                byte b = p[i];
                sum += (b & 0x0F) + (b >> 4);
            }
            if ((dimensions & 1) != 0)
            {
                sum += p[fullBytes] & 0x0F;
            }
        }

        return sum;
    }

    /// <summary>
    /// Computes the affine quantized dot product between two 4-bit packed vectors
    /// <c>sum((minA + scaleA * qA[i]) * (minB + scaleB * qB[i]))</c> using SIMD/SWAR nibble dot and sum kernels.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float DotUInt4PackedQuantized(
        ReadOnlySpan<byte> aPacked,
        float scaleA,
        float minA,
        ReadOnlySpan<byte> bPacked,
        float scaleB,
        float minB,
        int dimensions)
    {
        if (dimensions == 0) return 0f;

        int dotQ = DotUInt4Packed(aPacked, bPacked, dimensions);
        double result = (double)scaleA * scaleB * dotQ;

        if (minA != 0f)
        {
            int sumB = SumUInt4Packed(bPacked, dimensions);
            result += (double)minA * scaleB * sumB;
        }
        if (minB != 0f)
        {
            int sumA = SumUInt4Packed(aPacked, dimensions);
            result += (double)minB * scaleA * sumA;
        }
        if (minA != 0f && minB != 0f)
        {
            result += (double)dimensions * minA * minB;
        }

        return (float)result;
    }
}
