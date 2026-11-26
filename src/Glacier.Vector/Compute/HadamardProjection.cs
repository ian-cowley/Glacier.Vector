using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Glacier.Vector.Compute;

/// <summary>
/// Per-vector quantization metadata emitted by <see cref="HadamardProjection.RotateAndBinarize"/>
/// for RaBitQ-style asymmetric inner-product and cosine estimation.
/// </summary>
/// <param name="Norm">Exact Euclidean (L2) norm of the source vector.</param>
/// <param name="CentroidInnerProduct">
/// Exact inner product <c>&lt;x_unit, x_bar&gt;</c> between the rotated unit vector and its binary hypercube vertex
/// <c>x_bar ∈ {±1/sqrt(D)}^D</c>, equal to <c>||r||_1 / (||r||_2 * sqrt(D))</c>.
/// </param>
/// <param name="PaddedDimensions">Power-of-2 Hadamard rotation dimension <c>D</c>.</param>
public readonly record struct BinaryRotationMetadata(
    float Norm,
    float CentroidInnerProduct,
    int PaddedDimensions);

/// <summary>
/// High-speed SIMD Fast Walsh-Hadamard Random Projection and RaBitQ-style 1-bit sign-quantized rotation
/// for high-dimensional (<c>512..4096+</c>) vector search and 32x binary compression.
/// </summary>
public static unsafe class HadamardProjection
{
    private const int StackAllocLimit = 4096;

    /// <summary>
    /// Returns the smallest power of 2 greater than or equal to <c>Math.Max(16, dimensions)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetPaddedDimension(int dimensions)
    {
        if (dimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Dimensions must be positive.");
        int minDim = Math.Max(16, dimensions);
        return (int)BitOperations.RoundUpToPowerOf2((uint)minDim);
    }

    /// <summary>
    /// Returns the number of 64-bit <see cref="ulong"/> words required to store packed 1-bit codes
    /// for <paramref name="dimensions"/> (after power-of-2 padding).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int GetPackedWordCount(int dimensions)
    {
        int padded = GetPaddedDimension(dimensions);
        return (padded + 63) >> 6;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsPowerOfTwo(int v) => v > 0 && (v & (v - 1)) == 0;

    /// <summary>
    /// Deterministically fills <paramref name="signs"/> with Rademacher <c>+1.0f</c> / <c>-1.0f</c>
    /// values via SplitMix64 for reproducible orthogonal random projections.
    /// </summary>
    public static void GenerateSigns(Span<float> signs, ulong seed)
    {
        ulong state = seed;
        int i = 0;
        while (i < signs.Length)
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;

            int bits = Math.Min(64, signs.Length - i);
            for (int b = 0; b < bits; b++)
            {
                signs[i++] = ((z >> b) & 1UL) == 0 ? 1.0f : -1.0f;
            }
        }
    }

    /// <summary>
    /// Applies the normalized self-inverse Sylvester Fast Walsh-Hadamard Transform (<c>H * H = I</c>)
    /// in place over a power-of-2 span using AVX-512 / AVX2 SIMD butterflies.
    /// Preserves all inner products and L2 norms by Parseval's theorem (<c>&lt;Hx, Hy&gt; == &lt;x, y&gt;</c>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyFwhtInPlace(Span<float> data)
    {
        int n = data.Length;
        if (n < 2 || !IsPowerOfTwo(n))
            throw new ArgumentException($"FWHT length ({n}) must be a power of 2 >= 2.", nameof(data));

        float scale = (float)(1.0 / Math.Sqrt(n));
        fixed (float* p = data)
        {
            if (Avx512F.IsSupported && n >= 16)
            {
                FwhtAvx512(p, n, scale);
            }
            else if (Avx2.IsSupported && n >= 16)
            {
                FwhtAvx2(p, n, scale);
            }
            else
            {
                FwhtScalar(p, n, scale);
            }
        }
    }

    /// <summary>
    /// Applies an orthogonal Randomized Hadamard Projection <c>destination = H * (signs ⊙ pad(source))</c>.
    /// <paramref name="destination"/> must have a power-of-2 length &gt;= <c>source.Length</c> (and &gt;= 2).
    /// Any trailing positions <c>[source.Length..destination.Length)</c> are zero-padded prior to rotation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyRandomRotation(
        ReadOnlySpan<float> source,
        Span<float> destination,
        ReadOnlySpan<float> signs)
    {
        int padded = destination.Length;
        if (padded < 2 || !IsPowerOfTwo(padded))
            throw new ArgumentException($"Destination length ({padded}) must be a power of 2 >= 2.", nameof(destination));
        if (source.Length > padded)
            throw new ArgumentException($"Source length ({source.Length}) exceeds destination length ({padded}).", nameof(source));
        if (signs.Length < padded)
            throw new ArgumentException($"Signs length ({signs.Length}) must be at least destination length ({padded}).", nameof(signs));

        fixed (float* pSrc = source)
        fixed (float* pDst = destination)
        fixed (float* pSigns = signs)
        {
            MultiplyCopySimd(pSrc, pSigns, pDst, source.Length);
            if (source.Length < padded)
            {
                destination.Slice(source.Length).Clear();
            }
        }

        ApplyFwhtInPlace(destination);
    }

    /// <summary>
    /// Applies the exact inverse of <see cref="ApplyRandomRotation"/>:
    /// <c>x = (signs ⊙ (H * rotated))[0..destination.Length)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static void ApplyRandomRotationInverse(
        ReadOnlySpan<float> rotated,
        Span<float> destination,
        ReadOnlySpan<float> signs)
    {
        int padded = rotated.Length;
        if (padded < 2 || !IsPowerOfTwo(padded))
            throw new ArgumentException($"Rotated length ({padded}) must be a power of 2 >= 2.", nameof(rotated));
        if (destination.Length > padded)
            throw new ArgumentException($"Destination length ({destination.Length}) exceeds rotated length ({padded}).", nameof(destination));
        if (signs.Length < padded)
            throw new ArgumentException($"Signs length ({signs.Length}) must be at least rotated length ({padded}).", nameof(signs));

        float[]? rented = padded > StackAllocLimit ? ArrayPool<float>.Shared.Rent(padded) : null;
        try
        {
            Span<float> work = rented != null ? rented.AsSpan(0, padded) : stackalloc float[padded];
            rotated.CopyTo(work);
            ApplyFwhtInPlace(work);

            fixed (float* pWork = work)
            fixed (float* pSigns = signs)
            fixed (float* pDst = destination)
            {
                MultiplyCopySimd(pWork, pSigns, pDst, destination.Length);
            }
        }
        finally
        {
            if (rented != null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Performs a RaBitQ-style orthogonal Hadamard random rotation followed by 1-bit sign quantization,
    /// packing 64 sign bits (<c>r[i] &gt;= 0 ? 1 : 0</c>) per <see cref="ulong"/> into <paramref name="packedBits"/>
    /// and returning exact norm and centroid inner-product metadata for unbiased asymmetric estimation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static BinaryRotationMetadata RotateAndBinarize(
        ReadOnlySpan<float> source,
        ReadOnlySpan<float> signs,
        Span<ulong> packedBits,
        Span<float> scratchBuffer = default)
    {
        int padded = signs.Length;
        if (padded < 16 || !IsPowerOfTwo(padded))
            throw new ArgumentException($"Signs length ({padded}) must be a power of 2 >= 16.", nameof(signs));
        if (source.Length > padded)
            throw new ArgumentException($"Source length ({source.Length}) exceeds signs length ({padded}).", nameof(source));

        int words = (padded + 63) >> 6;
        if (packedBits.Length < words)
            throw new ArgumentException($"packedBits length ({packedBits.Length}) must be at least {words}.", nameof(packedBits));

        float[]? rented = null;
        Span<float> stackBuf = (scratchBuffer.Length < padded && padded <= StackAllocLimit)
            ? stackalloc float[padded]
            : Span<float>.Empty;

        scoped Span<float> rotated;
        if (scratchBuffer.Length >= padded)
        {
            rotated = scratchBuffer.Slice(0, padded);
        }
        else if (!stackBuf.IsEmpty)
        {
            rotated = stackBuf;
        }
        else
        {
            rented = ArrayPool<float>.Shared.Rent(padded);
            rotated = rented.AsSpan(0, padded);
        }

        try
        {
            ApplyRandomRotation(source, rotated, signs);

            float normSq = 0f;
            float sumAbs = 0f;

            fixed (float* pRot = rotated)
            fixed (ulong* pBits = packedBits)
            {
                ComputeNormAndAbsSumSimd(pRot, padded, out normSq, out sumAbs);
                PackSignBitsSimd(pRot, padded, pBits, words);
            }

            float norm = MathF.Sqrt(normSq);
            float centroidIp = norm > 0f ? sumAbs / (norm * MathF.Sqrt(padded)) : 0f;
            return new BinaryRotationMetadata(norm, centroidIp, padded);
        }
        finally
        {
            if (rented != null) ArrayPool<float>.Shared.Return(rented);
        }
    }

    /// <summary>
    /// Estimates cosine similarity from the Hamming distance between two Hadamard-rotated 1-bit binary codes
    /// using Charikar's unbiased angular estimator <c>cos(pi * hammingDistance / dimensions)</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EstimateCosineFromHamming(int hammingDistance, int dimensions)
    {
        if (dimensions <= 0)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "Dimensions must be positive.");

        float ratio = Math.Clamp((float)hammingDistance / dimensions, 0f, 1f);
        return MathF.Cos(MathF.PI * ratio);
    }

    /// <summary>
    /// Computes Hamming distance between <paramref name="packedA"/> and <paramref name="packedB"/> and
    /// estimates cosine similarity via <see cref="EstimateCosineFromHamming(int, int)"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static float EstimateCosineFromHamming(ReadOnlySpan<ulong> packedA, ReadOnlySpan<ulong> packedB, int dimensions)
    {
        int words = (dimensions + 63) >> 6;
        int hamming = DistanceKernels.HammingDistance(packedA.Slice(0, words), packedB.Slice(0, words));
        return EstimateCosineFromHamming(hamming, dimensions);
    }

    /// <summary>
    /// Computes a RaBitQ-style asymmetric dot-product estimate <c>&lt;x, q&gt;</c> between an unquantized
    /// Hadamard-rotated query <paramref name="rotatedQuery"/> and a 1-bit binary-quantized database vector
    /// (<paramref name="packedDbBits"/> + <paramref name="dbMeta"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float EstimateAsymmetricDotProduct(
        ReadOnlySpan<float> rotatedQuery,
        ReadOnlySpan<ulong> packedDbBits,
        in BinaryRotationMetadata dbMeta)
    {
        int dim = dbMeta.PaddedDimensions;
        if (dim <= 0)
            throw new ArgumentException("Metadata PaddedDimensions must be positive.", nameof(dbMeta));
        if (rotatedQuery.Length < dim)
            throw new ArgumentException($"rotatedQuery length ({rotatedQuery.Length}) must be at least {dim}.", nameof(rotatedQuery));
        int words = (dim + 63) >> 6;
        if (packedDbBits.Length < words)
            throw new ArgumentException($"packedDbBits length ({packedDbBits.Length}) must be at least {words}.", nameof(packedDbBits));
        if (dbMeta.Norm <= 0f || dbMeta.CentroidInnerProduct <= 0f)
            return 0f;

        float signedSum = 0f;
        fixed (float* pQ = rotatedQuery)
        fixed (ulong* pBits = packedDbBits)
        {
            for (int w = 0; w < words; w++)
            {
                ulong word = pBits[w];
                int baseIdx = w << 6;
                int count = Math.Min(64, dim - baseIdx);

                if (Avx2.IsSupported && count == 64)
                {
                    var acc = Vector256<float>.Zero;
                    var bitMasks = Vector256.Create(1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u);
                    var signBit = Vector256.Create(0x80000000u);

                    for (int g = 0; g < 8; g++)
                    {
                        uint byteVal = (uint)((word >> (g * 8)) & 0xFFUL);
                        // bit == 1 -> signMask = 0 (+q); bit == 0 -> signMask = 0x80000000 (-q)
                        var vByte = Vector256.Create(byteVal);
                        var isZero = Avx2.CompareEqual(Avx2.And(vByte, bitMasks), Vector256<uint>.Zero);
                        var flip = Avx2.And(isZero, signBit);
                        var qv = Vector256.Load(pQ + baseIdx + g * 8);
                        var signedQ = Avx.Xor(qv, flip.AsSingle());
                        acc = Avx.Add(acc, signedQ);
                    }
                    signedSum += Vector256.Sum(acc);
                }
                else
                {
                    for (int b = 0; b < count; b++)
                    {
                        float qv = pQ[baseIdx + b];
                        signedSum += (((word >> b) & 1UL) != 0UL) ? qv : -qv;
                    }
                }
            }
        }

        // <x_bar, q_rot> = signedSum / sqrt(D); unbiased <x, q> = (||x|| / <x_unit, x_bar>) * <x_bar, q_rot>
        float barDotQ = signedSum / MathF.Sqrt(dim);
        return (dbMeta.Norm / dbMeta.CentroidInnerProduct) * barDotQ;
    }

    /// <summary>
    /// Estimates cosine similarity between a Hadamard-rotated query vector and a 1-bit binary-quantized
    /// database vector using the RaBitQ asymmetric estimator.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    public static float EstimateAsymmetricCosine(
        ReadOnlySpan<float> rotatedQuery,
        ReadOnlySpan<ulong> packedDbBits,
        in BinaryRotationMetadata dbMeta)
    {
        if (dbMeta.Norm <= 0f || dbMeta.CentroidInnerProduct <= 0f) return 0f;
        float qNormSq = DistanceKernels.DotProduct(rotatedQuery.Slice(0, dbMeta.PaddedDimensions), rotatedQuery.Slice(0, dbMeta.PaddedDimensions));
        if (qNormSq <= 0f) return 0f;

        float estDot = EstimateAsymmetricDotProduct(rotatedQuery, packedDbBits, in dbMeta);
        return Math.Clamp(estDot / (MathF.Sqrt(qNormSq) * dbMeta.Norm), -1f, 1f);
    }

    // =========================================================================
    // INTERNAL SIMD KERNELS
    // =========================================================================

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MultiplyCopySimd(float* src, float* signs, float* dst, int length)
    {
        int i = 0;
        if (Avx512F.IsSupported && length >= 16)
        {
            for (; i <= length - 16; i += 16)
                Vector512.Store(Avx512F.Multiply(Vector512.Load(src + i), Vector512.Load(signs + i)), dst + i);
        }
        else if (Avx2.IsSupported && length >= 8)
        {
            for (; i <= length - 8; i += 8)
                Vector256.Store(Avx.Multiply(Vector256.Load(src + i), Vector256.Load(signs + i)), dst + i);
        }
        for (; i < length; i++)
            dst[i] = src[i] * signs[i];
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void ComputeNormAndAbsSumSimd(float* data, int length, out float normSq, out float sumAbs)
    {
        int i = 0;
        normSq = 0f;
        sumAbs = 0f;

        if (Avx512F.IsSupported && length >= 16)
        {
            var accSq = Vector512<float>.Zero;
            var accAbs = Vector512<float>.Zero;
            var absMask = Vector512.Create(0x7FFFFFFFu);

            for (; i <= length - 16; i += 16)
            {
                var v = Vector512.Load(data + i);
                accSq = Avx512F.FusedMultiplyAdd(v, v, accSq);
                accAbs = Avx512F.Add(accAbs, Avx512F.And(v.AsUInt32(), absMask).AsSingle());
            }
            normSq += Vector512.Sum(accSq);
            sumAbs += Vector512.Sum(accAbs);
        }
        else if (Avx2.IsSupported && length >= 8)
        {
            var accSq = Vector256<float>.Zero;
            var accAbs = Vector256<float>.Zero;
            var absMask = Vector256.Create(0x7FFFFFFFu).AsSingle();

            for (; i <= length - 8; i += 8)
            {
                var v = Vector256.Load(data + i);
                accSq = Fma.IsSupported
                    ? Fma.MultiplyAdd(v, v, accSq)
                    : Avx.Add(accSq, Avx.Multiply(v, v));
                accAbs = Avx.Add(accAbs, Avx.And(v, absMask));
            }
            normSq += Vector256.Sum(accSq);
            sumAbs += Vector256.Sum(accAbs);
        }

        for (; i < length; i++)
        {
            float v = data[i];
            normSq += v * v;
            sumAbs += MathF.Abs(v);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void PackSignBitsSimd(float* data, int length, ulong* dst, int words)
    {
        int fullWords = length >> 6;
        if (Avx2.IsSupported)
        {
            var zero = Vector256<float>.Zero;
            for (int w = 0; w < fullWords; w++)
            {
                float* p = data + (w << 6);
                ulong m0 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p), zero));
                ulong m1 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 8), zero));
                ulong m2 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 16), zero));
                ulong m3 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 24), zero));
                ulong m4 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 32), zero));
                ulong m5 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 40), zero));
                ulong m6 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 48), zero));
                ulong m7 = (uint)Avx.MoveMask(Avx.CompareGreaterThanOrEqual(Vector256.Load(p + 56), zero));

                dst[w] = m0 | (m1 << 8) | (m2 << 16) | (m3 << 24)
                       | (m4 << 32) | (m5 << 40) | (m6 << 48) | (m7 << 56);
            }
        }
        else
        {
            for (int w = 0; w < fullWords; w++)
            {
                ulong word = 0UL;
                float* p = data + (w << 6);
                for (int b = 0; b < 64; b++)
                {
                    if (p[b] >= 0f) word |= 1UL << b;
                }
                dst[w] = word;
            }
        }

        int rem = length & 63;
        if (rem > 0)
        {
            ulong word = 0UL;
            float* p = data + (fullWords << 6);
            for (int b = 0; b < rem; b++)
            {
                if (p[b] >= 0f) word |= 1UL << b;
            }
            dst[fullWords] = word;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FwhtScalar(float* u, int n, float scale)
    {
        for (int h = 1; h < n; h <<= 1)
        {
            int step = h << 1;
            for (int i = 0; i < n; i += step)
            {
                for (int j = i; j < i + h; j++)
                {
                    float a = u[j];
                    float b = u[j + h];
                    u[j] = a + b;
                    u[j + h] = a - b;
                }
            }
        }
        for (int i = 0; i < n; i++) u[i] *= scale;
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FwhtAvx2(float* u, int n, float scale)
    {
        var sign1 = Vector256.Create(0f, -0.0f, 0f, -0.0f, 0f, -0.0f, 0f, -0.0f);
        var sign2 = Vector256.Create(0f, 0f, -0.0f, -0.0f, 0f, 0f, -0.0f, -0.0f);
        var sign4 = Vector256.Create(0f, 0f, 0f, 0f, -0.0f, -0.0f, -0.0f, -0.0f);
        var vScale = Vector256.Create(scale);

        bool scaleInFirstPass = n == 16;
        for (int i = 0; i < n; i += 16)
        {
            var v0 = Vector256.Load(u + i);
            var v1 = Vector256.Load(u + i + 8);

            v0 = Avx.Add(Avx.Xor(v0, sign1), Avx.Permute(v0, 0xB1));
            v1 = Avx.Add(Avx.Xor(v1, sign1), Avx.Permute(v1, 0xB1));

            v0 = Avx.Add(Avx.Xor(v0, sign2), Avx.Permute(v0, 0x4E));
            v1 = Avx.Add(Avx.Xor(v1, sign2), Avx.Permute(v1, 0x4E));

            v0 = Avx.Add(Avx.Xor(v0, sign4), Avx2.Permute2x128(v0, v0, 0x01));
            v1 = Avx.Add(Avx.Xor(v1, sign4), Avx2.Permute2x128(v1, v1, 0x01));

            var t0 = Avx.Add(v0, v1);
            var t1 = Avx.Subtract(v0, v1);

            if (scaleInFirstPass)
            {
                Vector256.Store(Avx.Multiply(t0, vScale), u + i);
                Vector256.Store(Avx.Multiply(t1, vScale), u + i + 8);
            }
            else
            {
                Vector256.Store(t0, u + i);
                Vector256.Store(t1, u + i + 8);
            }
        }

        if (scaleInFirstPass) return;

        for (int h = 16; h < n; h <<= 1)
        {
            int step = h << 1;
            if (step == n)
            {
                for (int j = 0; j < h; j += 16)
                {
                    var a0 = Vector256.Load(u + j);
                    var b0 = Vector256.Load(u + j + h);
                    var a1 = Vector256.Load(u + j + 8);
                    var b1 = Vector256.Load(u + j + h + 8);

                    Vector256.Store(Avx.Multiply(Avx.Add(a0, b0), vScale), u + j);
                    Vector256.Store(Avx.Multiply(Avx.Subtract(a0, b0), vScale), u + j + h);
                    Vector256.Store(Avx.Multiply(Avx.Add(a1, b1), vScale), u + j + 8);
                    Vector256.Store(Avx.Multiply(Avx.Subtract(a1, b1), vScale), u + j + h + 8);
                }
            }
            else
            {
                for (int i = 0; i < n; i += step)
                {
                    for (int j = i; j < i + h; j += 16)
                    {
                        var a0 = Vector256.Load(u + j);
                        var b0 = Vector256.Load(u + j + h);
                        var a1 = Vector256.Load(u + j + 8);
                        var b1 = Vector256.Load(u + j + h + 8);

                        Vector256.Store(Avx.Add(a0, b0), u + j);
                        Vector256.Store(Avx.Subtract(a0, b0), u + j + h);
                        Vector256.Store(Avx.Add(a1, b1), u + j + 8);
                        Vector256.Store(Avx.Subtract(a1, b1), u + j + h + 8);
                    }
                }
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void FwhtAvx512(float* u, int n, float scale)
    {
        var sign1 = Vector512.Create(
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u,
            0x00000000u, 0x80000000u, 0x00000000u, 0x80000000u);
        var sign2 = Vector512.Create(
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x80000000u, 0x80000000u);
        var sign4 = Vector512.Create(
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);
        var sign8 = Vector512.Create(
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u,
            0x80000000u, 0x80000000u, 0x80000000u, 0x80000000u);

        var perm4 = Vector512.Create(4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11);
        var perm8 = Vector512.Create(8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7);
        var vScale = Vector512.Create(scale);

        bool scaleInFirstPass = n == 16;
        for (int i = 0; i < n; i += 16)
        {
            var v = Vector512.Load(u + i);
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign1).AsSingle(), Avx512F.Shuffle(v, v, 0xB1));
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign2).AsSingle(), Avx512F.Shuffle(v, v, 0x4E));
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign4).AsSingle(), Vector512.Shuffle(v, perm4));
            v = Avx512F.Add(Avx512F.Xor(v.AsUInt32(), sign8).AsSingle(), Vector512.Shuffle(v, perm8));

            if (scaleInFirstPass)
                Vector512.Store(Avx512F.Multiply(v, vScale), u + i);
            else
                Vector512.Store(v, u + i);
        }

        if (scaleInFirstPass) return;

        for (int h = 16; h < n; h <<= 1)
        {
            int step = h << 1;
            if (step == n)
            {
                for (int j = 0; j < h; j += 16)
                {
                    var a = Vector512.Load(u + j);
                    var b = Vector512.Load(u + j + h);
                    Vector512.Store(Avx512F.Multiply(Avx512F.Add(a, b), vScale), u + j);
                    Vector512.Store(Avx512F.Multiply(Avx512F.Subtract(a, b), vScale), u + j + h);
                }
            }
            else
            {
                for (int i = 0; i < n; i += step)
                {
                    for (int j = i; j < i + h; j += 16)
                    {
                        var a = Vector512.Load(u + j);
                        var b = Vector512.Load(u + j + h);
                        Vector512.Store(Avx512F.Add(a, b), u + j);
                        Vector512.Store(Avx512F.Subtract(a, b), u + j + h);
                    }
                }
            }
        }
    }
}
