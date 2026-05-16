using Microsoft.Win32;
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Glacier.Vector.Compute
{
    /// <summary>
    /// Hardware-accelerated kernels for vector distance calculations.
    /// </summary>
    public static class DistanceKernels
    {
        /// <summary>
        /// Calculates the Dot Product of two vectors. 
        /// For normalized embeddings, Dot Product is equivalent to Cosine Similarity.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe float DotProduct(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
        {
            int length = target.Length;
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
                        acc0 = Vector512.Add(acc0, Vector512.Multiply(Vector512.Load(pTarget + i), Vector512.Load(pDb + i)));
                        acc1 = Vector512.Add(acc1, Vector512.Multiply(Vector512.Load(pTarget + i + 16), Vector512.Load(pDb + i + 16)));
                        acc2 = Vector512.Add(acc2, Vector512.Multiply(Vector512.Load(pTarget + i + 32), Vector512.Load(pDb + i + 32)));
                        acc3 = Vector512.Add(acc3, Vector512.Multiply(Vector512.Load(pTarget + i + 48), Vector512.Load(pDb + i + 48)));
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

                // 3. Scalar Cleanup (Catches remaining dimensions, e.g. for D=1536, this is skipped!)
                for (; i < length; i++)
                {
                    dotProduct += pTarget[i] * pDb[i];
                }
            }

            return dotProduct;
        }

        /// <summary>
        /// Calculates the L2 (Euclidean) distance squared between two vectors.
        /// Useful if embeddings are NOT normalized.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static unsafe float L2DistanceSquared(ReadOnlySpan<float> target, ReadOnlySpan<float> databaseVector)
        {
            int length = target.Length;
            float distance = 0f;

            fixed (float* pTarget = target)
            fixed (float* pDb = databaseVector)
            {
                int i = 0;

                if (Avx512F.IsSupported && length >= 16)
                {
                    var acc = Vector512<float>.Zero;
                    for (; i <= length - 16; i += 16)
                    {
                        var diff = Vector512.Subtract(Vector512.Load(pTarget + i), Vector512.Load(pDb + i));
                        acc = Vector512.Add(acc, Vector512.Multiply(diff, diff));
                    }
                    distance += Vector512.Sum(acc);
                }
                else if (Avx2.IsSupported && length >= 8)
                {
                    var acc = Vector256<float>.Zero;
                    for (; i <= length - 8; i += 8)
                    {
                        var diff = Vector256.Subtract(Vector256.Load(pTarget + i), Vector256.Load(pDb + i));
                        acc = Vector256.Add(acc, Vector256.Multiply(diff, diff));
                    }
                    distance += Vector256.Sum(acc);
                }

                for (; i < length; i++)
                {
                    float diff = pTarget[i] - pDb[i];
                    distance += diff * diff;
                }
            }

            return distance;
        }
    }
}