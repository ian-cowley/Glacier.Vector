namespace Glacier.Vector.Storage;

using System;
using System.Runtime.InteropServices;
using System.Threading;
using Glacier.Vector.Compute;

/// <summary>
/// High-performance 1-bit binary vector storage providing 32x memory compression
/// for streaming Hamming candidate filtering.
/// </summary>
public sealed class BinaryVectorStorage : IDisposable
{
    private readonly int _dimensions;
    private readonly int _wordsPerVector;
    private readonly Lock _writeLock = new();
    private ulong[] _data;
    private int _count;
    private bool _disposed;
    private int _disposeFlag;

    public int Dimensions => _dimensions;
    public int WordsPerVector => _wordsPerVector;
    public int Count => Volatile.Read(ref _count);

    public BinaryVectorStorage(int dimensions, int initialCapacity = 1024)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimensions, 1);
        _dimensions = dimensions;
        _wordsPerVector = (dimensions + 63) / 64;
        int capacity = Math.Max(16, initialCapacity);
        _data = new ulong[checked((int)((long)capacity * _wordsPerVector))];
    }

    public void Append(ReadOnlySpan<float> vector)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        if (vector.Length != _dimensions)
            throw new ArgumentException($"Dimension mismatch: {vector.Length} != {_dimensions}.");

        lock (_writeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            EnsureCapacityLocked(_count + 1);
            int offset = checked((int)((long)_count * _wordsPerVector));
            Span<ulong> dst = _data.AsSpan(offset, _wordsPerVector);
            DistanceKernels.PackBinaryVector(vector, dst);
            Volatile.Write(ref _count, _count + 1);
        }
    }

    public void AppendPacked(ReadOnlySpan<ulong> packedVector)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        if (packedVector.Length < _wordsPerVector)
            throw new ArgumentException($"Packed length {packedVector.Length} < {_wordsPerVector}.");

        lock (_writeLock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            EnsureCapacityLocked(_count + 1);
            int offset = checked((int)((long)_count * _wordsPerVector));
            Span<ulong> dst = _data.AsSpan(offset, _wordsPerVector);
            packedVector.Slice(0, _wordsPerVector).CopyTo(dst);

            int rem = _dimensions & 63;
            if (rem != 0)
            {
                dst[_wordsPerVector - 1] &= (1UL << rem) - 1UL;
            }

            Volatile.Write(ref _count, _count + 1);
        }
    }

    public ReadOnlySpan<ulong> GetPackedVector(int index)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
        int count = Volatile.Read(ref _count);
        if ((uint)index >= (uint)count)
            throw new ArgumentOutOfRangeException(nameof(index));

        ulong[] data = Volatile.Read(ref _data);
        int offset = checked((int)((long)index * _wordsPerVector));
        if (Volatile.Read(ref _disposed) || (long)offset + _wordsPerVector > data.Length)
            throw new ObjectDisposedException(nameof(BinaryVectorStorage));

        return data.AsSpan(offset, _wordsPerVector);
    }

    public ReadOnlySpan<uint> GetPackedVectorUInt(int index)
    {
        ReadOnlySpan<ulong> ulongs = GetPackedVector(index);
        return MemoryMarshal.Cast<ulong, uint>(ulongs);
    }

    private void EnsureCapacityLocked(int minCapacity)
    {
        int currentCapacity = _data.Length / _wordsPerVector;
        if (minCapacity > currentCapacity)
        {
            int newCapacity = Math.Max(minCapacity, currentCapacity * 2);
            var newData = new ulong[checked((int)((long)newCapacity * _wordsPerVector))];
            Array.Copy(_data, newData, (long)_count * _wordsPerVector);
            Volatile.Write(ref _data, newData);
        }
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposeFlag, 1) != 0) return;

        lock (_writeLock)
        {
            Volatile.Write(ref _disposed, true);
            Volatile.Write(ref _count, 0);
            Volatile.Write(ref _data, Array.Empty<ulong>());
        }
    }
}
