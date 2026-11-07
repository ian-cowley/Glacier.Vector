namespace Glacier.Vector.Storage;

using System;
using System.Runtime.InteropServices;
using Glacier.Vector.Compute;

/// <summary>
/// High-performance 1-bit binary vector storage providing 32x memory compression
/// for streaming Hamming candidate filtering.
/// </summary>
public sealed class BinaryVectorStorage : IDisposable
{
    private readonly int _dimensions;
    private readonly int _wordsPerVector;
    private ulong[] _data;
    private int _count;
    private bool _disposed;

    public int Dimensions => _dimensions;
    public int WordsPerVector => _wordsPerVector;
    public int Count => _count;

    public BinaryVectorStorage(int dimensions, int initialCapacity = 1024)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(dimensions, 1);
        _dimensions = dimensions;
        _wordsPerVector = (dimensions + 63) / 64;
        int capacity = Math.Max(16, initialCapacity);
        _data = new ulong[(long)capacity * _wordsPerVector];
    }

    public void Append(ReadOnlySpan<float> vector)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (vector.Length != _dimensions)
            throw new ArgumentException($"Dimension mismatch: {vector.Length} != {_dimensions}.");

        EnsureCapacity(_count + 1);
        Span<ulong> dst = _data.AsSpan(_count * _wordsPerVector, _wordsPerVector);
        DistanceKernels.PackBinaryVector(vector, dst);
        _count++;
    }

    public void AppendPacked(ReadOnlySpan<ulong> packedVector)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (packedVector.Length < _wordsPerVector)
            throw new ArgumentException($"Packed length {packedVector.Length} < {_wordsPerVector}.");

        EnsureCapacity(_count + 1);
        packedVector.Slice(0, _wordsPerVector).CopyTo(_data.AsSpan(_count * _wordsPerVector, _wordsPerVector));
        _count++;
    }

    public ReadOnlySpan<ulong> GetPackedVector(int index)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)index >= (uint)_count)
            throw new ArgumentOutOfRangeException(nameof(index));
        return _data.AsSpan(index * _wordsPerVector, _wordsPerVector);
    }

    public ReadOnlySpan<uint> GetPackedVectorUInt(int index)
    {
        ReadOnlySpan<ulong> ulongs = GetPackedVector(index);
        return MemoryMarshal.Cast<ulong, uint>(ulongs);
    }

    private void EnsureCapacity(int minCapacity)
    {
        int currentCapacity = _data.Length / _wordsPerVector;
        if (minCapacity > currentCapacity)
        {
            int newCapacity = Math.Max(minCapacity, currentCapacity * 2);
            var newData = new ulong[(long)newCapacity * _wordsPerVector];
            Array.Copy(_data, newData, (long)_count * _wordsPerVector);
            _data = newData;
        }
    }

    public void Dispose()
    {
        _disposed = true;
    }
}
