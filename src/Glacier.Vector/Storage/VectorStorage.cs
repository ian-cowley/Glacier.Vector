using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace Glacier.Vector.Storage
{
    /// <summary>
    /// Core abstraction for vector storage, allowing seamless switching 
    /// between RAM and Disk-backed (Memory-Mapped or Paged) storage.
    /// </summary>
    public interface IVectorStorage : IDisposable
    {
        int Dimensions { get; }
        int Count { get; }

        /// <summary>
        /// Unique runtime identifier for safe GPU VRAM cache keying (0 indicates non-cacheable storage).
        /// </summary>
        long StorageId => 0;

        /// <summary>
        /// Monotonically increasing mutation version incremented whenever vectors are appended or invalidated.
        /// </summary>
        long Version => 0;

        /// <summary>
        /// Retrieves a vector by its zero-based index without allocating memory.
        /// </summary>
        ReadOnlySpan<float> GetVector(int index);

        /// <summary>
        /// Appends a new vector to the storage.
        /// </summary>
        void Append(ReadOnlySpan<float> vector);

        /// <summary>
        /// Total number of contiguous memory chunks in storage.
        /// </summary>
        int ChunkCount => 1;

        /// <summary>
        /// Retrieves the contiguous memory span of a chunk for GPU / SIMD block operations.
        /// </summary>
        ReadOnlySpan<float> GetChunkSpan(int chunkIndex) => ReadOnlySpan<float>.Empty;

        /// <summary>
        /// Number of valid vectors in the specified chunk.
        /// </summary>
        int GetChunkVectorCount(int chunkIndex) => Count;
    }

    internal static class StorageIdGenerator
    {
        private static long s_nextId;
        public static long Next() => Interlocked.Increment(ref s_nextId);
    }

    /// <summary>
    /// Pure in-memory storage using a chunked array list.
    /// Prevents massive GC pauses associated with resizing giant arrays.
    /// </summary>
    public class InMemoryVectorStorage : IVectorStorage
    {
        public int Dimensions { get; }
        public int Count => Volatile.Read(ref _count);
        public long StorageId { get; } = StorageIdGenerator.Next();
        public long Version => Volatile.Read(ref _version);

        private int _count;
        private long _version = 1;
        private readonly int _vectorsPerChunk;
        private float[]?[] _chunks;
        private int _currentChunkIndex;
        private int _currentChunkVectorCount;
        private readonly object _appendLock = new();

        public int ChunkCount => Volatile.Read(ref _currentChunkIndex) + 1;

        public ReadOnlySpan<float> GetChunkSpan(int chunkIndex)
        {
            if ((uint)chunkIndex >= (uint)ChunkCount) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            float[]?[] chunks = Volatile.Read(ref _chunks);
            float[]? chunk = Volatile.Read(ref chunks[chunkIndex]);
            if (chunk == null) return ReadOnlySpan<float>.Empty;
            int count = GetChunkVectorCount(chunkIndex);
            return new ReadOnlySpan<float>(chunk, 0, count * Dimensions);
        }

        public int GetChunkVectorCount(int chunkIndex)
        {
            int currentChunk = Volatile.Read(ref _currentChunkIndex);
            if ((uint)chunkIndex > (uint)currentChunk) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            if (chunkIndex == currentChunk) return Volatile.Read(ref _currentChunkVectorCount);
            return _vectorsPerChunk;
        }

        // Default to ~65k vectors per chunk. 
        // For 1536 dims, this is ~402 MB per chunk.
        public InMemoryVectorStorage(int dimensions, int vectorsPerChunk = 65536)
        {
            if (dimensions <= 0) throw new ArgumentOutOfRangeException(nameof(dimensions));
            if (vectorsPerChunk <= 0) throw new ArgumentOutOfRangeException(nameof(vectorsPerChunk));

            Dimensions = dimensions;
            _vectorsPerChunk = vectorsPerChunk;
            _chunks = new float[1024][];
            _chunks[0] = new float[vectorsPerChunk * dimensions];
            _currentChunkIndex = 0;
            _currentChunkVectorCount = 0;
            _count = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetVector(int index)
        {
            if ((uint)index >= (uint)Volatile.Read(ref _count))
                throw new ArgumentOutOfRangeException(nameof(index));

            int chunkIdx = index / _vectorsPerChunk;
            int offsetInChunk = (index % _vectorsPerChunk) * Dimensions;
            float[]?[] chunks = Volatile.Read(ref _chunks);
            float[]? chunk = Volatile.Read(ref chunks[chunkIdx]);
            if (chunk == null)
                throw new ObjectDisposedException(nameof(InMemoryVectorStorage));

            return new ReadOnlySpan<float>(chunk, offsetInChunk, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            lock (_appendLock)
            {
                if (_currentChunkVectorCount == _vectorsPerChunk)
                {
                    int nextIndex = _currentChunkIndex + 1;
                    if (nextIndex >= _chunks.Length)
                    {
                        var newChunks = new float[_chunks.Length * 2][];
                        Array.Copy(_chunks, newChunks, _chunks.Length);
                        Volatile.Write(ref _chunks, newChunks);
                    }

                    var newChunk = new float[_vectorsPerChunk * Dimensions];
                    vector.CopyTo(new Span<float>(newChunk, 0, Dimensions));

                    // Publish populated chunk and reset per-chunk count BEFORE advancing _currentChunkIndex
                    Volatile.Write(ref _chunks[nextIndex], newChunk);
                    Volatile.Write(ref _currentChunkVectorCount, 1);
                    Volatile.Write(ref _currentChunkIndex, nextIndex);
                    Volatile.Write(ref _count, _count + 1);
                    Interlocked.Increment(ref _version);
                    return;
                }

                float[]? currentChunk = _chunks[_currentChunkIndex];
                if (currentChunk == null)
                    throw new ObjectDisposedException(nameof(InMemoryVectorStorage));

                int offset = _currentChunkVectorCount * Dimensions;
                vector.CopyTo(new Span<float>(currentChunk, offset, Dimensions));

                Volatile.Write(ref _currentChunkVectorCount, _currentChunkVectorCount + 1);
                Volatile.Write(ref _count, _count + 1);
                Interlocked.Increment(ref _version);
            }
        }

        public void Dispose()
        {
            lock (_appendLock)
            {
                Array.Clear(_chunks, 0, _chunks.Length);
                Volatile.Write(ref _count, 0);
                Volatile.Write(ref _currentChunkVectorCount, 0);
                Volatile.Write(ref _currentChunkIndex, 0);
                Interlocked.Increment(ref _version);
            }
        }
    }

    /// <summary>
    /// Disk-backed storage using Memory-Mapped Files.
    /// Allows querying databases larger than physical RAM by relying on OS paging.
    /// </summary>
    public unsafe class MmfVectorStorage : IVectorStorage
    {
        public int Dimensions { get; }
        public int Count => Volatile.Read(ref _count);
        public long StorageId { get; } = StorageIdGenerator.Next();
        public long Version => Volatile.Read(ref _version);

        private int _count;
        private long _version = 1;
        private readonly string _filePath;
        private readonly string _metaPath;
        private FileStream _fileStream = null!;
        private MemoryMappedFile _mmf = null!;
        private MemoryMappedViewAccessor _accessor = null!;
        private IntPtr _basePointer;
        private bool _disposed;

        // Start with a 64MB mapping segment and grow in 64MB increments to stay gentle on CI runners while amortizing remaps
        private const long GrowSize = 64L * 1024 * 1024;
        private long _currentCapacityBytes;
        private readonly List<MemoryMappedFile> _oldMmfs = new();
        private readonly List<MemoryMappedViewAccessor> _oldAccessors = new();
        private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);

        public MmfVectorStorage(string filePath, int dimensions)
        {
            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentException("File path is required.", nameof(filePath));
            if (dimensions <= 0) throw new ArgumentOutOfRangeException(nameof(dimensions));

            Dimensions = dimensions;
            _filePath = filePath;
            _metaPath = filePath + ".meta";

            bool fileExists = File.Exists(filePath);
            _fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);

            long bytesPerVector = (long)dimensions * sizeof(float);
            if (fileExists && _fileStream.Length > 0)
            {
                long fileLength = _fileStream.Length;
                long maxVectorsInFile = fileLength / bytesPerVector;

                if (File.Exists(_metaPath))
                {
                    try
                    {
                        byte[] metaBytes = File.ReadAllBytes(_metaPath);
                        if (metaBytes.Length >= sizeof(int))
                        {
                            int recordedCount = BitConverter.ToInt32(metaBytes, 0);
                            _count = (int)Math.Clamp(recordedCount, 0, maxVectorsInFile);
                        }
                        else
                        {
                            _count = (int)maxVectorsInFile;
                        }
                    }
                    catch
                    {
                        _count = (int)maxVectorsInFile;
                    }
                }
                else
                {
                    _count = (int)maxVectorsInFile;
                }

                _currentCapacityBytes = Math.Max(fileLength, bytesPerVector);
                if (_fileStream.Length < _currentCapacityBytes)
                {
                    _fileStream.SetLength(_currentCapacityBytes);
                }
            }
            else
            {
                _count = 0;
                _currentCapacityBytes = Math.Max(GrowSize, bytesPerVector);
                _fileStream.SetLength(_currentCapacityBytes);
                WriteCountMetadata(0);
            }

            InitializeMapping();
        }

        private void WriteCountMetadata(int count)
        {
            try
            {
                Span<byte> buf = stackalloc byte[sizeof(int)];
                BitConverter.TryWriteBytes(buf, count);
                using var fs = new FileStream(_metaPath, FileMode.Create, FileAccess.Write, FileShare.ReadWrite);
                fs.Write(buf);
            }
            catch
            {
                // Best-effort crash recovery sidecar
            }
        }

        private void InitializeMapping()
        {
            _mmf = MemoryMappedFile.CreateFromFile(_fileStream, null, _currentCapacityBytes, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, false);
            _accessor = _mmf.CreateViewAccessor(0, _currentCapacityBytes, MemoryMappedFileAccess.ReadWrite);

            byte* newPtr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref newPtr);
            Volatile.Write(ref _basePointer, (IntPtr)newPtr);
        }

        public int ChunkCount => 1;

        public ReadOnlySpan<float> GetChunkSpan(int chunkIndex)
        {
            if (chunkIndex != 0) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            int count = Volatile.Read(ref _count);
            byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
            if (basePtr == null || count == 0) return ReadOnlySpan<float>.Empty;
            return new ReadOnlySpan<float>((float*)basePtr, count * Dimensions);
        }

        public int GetChunkVectorCount(int chunkIndex)
        {
            if (chunkIndex != 0) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            return Volatile.Read(ref _count);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetVector(int index)
        {
            if ((uint)index >= (uint)Volatile.Read(ref _count))
                throw new ArgumentOutOfRangeException(nameof(index));

            byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
            if (basePtr == null)
            {
                _rwLock.EnterReadLock();
                try
                {
                    basePtr = (byte*)Volatile.Read(ref _basePointer);
                }
                finally
                {
                    _rwLock.ExitReadLock();
                }
            }

            if (basePtr == null)
                throw new ObjectDisposedException(nameof(MmfVectorStorage), "Storage is unmapped or disposed.");

            float* floatPtr = (float*)basePtr;
            long offset = (long)index * Dimensions;
            return new ReadOnlySpan<float>(floatPtr + offset, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            _rwLock.EnterWriteLock();
            try
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                long bytesRequired = (long)(_count + 1) * Dimensions * sizeof(float);

                if (bytesRequired > _currentCapacityBytes)
                {
                    ExpandMapping(bytesRequired);
                }

                byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
                float* floatPtr = (float*)basePtr;
                long offset = (long)_count * Dimensions;

                fixed (float* src = vector)
                {
                    Unsafe.CopyBlockUnaligned(
                        floatPtr + offset,
                        src,
                        (uint)(Dimensions * sizeof(float)));
                }

                Volatile.Write(ref _count, _count + 1);
                Interlocked.Increment(ref _version);
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }

        private void ExpandMapping(long minRequiredBytes)
        {
            if (_accessor != null) _oldAccessors.Add(_accessor);
            if (_mmf != null) _oldMmfs.Add(_mmf);

            while (_currentCapacityBytes < minRequiredBytes)
            {
                _currentCapacityBytes += GrowSize;
            }
            _fileStream.SetLength(_currentCapacityBytes);

            InitializeMapping();
        }

        public void Dispose()
        {
            if (_disposed) return;

            _rwLock.EnterWriteLock();
            try
            {
                if (_disposed) return;
                _disposed = true;

                if (_basePointer != IntPtr.Zero)
                {
                    _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
                    Volatile.Write(ref _basePointer, IntPtr.Zero);
                }
                _accessor?.Dispose();
                _mmf?.Dispose();

                foreach (var acc in _oldAccessors)
                {
                    try { acc.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
                    acc.Dispose();
                }
                _oldAccessors.Clear();

                foreach (var m in _oldMmfs)
                {
                    m.Dispose();
                }
                _oldMmfs.Clear();

                long exactBytes = (long)_count * Dimensions * sizeof(float);
                try { _fileStream?.SetLength(exactBytes); } catch { }
                _fileStream?.Dispose();

                if (File.Exists(_metaPath))
                {
                    try { File.Delete(_metaPath); } catch { }
                }

                Interlocked.Increment(ref _version);
            }
            finally
            {
                _rwLock.ExitWriteLock();
                _rwLock.Dispose();
            }
        }
    }

    /// <summary>
    /// Disk-backed storage using a software-controlled LRU page cache.
    /// Enforces a strict memory limit, making it ideal for low-memory or embedded systems.
    /// </summary>
    public class PagedVectorStorage : IVectorStorage
    {
        public int Dimensions { get; }
        public int Count => Volatile.Read(ref _count);
        public int ChunkCount => 0;

        private int _count;
        private readonly string _filePath;
        private readonly FileStream _fileStream;
        private readonly object _lock = new object();

        // Software page cache parameters
        private readonly int _pageSize;
        private readonly byte[][] _pages;
        private readonly long[] _pageTags;
        private readonly long[] _pageAccess;
        private long _accessCounter = 0;

        [ThreadStatic]
        private static float[][]? _threadLocalBuffers;
        [ThreadStatic]
        private static int _bufferIndex;

        public PagedVectorStorage(string filePath, int dimensions, int maxMemoryBytes = 256 * 1024)
        {
            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentException("File path is required.", nameof(filePath));
            if (dimensions <= 0) throw new ArgumentOutOfRangeException(nameof(dimensions));

            Dimensions = dimensions;
            _filePath = filePath;

            int bytesPerVector = dimensions * sizeof(float);
            _pageSize = Math.Max(4096, ((4096 + bytesPerVector - 1) / bytesPerVector) * bytesPerVector);

            bool fileExists = File.Exists(filePath);
            _fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.RandomAccess);

            if (fileExists && _fileStream.Length > 0)
            {
                _count = (int)(_fileStream.Length / bytesPerVector);
            }
            else
            {
                _count = 0;
            }

            int numPages = Math.Max(1, maxMemoryBytes / _pageSize);
            _pages = new byte[numPages][];
            for (int i = 0; i < numPages; i++) _pages[i] = new byte[_pageSize];
            _pageTags = new long[numPages];
            Array.Fill(_pageTags, -1);
            _pageAccess = new long[numPages];
        }

        private ReadOnlySpan<byte> GetPage(long pageIndex)
        {
            int cacheSlot = -1;
            int lruSlot = 0;
            long minAccess = long.MaxValue;

            for (int i = 0; i < _pageTags.Length; i++)
            {
                if (_pageTags[i] == pageIndex)
                {
                    cacheSlot = i;
                    break;
                }
                if (_pageAccess[i] < minAccess)
                {
                    minAccess = _pageAccess[i];
                    lruSlot = i;
                }
            }

            if (cacheSlot == -1)
            {
                cacheSlot = lruSlot;
                _pageTags[cacheSlot] = pageIndex;
                long fileOffset = pageIndex * _pageSize;

                long remainingBytes = _fileStream.Length - fileOffset;
                int bytesToRead = (int)Math.Clamp(remainingBytes, 0, _pageSize);

                if (bytesToRead > 0)
                {
                    RandomAccess.Read(_fileStream.SafeFileHandle, _pages[cacheSlot].AsSpan(0, bytesToRead), fileOffset);
                }
                if (bytesToRead < _pageSize)
                {
                    _pages[cacheSlot].AsSpan(bytesToRead).Clear();
                }
            }

            _pageAccess[cacheSlot] = ++_accessCounter;
            return _pages[cacheSlot];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetVector(int index)
        {
            if ((uint)index >= (uint)Volatile.Read(ref _count))
                throw new ArgumentOutOfRangeException(nameof(index));

            if (_threadLocalBuffers == null)
            {
                _threadLocalBuffers = new float[4][];
            }

            _bufferIndex = (_bufferIndex + 1) & 3;

            if (_threadLocalBuffers[_bufferIndex] == null || _threadLocalBuffers[_bufferIndex].Length < Dimensions)
            {
                _threadLocalBuffers[_bufferIndex] = new float[Dimensions];
            }

            float[] buffer = _threadLocalBuffers[_bufferIndex];

            long byteOffset = (long)index * Dimensions * sizeof(float);
            long pageIndex = byteOffset / _pageSize;
            int inPageOffset = (int)(byteOffset % _pageSize);

            lock (_lock)
            {
                ReadOnlySpan<byte> pageSpan = GetPage(pageIndex);
                var floatSpan = MemoryMarshal.Cast<byte, float>(pageSpan.Slice(inPageOffset, Dimensions * sizeof(float)));
                floatSpan.CopyTo(buffer);
            }

            return buffer.AsSpan(0, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            int bytesPerVector = Dimensions * sizeof(float);

            lock (_lock)
            {
                long fileOffset = (long)_count * bytesPerVector;
                var byteSpan = MemoryMarshal.AsBytes(vector);
                RandomAccess.Write(_fileStream.SafeFileHandle, byteSpan, fileOffset);

                long pageIndex = fileOffset / _pageSize;
                for (int i = 0; i < _pageTags.Length; i++)
                {
                    if (_pageTags[i] == pageIndex)
                    {
                        _pageTags[i] = -1;
                        break;
                    }
                }

                Volatile.Write(ref _count, _count + 1);
            }
        }

        public void Dispose()
        {
            lock (_lock)
            {
                _fileStream?.Dispose();
            }
        }
    }
}
