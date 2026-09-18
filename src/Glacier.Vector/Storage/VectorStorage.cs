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
    /// between RAM and Disk-backed (Memory-Mapped) storage.
    /// </summary>
    public interface IVectorStorage : IDisposable
    {
        int Dimensions { get; }
        int Count { get; }

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

    /// <summary>
    /// Pure in-memory storage using a chunked array list.
    /// Prevents massive GC pauses associated with resizing giant arrays.
    /// </summary>
    public class InMemoryVectorStorage : IVectorStorage
    {
        public int Dimensions { get; }
        public int Count { get; private set; }

        private readonly int _vectorsPerChunk;
        private float[]?[] _chunks;
        private int _currentChunkIndex;
        private int _currentChunkVectorCount;
        private readonly object _appendLock = new();

        public int ChunkCount => Volatile.Read(ref _currentChunkIndex) + 1;

        public ReadOnlySpan<float> GetChunkSpan(int chunkIndex)
        {
            if ((uint)chunkIndex >= (uint)ChunkCount) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            float[]? chunk = Volatile.Read(ref _chunks[chunkIndex]);
            if (chunk == null) return ReadOnlySpan<float>.Empty;
            int count = GetChunkVectorCount(chunkIndex);
            return new ReadOnlySpan<float>(chunk, 0, count * Dimensions);
        }

        public int GetChunkVectorCount(int chunkIndex)
        {
            if ((uint)chunkIndex >= (uint)ChunkCount) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            if (chunkIndex == Volatile.Read(ref _currentChunkIndex)) return Volatile.Read(ref _currentChunkVectorCount);
            return _vectorsPerChunk;
        }

        // Default to ~65k vectors per chunk. 
        // For 1536 dims, this is exactly 402 MB per chunk.
        public InMemoryVectorStorage(int dimensions, int vectorsPerChunk = 65536)
        {
            Dimensions = dimensions;
            _vectorsPerChunk = vectorsPerChunk;
            _chunks = new float[1024][];
            _chunks[0] = new float[vectorsPerChunk * dimensions];
            _currentChunkIndex = 0;
            _currentChunkVectorCount = 0;
            Count = 0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetVector(int index)
        {
            // Fast math to find which array holds our vector, and where it starts
            int chunkIdx = index / _vectorsPerChunk;
            int offsetInChunk = (index % _vectorsPerChunk) * Dimensions;
            float[] chunk = Volatile.Read(ref _chunks[chunkIdx])!;

            return new ReadOnlySpan<float>(chunk, offsetInChunk, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            lock (_appendLock)
            {
                // If the current chunk is full, allocate a new one
                if (_currentChunkVectorCount == _vectorsPerChunk)
                {
                    int nextIndex = _currentChunkIndex + 1;
                    if (nextIndex >= _chunks.Length)
                    {
                        var newChunks = new float[_chunks.Length * 2][];
                        Array.Copy(_chunks, newChunks, _chunks.Length);
                        _chunks = newChunks;
                    }

                    Volatile.Write(ref _chunks[nextIndex], new float[_vectorsPerChunk * Dimensions]);
                    Volatile.Write(ref _currentChunkIndex, nextIndex);
                    Volatile.Write(ref _currentChunkVectorCount, 0);
                }

                // Copy the data directly into the pre-allocated chunk
                int offset = _currentChunkVectorCount * Dimensions;
                float[] chunk = _chunks[_currentChunkIndex]!;
                vector.CopyTo(new Span<float>(chunk, offset, Dimensions));

                _currentChunkVectorCount++;
                Count++;
            }
        }

        public void Dispose()
        {
            lock (_appendLock)
            {
                Array.Clear(_chunks, 0, _chunks.Length);
                Count = 0;
            }
        }
    }

    /// <summary>
    /// Disk-backed storage using Memory-Mapped Files.
    /// Allows querying databases much larger than physical RAM by relying on OS paging.
    /// </summary>
    public unsafe class MmfVectorStorage : IVectorStorage
    {
        public int Dimensions { get; }
        public int Count { get; private set; }

        private readonly string _filePath;
        private FileStream _fileStream = null!;
        private MemoryMappedFile _mmf = null!;
        private MemoryMappedViewAccessor _accessor = null!;
        private IntPtr _basePointer;

        // Start with a 1GB file, grow in 1GB chunks to avoid constant file resizing
        private const long GrowSize = 1024 * 1024 * 1024;
        private long _currentCapacityBytes;
        private readonly List<MemoryMappedFile> _oldMmfs = new();
        private readonly List<MemoryMappedViewAccessor> _oldAccessors = new();
        private readonly ReaderWriterLockSlim _rwLock = new(LockRecursionPolicy.NoRecursion);

        public MmfVectorStorage(string filePath, int dimensions)
        {
            Dimensions = dimensions;
            _filePath = filePath;

            bool fileExists = File.Exists(filePath);
            _fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite);

            if (fileExists && _fileStream.Length > 0)
            {
                // Load existing database
                long fileLength = _fileStream.Length;
                long bytesPerVector = dimensions * sizeof(float);
                Count = (int)(fileLength / bytesPerVector);
                _currentCapacityBytes = fileLength;
            }
            else
            {
                // New database
                Count = 0;
                _currentCapacityBytes = GrowSize;
                _fileStream.SetLength(_currentCapacityBytes);
            }

            InitializeMapping();
        }

        private void InitializeMapping()
        {
            _mmf = MemoryMappedFile.CreateFromFile(_fileStream, null, _currentCapacityBytes, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, false);
            _accessor = _mmf.CreateViewAccessor(0, _currentCapacityBytes, MemoryMappedFileAccess.ReadWrite);

            // Acquire raw unmanaged pointer to local variable first, then atomically publish
            byte* newPtr = null;
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref newPtr);
            Volatile.Write(ref _basePointer, (IntPtr)newPtr);
        }

        public int ChunkCount => 1;

        public ReadOnlySpan<float> GetChunkSpan(int chunkIndex)
        {
            if (chunkIndex != 0) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
            if (basePtr == null || Count == 0) return ReadOnlySpan<float>.Empty;
            return new ReadOnlySpan<float>((float*)basePtr, Count * Dimensions);
        }

        public int GetChunkVectorCount(int chunkIndex) => Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetVector(int index)
        {
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
                throw new InvalidOperationException("Storage is unmapped or disposed.");

            // Cast the byte pointer to a float pointer, then offset directly to the target vector
            float* floatPtr = (float*)basePtr;
            long offset = (long)index * Dimensions;

            // Returns a span pointing directly to the hard drive cache / physical memory
            return new ReadOnlySpan<float>(floatPtr + offset, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            _rwLock.EnterWriteLock();
            try
            {
                long bytesRequired = (long)(Count + 1) * Dimensions * sizeof(float);

                // If we run out of space in the mapped file, grow the file and map the expansion
                if (bytesRequired > _currentCapacityBytes)
                {
                    ExpandMapping();
                }

                // Copy memory directly to the mapped file buffer
                byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
                float* floatPtr = (float*)basePtr;
                long offset = (long)Count * Dimensions;

                fixed (float* src = vector)
                {
                    System.Runtime.CompilerServices.Unsafe.CopyBlockUnaligned(
                        floatPtr + offset,
                        src,
                        (uint)(Dimensions * sizeof(float)));
                }

                Count++;
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }

        private void ExpandMapping()
        {
            // Keep previous mapping active to prevent 0xC0000005 access violations in active readers
            if (_accessor != null) _oldAccessors.Add(_accessor);
            if (_mmf != null) _oldMmfs.Add(_mmf);

            // Grow the physical file on disk by 1GB
            _currentCapacityBytes += GrowSize;
            _fileStream.SetLength(_currentCapacityBytes);

            // Remap the file into virtual memory
            InitializeMapping();
        }

        public void Dispose()
        {
            _rwLock.EnterWriteLock();
            try
            {
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

                // Truncate the file to exact size to save disk space before closing
                long exactBytes = (long)Count * Dimensions * sizeof(float);
                try { _fileStream?.SetLength(exactBytes); } catch { }
                _fileStream?.Dispose();
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
        public int Count { get; private set; }

        private readonly string _filePath;
        private readonly FileStream _fileStream;
        private readonly object _lock = new object();

        // Software page cache parameters
        private readonly int _pageSize;
        private readonly byte[][] _pages;
        private readonly long[] _pageTags;
        private readonly int[] _pageAccess;
        private int _accessCounter = 0;

        [ThreadStatic]
        private static float[][]? _threadLocalBuffers;
        [ThreadStatic]
        private static int _bufferIndex;

        public PagedVectorStorage(string filePath, int dimensions, int maxMemoryBytes = 256 * 1024)
        {
            Dimensions = dimensions;
            _filePath = filePath;

            int bytesPerVector = dimensions * sizeof(float);
            // Align page size to be a multiple of vector size, >= 4096 bytes
            _pageSize = Math.Max(4096, ((4096 + bytesPerVector - 1) / bytesPerVector) * bytesPerVector);

            bool fileExists = File.Exists(filePath);
            _fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.RandomAccess);

            if (fileExists && _fileStream.Length > 0)
            {
                Count = (int)(_fileStream.Length / bytesPerVector);
            }
            else
            {
                Count = 0;
            }

            // Initialize Page Cache
            int numPages = Math.Max(1, maxMemoryBytes / _pageSize);
            _pages = new byte[numPages][];
            for (int i = 0; i < numPages; i++) _pages[i] = new byte[_pageSize];
            _pageTags = new long[numPages];
            Array.Fill(_pageTags, -1);
            _pageAccess = new int[numPages];
        }

        private ReadOnlySpan<byte> GetPage(long pageIndex)
        {
            int cacheSlot = -1;
            int lruSlot = 0;
            int minAccess = int.MaxValue;

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
                // Cache miss: evict LRU slot and load new page from disk
                cacheSlot = lruSlot;
                _pageTags[cacheSlot] = pageIndex;
                long fileOffset = pageIndex * _pageSize;

                long remainingBytes = _fileStream.Length - fileOffset;
                int bytesToRead = (int)Math.Min(_pageSize, remainingBytes);

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
            long fileOffset = (long)Count * bytesPerVector;

            lock (_lock)
            {
                // Write vector directly to disk
                var byteSpan = MemoryMarshal.AsBytes(vector);
                RandomAccess.Write(_fileStream.SafeFileHandle, byteSpan, fileOffset);

                // Invalidate any page in the cache that covers this offset
                long pageIndex = fileOffset / _pageSize;
                for (int i = 0; i < _pageTags.Length; i++)
                {
                    if (_pageTags[i] == pageIndex)
                    {
                        _pageTags[i] = -1; // Invalidate cache slot
                        break;
                    }
                }

                Count++;
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