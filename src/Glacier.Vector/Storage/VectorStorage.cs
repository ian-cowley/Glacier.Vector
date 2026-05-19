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
        private readonly List<float[]> _chunks;
        private int _currentChunkIndex;
        private int _currentChunkVectorCount;

        // Default to ~65k vectors per chunk. 
        // For 1536 dims, this is exactly 402 MB per chunk.
        public InMemoryVectorStorage(int dimensions, int vectorsPerChunk = 65536)
        {
            Dimensions = dimensions;
            _vectorsPerChunk = vectorsPerChunk;
            _chunks = new List<float[]> { new float[vectorsPerChunk * dimensions] };
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

            return new ReadOnlySpan<float>(_chunks[chunkIdx], offsetInChunk, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            // If the current chunk is full, allocate a new one
            if (_currentChunkVectorCount == _vectorsPerChunk)
            {
                _chunks.Add(new float[_vectorsPerChunk * Dimensions]);
                _currentChunkIndex++;
                _currentChunkVectorCount = 0;
            }

            // Copy the data directly into the pre-allocated chunk
            int offset = _currentChunkVectorCount * Dimensions;
            vector.CopyTo(new Span<float>(_chunks[_currentChunkIndex], offset, Dimensions));

            _currentChunkVectorCount++;
            Count++;
        }

        public void Dispose()
        {
            _chunks.Clear();
            Count = 0;
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
        private FileStream _fileStream;
        private MemoryMappedFile _mmf;
        private MemoryMappedViewAccessor _accessor;
        private byte* _basePointer;

        // Start with a 1GB file, grow in 1GB chunks to avoid constant file resizing
        private const long GrowSize = 1024 * 1024 * 1024;
        private long _currentCapacityBytes;

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

            // Acquire raw unmanaged pointer to the mapped memory
            _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref _basePointer);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public ReadOnlySpan<float> GetVector(int index)
        {
            // Cast the byte pointer to a float pointer, then offset directly to the target vector
            float* floatPtr = (float*)_basePointer;
            long offset = (long)index * Dimensions;

            // Returns a span pointing directly to the hard drive cache / physical memory
            return new ReadOnlySpan<float>(floatPtr + offset, Dimensions);
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            long bytesRequired = (long)(Count + 1) * Dimensions * sizeof(float);

            // If we run out of space in the mapped file, we must unmap, grow the file, and remap
            if (bytesRequired > _currentCapacityBytes)
            {
                ExpandMapping();
            }

            // Copy memory directly to the mapped file buffer
            float* floatPtr = (float*)_basePointer;
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

        private void ExpandMapping()
        {
            // Release the current view and mapping lock
            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
            _accessor.Dispose();
            _mmf.Dispose();

            // Grow the physical file on disk by 1GB
            _currentCapacityBytes += GrowSize;
            _fileStream.SetLength(_currentCapacityBytes);

            // Remap the file into virtual memory
            InitializeMapping();
        }

        public void Dispose()
        {
            if (_basePointer != null)
            {
                _accessor?.SafeMemoryMappedViewHandle.ReleasePointer();
                _basePointer = null;
            }
            _accessor?.Dispose();
            _mmf?.Dispose();

            // Truncate the file to exact size to save disk space before closing
            long exactBytes = (long)Count * Dimensions * sizeof(float);
            _fileStream?.SetLength(exactBytes);
            _fileStream?.Dispose();
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
        private static float[]? _threadLocalBuffer;

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
            if (_threadLocalBuffer == null || _threadLocalBuffer.Length < Dimensions)
            {
                _threadLocalBuffer = new float[Dimensions];
            }

            long byteOffset = (long)index * Dimensions * sizeof(float);
            long pageIndex = byteOffset / _pageSize;
            int inPageOffset = (int)(byteOffset % _pageSize);

            lock (_lock)
            {
                ReadOnlySpan<byte> pageSpan = GetPage(pageIndex);
                var floatSpan = MemoryMarshal.Cast<byte, float>(pageSpan.Slice(inPageOffset, Dimensions * sizeof(float)));
                floatSpan.CopyTo(_threadLocalBuffer);
            }

            return _threadLocalBuffer.AsSpan(0, Dimensions);
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