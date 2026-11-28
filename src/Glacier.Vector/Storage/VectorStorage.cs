using System;
using System.Buffers.Binary;
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
        /// Copies the vector at the specified zero-based index into <paramref name="destination"/>.
        /// </summary>
        void ReadVector(int index, Span<float> destination) => GetVector(index).CopyTo(destination);

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
            return new ReadOnlySpan<float>(chunk, 0, checked((int)((long)count * Dimensions)));
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
            _chunks[0] = new float[checked((int)((long)vectorsPerChunk * dimensions))];
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
            int offsetInChunk = checked((int)((long)(index % _vectorsPerChunk) * Dimensions));
            float[]?[] chunks = Volatile.Read(ref _chunks);
            float[]? chunk = Volatile.Read(ref chunks[chunkIdx]);
            if (chunk == null)
                throw new ObjectDisposedException(nameof(InMemoryVectorStorage));

            return new ReadOnlySpan<float>(chunk, offsetInChunk, Dimensions);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReadVector(int index, Span<float> destination)
        {
            if (destination.Length < Dimensions)
                throw new ArgumentException($"Destination span length ({destination.Length}) is smaller than Dimensions ({Dimensions}).", nameof(destination));
            GetVector(index).CopyTo(destination);
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

                    var newChunk = new float[checked((int)((long)_vectorsPerChunk * Dimensions))];
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

                int offset = checked((int)((long)_currentChunkVectorCount * Dimensions));
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
        private const uint HeaderMagic = 0x564D4C47u; // "GLMV"
        private const int HeaderVersion = 1;
        private const int HeaderSize = 64;

        public int Dimensions { get; }
        public int Count => Volatile.Read(ref _count);
        public long StorageId { get; } = StorageIdGenerator.Next();
        public long Version => Volatile.Read(ref _version);

        private int _count;
        private long _version = 1;
        private readonly string _filePath;
        private readonly string _metaPath;
        private readonly int _dataOffset;
        private FileStream _fileStream = null!;
        private MemoryMappedFile? _mmf;
        private MemoryMappedViewAccessor? _accessor;
        private IntPtr _basePointer;
        private bool _disposed;
        private int _disposeFlag;

        // Start with a 64MB mapping segment and grow in 64MB increments to stay gentle on CI runners while amortizing remaps
        private const long GrowSize = 64L * 1024 * 1024;
        private long _currentCapacityBytes;
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

            try
            {
                long bytesPerVector = (long)dimensions * sizeof(float);
                if (fileExists && _fileStream.Length > 0)
                {
                    long fileLength = _fileStream.Length;
                    Span<byte> headerBuf = stackalloc byte[HeaderSize];
                    _fileStream.Position = 0;
                    int headerRead = ReadAtLeastBytes(_fileStream, headerBuf);
                    _fileStream.Position = 0;

                    if (headerRead >= HeaderSize &&
                        BinaryPrimitives.ReadUInt32LittleEndian(headerBuf) == HeaderMagic &&
                        BinaryPrimitives.ReadInt32LittleEndian(headerBuf.Slice(4)) == HeaderVersion)
                    {
                        int fileDims = BinaryPrimitives.ReadInt32LittleEndian(headerBuf.Slice(8));
                        if (fileDims != dimensions)
                            throw new InvalidDataException($"MMF header dimension mismatch: expected {dimensions}, found {fileDims}.");

                        _dataOffset = HeaderSize;
                        long maxVectorsInFile = Math.Max(0L, (fileLength - HeaderSize) / bytesPerVector);
                        long recordedCount = BinaryPrimitives.ReadInt64LittleEndian(headerBuf.Slice(16));
                        _count = (int)Math.Min(int.MaxValue, Math.Clamp(recordedCount, 0L, maxVectorsInFile));
                        _currentCapacityBytes = Math.Max(fileLength, HeaderSize + bytesPerVector);
                    }
                    else
                    {
                        // Legacy headerless file compatibility (e.g., raw pre-sized binary vector files)
                        _dataOffset = 0;
                        long maxVectorsInFile = fileLength / bytesPerVector;

                        if (File.Exists(_metaPath))
                        {
                            try
                            {
                                byte[] metaBytes = File.ReadAllBytes(_metaPath);
                                if (metaBytes.Length >= sizeof(int))
                                {
                                    int recordedCount = BitConverter.ToInt32(metaBytes, 0);
                                    _count = (int)Math.Min(int.MaxValue, Math.Clamp((long)recordedCount, 0L, maxVectorsInFile));
                                }
                                else
                                {
                                    _count = (int)Math.Min(int.MaxValue, maxVectorsInFile);
                                }
                            }
                            catch
                            {
                                _count = (int)Math.Min(int.MaxValue, maxVectorsInFile);
                            }
                        }
                        else
                        {
                            _count = (int)Math.Min(int.MaxValue, maxVectorsInFile);
                        }

                        _currentCapacityBytes = Math.Max(fileLength, bytesPerVector);
                    }

                    if (_fileStream.Length < _currentCapacityBytes)
                    {
                        _fileStream.SetLength(_currentCapacityBytes);
                    }
                }
                else
                {
                    _dataOffset = HeaderSize;
                    _count = 0;
                    _currentCapacityBytes = Math.Max(GrowSize, HeaderSize + bytesPerVector);
                    _fileStream.SetLength(_currentCapacityBytes);

                    Span<byte> headerBuf = stackalloc byte[HeaderSize];
                    headerBuf.Clear();
                    BinaryPrimitives.WriteUInt32LittleEndian(headerBuf, HeaderMagic);
                    BinaryPrimitives.WriteInt32LittleEndian(headerBuf.Slice(4), HeaderVersion);
                    BinaryPrimitives.WriteInt32LittleEndian(headerBuf.Slice(8), dimensions);
                    BinaryPrimitives.WriteInt64LittleEndian(headerBuf.Slice(16), 0L);
                    BinaryPrimitives.WriteInt64LittleEndian(headerBuf.Slice(24), _version);
                    _fileStream.Position = 0;
                    _fileStream.Write(headerBuf);
                    _fileStream.Flush(flushToDisk: true);
                    _fileStream.Position = 0;
                }

                InitializeMapping();
            }
            catch
            {
                ReleaseMappingHandles();
                try { _fileStream?.Dispose(); } catch { }
                _rwLock.Dispose();
                throw;
            }
        }

        private static int ReadAtLeastBytes(Stream stream, Span<byte> buffer)
        {
            int total = 0;
            while (total < buffer.Length)
            {
                int read = stream.Read(buffer.Slice(total));
                if (read == 0) break;
                total += read;
            }
            return total;
        }

        private void WriteCountMetadata(int count)
        {
            string tempPath = $"{_metaPath}.{Guid.NewGuid():N}.tmp";
            try
            {
                Span<byte> buf = stackalloc byte[sizeof(int)];
                BitConverter.TryWriteBytes(buf, count);
                using (var fs = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
                {
                    fs.Write(buf);
                    fs.Flush(flushToDisk: true);
                }
                File.Move(tempPath, _metaPath, overwrite: true);
            }
            catch
            {
                // Best-effort crash recovery sidecar
                try { if (File.Exists(tempPath)) File.Delete(tempPath); } catch { }
            }
        }

        private void ReleaseMappingHandles()
        {
            if (_basePointer != IntPtr.Zero)
            {
                try { _accessor?.Flush(); } catch { }
                try { _accessor?.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
                Volatile.Write(ref _basePointer, IntPtr.Zero);
            }
            try { _accessor?.Dispose(); } catch { }
            _accessor = null;
            try { _mmf?.Dispose(); } catch { }
            _mmf = null;
        }

        private void InitializeMapping()
        {
            MemoryMappedFile? mmf = null;
            MemoryMappedViewAccessor? accessor = null;
            bool pointerAcquired = false;
            try
            {
                mmf = MemoryMappedFile.CreateFromFile(_fileStream, null, _currentCapacityBytes, MemoryMappedFileAccess.ReadWrite, HandleInheritability.None, leaveOpen: true);
                accessor = mmf.CreateViewAccessor(0, _currentCapacityBytes, MemoryMappedFileAccess.ReadWrite);

                byte* newPtr = null;
                accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref newPtr);
                pointerAcquired = true;

                _mmf = mmf;
                _accessor = accessor;
                Volatile.Write(ref _basePointer, (IntPtr)newPtr);
            }
            catch
            {
                if (pointerAcquired && accessor != null)
                {
                    try { accessor.SafeMemoryMappedViewHandle.ReleasePointer(); } catch { }
                }
                try { accessor?.Dispose(); } catch { }
                try { mmf?.Dispose(); } catch { }
                throw;
            }
        }

        public int ChunkCount => 1;

        public ReadOnlySpan<float> GetChunkSpan(int chunkIndex)
        {
            if (chunkIndex != 0) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            int count = Volatile.Read(ref _count);
            byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
            if (basePtr == null || count == 0) return ReadOnlySpan<float>.Empty;
            int totalFloats = checked((int)((long)count * Dimensions));
            return new ReadOnlySpan<float>((float*)(basePtr + _dataOffset), totalFloats);
        }

        public int GetChunkVectorCount(int chunkIndex)
        {
            if (chunkIndex != 0) throw new ArgumentOutOfRangeException(nameof(chunkIndex));
            return Volatile.Read(ref _count);
        }

        public ReadOnlySpan<float> GetVector(int index)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            if ((uint)index >= (uint)Volatile.Read(ref _count))
                throw new ArgumentOutOfRangeException(nameof(index));

            float[] copy = new float[Dimensions];
            ReadVector(index, copy);
            return copy;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReadVector(int index, Span<float> destination)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            if (destination.Length < Dimensions)
                throw new ArgumentException($"Destination span length ({destination.Length}) is smaller than Dimensions ({Dimensions}).", nameof(destination));

            try
            {
                _rwLock.EnterReadLock();
            }
            catch (ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(MmfVectorStorage), "Storage is unmapped or disposed.");
            }

            try
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if ((uint)index >= (uint)_count)
                    throw new ArgumentOutOfRangeException(nameof(index));

                byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
                if (basePtr == null)
                    throw new ObjectDisposedException(nameof(MmfVectorStorage), "Storage is unmapped or disposed.");

                float* floatPtr = (float*)(basePtr + _dataOffset + (long)index * Dimensions * sizeof(float));
                new ReadOnlySpan<float>(floatPtr, Dimensions).CopyTo(destination);
            }
            finally
            {
                _rwLock.ExitReadLock();
            }
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            try
            {
                _rwLock.EnterWriteLock();
            }
            catch (ObjectDisposedException)
            {
                throw new ObjectDisposedException(nameof(MmfVectorStorage), "Storage is unmapped or disposed.");
            }

            try
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                long bytesRequired = _dataOffset + (long)(_count + 1) * Dimensions * sizeof(float);

                if (bytesRequired > _currentCapacityBytes)
                {
                    ExpandMapping(bytesRequired);
                }

                byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
                if (basePtr == null)
                    throw new ObjectDisposedException(nameof(MmfVectorStorage), "Storage is unmapped or disposed.");

                float* floatPtr = (float*)(basePtr + _dataOffset + (long)_count * Dimensions * sizeof(float));

                fixed (float* src = vector)
                {
                    Unsafe.CopyBlockUnaligned(
                        floatPtr,
                        src,
                        (uint)(Dimensions * sizeof(float)));
                }

                Thread.MemoryBarrier();

                int newCount = _count + 1;
                long newVersion = _version + 1;
                if (_dataOffset == HeaderSize)
                {
                    Volatile.Write(ref *(long*)(basePtr + 16), (long)newCount);
                    Volatile.Write(ref *(long*)(basePtr + 24), newVersion);
                }
                else
                {
                    WriteCountMetadata(newCount);
                }

                Volatile.Write(ref _count, newCount);
                Volatile.Write(ref _version, newVersion);
            }
            finally
            {
                _rwLock.ExitWriteLock();
            }
        }

        private void ExpandMapping(long minRequiredBytes)
        {
            ReleaseMappingHandles();

            while (_currentCapacityBytes < minRequiredBytes)
            {
                _currentCapacityBytes += GrowSize;
            }
            _fileStream.SetLength(_currentCapacityBytes);

            InitializeMapping();
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposeFlag, 1) != 0) return;

            try
            {
                _rwLock.EnterWriteLock();
            }
            catch (ObjectDisposedException)
            {
                return;
            }

            try
            {
                if (_disposed) return;
                _disposed = true;

                byte* basePtr = (byte*)Volatile.Read(ref _basePointer);
                if (_dataOffset == HeaderSize && basePtr != null)
                {
                    Volatile.Write(ref *(long*)(basePtr + 16), (long)_count);
                    Volatile.Write(ref *(long*)(basePtr + 24), _version);
                }

                ReleaseMappingHandles();

                long exactBytes = _dataOffset + (long)_count * Dimensions * sizeof(float);
                try { _fileStream?.SetLength(exactBytes); } catch { }
                try { _fileStream?.Flush(flushToDisk: true); } catch { }
                _fileStream?.Dispose();

                if (_dataOffset == 0)
                {
                    WriteCountMetadata(_count);
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
        private bool _disposed;
        private int _disposeFlag;
        private readonly string _filePath;
        private readonly FileStream _fileStream;
        private readonly object _lock = new object();

        // Software page cache parameters
        private readonly int _pageSize;
        private readonly byte[][] _pages;
        private readonly long[] _pageTags;
        private readonly long[] _pageAccess;
        private long _accessCounter = 0;

        public PagedVectorStorage(string filePath, int dimensions, int maxMemoryBytes = 256 * 1024)
        {
            if (string.IsNullOrWhiteSpace(filePath)) throw new ArgumentException("File path is required.", nameof(filePath));
            if (dimensions <= 0) throw new ArgumentOutOfRangeException(nameof(dimensions));
            if (maxMemoryBytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxMemoryBytes));

            Dimensions = dimensions;
            _filePath = filePath;

            long bytesPerVector = (long)dimensions * sizeof(float);
            long pageSizeLong = Math.Max(4096L, ((4096L + bytesPerVector - 1L) / bytesPerVector) * bytesPerVector);
            _pageSize = checked((int)pageSizeLong);

            bool fileExists = File.Exists(filePath);
            _fileStream = new FileStream(filePath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.ReadWrite, 4096, FileOptions.RandomAccess);

            if (fileExists && _fileStream.Length > 0)
            {
                _count = checked((int)(_fileStream.Length / bytesPerVector));
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

        public ReadOnlySpan<float> GetVector(int index)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            if ((uint)index >= (uint)Volatile.Read(ref _count))
                throw new ArgumentOutOfRangeException(nameof(index));

            float[] copy = new float[Dimensions];
            ReadVector(index, copy);
            return copy;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public void ReadVector(int index, Span<float> destination)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            if (destination.Length < Dimensions)
                throw new ArgumentException($"Destination span length ({destination.Length}) is smaller than Dimensions ({Dimensions}).", nameof(destination));

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if ((uint)index >= (uint)_count)
                    throw new ArgumentOutOfRangeException(nameof(index));

                long byteOffset = (long)index * Dimensions * sizeof(float);
                long pageIndex = byteOffset / _pageSize;
                int inPageOffset = (int)(byteOffset % _pageSize);

                ReadOnlySpan<byte> pageSpan = GetPage(pageIndex);
                var floatSpan = MemoryMarshal.Cast<byte, float>(pageSpan.Slice(inPageOffset, Dimensions * sizeof(float)));
                floatSpan.CopyTo(destination);
            }
        }

        public void Append(ReadOnlySpan<float> vector)
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed), this);
            if (vector.Length != Dimensions)
                throw new ArgumentException($"Expected vector of dimension {Dimensions}, got {vector.Length}");

            long bytesPerVector = (long)Dimensions * sizeof(float);

            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
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
            if (Interlocked.Exchange(ref _disposeFlag, 1) != 0) return;

            lock (_lock)
            {
                if (_disposed) return;
                Volatile.Write(ref _disposed, true);
                _fileStream?.Dispose();
            }
        }
    }
}
