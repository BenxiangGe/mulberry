//===--- File.cpp ---------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/Runtime/String.h"

#include <cstdio>
#include <gc.h>
#include <limits>

namespace {

struct MulberryFile {
  std::FILE* stream;
  bool closed;
};

auto getFile(uint8_t* file) -> MulberryFile* {
  return reinterpret_cast<MulberryFile*>(file);
}

auto isOpen(MulberryFile* file) -> bool {
  return file && !file->closed && file->stream;
}

} // namespace

extern "C" uint8_t* mulberry_file_open(MulberryString path,
                                         MulberryString mode) {
  auto* stream = std::fopen(reinterpret_cast<const char*>(path.data),
                            reinterpret_cast<const char*>(mode.data));
  if (!stream)
    return nullptr;

  // Keep this wrapper alive after fclose so every File alias can observe the
  // same closed state without dereferencing a released native handle.
  auto* file = static_cast<MulberryFile*>(GC_malloc(sizeof(MulberryFile)));
  if (!file) {
    std::fclose(stream);
    return nullptr;
  }
  file->stream = stream;
  file->closed = false;
  return reinterpret_cast<uint8_t*>(file);
}

// MLIR's C interface lowering looks for `_mlir_ciface_*` wrapper symbols.
// Export the wrapper names directly so JIT symbol resolution stays trivial.
extern "C" uint8_t* _mlir_ciface_mulberry_file_open(MulberryString path,
                                                      MulberryString mode) {
  return mulberry_file_open(path, mode);
}

extern "C" bool mulberry_file_is_valid(uint8_t* file) {
  return getFile(file) != nullptr;
}

extern "C" bool _mlir_ciface_mulberry_file_is_valid(uint8_t* file) {
  return mulberry_file_is_valid(file);
}

extern "C" bool mulberry_file_is_closed(uint8_t* file) {
  auto* handle = getFile(file);
  return !handle || handle->closed;
}

extern "C" bool _mlir_ciface_mulberry_file_is_closed(uint8_t* file) {
  return mulberry_file_is_closed(file);
}

extern "C" bool mulberry_file_close(uint8_t* file) {
  auto* handle = getFile(file);
  if (!isOpen(handle))
    return false;

  auto* stream = handle->stream;
  handle->stream = nullptr;
  handle->closed = true;
  return std::fclose(stream) == 0;
}

extern "C" bool _mlir_ciface_mulberry_file_close(uint8_t* file) {
  return mulberry_file_close(file);
}

extern "C" bool mulberry_file_seek(uint8_t* file, uint64_t offset) {
  if (offset > static_cast<uint64_t>(std::numeric_limits<long>::max()))
    return false;
  auto* handle = getFile(file);
  if (!isOpen(handle))
    return false;
  return std::fseek(handle->stream, static_cast<long>(offset), SEEK_SET) ==
         0;
}

extern "C" bool _mlir_ciface_mulberry_file_seek(uint8_t* file,
                                                uint64_t offset) {
  return mulberry_file_seek(file, offset);
}

extern "C" uint64_t mulberry_file_tell(uint8_t* file) {
  auto* handle = getFile(file);
  if (!isOpen(handle))
    return std::numeric_limits<uint64_t>::max();

  auto position = std::ftell(handle->stream);
  // A valid ftell result is a nonnegative long, so UINT64_MAX can stay inside
  // the runtime ABI as an unambiguous failure marker.
  if (position < 0)
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(position);
}

extern "C" uint64_t _mlir_ciface_mulberry_file_tell(uint8_t* file) {
  return mulberry_file_tell(file);
}

extern "C" bool mulberry_file_tell_is_valid(uint64_t position) {
  return position != std::numeric_limits<uint64_t>::max();
}

extern "C" bool _mlir_ciface_mulberry_file_tell_is_valid(
    uint64_t position) {
  return mulberry_file_tell_is_valid(position);
}

extern "C" uint64_t mulberry_file_size(uint8_t* file) {
  // Bounds queries must not change the position used by the next read/write.
  auto* handle = getFile(file);
  if (!isOpen(handle))
    return std::numeric_limits<uint64_t>::max();

  auto currentPosition = std::ftell(handle->stream);
  if (currentPosition < 0)
    return std::numeric_limits<uint64_t>::max();
  if (std::fseek(handle->stream, 0, SEEK_END) != 0)
    return std::numeric_limits<uint64_t>::max();

  auto size = std::ftell(handle->stream);
  const bool restored =
      std::fseek(handle->stream, currentPosition, SEEK_SET) == 0;
  if (size < 0 || !restored)
    return std::numeric_limits<uint64_t>::max();
  return static_cast<uint64_t>(size);
}

extern "C" uint64_t _mlir_ciface_mulberry_file_size(uint8_t* file) {
  return mulberry_file_size(file);
}

extern "C" bool mulberry_file_size_is_valid(uint64_t size) {
  return size != std::numeric_limits<uint64_t>::max();
}

extern "C" bool _mlir_ciface_mulberry_file_size_is_valid(uint64_t size) {
  return mulberry_file_size_is_valid(size);
}

extern "C" bool mulberry_file_has_error(uint8_t* file) {
  auto* handle = getFile(file);
  return isOpen(handle) && std::ferror(handle->stream) != 0;
}

extern "C" bool _mlir_ciface_mulberry_file_has_error(uint8_t* file) {
  return mulberry_file_has_error(file);
}

extern "C" uint64_t mulberry_file_read(uint8_t* file, uint8_t* data,
                                       uint64_t byteSize) {
  auto* handle = getFile(file);
  if (!isOpen(handle))
    return 0;
  return static_cast<uint64_t>(std::fread(data, 1, byteSize, handle->stream));
}

extern "C" uint64_t _mlir_ciface_mulberry_file_read(uint8_t* file,
                                                    uint8_t* data,
                                                    uint64_t byteSize) {
  return mulberry_file_read(file, data, byteSize);
}

extern "C" uint64_t mulberry_file_write(uint8_t* file, uint8_t* data,
                                        uint64_t byteSize) {
  auto* handle = getFile(file);
  if (!isOpen(handle))
    return 0;
  return static_cast<uint64_t>(
      std::fwrite(data, 1, byteSize, handle->stream));
}

extern "C" uint64_t _mlir_ciface_mulberry_file_write(uint8_t* file,
                                                     uint8_t* data,
                                                     uint64_t byteSize) {
  return mulberry_file_write(file, data, byteSize);
}
