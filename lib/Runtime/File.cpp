//===--- File.cpp ---------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

struct String {
  uint64_t length;
  uint8_t* data;
};

} // namespace

extern "C" uint8_t* mulberry_file_open(String path, String mode) {
  return reinterpret_cast<uint8_t*>(
      std::fopen(reinterpret_cast<const char*>(path.data),
                 reinterpret_cast<const char*>(mode.data)));
}

// MLIR's C interface lowering looks for `_mlir_ciface_*` wrapper symbols.
// Export the wrapper names directly so JIT symbol resolution stays trivial.
extern "C" uint8_t* _mlir_ciface_mulberry_file_open(String path,
                                                    String mode) {
  return mulberry_file_open(path, mode);
}

extern "C" bool mulberry_file_is_valid(uint8_t* file) {
  return file != nullptr;
}

extern "C" bool _mlir_ciface_mulberry_file_is_valid(uint8_t* file) {
  return mulberry_file_is_valid(file);
}

extern "C" bool mulberry_file_close(uint8_t* file) {
  auto* handle = reinterpret_cast<std::FILE*>(file);
  return std::fclose(handle) == 0;
}

extern "C" bool _mlir_ciface_mulberry_file_close(uint8_t* file) {
  return mulberry_file_close(file);
}

extern "C" bool mulberry_file_seek(uint8_t* file, uint64_t offset) {
  if (offset > static_cast<uint64_t>(std::numeric_limits<long>::max()))
    return false;
  auto* handle = reinterpret_cast<std::FILE*>(file);
  return std::fseek(handle, static_cast<long>(offset), SEEK_SET) == 0;
}

extern "C" bool _mlir_ciface_mulberry_file_seek(uint8_t* file,
                                                uint64_t offset) {
  return mulberry_file_seek(file, offset);
}

extern "C" uint64_t mulberry_file_tell(uint8_t* file) {
  auto* handle = reinterpret_cast<std::FILE*>(file);
  auto position = std::ftell(handle);
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

extern "C" bool mulberry_file_has_error(uint8_t* file) {
  auto* handle = reinterpret_cast<std::FILE*>(file);
  return std::ferror(handle) != 0;
}

extern "C" bool _mlir_ciface_mulberry_file_has_error(uint8_t* file) {
  return mulberry_file_has_error(file);
}

extern "C" uint64_t mulberry_file_read(uint8_t* file, uint8_t* data,
                                       uint64_t byteSize) {
  auto* handle = reinterpret_cast<std::FILE*>(file);
  return static_cast<uint64_t>(std::fread(data, 1, byteSize, handle));
}

extern "C" uint64_t _mlir_ciface_mulberry_file_read(uint8_t* file,
                                                    uint8_t* data,
                                                    uint64_t byteSize) {
  return mulberry_file_read(file, data, byteSize);
}

extern "C" uint64_t mulberry_file_write(uint8_t* file, uint8_t* data,
                                        uint64_t byteSize) {
  auto* handle = reinterpret_cast<std::FILE*>(file);
  return static_cast<uint64_t>(std::fwrite(data, 1, byteSize, handle));
}

extern "C" uint64_t _mlir_ciface_mulberry_file_write(uint8_t* file,
                                                     uint8_t* data,
                                                     uint64_t byteSize) {
  return mulberry_file_write(file, data, byteSize);
}
