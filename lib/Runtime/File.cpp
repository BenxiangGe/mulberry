//===--- File.cpp ---------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>

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

extern "C" uint64_t mulberry_file_close(uint8_t* file) {
  auto* handle = reinterpret_cast<std::FILE*>(file);
  return static_cast<uint64_t>(std::fclose(handle));
}

extern "C" uint64_t _mlir_ciface_mulberry_file_close(uint8_t* file) {
  return mulberry_file_close(file);
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
