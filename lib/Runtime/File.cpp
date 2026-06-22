//===--- File.cpp ---------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>

namespace {

struct StringStorage {
  uint64_t length;
  uint8_t* data;
};

struct FileStorage {
  uint8_t* handle;
};

} // namespace

extern "C" void* mulberry_boehm_malloc(uint64_t size);

extern "C" FileStorage* mulberry_file_open(StringStorage* path,
                                           StringStorage* mode) {
  auto* file = static_cast<FileStorage*>(
      mulberry_boehm_malloc(sizeof(FileStorage)));
  file->handle = reinterpret_cast<uint8_t*>(
      std::fopen(reinterpret_cast<const char*>(path->data),
                 reinterpret_cast<const char*>(mode->data)));
  return file;
}

// MLIR's C interface lowering looks for `_mlir_ciface_*` wrapper symbols.
// Export the wrapper names directly so JIT symbol resolution stays trivial.
extern "C" FileStorage* _mlir_ciface_mulberry_file_open(StringStorage* path,
                                                        StringStorage* mode) {
  return mulberry_file_open(path, mode);
}

extern "C" uint64_t mulberry_file_close(FileStorage* file) {
  auto* handle = reinterpret_cast<std::FILE*>(file->handle);
  return static_cast<uint64_t>(std::fclose(handle));
}

extern "C" uint64_t _mlir_ciface_mulberry_file_close(FileStorage* file) {
  return mulberry_file_close(file);
}
