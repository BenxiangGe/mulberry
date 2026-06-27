//===--- Print.cpp --------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cinttypes>
#include <cstdint>
#include <cstdio>

// `extern fn printU64(...)` is lowered through MLIR's C-interface wrapper ABI,
// so runtime symbol lookup expects `_mlir_ciface_printU64`, not `printU64`.
extern "C" void _mlir_ciface_printU64(uint64_t value) {
  std::printf("%" PRIu64, value);
}

extern "C" void _mlir_ciface_printNewline() {
  std::printf("\n");
  std::fflush(stdout);
}
