//===--- Core.cpp ---------------------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>

extern "C" uint64_t _mlir_ciface_mulberry_u8_to_u64(uint8_t value) {
  return static_cast<uint64_t>(value);
}
