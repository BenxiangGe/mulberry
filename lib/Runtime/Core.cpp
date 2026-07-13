//===--- Core.cpp ---------------------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstring>

extern "C" uint8_t* _mlir_ciface_mulberry_ptr_offset(uint8_t* data,
                                                     uint64_t byteOffset) {
  return data + byteOffset;
}

extern "C" void _mlir_ciface_mulberry_zero_memory(uint8_t* data,
                                                  uint64_t byteSize) {
  std::memset(data, 0, byteSize);
}
