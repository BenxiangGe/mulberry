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

extern "C" uint64_t _mlir_ciface_mulberry_u8_to_u64(uint8_t value) {
  return static_cast<uint64_t>(value);
}

extern "C" uint8_t _mlir_ciface_mulberry_u64_to_u8(uint64_t value) {
  return static_cast<uint8_t>(value);
}

extern "C" float _mlir_ciface_mulberry_u64_to_f32(uint64_t value) {
  return static_cast<float>(value);
}

extern "C" void _mlir_ciface_mulberry_zero_memory(uint8_t* data,
                                                  uint64_t byteSize) {
  std::memset(data, 0, byteSize);
}
