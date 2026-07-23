//===--- String.h - Mulberry runtime String ABI -----------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_RUNTIME_STRING_H
#define MULBERRY_RUNTIME_STRING_H

#include <cstdint>

// This mirrors the value ABI for std.string.String at runtime boundaries.
struct MulberryString {
  uint64_t length;
  uint8_t* data;
};

#endif // MULBERRY_RUNTIME_STRING_H
