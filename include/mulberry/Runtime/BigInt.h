//===--- BigInt.h - Mulberry BigInt runtime ABI ----------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_RUNTIME_BIGINT_H
#define MULBERRY_RUNTIME_BIGINT_H

#include "mulberry/Runtime/String.h"

#include <cstdint>

// This is an internal runtime layout. The source language exposes only
// Integer; the semantic !bigint.int type does not depend on these fields.
struct MulberryBigInt {
  int8_t sign;
  uint64_t limbCount;
  uint32_t limbs[1];
};

extern "C" {

MulberryBigInt* mulberry_bigint_from_uint64(uint64_t value);
MulberryBigInt* mulberry_bigint_from_literal(const char* spelling,
                                             uint64_t length);

MulberryBigInt* mulberry_bigint_neg(const MulberryBigInt* value);
MulberryBigInt* mulberry_bigint_add(const MulberryBigInt* lhs,
                                    const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_sub(const MulberryBigInt* lhs,
                                    const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_mul(const MulberryBigInt* lhs,
                                    const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_div(const MulberryBigInt* lhs,
                                    const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_rem(const MulberryBigInt* lhs,
                                    const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_bit_and(const MulberryBigInt* lhs,
                                        const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_bit_or(const MulberryBigInt* lhs,
                                       const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_bit_xor(const MulberryBigInt* lhs,
                                        const MulberryBigInt* rhs);
MulberryBigInt* mulberry_bigint_shift_left(const MulberryBigInt* value,
                                           uint64_t count);
MulberryBigInt* mulberry_bigint_shift_right(const MulberryBigInt* value,
                                            uint64_t count);

// Returns -1, 0, or 1 for lhs < rhs, lhs == rhs, or lhs > rhs.
int32_t mulberry_bigint_compare(const MulberryBigInt* lhs,
                                const MulberryBigInt* rhs);
uint64_t mulberry_bigint_to_uint64(const MulberryBigInt* value);

// Source extern functions returning String use MLIR's C interface out
// parameter instead of the ordinary C++ aggregate return ABI.
void _mlir_ciface_mulberry_bigint_to_string(MulberryString* result,
                                            const MulberryBigInt* value);
void _mlir_ciface_mulberry_bigint_to_hex_string(MulberryString* result,
                                                const MulberryBigInt* value);

}

#endif // MULBERRY_RUNTIME_BIGINT_H
