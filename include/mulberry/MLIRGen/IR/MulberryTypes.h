//===--- MulberryTypes.h - Mulberry core dialect types --------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MULBERRYTYPES_H
#define MULBERRY_MULBERRYTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

#include <limits>
#include <string>

#define GET_TYPEDEF_CLASSES
#include "mulberry/MLIRGen/IR/MulberryOpsTypes.h.inc"

namespace mlir::mulberry_core {

inline bool operator==(const RecordType::Field& lhs,
                       const RecordType::Field& rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type;
}

inline llvm::hash_code hash_value(const RecordType::Field& field) {
  return llvm::hash_combine(field.name, field.type);
}

} // namespace mlir::mulberry_core

#endif // MULBERRY_MULBERRYTYPES_H
