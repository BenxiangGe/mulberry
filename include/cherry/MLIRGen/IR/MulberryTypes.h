//===--- MulberryTypes.h - Mulberry dialect types --------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MULBERRYTYPES_H
#define CHERRY_MULBERRYTYPES_H

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/TypeSwitch.h"

#include <limits>
#include <string>

#define GET_TYPEDEF_CLASSES
#include "cherry/MLIRGen/IR/MulberryOpsTypes.h.inc"

namespace mlir::mulberry {

inline bool operator==(const RecordType::Field& lhs,
                       const RecordType::Field& rhs) {
  return lhs.name == rhs.name && lhs.type == rhs.type;
}

inline llvm::hash_code hash_value(const RecordType::Field& field) {
  return llvm::hash_combine(field.name, field.type);
}

} // namespace mlir::mulberry

#endif // CHERRY_MULBERRYTYPES_H
