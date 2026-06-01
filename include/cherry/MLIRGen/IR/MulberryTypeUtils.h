//===--- MulberryTypeUtils.h -----------------------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_MLIRGEN_IR_MULBERRYTYPEUTILS_H
#define CHERRY_MLIRGEN_IR_MULBERRYTYPEUTILS_H

#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"

namespace mlir::cherry {

inline auto containsMulberryRecordType(Type type) -> bool {
  if (!type)
    return false;

  if (llvm::isa<mlir::mulberry::RecordType>(type))
    return true;

  if (auto ptrType = llvm::dyn_cast<mlir::mulberry::PtrType>(type))
    return containsMulberryRecordType(ptrType.getElementType());

  // Function boundary decisions need to catch records hidden in parameter or
  // result types, not just a top-level !mulberry.record.
  if (auto funcType = llvm::dyn_cast<FunctionType>(type)) {
    for (auto input : funcType.getInputs())
      if (containsMulberryRecordType(input))
        return true;
    for (auto result : funcType.getResults())
      if (containsMulberryRecordType(result))
        return true;
  }

  return false;
}

inline auto containsMulberryRecordType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsMulberryRecordType(type))
      return true;
  return false;
}

} // namespace mlir::cherry

#endif // CHERRY_MLIRGEN_IR_MULBERRYTYPEUTILS_H
