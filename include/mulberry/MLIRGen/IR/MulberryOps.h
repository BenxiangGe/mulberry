//===--- MulberryOps.h - Mulberry core dialect ops ------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MULBERRYOPS_H
#define MULBERRY_MULBERRYOPS_H

#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"

namespace mlir::mulberry_core {

inline constexpr llvm::StringLiteral kTransferTensorResultOwnershipAttr =
    "mulberry_core.transfer_tensor_result_ownership";

} // namespace mlir::mulberry_core

#define GET_OP_CLASSES
#include "mulberry/MLIRGen/IR/MulberryOps.h.inc"

#endif // MULBERRY_MULBERRYOPS_H
