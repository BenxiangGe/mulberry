//===--- MLIRGen.h - MLIR Generator -----------------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MLIRGEN_H
#define MULBERRY_MLIRGEN_H

#include "llvm/Support/LogicalResult.h"

namespace mlir {
class MLIRContext;
template <typename OpTy> class OwningOpRef;
class ModuleOp;
} // end namespace mlir

namespace llvm {
class SourceMgr;
} // end namespace llvm

namespace mulberry {
class Module;

auto mlirGen(const llvm::SourceMgr &sourceManager, mlir::MLIRContext &context,
             const Module &moduleAST, mlir::OwningOpRef<mlir::ModuleOp> &module)
    -> llvm::LogicalResult;

} // end namespace mulberry

#endif // MULBERRY_MLIRGEN_H
