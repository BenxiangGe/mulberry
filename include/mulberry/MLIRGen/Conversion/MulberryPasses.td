//===--- MulberryPasses.td - Mulberry passes --- -----------*- tablegen -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_PASSES_TD
#define MULBERRY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertMulberryToLLVM : Pass<"convert-mulberry-to-llvm"> {
  let summary = "Convert Mulberry storage-level IR to LLVM";
  let description = [{
    This pass delegates most lowering to MLIR's normal ConvertToLLVM dialect
    interfaces, but adds the memory-space conversion Mulberry needs for tensor
    ABI data pointers that use LLVM address-space attributes in the ptr dialect.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LowerMulberry : Pass<"lower-mulberry"> {
  let summary = "Lower high-level Mulberry IR to the current storage-level MLIR";
  let description = [{
    This pass stops at storage-level MLIR such as memref, linalg, math, and
    LLVM scalar/record storage. It does not perform final Mulberry-to-LLVM ABI
    lowering.
  }];
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect",
                           "LLVM::LLVMDialect", "linalg::LinalgDialect",
                           "math::MathDialect", "memref::MemRefDialect",
                           "scf::SCFDialect"];
}

#endif // MULBERRY_PASSES_TD
