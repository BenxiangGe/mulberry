//===--- CherryPasses.td - Cherry passes ------- -----------*- tablegen -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_PASSES_TD
#define CHERRY_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertCherryToSCF : Pass<"convert-cherry-to-scf"> {
  let summary = "Convert some Cherry ops to SCF ops";
  let dependentDialects = ["scf::SCFDialect"];
}

def ConvertCherryToArithCfFunc : Pass<"convert-cherry-to-arith-cf-func"> {
  let summary = "Convert some Cherry ops to arith, cf and func ops";
  let dependentDialects = ["arith::ArithDialect", "cf::ControlFlowDialect"];
}

def LowerCherryRuntime : Pass<"lower-cherry-runtime"> {
  let summary = "Lower Cherry runtime-facing ops";
  let description = [{
    This pass lowers Cherry ops that represent runtime calls after high-level
    Mulberry lowering has finished.
  }];
  let dependentDialects = ["arith::ArithDialect", "LLVM::LLVMDialect"];
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
                           "ptr::PtrDialect", "scf::SCFDialect"];
}

def PrepareMulberryBoundaries : Pass<"prepare-mulberry-boundaries"> {
  let summary = "Prepare supported Mulberry function boundaries";
  let description = [{
    This pass rewrites supported source-level boundaries before lower-mulberry,
    such as List<Tensor> -> list_desc<TensorDesc>. Unsupported boundary shapes
    still fail fast instead of hiding the conversion inside MLIRGen or
    lower-mulberry.
  }];
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect",
                           "mulberry::MulberryDialect"];
}

#endif // CHERRY_PASSES_TD
