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

def ConvertCherryNNToLinalg : Pass<"convert-cherry-nn-to-linalg"> {
  let summary = "Convert Cherry NN operations to Linalg operations";
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect",
                           "math::MathDialect"];
}

def LowerMulberry : Pass<"lower-mulberry"> {
  let summary = "Lower high-level Mulberry IR to storage-level MLIR";
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect",
                           "LLVM::LLVMDialect", "scf::SCFDialect"];
}

#endif // CHERRY_PASSES_TD
