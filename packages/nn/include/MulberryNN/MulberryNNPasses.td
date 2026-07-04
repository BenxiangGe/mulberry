//===--- MulberryNNPasses.td - Mulberry NN package passes ----*- tablegen -*-===//

#ifndef MULBERRY_NN_PASSES_TD
#define MULBERRY_NN_PASSES_TD

include "mlir/Pass/PassBase.td"

def PrepareMulberryNNCalls : Pass<"prepare-mulberry-nn-calls", "::mlir::ModuleOp"> {
  let summary = "Rewrite Mulberry NN package calls to Mulberry NN dialect ops";
  let description = [{
    Rewrites package-level calls such as `func.call @mulberry.nn.argmax` into
    `mulberry_nn.*` dialect ops. This pass belongs to the NN package; core
    Mulberry lowering must not know these package functions.
  }];
  let dependentDialects = ["arith::ArithDialect", "func::FuncDialect",
                           "mulberry_core::MulberryDialect",
                           "memref::MemRefDialect",
                           "mulberry_nn::MulberryNNDialect"];
}

def LowerMulberryNN : Pass<"lower-mulberry-nn", "::mlir::ModuleOp"> {
  let summary = "Lower Mulberry NN package ops to linalg";
  let description = [{
    Lowers the Mulberry NN package dialect to storage-level MLIR. This pass
    belongs to the NN package; core Mulberry lowering must not know these ops.
  }];
  let dependentDialects = ["arith::ArithDialect", "linalg::LinalgDialect",
                           "math::MathDialect", "memref::MemRefDialect"];
}

#endif // MULBERRY_NN_PASSES_TD
