//===--- ConvertCherryToArithCfFunc.cpp -----------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYTOARITHCFFUNC
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {
//===----------------------------------------------------------------------===//
// ConvertCherryToArithCfFunc
//===----------------------------------------------------------------------===//

struct ConvertCherryToArithCfFunc
    : public impl::ConvertCherryToArithCfFuncBase<ConvertCherryToArithCfFunc> {

  using impl::ConvertCherryToArithCfFuncBase<
      ConvertCherryToArithCfFunc>::ConvertCherryToArithCfFuncBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                           func::FuncDialect, scf::SCFDialect>();
    target.addIllegalDialect<cherry::CherryDialect>();
    target.addLegalOp<cherry::CastOp>();
    target.addLegalOp<cherry::PrintOp>();
    target.addLegalOp<cherry::StructWriteOp>();

    RewritePatternSet patterns(&getContext());

    auto f = getOperation();
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(f, target, patternSet))) {
      signalPassFailure();
    }
  }
};

} // end namespace
} // namespace mlir::cherry
