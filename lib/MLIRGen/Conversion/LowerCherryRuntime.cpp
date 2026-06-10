//===--- LowerCherryRuntime.cpp ------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERCHERRYRUNTIME
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

class PrintOpLowering : public OpRewritePattern<PrintOp> {
public:
  using OpRewritePattern<PrintOp>::OpRewritePattern;

  auto matchAndRewrite(PrintOp op, PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto inputType = op.getInput().getType();
    if (!inputType.isInteger(64))
      return rewriter.notifyMatchFailure(op, "print currently supports i64");

    auto moduleOp = op->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(op, "print needs a parent module");

    auto printFn = LLVM::lookupOrCreatePrintU64Fn(rewriter, moduleOp);
    if (failed(printFn))
      return failure();
    auto newlineFn = LLVM::lookupOrCreatePrintNewlineFn(rewriter, moduleOp);
    if (failed(newlineFn))
      return failure();

    LLVM::CallOp::create(rewriter, op.getLoc(), *printFn, op.getInput());
    LLVM::CallOp::create(rewriter, op.getLoc(), *newlineFn, ValueRange{});

    // The source-level print result is only a sequencing placeholder. The
    // runtime call carries the real side effect, so keep SSA users valid with 0.
    auto zero = arith::ConstantIntOp::create(rewriter, op.getLoc(), 0, 64);
    rewriter.replaceOp(op, zero.getResult());
    return success();
  }
};

struct LowerCherryRuntime
    : public impl::LowerCherryRuntimeBase<LowerCherryRuntime> {
  using impl::LowerCherryRuntimeBase<LowerCherryRuntime>::LowerCherryRuntimeBase;

  auto runOnOperation() -> void final {
    RewritePatternSet patterns(&getContext());
    patterns.add<PrintOpLowering>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
