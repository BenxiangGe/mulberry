//===--- LowerResultTry.cpp -----------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/Conversion/MulberryPasses.h"
#include "mulberry/MLIRGen/IR/MulberryDialect.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <iterator>
#include <vector>

namespace mlir::mulberry_core {

#define GEN_PASS_DEF_LOWERRESULTTRY
#include "mulberry/MLIRGen/Conversion/MulberryPasses.h.inc"

#undef DEBUG_TYPE
#define DEBUG_TYPE "lower-result-try"

namespace {

auto containsResultTry(Operation *operation) -> bool {
  auto containsTry = false;
  operation->walk([&](ResultTryOp) { containsTry = true; });
  return containsTry;
}

auto lowerContainingSCF(func::FuncOp function) -> LogicalResult {
  RewritePatternSet patterns(function.getContext());
  populateSCFToControlFlowConversionPatterns(patterns);

  ConversionTarget target(*function.getContext());
  // Tensor bufferization still needs unrelated loops in structured form. Only
  // the SCF ancestors that must carry an Err path to the function exit need CFG.
  target.addDynamicallyLegalOp<
      scf::ForallOp, scf::ForOp, scf::IfOp, scf::IndexSwitchOp,
      scf::ParallelOp, scf::WhileOp, scf::ExecuteRegionOp>(
      [](Operation *operation) { return !containsResultTry(operation); });
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  return applyPartialConversion(function, target, std::move(patterns));
}

auto lowerResultTry(ResultTryOp op, IRRewriter &rewriter) -> LogicalResult {
  auto function = op->getParentOfType<func::FuncOp>();
  if (!function || function.getFunctionType().getNumResults() != 1)
    return op.emitOpError("needs an enclosing single-result function");

  auto location = op.getLoc();
  auto input = op.getInput();
  std::vector<Type> valueTypes(op.getResultTypes().begin(),
                               op.getResultTypes().end());
  auto errorType = op.getErrorType();
  auto functionResultType = function.getFunctionType().getResult(0);

  auto *currentBlock = op->getBlock();
  auto *continuationBlock = rewriter.splitBlock(
      currentBlock, std::next(Block::iterator(op.getOperation())));
  if (!valueTypes.empty()) {
    auto continuationValue =
        continuationBlock->addArgument(valueTypes.front(), location);
    op.getResult(0).replaceAllUsesWith(continuationValue);
  }

  auto *region = currentBlock->getParent();
  auto *okBlock = rewriter.createBlock(
      region, Region::iterator(continuationBlock));
  auto *errorBlock = rewriter.createBlock(
      region, Region::iterator(continuationBlock));

  rewriter.setInsertionPoint(op);
  auto tag = DataTagOp::create(
      rewriter, location, rewriter.getI64Type(), input);
  auto okTag = arith::ConstantIntOp::create(
      rewriter, location, 0, 64);
  auto isOk = arith::CmpIOp::create(
      rewriter, location, arith::CmpIPredicate::eq, tag, okTag);
  cf::CondBranchOp::create(
      rewriter, location, isOk, okBlock, ValueRange{}, errorBlock,
      ValueRange{});

  rewriter.setInsertionPointToEnd(okBlock);
  auto ok = DataUnpackOp::create(
      rewriter, location, valueTypes, "std.result.Ok", 0, input);
  cf::BranchOp::create(rewriter, location, continuationBlock,
                       ok.getPayloads());

  rewriter.setInsertionPointToEnd(errorBlock);
  std::vector<Type> errorTypes;
  if (!llvm::isa<NoneType>(errorType))
    errorTypes.push_back(errorType);
  auto error = DataUnpackOp::create(
      rewriter, location, errorTypes, "std.result.Err", 1, input);
  auto propagated = DataConstructOp::create(
      rewriter, location, functionResultType, "std.result.Err", 1,
      error.getPayloads());
  // Native Tensor storage has a GC finalizer, so locals skipped by this direct
  // error return safely fall back to GC-managed release.
  func::ReturnOp::create(rewriter, location, propagated.getResult());

  LLVM_DEBUG(llvm::dbgs() << "lower `?` in function `"
                          << function.getSymName() << "`\n");
  rewriter.eraseOp(op);
  return success();
}

struct LowerResultTry : public impl::LowerResultTryBase<LowerResultTry> {
  using impl::LowerResultTryBase<LowerResultTry>::LowerResultTryBase;

  auto runOnOperation() -> void final {
    std::vector<func::FuncOp> functions;
    getOperation().walk([&](ResultTryOp op) {
      auto function = op->getParentOfType<func::FuncOp>();
      if (std::find(functions.begin(), functions.end(), function) ==
          functions.end())
        functions.push_back(function);
    });

    for (auto function : functions) {
      if (failed(lowerContainingSCF(function))) {
        signalPassFailure();
        return;
      }

      std::vector<ResultTryOp> tryOps;
      function.walk([&](ResultTryOp op) { tryOps.push_back(op); });
      IRRewriter rewriter(&getContext());
      for (auto op : tryOps) {
        if (failed(lowerResultTry(op, rewriter))) {
          signalPassFailure();
          return;
        }
      }
    }
  }
};

} // namespace
} // namespace mlir::mulberry_core
