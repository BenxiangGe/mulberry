//===--- LowerMulberryList.cpp - Lower Mulberry list ops -----------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/Basic/Logging.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "LowerMulberryList"

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERMULBERRYLIST
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

auto getListCreate(Value list) -> mulberry::ListCreateOp {
  if (!list)
    return {};
  return list.getDefiningOp<mulberry::ListCreateOp>();
}

class ListSizeOpLowering : public OpRewritePattern<mulberry::ListSizeOp> {
public:
  using OpRewritePattern<mulberry::ListSizeOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListSizeOp op,
                       PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto createOp = getListCreate(op.getList());
    if (!createOp)
      return rewriter.notifyMatchFailure(
          op, "list.size currently requires list.create source");

    auto size =
        arith::ConstantIntOp::create(rewriter, op.getLoc(),
                                     createOp.getElements().size(), 64);
    rewriter.replaceOp(op, size);
    return success();
  }
};

class ListGetOpLowering : public OpRewritePattern<mulberry::ListGetOp> {
public:
  using OpRewritePattern<mulberry::ListGetOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListGetOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto createOp = getListCreate(op.getList());
    if (!createOp)
      return rewriter.notifyMatchFailure(
          op, "list.get currently requires list.create source");

    auto elements = createOp.getElements();
    if (elements.empty())
      return rewriter.notifyMatchFailure(op, "cannot lower empty list.get");

    std::vector<int64_t> cases;
    for (size_t i = 0; i < elements.size(); ++i)
      cases.push_back(i);

    // TODO: replace this small static-list lowering with real runtime list
    // storage. A general List<T> should lower to a descriptor like
    // {length, data}, and list[i] should load data[i] instead of expanding
    // every element into an scf.index_switch case.
    auto switchOp = scf::IndexSwitchOp::create(
        rewriter, op.getLoc(), TypeRange{op.getResult().getType()},
        op.getIndex(), cases, cases.size());

    // TODO: replace this with a runtime bounds trap when Mulberry has one.
    // MLIR requires a default region for scf.index_switch, but Sema only
    // proves the index type, not the runtime value range.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      auto &defaultRegion = switchOp.getDefaultRegion();
      auto &block = defaultRegion.emplaceBlock();
      rewriter.setInsertionPointToStart(&block);
      scf::YieldOp::create(rewriter, op.getLoc(), elements.front());
    }

    for (const auto &element : llvm::enumerate(elements)) {
      OpBuilder::InsertionGuard guard(rewriter);
      auto &caseRegion = switchOp.getCaseRegions()[element.index()];
      auto &block = caseRegion.emplaceBlock();
      rewriter.setInsertionPointToStart(&block);
      scf::YieldOp::create(rewriter, op.getLoc(), element.value());
    }

    rewriter.replaceOp(op, switchOp.getResult(0));
    return success();
  }
};

class ListCreateOpLowering
    : public OpRewritePattern<mulberry::ListCreateOp> {
public:
  using OpRewritePattern<mulberry::ListCreateOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::ListCreateOp op,
                       PatternRewriter &rewriter) const
      -> LogicalResult final {
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "list.create still has users");

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMulberryList
    : public impl::LowerMulberryListBase<LowerMulberryList> {
  using impl::LowerMulberryListBase<LowerMulberryList>::LowerMulberryListBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect>();
    target.addIllegalOp<mulberry::ListGetOp, mulberry::ListSizeOp>();
    target.addLegalOp<mulberry::ListCreateOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<ListGetOpLowering, ListSizeOpLowering>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();

    RewritePatternSet cleanupPatterns(&getContext());
    cleanupPatterns.add<ListCreateOpLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(cleanupPatterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
