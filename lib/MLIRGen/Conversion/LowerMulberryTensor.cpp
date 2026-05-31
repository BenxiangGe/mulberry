//===--- LowerMulberryTensor.cpp - Lower Mulberry tensor ops --------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/Conversion/TensorDescriptorLowering.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "LowerMulberryTensor"

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERMULBERRYTENSOR
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

class TensorUnpackOpLowering
    : public OpRewritePattern<mulberry::TensorUnpackOp> {
public:
  using OpRewritePattern<mulberry::TensorUnpackOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::TensorUnpackOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    // Round-trip through MLIR memref metadata instead of returning the source
    // memref directly. This pins the reconstruction path needed by future
    // stored tensor descriptors.
    // See: https://mlir.llvm.org/docs/Dialects/MemRef/
    auto packOp = op.getTensor().getDefiningOp<mulberry::TensorPackOp>();
    if (!packOp)
      return rewriter.notifyMatchFailure(
          op, "tensor.unpack currently requires tensor.pack source");

    auto resultType = llvm::cast<MemRefType>(op.getResult().getType());
    auto metadata = memref::ExtractStridedMetadataOp::create(
        rewriter, op.getLoc(), packOp.getTensor());
    auto result = reconstructTensorFromMetadata(op.getLoc(), resultType,
                                                metadata, rewriter);
    rewriter.replaceOp(op, result);
    return success();
  }
};

class TensorPackOpLowering : public OpRewritePattern<mulberry::TensorPackOp> {
public:
  using OpRewritePattern<mulberry::TensorPackOp>::OpRewritePattern;

  auto matchAndRewrite(mulberry::TensorPackOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    // Remaining uses mean the descriptor crossed an IR boundary that this pass
    // does not know how to materialize yet.
    if (!op->use_empty())
      return rewriter.notifyMatchFailure(op, "tensor.pack still has users");

    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMulberryTensor
    : public impl::LowerMulberryTensorBase<LowerMulberryTensor> {
  using impl::LowerMulberryTensorBase<LowerMulberryTensor>::LowerMulberryTensorBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, memref::MemRefDialect>();
    target.addIllegalOp<mulberry::TensorUnpackOp>();
    target.addLegalOp<mulberry::TensorPackOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<TensorUnpackOpLowering>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();

    RewritePatternSet cleanupPatterns(&getContext());
    cleanupPatterns.add<TensorPackOpLowering>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(),
                                     std::move(cleanupPatterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
