#include "cherry/Basic/Logging.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

#define DEBUG_TYPE "ConvertCherryNNToLinalg"

namespace mlir::cherry {

#define GEN_PASS_DEF_CONVERTCHERRYNNTOLINALG
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

class MatmulOpLowering : public OpRewritePattern<cherry_nn::MatmulOp> {
public:
  using OpRewritePattern<cherry_nn::MatmulOp>::OpRewritePattern;

  auto matchAndRewrite(cherry_nn::MatmulOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    auto outType = cast<MemRefType>(op.getOut().getType());
    auto elementType = cast<FloatType>(outType.getElementType());
    auto zero = arith::ConstantFloatOp::create(rewriter, loc, elementType, llvm::APFloat::getZero(elementType.getFloatSemantics()));

    linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{op.getOut()});
    linalg::MatmulOp::create(rewriter, loc, TypeRange{}, ValueRange{op.getLhs(), op.getRhs()}, ValueRange{op.getOut()});

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertCherryNNToLinalg : public impl::ConvertCherryNNToLinalgBase<ConvertCherryNNToLinalg> {

  using impl::ConvertCherryNNToLinalgBase<
      ConvertCherryNNToLinalg>::ConvertCherryNNToLinalgBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpLowering>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
