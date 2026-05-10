#include "cherry/Basic/Logging.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
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
    auto zero = arith::ConstantFloatOp::create(
        rewriter, loc, elementType,
        llvm::APFloat::getZero(elementType.getFloatSemantics()));

    linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                           ValueRange{op.getOut()});
    linalg::MatmulOp::create(rewriter, loc, TypeRange{},
                             ValueRange{op.getLhs(), op.getRhs()},
                             ValueRange{op.getOut()});

    rewriter.eraseOp(op);

    return success();
  }
};

class MataddOpLowering : public OpRewritePattern<cherry_nn::MataddOp> {
public:
  using OpRewritePattern<cherry_nn::MataddOp>::OpRewritePattern;

  auto matchAndRewrite(cherry_nn::MataddOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    linalg::AddOp::create(rewriter, loc, TypeRange{},
                          ValueRange{op.getLhs(), op.getRhs()},
                          ValueRange{op.getOut()});

    rewriter.eraseOp(op);

    return success();
  }
};

class TransposeOpLowering : public OpRewritePattern<cherry_nn::TransposeOp> {
public:
  using OpRewritePattern<cherry_nn::TransposeOp>::OpRewritePattern;

  auto matchAndRewrite(cherry_nn::TransposeOp op,
                       PatternRewriter &rewriter) const -> LogicalResult final {
    auto loc = op.getLoc();
    linalg::TransposeOp::create(rewriter, loc, op.getInput(), op.getOut(),
                                llvm::ArrayRef<int64_t>{1, 0});

    rewriter.eraseOp(op);

    return success();
  }
};

class ExpOpLowering : public OpRewritePattern<cherry_nn::ExpOp> {
public:
  using OpRewritePattern<cherry_nn::ExpOp>::OpRewritePattern;

  auto matchAndRewrite(cherry_nn::ExpOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    linalg::MapOp::create(
        rewriter, loc, ValueRange{op.getInput()}, op.getOut(),
        [](OpBuilder &builder, Location location, ValueRange args) {
          auto exp = math::ExpOp::create(builder, location, args.front(),
                                         arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, exp.getResult());
        });

    rewriter.eraseOp(op);

    return success();
  }
};

class SigmoidOpLowering : public OpRewritePattern<cherry_nn::SigmoidOp> {
public:
  using OpRewritePattern<cherry_nn::SigmoidOp>::OpRewritePattern;

  auto matchAndRewrite(cherry_nn::SigmoidOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    linalg::MapOp::create(
        rewriter, loc, ValueRange{op.getInput()}, op.getOut(),
        [](OpBuilder &builder, Location location, ValueRange args) {
          auto input = args.front();
          auto elementType = input.getType();
          auto oneAttr = builder.getFloatAttr(elementType, 1.0);
          auto one = arith::ConstantOp::create(builder, location, oneAttr);
          auto neg = arith::NegFOp::create(builder, location, input,
                                           arith::FastMathFlagsAttr{});
          auto exp = math::ExpOp::create(builder, location, neg.getResult(),
                                         arith::FastMathFlagsAttr{});
          auto denominator = arith::AddFOp::create(
              builder, location, one.getResult(), exp.getResult(),
              arith::FastMathFlagsAttr{});
          auto sigmoid = arith::DivFOp::create(
              builder, location, one.getResult(), denominator.getResult(),
              arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, sigmoid.getResult());
        });

    rewriter.eraseOp(op);

    return success();
  }
};

struct ConvertCherryNNToLinalg
    : public impl::ConvertCherryNNToLinalgBase<ConvertCherryNNToLinalg> {

  using impl::ConvertCherryNNToLinalgBase<
      ConvertCherryNNToLinalg>::ConvertCherryNNToLinalgBase;

  auto runOnOperation() -> void final {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           math::MathDialect>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();
    target.addLegalOp<cherry_nn::CastOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpLowering, MataddOpLowering, TransposeOpLowering,
                 ExpOpLowering, SigmoidOpLowering>(&getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
