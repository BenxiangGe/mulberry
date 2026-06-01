#include "cherry/Basic/Logging.h"
#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
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

class ArgmaxOpLowering : public OpRewritePattern<cherry_nn::ArgmaxOp> {
public:
  using OpRewritePattern<cherry_nn::ArgmaxOp>::OpRewritePattern;

  auto matchAndRewrite(cherry_nn::ArgmaxOp op, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    auto input = op.getInput();
    auto inputType = dyn_cast<MemRefType>(input.getType());
    if (!inputType)
      return failure();

    auto rank = inputType.getRank();
    if (rank != 1 && rank != 2)
      return failure();

    auto elementType = dyn_cast<FloatType>(inputType.getElementType());
    if (!elementType)
      return failure();

    auto makeIndex = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };

    SmallVector<Value, 2> firstIndices{makeIndex(0)};
    if (rank == 2)
      firstIndices.push_back(makeIndex(0));

    auto bestValueType = MemRefType::get({}, elementType);
    auto bestIndexType = MemRefType::get({}, rewriter.getIndexType());
    auto bestValue = memref::AllocOp::create(rewriter, loc, bestValueType);
    auto bestIndex = memref::AllocOp::create(rewriter, loc, bestIndexType);
    auto firstValue =
        memref::LoadOp::create(rewriter, loc, input, firstIndices).getResult();
    auto zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    memref::StoreOp::create(rewriter, loc, firstValue, bestValue);
    memref::StoreOp::create(rewriter, loc, zeroIndex, bestIndex);

    Value dim1;
    if (rank == 2) {
      if (inputType.isDynamicDim(1))
        dim1 = memref::DimOp::create(rewriter, loc, input, 1);
      else
        dim1 = arith::ConstantIndexOp::create(rewriter, loc,
                                              inputType.getDimSize(1));
    }

    auto inputMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    auto scalarMap = AffineMap::get(rank, 0, {}, rewriter.getContext());
    SmallVector<AffineMap, 3> indexingMaps{inputMap, scalarMap, scalarMap};
    SmallVector<utils::IteratorType, 2> iteratorTypes(
        rank, utils::IteratorType::reduction);

    linalg::GenericOp::create(
        rewriter, loc, ValueRange{input}, ValueRange{bestValue, bestIndex},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &builder, Location location, ValueRange args) {
          auto candidate = args[0];
          auto previousBestValue = args[1];
          auto previousBestIndex = args[2];

          Value currentIndex = linalg::IndexOp::create(builder, location, 0);
          if (rank == 2) {
            auto row = currentIndex;
            auto col = linalg::IndexOp::create(builder, location, 1);
            auto rowOffset =
                arith::MulIOp::create(builder, location, row, dim1);
            currentIndex =
                arith::AddIOp::create(builder, location, rowOffset, col);
          }

          auto isGreater = arith::CmpFOp::create(
              builder, location, arith::CmpFPredicate::OGT, candidate,
              previousBestValue);
          auto nextBestValue =
              arith::SelectOp::create(builder, location, isGreater.getResult(),
                                      candidate, previousBestValue)
                  .getResult();
          auto nextBestIndex =
              arith::SelectOp::create(builder, location, isGreater.getResult(),
                                      currentIndex, previousBestIndex)
                  .getResult();
          linalg::YieldOp::create(builder, location,
                                  ValueRange{nextBestValue, nextBestIndex});
        });

    auto resultIndex =
        memref::LoadOp::create(rewriter, loc, bestIndex, ValueRange{})
            .getResult();
    auto result =
        arith::IndexCastOp::create(rewriter, loc, op.getResult().getType(),
                                   resultIndex)
            .getResult();
    rewriter.replaceOp(op, result);

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
                           math::MathDialect, memref::MemRefDialect>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulOpLowering, MataddOpLowering, TransposeOpLowering,
                 ExpOpLowering, SigmoidOpLowering, ArgmaxOpLowering>(
        &getContext());

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
