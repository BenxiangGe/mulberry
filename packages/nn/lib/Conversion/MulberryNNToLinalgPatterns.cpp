//===--- MulberryNNToLinalgPatterns.cpp -----------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "MulberryNN/MulberryNNToLinalgPatterns.h"
#include "MulberryNN/MulberryNNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

namespace mlir::mulberry_nn {
namespace {

class MatmulConversion : public OpConversionPattern<mulberry_nn::MatmulOp> {
public:
  using OpConversionPattern<mulberry_nn::MatmulOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MatmulOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    auto out = adaptor.getOut();
    auto outType = llvm::cast<MemRefType>(out.getType());
    auto elementType = llvm::cast<FloatType>(outType.getElementType());
    auto zero = arith::ConstantFloatOp::create(
        rewriter, loc, elementType,
        llvm::APFloat::getZero(elementType.getFloatSemantics()));

    linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{out});

    // Use explicit generic contraction so the lowered IR owns the canonical
    // `out + lhs * rhs` body instead of relying on named matmul region magic.
    auto context = rewriter.getContext();
    auto m = rewriter.getAffineDimExpr(0);
    auto n = rewriter.getAffineDimExpr(1);
    auto k = rewriter.getAffineDimExpr(2);
    std::vector<AffineMap> indexingMaps{
        AffineMap::get(3, 0, {m, k}, context),
        AffineMap::get(3, 0, {k, n}, context),
        AffineMap::get(3, 0, {m, n}, context)};
    std::vector<utils::IteratorType> iteratorTypes{
        utils::IteratorType::parallel, utils::IteratorType::parallel,
        utils::IteratorType::reduction};

    linalg::GenericOp::create(
        rewriter, loc, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{out}, indexingMaps, iteratorTypes,
        [](OpBuilder& builder, Location location, ValueRange args) {
          auto product =
              arith::MulFOp::create(builder, location, args[0], args[1]);
          auto sum = arith::AddFOp::create(builder, location, args[2],
                                           product.getResult());
          linalg::YieldOp::create(builder, location, sum.getResult());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

class MataddConversion : public OpConversionPattern<mulberry_nn::MataddOp> {
public:
  using OpConversionPattern<mulberry_nn::MataddOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MataddOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::AddOp::create(rewriter, op.getLoc(), TypeRange{},
                          ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                          ValueRange{adaptor.getOut()});
    rewriter.eraseOp(op);
    return success();
  }
};

class MatsubConversion : public OpConversionPattern<mulberry_nn::MatsubOp> {
public:
  using OpConversionPattern<mulberry_nn::MatsubOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MatsubOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::SubOp::create(rewriter, op.getLoc(), TypeRange{},
                          ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                          ValueRange{adaptor.getOut()});
    rewriter.eraseOp(op);
    return success();
  }
};

class HadamardConversion
    : public OpConversionPattern<mulberry_nn::HadamardOp> {
public:
  using OpConversionPattern<mulberry_nn::HadamardOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::HadamardOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::MulOp::create(rewriter, op.getLoc(), TypeRange{},
                          ValueRange{adaptor.getLhs(), adaptor.getRhs()},
                          ValueRange{adaptor.getOut()});
    rewriter.eraseOp(op);
    return success();
  }
};

class MatscaleConversion
    : public OpConversionPattern<mulberry_nn::MatscaleOp> {
public:
  using OpConversionPattern<mulberry_nn::MatscaleOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MatscaleOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    // linalg.map only maps shaped operands; the scalar scale is captured as a
    // surrounding SSA value instead of materializing a temporary tensor.
    linalg::MapOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()},
        adaptor.getOut(),
        [&](OpBuilder& builder, Location location, ValueRange args) {
          auto scaled = arith::MulFOp::create(builder, location, args.front(),
                                              adaptor.getScale());
          linalg::YieldOp::create(builder, location, scaled.getResult());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

class TransposeConversion
    : public OpConversionPattern<mulberry_nn::TransposeOp> {
public:
  using OpConversionPattern<mulberry_nn::TransposeOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::TransposeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::TransposeOp::create(rewriter, op.getLoc(), adaptor.getInput(),
                                adaptor.getOut(),
                                llvm::ArrayRef<int64_t>{1, 0});
    rewriter.eraseOp(op);
    return success();
  }
};

class ExpConversion : public OpConversionPattern<mulberry_nn::ExpOp> {
public:
  using OpConversionPattern<mulberry_nn::ExpOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::ExpOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::MapOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()},
        adaptor.getOut(),
        [](OpBuilder& builder, Location location, ValueRange args) {
          auto exp = math::ExpOp::create(builder, location, args.front(),
                                         arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, exp.getResult());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

class SigmoidConversion : public OpConversionPattern<mulberry_nn::SigmoidOp> {
public:
  using OpConversionPattern<mulberry_nn::SigmoidOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::SigmoidOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::MapOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()},
        adaptor.getOut(),
        [](OpBuilder& builder, Location location, ValueRange args) {
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

class SigmoidPrimeConversion
    : public OpConversionPattern<mulberry_nn::SigmoidPrimeOp> {
public:
  using OpConversionPattern<mulberry_nn::SigmoidPrimeOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::SigmoidPrimeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    linalg::MapOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()},
        adaptor.getOut(),
        [](OpBuilder& builder, Location location, ValueRange args) {
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
          auto oneMinusSigmoid = arith::SubFOp::create(
              builder, location, one.getResult(), sigmoid.getResult(),
              arith::FastMathFlagsAttr{});
          // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x)).
          auto derivative = arith::MulFOp::create(
              builder, location, sigmoid.getResult(),
              oneMinusSigmoid.getResult(), arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, derivative.getResult());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

class ArgmaxConversion : public OpConversionPattern<mulberry_nn::ArgmaxOp> {
public:
  using OpConversionPattern<mulberry_nn::ArgmaxOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::ArgmaxOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto loc = op.getLoc();
    auto input = adaptor.getInput();
    auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(
          op, "argmax currently needs a ranked memref input");

    auto rank = inputType.getRank();
    if (rank != 1 && rank != 2)
      return rewriter.notifyMatchFailure(
          op, "argmax currently supports rank-1 or rank-2 inputs");

    auto elementType = llvm::dyn_cast<FloatType>(inputType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "argmax currently needs a float input element type");

    auto makeIndex = [&](int64_t value) -> Value {
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    };
    auto getDim = [&](int64_t index) -> Value {
      if (!inputType.isDynamicDim(index))
        return makeIndex(inputType.getDimSize(index));

      auto indexValue = makeIndex(index);
      return memref::DimOp::create(rewriter, loc, input, indexValue);
    };
    auto flatColCount = rank == 2 ? getDim(1) : Value{};

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

    auto inputMap =
        AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
    auto scalarMap = AffineMap::get(rank, 0, {}, rewriter.getContext());
    SmallVector<AffineMap, 3> indexingMaps{inputMap, scalarMap, scalarMap};
    SmallVector<utils::IteratorType, 2> iteratorTypes(
        rank, utils::IteratorType::reduction);

    linalg::GenericOp::create(
        rewriter, loc, ValueRange{input}, ValueRange{bestValue, bestIndex},
        indexingMaps, iteratorTypes,
        [&](OpBuilder& builder, Location location, ValueRange args) {
          auto candidate = args[0];
          auto previousBestValue = args[1];
          auto previousBestIndex = args[2];

          Value currentIndex = linalg::IndexOp::create(builder, location, 0);
          if (rank == 2) {
            auto row = currentIndex;
            auto col = linalg::IndexOp::create(builder, location, 1);
            auto rowOffset =
                arith::MulIOp::create(builder, location, row, flatColCount);
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
    auto result = arith::IndexCastOp::create(
        rewriter, loc, op.getResult().getType(), resultIndex);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

} // namespace

auto populateMulberryNNToLinalgPatterns(const TypeConverter& typeConverter,
                                      RewritePatternSet& patterns) -> void {
  patterns.add<MatmulConversion, MataddConversion, MatsubConversion,
               HadamardConversion, MatscaleConversion, TransposeConversion,
               ExpConversion, SigmoidConversion, SigmoidPrimeConversion,
               ArgmaxConversion>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::mulberry_nn
