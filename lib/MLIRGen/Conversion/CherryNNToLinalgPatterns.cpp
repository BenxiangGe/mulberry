//===--- CherryNNToLinalgPatterns.cpp -------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "CherryNNToLinalgPatterns.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

namespace mlir::cherry {
namespace {

template <typename Rewriter>
static auto lowerMatmul(cherry_nn::MatmulOp op, Value lhs, Value rhs,
                        Value out, Rewriter& rewriter) -> LogicalResult {
  auto loc = op.getLoc();
  auto outType = llvm::cast<MemRefType>(out.getType());
  auto elementType = llvm::cast<FloatType>(outType.getElementType());
  auto zero = arith::ConstantFloatOp::create(
      rewriter, loc, elementType,
      llvm::APFloat::getZero(elementType.getFloatSemantics()));

  linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{out});
  linalg::MatmulOp::create(rewriter, loc, TypeRange{},
                           ValueRange{lhs, rhs}, ValueRange{out});
  rewriter.eraseOp(op);
  return success();
}

template <typename Rewriter>
static auto lowerMatadd(cherry_nn::MataddOp op, Value lhs, Value rhs,
                        Value out, Rewriter& rewriter) -> LogicalResult {
  linalg::AddOp::create(rewriter, op.getLoc(), TypeRange{},
                        ValueRange{lhs, rhs}, ValueRange{out});
  rewriter.eraseOp(op);
  return success();
}

template <typename Rewriter>
static auto lowerTranspose(cherry_nn::TransposeOp op, Value input, Value out,
                           Rewriter& rewriter) -> LogicalResult {
  linalg::TransposeOp::create(rewriter, op.getLoc(), input, out,
                              llvm::ArrayRef<int64_t>{1, 0});
  rewriter.eraseOp(op);
  return success();
}

template <typename Rewriter>
static auto lowerExp(cherry_nn::ExpOp op, Value input, Value out,
                     Rewriter& rewriter) -> LogicalResult {
  linalg::MapOp::create(
      rewriter, op.getLoc(), ValueRange{input}, out,
      [](OpBuilder& builder, Location location, ValueRange args) {
        auto exp = math::ExpOp::create(builder, location, args.front(),
                                       arith::FastMathFlagsAttr{});
        linalg::YieldOp::create(builder, location, exp.getResult());
      });
  rewriter.eraseOp(op);
  return success();
}

template <typename Rewriter>
static auto lowerSigmoid(cherry_nn::SigmoidOp op, Value input, Value out,
                         Rewriter& rewriter) -> LogicalResult {
  linalg::MapOp::create(
      rewriter, op.getLoc(), ValueRange{input}, out,
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

template <typename Rewriter>
static auto lowerArgmax(cherry_nn::ArgmaxOp op, Value input,
                        Rewriter& rewriter) -> LogicalResult {
  auto loc = op.getLoc();
  auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
  if (!inputType || !inputType.hasStaticShape())
    return rewriter.notifyMatchFailure(
        op, "argmax currently needs a static memref input");

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
          auto dim1 = arith::ConstantIndexOp::create(
              builder, location, inputType.getDimSize(1));
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
  auto result = arith::IndexCastOp::create(
      rewriter, loc, op.getResult().getType(), resultIndex);
  rewriter.replaceOp(op, result.getResult());
  return success();
}

class MatmulConversion : public OpConversionPattern<cherry_nn::MatmulOp> {
public:
  using OpConversionPattern<cherry_nn::MatmulOp>::OpConversionPattern;

  auto matchAndRewrite(cherry_nn::MatmulOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    return lowerMatmul(op, adaptor.getLhs(), adaptor.getRhs(),
                       adaptor.getOut(), rewriter);
  }
};

class MataddConversion : public OpConversionPattern<cherry_nn::MataddOp> {
public:
  using OpConversionPattern<cherry_nn::MataddOp>::OpConversionPattern;

  auto matchAndRewrite(cherry_nn::MataddOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    return lowerMatadd(op, adaptor.getLhs(), adaptor.getRhs(),
                       adaptor.getOut(), rewriter);
  }
};

class TransposeConversion
    : public OpConversionPattern<cherry_nn::TransposeOp> {
public:
  using OpConversionPattern<cherry_nn::TransposeOp>::OpConversionPattern;

  auto matchAndRewrite(cherry_nn::TransposeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    return lowerTranspose(op, adaptor.getInput(), adaptor.getOut(), rewriter);
  }
};

class ExpConversion : public OpConversionPattern<cherry_nn::ExpOp> {
public:
  using OpConversionPattern<cherry_nn::ExpOp>::OpConversionPattern;

  auto matchAndRewrite(cherry_nn::ExpOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    return lowerExp(op, adaptor.getInput(), adaptor.getOut(), rewriter);
  }
};

class SigmoidConversion : public OpConversionPattern<cherry_nn::SigmoidOp> {
public:
  using OpConversionPattern<cherry_nn::SigmoidOp>::OpConversionPattern;

  auto matchAndRewrite(cherry_nn::SigmoidOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    return lowerSigmoid(op, adaptor.getInput(), adaptor.getOut(), rewriter);
  }
};

class ArgmaxConversion : public OpConversionPattern<cherry_nn::ArgmaxOp> {
public:
  using OpConversionPattern<cherry_nn::ArgmaxOp>::OpConversionPattern;

  auto matchAndRewrite(cherry_nn::ArgmaxOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    return lowerArgmax(op, adaptor.getInput(), rewriter);
  }
};

} // namespace

auto populateCherryNNToLinalgPatterns(const TypeConverter& typeConverter,
                                      RewritePatternSet& patterns) -> void {
  patterns.add<MatmulConversion, MataddConversion, TransposeConversion,
               ExpConversion, SigmoidConversion, ArgmaxConversion>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::cherry
