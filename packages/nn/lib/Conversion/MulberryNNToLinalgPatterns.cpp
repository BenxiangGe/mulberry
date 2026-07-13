//===--- MulberryNNToLinalgPatterns.cpp -----------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "MulberryNN/MulberryNNToLinalgPatterns.h"
#include "MulberryNN/MulberryNNOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/APFloat.h"

namespace mlir::mulberry_nn {
namespace {

static auto createFloatZero(ConversionPatternRewriter& rewriter, Location loc,
                            FloatType type) -> Value {
  return arith::ConstantFloatOp::create(
      rewriter, loc, type, llvm::APFloat::getZero(type.getFloatSemantics()));
}

static auto createNegativeInfinity(ConversionPatternRewriter& rewriter,
                                   Location loc, FloatType type) -> Value {
  return arith::ConstantFloatOp::create(
      rewriter, loc, type,
      llvm::APFloat::getInf(type.getFloatSemantics(), /*Negative=*/true));
}

static auto createI64VectorAttr(ConversionPatternRewriter& rewriter,
                                ArrayRef<int64_t> values)
    -> DenseIntElementsAttr {
  auto type = RankedTensorType::get({static_cast<int64_t>(values.size())},
                                    rewriter.getI64Type());
  return DenseIntElementsAttr::get(type, values);
}

static auto getMemRefDim(OpBuilder& builder, Location loc, Value value,
                         int64_t dim) -> Value {
  auto type = llvm::cast<MemRefType>(value.getType());
  if (!type.isDynamicDim(dim))
    return arith::ConstantIndexOp::create(builder, loc, type.getDimSize(dim));

  auto index = arith::ConstantIndexOp::create(builder, loc, dim);
  return memref::DimOp::create(builder, loc, value, index);
}

static auto buildZeroBasedLoopNest(
    OpBuilder& builder, Location loc, ValueRange upperBounds,
    function_ref<void(OpBuilder&, Location, ValueRange)> bodyBuilder) -> void {
  std::vector<Value> lowerBounds;
  std::vector<Value> steps;
  for (size_t index = 0; index < upperBounds.size(); ++index) {
    lowerBounds.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
    steps.push_back(arith::ConstantIndexOp::create(builder, loc, 1));
  }
  scf::buildLoopNest(builder, loc, lowerBounds, upperBounds, steps,
                     bodyBuilder);
}

static auto createWindowInputIndex(OpBuilder& builder, Location loc,
                                   Value outputIndex, Value kernelIndex,
                                   int64_t stride, int64_t dilation,
                                   int64_t padding) -> Value {
  auto strideValue = arith::ConstantIndexOp::create(builder, loc, stride);
  auto dilationValue = arith::ConstantIndexOp::create(builder, loc, dilation);
  auto paddingValue = arith::ConstantIndexOp::create(builder, loc, padding);
  auto outputOffset =
      arith::MulIOp::create(builder, loc, outputIndex, strideValue);
  auto kernelOffset =
      arith::MulIOp::create(builder, loc, kernelIndex, dilationValue);
  auto paddedIndex =
      arith::AddIOp::create(builder, loc, outputOffset, kernelOffset);
  return arith::SubIOp::create(builder, loc, paddedIndex, paddingValue);
}

static auto createInBounds(OpBuilder& builder, Location loc, Value index,
                           Value upperBound) -> Value {
  auto zero = arith::ConstantIndexOp::create(builder, loc, 0);
  auto nonNegative = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::sge, index, zero);
  auto belowUpperBound = arith::CmpIOp::create(
      builder, loc, arith::CmpIPredicate::slt, index, upperBound);
  return arith::AndIOp::create(builder, loc, nonNegative, belowUpperBound);
}

static auto createRowBuffer(ConversionPatternRewriter& rewriter, Location loc,
                            Value matrix, FloatType elementType) -> Value {
  auto matrixType = llvm::cast<MemRefType>(matrix.getType());
  std::vector<int64_t> shape{matrixType.getDimSize(0)};
  std::vector<Value> dynamicSizes;
  if (matrixType.isDynamicDim(0)) {
    auto rowIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    dynamicSizes.push_back(
        memref::DimOp::create(rewriter, loc, matrix, rowIndex));
  }

  auto bufferType = MemRefType::get(shape, elementType);
  return memref::AllocOp::create(rewriter, loc, bufferType, dynamicSizes);
}

struct PaddedMemRef {
  Value value;
  Value allocation;
};

static auto hasNonZeroPadding(ArrayRef<int64_t> padding) -> bool {
  for (int64_t value : padding)
    if (value != 0)
      return true;
  return false;
}

static auto createPaddedNchwInput(ConversionPatternRewriter& rewriter,
                                  Location loc, Value input,
                                  ArrayRef<int64_t> padding, Value padValue)
    -> FailureOr<PaddedMemRef> {
  if (!hasNonZeroPadding(padding))
    return PaddedMemRef{input, {}};

  auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
  if (!inputType || inputType.getRank() != 4 || padding.size() != 4)
    return failure();

  std::vector<int64_t> paddedShape(inputType.getShape().begin(),
                                   inputType.getShape().end());
  int64_t heightPadding = padding[0] + padding[1];
  int64_t widthPadding = padding[2] + padding[3];
  if (!inputType.isDynamicDim(2))
    paddedShape[2] += heightPadding;
  if (!inputType.isDynamicDim(3))
    paddedShape[3] += widthPadding;

  std::vector<Value> dynamicSizes;
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    if (!ShapedType::isDynamic(paddedShape[dim]))
      continue;

    auto dimIndex = arith::ConstantIndexOp::create(rewriter, loc, dim);
    Value size = memref::DimOp::create(rewriter, loc, input, dimIndex);
    int64_t addedSize = dim == 2 ? heightPadding : dim == 3 ? widthPadding : 0;
    if (addedSize != 0) {
      auto paddingSize =
          arith::ConstantIndexOp::create(rewriter, loc, addedSize);
      size = arith::AddIOp::create(rewriter, loc, size, paddingSize);
    }
    dynamicSizes.push_back(size);
  }

  auto layout = MemRefLayoutAttrInterface{};
  auto paddedType = MemRefType::get(paddedShape, inputType.getElementType(),
                                    layout, inputType.getMemorySpace());
  Value padded =
      memref::AllocOp::create(rewriter, loc, paddedType, dynamicSizes);
  linalg::FillOp::create(rewriter, loc, ValueRange{padValue},
                         ValueRange{padded});

  // Named linalg window ops have no padding attribute. Copy the input into an
  // interior subview so their affine indexing can operate on a padded memref.
  std::vector<OpFoldResult> offsets{
      rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
      rewriter.getIndexAttr(padding[0]), rewriter.getIndexAttr(padding[2])};
  auto sizes = memref::getMixedSizes(rewriter, loc, input);
  std::vector<OpFoldResult> strides(inputType.getRank(),
                                    rewriter.getIndexAttr(1));
  Value interior =
      memref::SubViewOp::create(rewriter, loc, padded, offsets, sizes, strides);
  memref::CopyOp::create(rewriter, loc, input, interior);
  return PaddedMemRef{padded, padded};
}

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
    auto zero = createFloatZero(rewriter, loc, elementType);

    linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{out});

    // Use explicit generic contraction so the lowered IR owns the canonical
    // `out + lhs * rhs` body instead of relying on named matmul region magic.
    auto context = rewriter.getContext();
    auto m = rewriter.getAffineDimExpr(0);
    auto n = rewriter.getAffineDimExpr(1);
    auto k = rewriter.getAffineDimExpr(2);
    std::vector<AffineMap> indexingMaps{AffineMap::get(3, 0, {m, k}, context),
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

class HadamardConversion : public OpConversionPattern<mulberry_nn::HadamardOp> {
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

class MatscaleConversion : public OpConversionPattern<mulberry_nn::MatscaleOp> {
public:
  using OpConversionPattern<mulberry_nn::MatscaleOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MatscaleOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    // linalg.map only maps shaped operands; the scalar scale is captured as a
    // surrounding SSA value instead of materializing a temporary tensor.
    linalg::MapOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, adaptor.getOut(),
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
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, adaptor.getOut(),
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
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, adaptor.getOut(),
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
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, adaptor.getOut(),
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

class ReluConversion : public OpConversionPattern<mulberry_nn::ReluOp> {
public:
  using OpConversionPattern<mulberry_nn::ReluOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::ReluOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto outType = llvm::dyn_cast<MemRefType>(adaptor.getOut().getType());
    auto elementType = outType
                           ? llvm::dyn_cast<FloatType>(outType.getElementType())
                           : FloatType{};
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "relu needs a floating-point memref output");

    auto zero = createFloatZero(rewriter, op.getLoc(), elementType);
    linalg::MapOp::create(
        rewriter, op.getLoc(), ValueRange{adaptor.getInput()}, adaptor.getOut(),
        [&](OpBuilder& builder, Location location, ValueRange args) {
          auto result =
              arith::MaximumFOp::create(builder, location, args.front(), zero);
          linalg::YieldOp::create(builder, location, result.getResult());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

class ReluBackwardConversion
    : public OpConversionPattern<mulberry_nn::ReluBackwardOp> {
public:
  using OpConversionPattern<mulberry_nn::ReluBackwardOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::ReluBackwardOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto gradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getInputGradient().getType());
    auto elementType =
        gradientType ? llvm::dyn_cast<FloatType>(gradientType.getElementType())
                     : FloatType{};
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "relu_backward needs a floating-point memref destination");

    auto zero = createFloatZero(rewriter, op.getLoc(), elementType);
    linalg::MapOp::create(
        rewriter, op.getLoc(),
        ValueRange{adaptor.getInput(), adaptor.getOutputGradient()},
        adaptor.getInputGradient(),
        [&](OpBuilder& builder, Location location, ValueRange args) {
          auto active = arith::CmpFOp::create(
              builder, location, arith::CmpFPredicate::OGT, args[0], zero);
          auto gradient =
              arith::SelectOp::create(builder, location, active, args[1], zero);
          linalg::YieldOp::create(builder, location, gradient.getResult());
        });
    rewriter.eraseOp(op);
    return success();
  }
};

class SoftmaxConversion : public OpConversionPattern<mulberry_nn::SoftmaxOp> {
public:
  using OpConversionPattern<mulberry_nn::SoftmaxOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::SoftmaxOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto inputType = llvm::dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto outType = llvm::dyn_cast<MemRefType>(adaptor.getOut().getType());
    if (!inputType || !outType || inputType.getRank() != 2 ||
        outType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "softmax needs rank-2 memref input and output");

    auto elementType = llvm::dyn_cast<FloatType>(inputType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "softmax needs floating-point elements");

    auto loc = op.getLoc();
    // Stable softmax: reduce each row's maximum, exponentiate x - max, reduce
    // the row sum, then normalize. This avoids overflow for large logits.
    auto rowMax =
        createRowBuffer(rewriter, loc, adaptor.getInput(), elementType);
    auto rowSum =
        createRowBuffer(rewriter, loc, adaptor.getInput(), elementType);
    auto negativeInfinity = createNegativeInfinity(rewriter, loc, elementType);
    auto zero = createFloatZero(rewriter, loc, elementType);
    linalg::FillOp::create(rewriter, loc, ValueRange{negativeInfinity},
                           ValueRange{rowMax});
    linalg::FillOp::create(rewriter, loc, ValueRange{zero}, ValueRange{rowSum});

    auto context = rewriter.getContext();
    auto row = rewriter.getAffineDimExpr(0);
    auto column = rewriter.getAffineDimExpr(1);
    auto matrixMap = AffineMap::get(2, 0, {row, column}, context);
    auto rowMap = AffineMap::get(2, 0, {row}, context);
    std::vector<AffineMap> reductionMaps{matrixMap, rowMap};
    std::vector<utils::IteratorType> reductionIterators{
        utils::IteratorType::parallel, utils::IteratorType::reduction};

    linalg::GenericOp::create(
        rewriter, loc, ValueRange{adaptor.getInput()}, ValueRange{rowMax},
        reductionMaps, reductionIterators,
        [](OpBuilder& builder, Location location, ValueRange args) {
          auto maximum =
              arith::MaximumFOp::create(builder, location, args[0], args[1]);
          linalg::YieldOp::create(builder, location, maximum.getResult());
        });

    std::vector<AffineMap> pointwiseMaps{matrixMap, rowMap, matrixMap};
    std::vector<utils::IteratorType> pointwiseIterators{
        utils::IteratorType::parallel, utils::IteratorType::parallel};
    linalg::GenericOp::create(
        rewriter, loc, ValueRange{adaptor.getInput(), rowMax},
        ValueRange{adaptor.getOut()}, pointwiseMaps, pointwiseIterators,
        [](OpBuilder& builder, Location location, ValueRange args) {
          auto shifted = arith::SubFOp::create(
              builder, location, args[0], args[1], arith::FastMathFlagsAttr{});
          auto exponential =
              math::ExpOp::create(builder, location, shifted.getResult(),
                                  arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, exponential.getResult());
        });

    linalg::GenericOp::create(
        rewriter, loc, ValueRange{adaptor.getOut()}, ValueRange{rowSum},
        reductionMaps, reductionIterators,
        [](OpBuilder& builder, Location location, ValueRange args) {
          auto sum = arith::AddFOp::create(builder, location, args[0], args[1],
                                           arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, sum.getResult());
        });

    linalg::GenericOp::create(
        rewriter, loc, ValueRange{adaptor.getOut(), rowSum},
        ValueRange{adaptor.getOut()}, pointwiseMaps, pointwiseIterators,
        [](OpBuilder& builder, Location location, ValueRange args) {
          auto probability = arith::DivFOp::create(
              builder, location, args[0], args[1], arith::FastMathFlagsAttr{});
          linalg::YieldOp::create(builder, location, probability.getResult());
        });

    rewriter.eraseOp(op);
    return success();
  }
};

class SoftmaxCrossEntropyConversion
    : public OpConversionPattern<mulberry_nn::SoftmaxCrossEntropyOp> {
public:
  using OpConversionPattern<
      mulberry_nn::SoftmaxCrossEntropyOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::SoftmaxCrossEntropyOp op,
                       OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto logitsType =
        llvm::dyn_cast<MemRefType>(adaptor.getLogits().getType());
    auto expectedType =
        llvm::dyn_cast<MemRefType>(adaptor.getExpected().getType());
    if (!logitsType || logitsType.getRank() != 2 || !expectedType ||
        expectedType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "softmax cross-entropy needs rank-2 memref operands");

    auto elementType = llvm::dyn_cast<FloatType>(logitsType.getElementType());
    if (!elementType || expectedType.getElementType() != elementType)
      return rewriter.notifyMatchFailure(
          op, "softmax cross-entropy needs matching floating-point elements");

    auto loc = op.getLoc();
    auto zeroIndex = arith::ConstantIndexOp::create(rewriter, loc, 0);
    auto oneIndex = arith::ConstantIndexOp::create(rewriter, loc, 1);
    auto rowCount = getMemRefDim(rewriter, loc, adaptor.getLogits(), 0);
    auto columnCount = getMemRefDim(rewriter, loc, adaptor.getLogits(), 1);
    auto zero = createFloatZero(rewriter, loc, elementType);
    auto negativeInfinity = createNegativeInfinity(rewriter, loc, elementType);

    // Compute log-sum-exp directly from logits. Keeping every reduction in SSA
    // avoids both an intermediate probability tensor and temporary buffers.
    auto totalLoss = scf::buildLoopNest(
        rewriter, loc, ValueRange{zeroIndex}, ValueRange{rowCount},
        ValueRange{oneIndex}, ValueRange{zero},
        [&](OpBuilder& rowBuilder, Location rowLocation, ValueRange rowIndices,
            ValueRange previousTotal) -> scf::ValueVector {
          auto rowMax = scf::buildLoopNest(
              rowBuilder, rowLocation, ValueRange{zeroIndex},
              ValueRange{columnCount}, ValueRange{oneIndex},
              ValueRange{negativeInfinity},
              [&](OpBuilder& columnBuilder, Location columnLocation,
                  ValueRange columnIndices,
                  ValueRange previousMax) -> scf::ValueVector {
                std::vector<Value> indices{rowIndices[0], columnIndices[0]};
                auto logit = memref::LoadOp::create(
                    columnBuilder, columnLocation, adaptor.getLogits(),
                    indices);
                auto maximum = arith::MaximumFOp::create(
                    columnBuilder, columnLocation, logit, previousMax[0]);
                return {maximum.getResult()};
              });

          auto rowStatistics = scf::buildLoopNest(
              rowBuilder, rowLocation, ValueRange{zeroIndex},
              ValueRange{columnCount}, ValueRange{oneIndex},
              ValueRange{zero, zero},
              [&](OpBuilder& columnBuilder, Location columnLocation,
                  ValueRange columnIndices,
                  ValueRange previous) -> scf::ValueVector {
                std::vector<Value> indices{rowIndices[0], columnIndices[0]};
                auto logit = memref::LoadOp::create(
                    columnBuilder, columnLocation, adaptor.getLogits(),
                    indices);
                auto expected = memref::LoadOp::create(
                    columnBuilder, columnLocation, adaptor.getExpected(),
                    indices);
                auto shifted = arith::SubFOp::create(
                    columnBuilder, columnLocation, logit,
                    rowMax.results[0], arith::FastMathFlagsAttr{});
                auto exponential = math::ExpOp::create(
                    columnBuilder, columnLocation, shifted.getResult(),
                    arith::FastMathFlagsAttr{});
                auto exponentialSum = arith::AddFOp::create(
                    columnBuilder, columnLocation, previous[0], exponential,
                    arith::FastMathFlagsAttr{});
                auto weightedLogit = arith::MulFOp::create(
                    columnBuilder, columnLocation, expected, logit,
                    arith::FastMathFlagsAttr{});
                auto weightedLogitSum = arith::AddFOp::create(
                    columnBuilder, columnLocation, previous[1], weightedLogit,
                    arith::FastMathFlagsAttr{});
                return {exponentialSum.getResult(),
                        weightedLogitSum.getResult()};
              });

          auto logSum = math::LogOp::create(
              rowBuilder, rowLocation, rowStatistics.results[0]);
          auto logSumExp = arith::AddFOp::create(
              rowBuilder, rowLocation, logSum, rowMax.results[0],
              arith::FastMathFlagsAttr{});
          auto rowLoss = arith::SubFOp::create(
              rowBuilder, rowLocation, logSumExp,
              rowStatistics.results[1], arith::FastMathFlagsAttr{});
          auto nextTotal = arith::AddFOp::create(
              rowBuilder, rowLocation, previousTotal[0], rowLoss,
              arith::FastMathFlagsAttr{});
          return {nextTotal.getResult()};
        });

    auto rowCountI64 = arith::IndexCastUIOp::create(
        rewriter, loc, rewriter.getI64Type(), rowCount);
    auto rowCountFloat = arith::UIToFPOp::create(
        rewriter, loc, elementType, rowCountI64);
    auto meanLoss = arith::DivFOp::create(
        rewriter, loc, totalLoss.results[0], rowCountFloat,
        arith::FastMathFlagsAttr{});
    rewriter.replaceOp(op, meanLoss.getResult());
    return success();
  }
};

class Conv2DConversion : public OpConversionPattern<mulberry_nn::Conv2DOp> {
public:
  using OpConversionPattern<mulberry_nn::Conv2DOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::Conv2DOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto inputType = llvm::dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto weightType = llvm::dyn_cast<MemRefType>(adaptor.getWeight().getType());
    auto biasType = llvm::dyn_cast<MemRefType>(adaptor.getBias().getType());
    auto outType = llvm::dyn_cast<MemRefType>(adaptor.getOut().getType());
    if (!inputType || !weightType || !biasType || !outType)
      return rewriter.notifyMatchFailure(
          op, "conv2d needs memref input, weight, bias and output");

    auto elementType = llvm::dyn_cast<FloatType>(outType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "conv2d needs floating-point elements");

    auto loc = op.getLoc();
    PaddedMemRef paddedInput{adaptor.getInput(), {}};
    if (hasNonZeroPadding(op.getPadding())) {
      auto zero = createFloatZero(rewriter, loc, elementType);
      auto result = createPaddedNchwInput(rewriter, loc, adaptor.getInput(),
                                          op.getPadding(), zero);
      if (failed(result))
        return rewriter.notifyMatchFailure(op,
                                           "conv2d could not pad its input");
      paddedInput = *result;
    }

    auto context = rewriter.getContext();
    auto outputChannel = rewriter.getAffineDimExpr(1);
    auto biasMap = AffineMap::get(4, 0, {outputChannel}, context);
    auto outputMap = AffineMap::getMultiDimIdentityMap(4, context);
    std::vector<AffineMap> biasMaps{biasMap, outputMap};
    std::vector<utils::IteratorType> parallelIterators(
        4, utils::IteratorType::parallel);
    // The named convolution accumulates into `out`; initialize each output
    // channel with its bias so convolution and bias addition stay one pipeline.
    linalg::GenericOp::create(
        rewriter, loc, ValueRange{adaptor.getBias()},
        ValueRange{adaptor.getOut()}, biasMaps, parallelIterators,
        [](OpBuilder& builder, Location location, ValueRange args) {
          linalg::YieldOp::create(builder, location, args.front());
        });

    auto strides = createI64VectorAttr(rewriter, op.getStrides());
    auto dilations = createI64VectorAttr(rewriter, op.getDilations());
    linalg::Conv2DNchwFchwOp::create(
        rewriter, loc, TypeRange{},
        ValueRange{paddedInput.value, adaptor.getWeight()},
        ValueRange{adaptor.getOut()}, strides, dilations);

    rewriter.eraseOp(op);
    return success();
  }
};

class Conv2DBackwardConversion
    : public OpConversionPattern<mulberry_nn::Conv2DBackwardOp> {
public:
  using OpConversionPattern<mulberry_nn::Conv2DBackwardOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::Conv2DBackwardOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto inputType = llvm::dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto weightType = llvm::dyn_cast<MemRefType>(adaptor.getWeight().getType());
    auto outputGradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getOutputGradient().getType());
    auto inputGradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getInputGradient().getType());
    auto weightGradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getWeightGradient().getType());
    auto biasGradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getBiasGradient().getType());
    if (!inputType || inputType.getRank() != 4 || !weightType ||
        weightType.getRank() != 4 || !outputGradientType ||
        outputGradientType.getRank() != 4 || !inputGradientType ||
        inputGradientType.getRank() != 4 || !weightGradientType ||
        weightGradientType.getRank() != 4 || !biasGradientType ||
        biasGradientType.getRank() != 1)
      return rewriter.notifyMatchFailure(
          op, "conv2d_backward needs rank-4 memrefs and a rank-1 bias "
              "gradient");

    auto elementType =
        llvm::dyn_cast<FloatType>(inputGradientType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "conv2d_backward needs floating-point elements");

    auto loc = op.getLoc();
    auto zero = createFloatZero(rewriter, loc, elementType);
    linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                           ValueRange{adaptor.getInputGradient()});
    linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                           ValueRange{adaptor.getWeightGradient()});
    linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                           ValueRange{adaptor.getBiasGradient()});

    std::vector<Value> outputBounds;
    for (int64_t dim = 0; dim < 4; ++dim)
      outputBounds.push_back(
          getMemRefDim(rewriter, loc, adaptor.getOutputGradient(), dim));

    // Bias gradient reduces output gradients over batch and spatial axes.
    buildZeroBasedLoopNest(
        rewriter, loc, outputBounds,
        [&](OpBuilder& builder, Location location, ValueRange indices) {
          auto outputGradient = memref::LoadOp::create(
              builder, location, adaptor.getOutputGradient(), indices);
          auto biasGradient = memref::LoadOp::create(
              builder, location, adaptor.getBiasGradient(), indices[1]);
          auto updated = arith::AddFOp::create(builder, location, biasGradient,
                                               outputGradient);
          memref::StoreOp::create(builder, location, updated,
                                  adaptor.getBiasGradient(), indices[1]);
        });

    std::vector<Value> convolutionBounds = outputBounds;
    convolutionBounds.push_back(
        getMemRefDim(rewriter, loc, adaptor.getInput(), 1));
    convolutionBounds.push_back(
        getMemRefDim(rewriter, loc, adaptor.getWeight(), 2));
    convolutionBounds.push_back(
        getMemRefDim(rewriter, loc, adaptor.getWeight(), 3));

    auto inputHeight = getMemRefDim(rewriter, loc, adaptor.getInput(), 2);
    auto inputWidth = getMemRefDim(rewriter, loc, adaptor.getInput(), 3);
    auto padding = op.getPadding();
    auto strides = op.getStrides();
    auto dilations = op.getDilations();
    // Walk each forward convolution contribution once. The bounds check skips
    // coordinates that belonged to forward padding.
    buildZeroBasedLoopNest(
        rewriter, loc, convolutionBounds,
        [&](OpBuilder& builder, Location location, ValueRange indices) {
          auto inputHeightIndex =
              createWindowInputIndex(builder, location, indices[2], indices[5],
                                     strides[0], dilations[0], padding[0]);
          auto inputWidthIndex =
              createWindowInputIndex(builder, location, indices[3], indices[6],
                                     strides[1], dilations[1], padding[2]);
          auto heightInBounds =
              createInBounds(builder, location, inputHeightIndex, inputHeight);
          auto widthInBounds =
              createInBounds(builder, location, inputWidthIndex, inputWidth);
          auto inBounds = arith::AndIOp::create(builder, location,
                                                heightInBounds, widthInBounds);

          scf::IfOp::create(
              builder, location, inBounds,
              [&](OpBuilder& bodyBuilder, Location bodyLocation) {
                std::vector<Value> outputIndices{indices[0], indices[1],
                                                 indices[2], indices[3]};
                std::vector<Value> inputIndices{
                    indices[0], indices[4], inputHeightIndex, inputWidthIndex};
                std::vector<Value> weightIndices{indices[1], indices[4],
                                                 indices[5], indices[6]};

                auto outputGradient = memref::LoadOp::create(
                    bodyBuilder, bodyLocation, adaptor.getOutputGradient(),
                    outputIndices);
                auto input =
                    memref::LoadOp::create(bodyBuilder, bodyLocation,
                                           adaptor.getInput(), inputIndices);
                auto weight =
                    memref::LoadOp::create(bodyBuilder, bodyLocation,
                                           adaptor.getWeight(), weightIndices);

                auto inputGradient = memref::LoadOp::create(
                    bodyBuilder, bodyLocation, adaptor.getInputGradient(),
                    inputIndices);
                auto inputContribution = arith::MulFOp::create(
                    bodyBuilder, bodyLocation, outputGradient, weight);
                auto updatedInputGradient =
                    arith::AddFOp::create(bodyBuilder, bodyLocation,
                                          inputGradient, inputContribution);
                memref::StoreOp::create(
                    bodyBuilder, bodyLocation, updatedInputGradient,
                    adaptor.getInputGradient(), inputIndices);

                auto weightGradient = memref::LoadOp::create(
                    bodyBuilder, bodyLocation, adaptor.getWeightGradient(),
                    weightIndices);
                auto weightContribution = arith::MulFOp::create(
                    bodyBuilder, bodyLocation, outputGradient, input);
                auto updatedWeightGradient =
                    arith::AddFOp::create(bodyBuilder, bodyLocation,
                                          weightGradient, weightContribution);
                memref::StoreOp::create(
                    bodyBuilder, bodyLocation, updatedWeightGradient,
                    adaptor.getWeightGradient(), weightIndices);
                scf::YieldOp::create(bodyBuilder, bodyLocation);
              });
        });

    rewriter.eraseOp(op);
    return success();
  }
};

class MaxPool2DConversion
    : public OpConversionPattern<mulberry_nn::MaxPool2DOp> {
public:
  using OpConversionPattern<mulberry_nn::MaxPool2DOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MaxPool2DOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto inputType = llvm::dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto outType = llvm::dyn_cast<MemRefType>(adaptor.getOut().getType());
    if (!inputType || !outType)
      return rewriter.notifyMatchFailure(
          op, "max_pool2d needs memref input and output");

    auto elementType = llvm::dyn_cast<FloatType>(outType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "max_pool2d needs floating-point elements");

    auto loc = op.getLoc();
    auto negativeInfinity = createNegativeInfinity(rewriter, loc, elementType);
    PaddedMemRef paddedInput{adaptor.getInput(), {}};
    if (hasNonZeroPadding(op.getPadding())) {
      auto result = createPaddedNchwInput(rewriter, loc, adaptor.getInput(),
                                          op.getPadding(), negativeInfinity);
      if (failed(result))
        return rewriter.notifyMatchFailure(
            op, "max_pool2d could not pad its input");
      paddedInput = *result;
    }

    linalg::FillOp::create(rewriter, loc, ValueRange{negativeInfinity},
                           ValueRange{adaptor.getOut()});
    auto kernel = op.getKernel();
    auto windowType = MemRefType::get({kernel[0], kernel[1]}, elementType);
    // linalg pooling reads only the window operand's shape; its values are not
    // part of the max reduction.
    Value window = memref::AllocaOp::create(rewriter, loc, windowType);
    auto strides = createI64VectorAttr(rewriter, op.getStrides());
    auto dilations = createI64VectorAttr(rewriter, ArrayRef<int64_t>{1, 1});
    linalg::PoolingNchwMaxOp::create(
        rewriter, loc, TypeRange{}, ValueRange{paddedInput.value, window},
        ValueRange{adaptor.getOut()}, strides, dilations);

    rewriter.eraseOp(op);
    return success();
  }
};

class MaxPool2DBackwardConversion
    : public OpConversionPattern<mulberry_nn::MaxPool2DBackwardOp> {
public:
  using OpConversionPattern<
      mulberry_nn::MaxPool2DBackwardOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry_nn::MaxPool2DBackwardOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto inputType = llvm::dyn_cast<MemRefType>(adaptor.getInput().getType());
    auto outputGradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getOutputGradient().getType());
    auto inputGradientType =
        llvm::dyn_cast<MemRefType>(adaptor.getInputGradient().getType());
    if (!inputType || inputType.getRank() != 4 || !outputGradientType ||
        outputGradientType.getRank() != 4 || !inputGradientType ||
        inputGradientType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "max_pool2d_backward needs rank-4 memrefs");

    auto elementType =
        llvm::dyn_cast<FloatType>(inputGradientType.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "max_pool2d_backward needs floating-point elements");

    auto loc = op.getLoc();
    auto zero = createFloatZero(rewriter, loc, elementType);
    linalg::FillOp::create(rewriter, loc, ValueRange{zero},
                           ValueRange{adaptor.getInputGradient()});

    std::vector<Value> outputBounds;
    for (int64_t dim = 0; dim < 4; ++dim)
      outputBounds.push_back(
          getMemRefDim(rewriter, loc, adaptor.getOutputGradient(), dim));
    auto inputHeight = getMemRefDim(rewriter, loc, adaptor.getInput(), 2);
    auto inputWidth = getMemRefDim(rewriter, loc, adaptor.getInput(), 3);
    auto kernel = op.getKernel();
    auto padding = op.getPadding();
    auto strides = op.getStrides();

    buildZeroBasedLoopNest(
        rewriter, loc, outputBounds,
        [&](OpBuilder& builder, Location location, ValueRange outputIndices) {
          auto falseValue =
              arith::ConstantIntOp::create(builder, location, 0, 1);
          auto initialHeight =
              arith::ConstantIndexOp::create(builder, location, 0);
          auto initialWidth =
              arith::ConstantIndexOp::create(builder, location, 0);
          std::vector<Value> innerLowerBounds{
              arith::ConstantIndexOp::create(builder, location, 0),
              arith::ConstantIndexOp::create(builder, location, 0)};
          std::vector<Value> innerUpperBounds{
              arith::ConstantIndexOp::create(builder, location, kernel[0]),
              arith::ConstantIndexOp::create(builder, location, kernel[1])};
          std::vector<Value> innerSteps{
              arith::ConstantIndexOp::create(builder, location, 1),
              arith::ConstantIndexOp::create(builder, location, 1)};

          // Only a strictly greater candidate replaces the loop-carried
          // winner, so row-major traversal deterministically keeps the first
          // maximum when values tie.
          auto winners = scf::buildLoopNest(
              builder, location, innerLowerBounds, innerUpperBounds, innerSteps,
              ValueRange{falseValue, zero, initialHeight, initialWidth},
              [&](OpBuilder& bodyBuilder, Location bodyLocation,
                  ValueRange kernelIndices,
                  ValueRange previous) -> scf::ValueVector {
                auto height = createWindowInputIndex(
                    bodyBuilder, bodyLocation, outputIndices[2],
                    kernelIndices[0], strides[0], /*dilation=*/1, padding[0]);
                auto width = createWindowInputIndex(
                    bodyBuilder, bodyLocation, outputIndices[3],
                    kernelIndices[1], strides[1], /*dilation=*/1, padding[2]);
                auto heightInBounds = createInBounds(bodyBuilder, bodyLocation,
                                                     height, inputHeight);
                auto widthInBounds = createInBounds(bodyBuilder, bodyLocation,
                                                    width, inputWidth);
                auto inBounds = arith::AndIOp::create(
                    bodyBuilder, bodyLocation, heightInBounds, widthInBounds);

                auto next = scf::IfOp::create(
                    bodyBuilder, bodyLocation, inBounds,
                    [&](OpBuilder& thenBuilder, Location thenLocation) {
                      std::vector<Value> inputIndices{
                          outputIndices[0], outputIndices[1], height, width};
                      auto candidate = memref::LoadOp::create(
                          thenBuilder, thenLocation, adaptor.getInput(),
                          inputIndices);
                      auto trueValue = arith::ConstantIntOp::create(
                          thenBuilder, thenLocation, 1, 1);
                      auto notFound = arith::XOrIOp::create(
                          thenBuilder, thenLocation, previous[0], trueValue);
                      auto greater = arith::CmpFOp::create(
                          thenBuilder, thenLocation, arith::CmpFPredicate::OGT,
                          candidate, previous[1]);
                      auto shouldUpdate = arith::OrIOp::create(
                          thenBuilder, thenLocation, notFound, greater);
                      auto best = arith::SelectOp::create(
                          thenBuilder, thenLocation, shouldUpdate, candidate,
                          previous[1]);
                      auto bestHeight = arith::SelectOp::create(
                          thenBuilder, thenLocation, shouldUpdate, height,
                          previous[2]);
                      auto bestWidth = arith::SelectOp::create(
                          thenBuilder, thenLocation, shouldUpdate, width,
                          previous[3]);
                      scf::YieldOp::create(
                          thenBuilder, thenLocation,
                          ValueRange{trueValue, best, bestHeight, bestWidth});
                    },
                    [&](OpBuilder& elseBuilder, Location elseLocation) {
                      scf::YieldOp::create(elseBuilder, elseLocation, previous);
                    });
                return {next.getResult(0), next.getResult(1), next.getResult(2),
                        next.getResult(3)};
              });

          scf::IfOp::create(
              builder, location, winners.results[0],
              [&](OpBuilder& bodyBuilder, Location bodyLocation) {
                std::vector<Value> outputGradientIndices{
                    outputIndices[0], outputIndices[1], outputIndices[2],
                    outputIndices[3]};
                std::vector<Value> inputGradientIndices{
                    outputIndices[0], outputIndices[1], winners.results[2],
                    winners.results[3]};
                auto outputGradient = memref::LoadOp::create(
                    bodyBuilder, bodyLocation, adaptor.getOutputGradient(),
                    outputGradientIndices);
                auto inputGradient = memref::LoadOp::create(
                    bodyBuilder, bodyLocation, adaptor.getInputGradient(),
                    inputGradientIndices);
                auto updated = arith::AddFOp::create(
                    bodyBuilder, bodyLocation, inputGradient, outputGradient);
                memref::StoreOp::create(bodyBuilder, bodyLocation, updated,
                                        adaptor.getInputGradient(),
                                        inputGradientIndices);
                scf::YieldOp::create(bodyBuilder, bodyLocation);
              });
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

          auto isGreater = arith::CmpFOp::create(builder, location,
                                                 arith::CmpFPredicate::OGT,
                                                 candidate, previousBestValue);
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
               ReluConversion, ReluBackwardConversion, SoftmaxConversion,
               SoftmaxCrossEntropyConversion, Conv2DConversion,
               Conv2DBackwardConversion, MaxPool2DConversion,
               MaxPool2DBackwardConversion, ArgmaxConversion>(
      typeConverter, patterns.getContext());
}

} // namespace mlir::mulberry_nn
