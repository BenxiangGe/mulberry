#include "MulberryNN/MulberryNNOps.h"
#include "MulberryNN/MulberryNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

#include <optional>
#include <vector>

using namespace mlir;
using namespace mlir::mulberry_nn;

#define GET_OP_CLASSES
#include "MulberryNN/MulberryNNOps.cpp.inc"

namespace {

enum class TensorRepresentation {
  CoreTensor,
  MemRef,
};

struct TensorInfo {
  ArrayRef<int64_t> shape;
  Type elementType;
  TensorRepresentation representation;
};

static auto getTensorInfo(Operation* op, Type type, const char* name)
    -> FailureOr<TensorInfo> {
  TensorInfo info;
  if (auto tensorType = llvm::dyn_cast<mulberry_core::TensorType>(type)) {
    info = TensorInfo{tensorType.getShape(), tensorType.getElementType(),
                      TensorRepresentation::CoreTensor};
  } else if (auto memRefType = llvm::dyn_cast<MemRefType>(type)) {
    info = TensorInfo{memRefType.getShape(), memRefType.getElementType(),
                      TensorRepresentation::MemRef};
  } else {
    op->emitOpError() << name << " must be a core tensor or ranked memref";
    return failure();
  }

  if (!info.elementType.isF32()) {
    op->emitOpError() << name << " must have Float32 elements";
    return failure();
  }
  return info;
}

static auto verifySameRepresentation(Operation* op,
                                     const std::vector<TensorInfo>& tensors)
    -> LogicalResult {
  for (size_t index = 1; index < tensors.size(); ++index) {
    if (tensors[index].representation != tensors.front().representation)
      return op->emitOpError("tensor operands must use the same core-tensor or "
                             "memref representation");
  }
  return success();
}

static auto verifyRank(Operation* op, const TensorInfo& tensor,
                       size_t expectedRank, const char* name) -> LogicalResult {
  if (tensor.shape.size() != expectedRank)
    return op->emitOpError() << name << " must have rank " << expectedRank;
  return success();
}

static auto verifyCompatibleDim(Operation* op, int64_t lhs, int64_t rhs,
                                const char* description) -> LogicalResult {
  if (!ShapedType::isDynamic(lhs) && !ShapedType::isDynamic(rhs) && lhs != rhs)
    return op->emitOpError() << description << " must match";
  return success();
}

static auto verifyCompatibleShape(Operation* op, const TensorInfo& lhs,
                                  const TensorInfo& rhs,
                                  const char* description) -> LogicalResult {
  if (lhs.shape.size() != rhs.shape.size())
    return op->emitOpError() << description << " must have the same rank";

  for (auto [lhsDim, rhsDim] : llvm::zip(lhs.shape, rhs.shape)) {
    if (failed(verifyCompatibleDim(op, lhsDim, rhsDim, description)))
      return failure();
  }
  return success();
}

static auto verifyPositive2D(Operation* op, ArrayRef<int64_t> values,
                             const char* name) -> LogicalResult {
  if (values.size() != 2)
    return op->emitOpError() << name << " must contain [height, width]";
  for (int64_t value : values)
    if (value <= 0)
      return op->emitOpError() << name << " values must be positive";
  return success();
}

static auto verifyPadding2D(Operation* op, ArrayRef<int64_t> padding)
    -> LogicalResult {
  if (padding.size() != 4)
    return op->emitOpError("padding must contain [top, bottom, left, right]");
  for (int64_t value : padding)
    if (value < 0)
      return op->emitOpError("padding values must be non-negative");
  return success();
}

static auto verifyWindowOutputDim(Operation* op, int64_t inputSize,
                                  int64_t kernelSize, int64_t outputSize,
                                  int64_t padBefore, int64_t padAfter,
                                  int64_t stride, int64_t dilation,
                                  const char* dimension) -> LogicalResult {
  if (ShapedType::isDynamic(inputSize) || ShapedType::isDynamic(kernelSize) ||
      ShapedType::isDynamic(outputSize))
    return success();

  if (kernelSize <= 0)
    return op->emitOpError() << dimension << " kernel must be non-empty";

  int64_t effectiveKernel = (kernelSize - 1) * dilation + 1;
  int64_t paddedInput = inputSize + padBefore + padAfter;
  if (paddedInput < effectiveKernel)
    return op->emitOpError()
           << dimension << " kernel does not fit the padded input";

  int64_t expectedOutput = (paddedInput - effectiveKernel) / stride + 1;
  if (outputSize != expectedOutput)
    return op->emitOpError()
           << dimension << " output must have size " << expectedOutput;
  return success();
}

static auto verifyConv2DShape(Operation* op, const TensorInfo& input,
                              const TensorInfo& weight,
                              const TensorInfo& output,
                              ArrayRef<int64_t> padding,
                              ArrayRef<int64_t> strides,
                              ArrayRef<int64_t> dilations) -> LogicalResult {
  if (failed(verifyCompatibleDim(op, input.shape[0], output.shape[0],
                                 "input and output batch dimensions")) ||
      failed(verifyCompatibleDim(op, input.shape[1], weight.shape[1],
                                 "input channels")) ||
      failed(verifyCompatibleDim(op, weight.shape[0], output.shape[1],
                                 "weight and output channels")) ||
      failed(verifyWindowOutputDim(op, input.shape[2], weight.shape[2],
                                   output.shape[2], padding[0], padding[1],
                                   strides[0], dilations[0], "height")) ||
      failed(verifyWindowOutputDim(op, input.shape[3], weight.shape[3],
                                   output.shape[3], padding[2], padding[3],
                                   strides[1], dilations[1], "width")))
    return failure();
  return success();
}

static auto verifyMaxPool2DShape(Operation* op, const TensorInfo& input,
                                 const TensorInfo& output,
                                 ArrayRef<int64_t> kernel,
                                 ArrayRef<int64_t> padding,
                                 ArrayRef<int64_t> strides) -> LogicalResult {
  if (failed(verifyCompatibleDim(op, input.shape[0], output.shape[0],
                                 "input and output batch dimensions")) ||
      failed(verifyCompatibleDim(op, input.shape[1], output.shape[1],
                                 "input and output channels")) ||
      failed(verifyWindowOutputDim(op, input.shape[2], kernel[0],
                                   output.shape[2], padding[0], padding[1],
                                   strides[0], /*dilation=*/1, "height")) ||
      failed(verifyWindowOutputDim(op, input.shape[3], kernel[1],
                                   output.shape[3], padding[2], padding[3],
                                   strides[1], /*dilation=*/1, "width")))
    return failure();
  return success();
}

static auto verifyElementwiseOp(Operation* op, Type inputType, Type outType,
                                std::optional<size_t> expectedRank)
    -> LogicalResult {
  auto input = getTensorInfo(op, inputType, "input");
  auto out = getTensorInfo(op, outType, "output");
  if (failed(input) || failed(out))
    return failure();

  std::vector<TensorInfo> tensors{*input, *out};
  if (failed(verifySameRepresentation(op, tensors)))
    return failure();
  if (expectedRank && failed(verifyRank(op, *input, *expectedRank, "input")))
    return failure();
  return verifyCompatibleShape(op, *input, *out, "input and output dimensions");
}

} // namespace

auto ReluOp::verify() -> LogicalResult {
  return verifyElementwiseOp(getOperation(), getInput().getType(),
                             getOut().getType(), std::nullopt);
}

auto SoftmaxOp::verify() -> LogicalResult {
  return verifyElementwiseOp(getOperation(), getInput().getType(),
                             getOut().getType(), 2);
}

auto SoftmaxCrossEntropyOp::verify() -> LogicalResult {
  auto logits =
      getTensorInfo(getOperation(), getLogits().getType(), "logits");
  auto expected =
      getTensorInfo(getOperation(), getExpected().getType(), "expected");
  if (failed(logits) || failed(expected))
    return failure();

  std::vector<TensorInfo> tensors{*logits, *expected};
  if (failed(verifySameRepresentation(getOperation(), tensors)) ||
      failed(verifyRank(getOperation(), *logits, 2, "logits")) ||
      failed(verifyRank(getOperation(), *expected, 2, "expected")))
    return failure();
  return verifyCompatibleShape(getOperation(), *logits, *expected,
                               "logits and expected dimensions");
}

auto Conv2DOp::verify() -> LogicalResult {
  auto input = getTensorInfo(getOperation(), getInput().getType(), "input");
  auto weight = getTensorInfo(getOperation(), getWeight().getType(), "weight");
  auto bias = getTensorInfo(getOperation(), getBias().getType(), "bias");
  auto out = getTensorInfo(getOperation(), getOut().getType(), "output");
  if (failed(input) || failed(weight) || failed(bias) || failed(out))
    return failure();

  std::vector<TensorInfo> tensors{*input, *weight, *bias, *out};
  if (failed(verifySameRepresentation(getOperation(), tensors)) ||
      failed(verifyRank(getOperation(), *input, 4, "input")) ||
      failed(verifyRank(getOperation(), *weight, 4, "weight")) ||
      failed(verifyRank(getOperation(), *bias, 1, "bias")) ||
      failed(verifyRank(getOperation(), *out, 4, "output")) ||
      failed(verifyPadding2D(getOperation(), getPadding())) ||
      failed(verifyPositive2D(getOperation(), getStrides(), "strides")) ||
      failed(verifyPositive2D(getOperation(), getDilations(), "dilations")))
    return failure();

  if (failed(verifyCompatibleDim(getOperation(), weight->shape[0],
                                 bias->shape[0], "weight and bias channels")))
    return failure();

  return verifyConv2DShape(getOperation(), *input, *weight, *out, getPadding(),
                           getStrides(), getDilations());
}

auto MaxPool2DOp::verify() -> LogicalResult {
  auto input = getTensorInfo(getOperation(), getInput().getType(), "input");
  auto out = getTensorInfo(getOperation(), getOut().getType(), "output");
  if (failed(input) || failed(out))
    return failure();

  std::vector<TensorInfo> tensors{*input, *out};
  if (failed(verifySameRepresentation(getOperation(), tensors)) ||
      failed(verifyRank(getOperation(), *input, 4, "input")) ||
      failed(verifyRank(getOperation(), *out, 4, "output")) ||
      failed(verifyPositive2D(getOperation(), getKernel(), "kernel")) ||
      failed(verifyPadding2D(getOperation(), getPadding())) ||
      failed(verifyPositive2D(getOperation(), getStrides(), "strides")))
    return failure();

  return verifyMaxPool2DShape(getOperation(), *input, *out, getKernel(),
                              getPadding(), getStrides());
}

auto ReluBackwardOp::verify() -> LogicalResult {
  auto input = getTensorInfo(getOperation(), getInput().getType(), "input");
  auto outputGradient = getTensorInfo(
      getOperation(), getOutputGradient().getType(), "output gradient");
  auto inputGradient = getTensorInfo(
      getOperation(), getInputGradient().getType(), "input gradient");
  if (failed(input) || failed(outputGradient) || failed(inputGradient))
    return failure();

  std::vector<TensorInfo> tensors{*input, *outputGradient, *inputGradient};
  if (failed(verifySameRepresentation(getOperation(), tensors)) ||
      failed(verifyCompatibleShape(getOperation(), *input, *outputGradient,
                                   "input and output-gradient dimensions")) ||
      failed(verifyCompatibleShape(getOperation(), *input, *inputGradient,
                                   "input and input-gradient dimensions")))
    return failure();
  return success();
}

auto Conv2DBackwardOp::verify() -> LogicalResult {
  auto input = getTensorInfo(getOperation(), getInput().getType(), "input");
  auto weight = getTensorInfo(getOperation(), getWeight().getType(), "weight");
  auto outputGradient = getTensorInfo(
      getOperation(), getOutputGradient().getType(), "output gradient");
  auto inputGradient = getTensorInfo(
      getOperation(), getInputGradient().getType(), "input gradient");
  auto weightGradient = getTensorInfo(
      getOperation(), getWeightGradient().getType(), "weight gradient");
  auto biasGradient = getTensorInfo(getOperation(), getBiasGradient().getType(),
                                    "bias gradient");
  if (failed(input) || failed(weight) || failed(outputGradient) ||
      failed(inputGradient) || failed(weightGradient) || failed(biasGradient))
    return failure();

  std::vector<TensorInfo> tensors{*input,          *weight,
                                  *outputGradient, *inputGradient,
                                  *weightGradient, *biasGradient};
  if (failed(verifySameRepresentation(getOperation(), tensors)) ||
      failed(verifyRank(getOperation(), *input, 4, "input")) ||
      failed(verifyRank(getOperation(), *weight, 4, "weight")) ||
      failed(
          verifyRank(getOperation(), *outputGradient, 4, "output gradient")) ||
      failed(verifyRank(getOperation(), *inputGradient, 4, "input gradient")) ||
      failed(
          verifyRank(getOperation(), *weightGradient, 4, "weight gradient")) ||
      failed(verifyRank(getOperation(), *biasGradient, 1, "bias gradient")) ||
      failed(verifyPadding2D(getOperation(), getPadding())) ||
      failed(verifyPositive2D(getOperation(), getStrides(), "strides")) ||
      failed(verifyPositive2D(getOperation(), getDilations(), "dilations")) ||
      failed(verifyCompatibleShape(getOperation(), *input, *inputGradient,
                                   "input and input-gradient dimensions")) ||
      failed(verifyCompatibleShape(getOperation(), *weight, *weightGradient,
                                   "weight and weight-gradient dimensions")) ||
      failed(verifyCompatibleDim(getOperation(), weight->shape[0],
                                 biasGradient->shape[0],
                                 "weight and bias-gradient channels")) ||
      failed(verifyConv2DShape(getOperation(), *input, *weight, *outputGradient,
                               getPadding(), getStrides(), getDilations())))
    return failure();
  return success();
}

auto MaxPool2DBackwardOp::verify() -> LogicalResult {
  auto input = getTensorInfo(getOperation(), getInput().getType(), "input");
  auto outputGradient = getTensorInfo(
      getOperation(), getOutputGradient().getType(), "output gradient");
  auto inputGradient = getTensorInfo(
      getOperation(), getInputGradient().getType(), "input gradient");
  if (failed(input) || failed(outputGradient) || failed(inputGradient))
    return failure();

  std::vector<TensorInfo> tensors{*input, *outputGradient, *inputGradient};
  if (failed(verifySameRepresentation(getOperation(), tensors)) ||
      failed(verifyRank(getOperation(), *input, 4, "input")) ||
      failed(
          verifyRank(getOperation(), *outputGradient, 4, "output gradient")) ||
      failed(verifyRank(getOperation(), *inputGradient, 4, "input gradient")) ||
      failed(verifyPositive2D(getOperation(), getKernel(), "kernel")) ||
      failed(verifyPadding2D(getOperation(), getPadding())) ||
      failed(verifyPositive2D(getOperation(), getStrides(), "strides")) ||
      failed(verifyCompatibleShape(getOperation(), *input, *inputGradient,
                                   "input and input-gradient dimensions")) ||
      failed(verifyMaxPool2DShape(getOperation(), *input, *outputGradient,
                                  getKernel(), getPadding(), getStrides())))
    return failure();
  return success();
}
