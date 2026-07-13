//===--- MulberryNNPasses.cpp - Mulberry NN package passes ----------------===//

#include "MulberryNN/MulberryNNPasses.h"
#include "MulberryNN/MulberryNNDialect.h"
#include "MulberryNN/MulberryNNOps.h"
#include "MulberryNN/MulberryNNToLinalgPatterns.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"

namespace mlir::mulberry_nn {

#define GEN_PASS_DEF_PREPAREMULBERRYNNCALLS
#define GEN_PASS_DEF_LOWERMULBERRYNN
#include "MulberryNN/MulberryNNPasses.h.inc"

namespace {

static auto getCallee(func::CallOp call) -> StringRef {
  return call.getCallee();
}

static auto getPublicCalleeName(StringRef name) -> std::string {
  std::string publicName = "mulberry.nn.";
  publicName += name.str();
  return publicName;
}

static auto getTensorCalleeName(StringRef name) -> std::string {
  std::string callee = "mulberry.nn.__tensor.";
  callee += name.str();
  return callee;
}

static auto isPublicMulberryNNCall(func::CallOp call, StringRef name) -> bool {
  return getCallee(call) == getPublicCalleeName(name);
}

static auto isTensorMulberryNNCall(func::CallOp call, StringRef name) -> bool {
  return getCallee(call) == getTensorCalleeName(name);
}

static auto getOrCreateTensorCallee(PatternRewriter &rewriter, Location loc,
                                    func::CallOp call, StringRef name,
                                    TypeRange inputs, TypeRange results)
    -> FlatSymbolRefAttr {
  auto module = call->getParentOfType<ModuleOp>();
  auto callee = getTensorCalleeName(name);
  if (!module.lookupSymbol<func::FuncOp>(callee)) {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(module.getBody());
    auto funcType = rewriter.getFunctionType(inputs, results);
    auto func = func::FuncOp::create(loc, callee, funcType);
    SymbolTable::setSymbolVisibility(func, SymbolTable::Visibility::Private);
    rewriter.insert(func);
  }
  return SymbolRefAttr::get(rewriter.getContext(), callee);
}

static auto allocateLike(PatternRewriter &rewriter, Location loc, Value source,
                         MemRefType resultType) -> Value {
  std::vector<Value> dynamicSizes;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;
    auto dimIndex = arith::ConstantIndexOp::create(rewriter, loc, index);
    dynamicSizes.push_back(
        memref::DimOp::create(rewriter, loc, source, dimIndex));
  }
  return memref::AllocOp::create(rewriter, loc, resultType, dynamicSizes);
}

static auto createDim(PatternRewriter &rewriter, Location loc, Value source,
                      int64_t index) -> Value {
  auto dimIndex = arith::ConstantIndexOp::create(rewriter, loc, index);
  return memref::DimOp::create(rewriter, loc, source, dimIndex);
}

static auto allocateMatmul(PatternRewriter &rewriter, Location loc, Value lhs,
                           Value rhs, MemRefType resultType) -> Value {
  std::vector<Value> dynamicSizes;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;

    if (index == 0) {
      dynamicSizes.push_back(createDim(rewriter, loc, lhs, 0));
    } else if (index == 1) {
      dynamicSizes.push_back(createDim(rewriter, loc, rhs, 1));
    } else {
      return nullptr;
    }
  }
  return memref::AllocOp::create(rewriter, loc, resultType, dynamicSizes);
}

static auto allocateTranspose(PatternRewriter &rewriter, Location loc,
                              Value input, MemRefType resultType) -> Value {
  std::vector<Value> dynamicSizes;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;

    if (index == 0) {
      dynamicSizes.push_back(createDim(rewriter, loc, input, 1));
    } else if (index == 1) {
      dynamicSizes.push_back(createDim(rewriter, loc, input, 0));
    } else {
      return nullptr;
    }
  }
  return memref::AllocOp::create(rewriter, loc, resultType, dynamicSizes);
}

static auto createWindowOutputSize(PatternRewriter &rewriter, Location loc,
                                   Value inputSize, Value kernelSize,
                                   int64_t stride) -> Value {
  Value outputSize =
      arith::SubIOp::create(rewriter, loc, inputSize, kernelSize);
  if (stride != 1) {
    auto strideValue = arith::ConstantIndexOp::create(rewriter, loc, stride);
    outputSize = arith::DivUIOp::create(rewriter, loc, outputSize, strideValue);
  }
  auto one = arith::ConstantIndexOp::create(rewriter, loc, 1);
  return arith::AddIOp::create(rewriter, loc, outputSize, one);
}

static auto allocateConv2D(PatternRewriter &rewriter, Location loc, Value input,
                           Value weight, MemRefType resultType) -> Value {
  auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
  auto weightType = llvm::dyn_cast<MemRefType>(weight.getType());
  if (!inputType || !weightType || inputType.getRank() != 4 ||
      weightType.getRank() != 4 || resultType.getRank() != 4)
    return nullptr;

  std::vector<Value> dynamicSizes;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;

    if (index == 0) {
      dynamicSizes.push_back(createDim(rewriter, loc, input, 0));
    } else if (index == 1) {
      dynamicSizes.push_back(createDim(rewriter, loc, weight, 0));
    } else if (index == 2) {
      dynamicSizes.push_back(createWindowOutputSize(
          rewriter, loc, createDim(rewriter, loc, input, 2),
          createDim(rewriter, loc, weight, 2), 1));
    } else if (index == 3) {
      dynamicSizes.push_back(createWindowOutputSize(
          rewriter, loc, createDim(rewriter, loc, input, 3),
          createDim(rewriter, loc, weight, 3), 1));
    } else {
      return nullptr;
    }
  }
  return memref::AllocOp::create(rewriter, loc, resultType, dynamicSizes);
}

static auto allocateMaxPool2D(PatternRewriter &rewriter, Location loc,
                              Value input, int64_t height, int64_t width,
                              MemRefType resultType) -> Value {
  auto inputType = llvm::dyn_cast<MemRefType>(input.getType());
  if (!inputType || inputType.getRank() != 4 || resultType.getRank() != 4)
    return nullptr;

  std::vector<Value> dynamicSizes;
  for (auto [index, dim] : llvm::enumerate(resultType.getShape())) {
    if (!ShapedType::isDynamic(dim))
      continue;

    if (index == 0 || index == 1) {
      dynamicSizes.push_back(createDim(rewriter, loc, input, index));
    } else if (index == 2) {
      auto kernel = arith::ConstantIndexOp::create(rewriter, loc, height);
      dynamicSizes.push_back(createWindowOutputSize(
          rewriter, loc, createDim(rewriter, loc, input, 2), kernel, height));
    } else if (index == 3) {
      auto kernel = arith::ConstantIndexOp::create(rewriter, loc, width);
      dynamicSizes.push_back(createWindowOutputSize(
          rewriter, loc, createDim(rewriter, loc, input, 3), kernel, width));
    } else {
      return nullptr;
    }
  }
  return memref::AllocOp::create(rewriter, loc, resultType, dynamicSizes);
}

static auto getPositiveI64Constant(Value value) -> FailureOr<int64_t> {
  llvm::APInt constant;
  if (!matchPattern(value, m_ConstantInt(&constant)) || constant.isZero() ||
      constant.getActiveBits() > 63)
    return failure();
  return static_cast<int64_t>(constant.getZExtValue());
}

static auto isListU64Reference(Type type) -> bool {
  auto ptrType = llvm::dyn_cast<mlir::mulberry_core::PtrType>(type);
  if (!ptrType)
    return false;

  auto recordType =
      llvm::dyn_cast<mlir::mulberry_core::RecordType>(ptrType.getPointeeType());
  if (!recordType)
    return false;

  auto dataType = llvm::dyn_cast<mlir::mulberry_core::PtrType>(
      recordType.getFieldType("data"));
  auto lengthType = recordType.getFieldType("length");
  auto capacityType = recordType.getFieldType("capacity");
  return lengthType && lengthType.isInteger(64) && capacityType &&
         capacityType.isInteger(64) && dataType &&
         dataType.getPointeeType().isInteger(64);
}

static auto isTensorF32RecordABI(Type type) -> bool {
  auto recordType = llvm::dyn_cast<mlir::mulberry_core::RecordType>(type);
  if (!recordType)
    return false;

  auto ptrType = llvm::dyn_cast<mlir::mulberry_core::PtrType>(
      recordType.getFieldType("data"));
  if (!ptrType || !ptrType.getPointeeType().isF32())
    return false;

  auto rankType = recordType.getFieldType("rank");
  auto numelType = recordType.getFieldType("numel");
  return rankType && rankType.isInteger(64) && numelType &&
         numelType.isInteger(64) &&
         isListU64Reference(recordType.getFieldType("sizes")) &&
         isListU64Reference(recordType.getFieldType("strides"));
}

static auto getTensorType(MLIRContext *context, size_t rank)
    -> mlir::mulberry_core::TensorType {
  std::vector<int64_t> shape(rank, ShapedType::kDynamic);
  return mlir::mulberry_core::TensorType::get(context, shape,
                                              Float32Type::get(context));
}

static auto createTensorView(PatternRewriter &rewriter, Location loc,
                             Value tensorRecord, size_t rank) -> Value {
  auto tensorType = getTensorType(rewriter.getContext(), rank);
  return mlir::mulberry_core::TensorViewOp::create(rewriter, loc, tensorType,
                                                   tensorRecord);
}

static auto createTensorCall(PatternRewriter &rewriter, Location loc,
                             func::CallOp call, StringRef name,
                             ValueRange inputs, TypeRange results)
    -> func::CallOp {
  auto callee = getOrCreateTensorCallee(rewriter, loc, call, name,
                                        inputs.getTypes(), results);
  return func::CallOp::create(rewriter, loc, callee, results, inputs);
}

static auto packTensor(PatternRewriter &rewriter, Location loc, Value tensor,
                       Type tensorRecordType) -> Value {
  Value packed = mlir::mulberry_core::TensorPackOp::create(
      rewriter, loc, tensorRecordType, tensor);
  mlir::mulberry_core::TensorReleaseOp::create(rewriter, loc, tensor);
  return packed;
}

enum class BinaryOperation {
  Matmul,
  Add,
  Subtract,
  Hadamard,
};

static auto rewriteBinaryCall(func::CallOp call, PatternRewriter &rewriter,
                              StringRef name, BinaryOperation operation)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, name))
    return failure();
  if (call.getNumOperands() != 2 || call.getNumResults() != 1)
    return failure();
  auto lhsType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto rhsType = llvm::dyn_cast<MemRefType>(call.getOperand(1).getType());
  auto resultType = llvm::dyn_cast<MemRefType>(call.getResult(0).getType());
  if (!lhsType || !rhsType || !resultType)
    return failure();

  auto loc = call.getLoc();
  Value out;
  if (operation == BinaryOperation::Matmul) {
    out = allocateMatmul(rewriter, loc, call.getOperand(0), call.getOperand(1),
                         resultType);
    if (!out)
      return failure();
    MatmulOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                     out);
  } else if (operation == BinaryOperation::Add) {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    MataddOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                     out);
  } else if (operation == BinaryOperation::Subtract) {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    MatsubOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                     out);
  } else if (operation == BinaryOperation::Hadamard) {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    HadamardOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                       out);
  } else {
    return failure();
  }
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteBinaryRecordABICall(func::CallOp call,
                                       PatternRewriter &rewriter,
                                       StringRef name, size_t rank)
    -> LogicalResult {
  if (!isPublicMulberryNNCall(call, name))
    return failure();
  if (call.getNumOperands() != 2 || call.getNumResults() != 1)
    return failure();
  if (!isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !isTensorF32RecordABI(call.getOperand(1).getType()) ||
      !isTensorF32RecordABI(call.getResult(0).getType()))
    return failure();

  auto loc = call.getLoc();
  auto lhs = createTensorView(rewriter, loc, call.getOperand(0), rank);
  auto rhs = createTensorView(rewriter, loc, call.getOperand(1), rank);
  auto resultType = getTensorType(rewriter.getContext(), rank);
  auto result = createTensorCall(rewriter, loc, call, name,
                                 ValueRange{lhs, rhs}, TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

static auto rewriteScaleCall(func::CallOp call, PatternRewriter &rewriter,
                             StringRef name) -> LogicalResult {
  if (!isTensorMulberryNNCall(call, name) || call.getNumOperands() != 2 ||
      call.getNumResults() != 1)
    return failure();

  auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto resultType = llvm::dyn_cast<MemRefType>(call.getResult(0).getType());
  if (!inputType || !resultType || !call.getOperand(1).getType().isF32())
    return failure();

  auto out =
      allocateLike(rewriter, call.getLoc(), call.getOperand(0), resultType);
  MatscaleOp::create(rewriter, call.getLoc(), call.getOperand(0),
                     call.getOperand(1), out);
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteScaleRecordABICall(func::CallOp call,
                                      PatternRewriter &rewriter, StringRef name,
                                      size_t rank) -> LogicalResult {
  if (!isPublicMulberryNNCall(call, name) || call.getNumOperands() != 2 ||
      call.getNumResults() != 1 ||
      !isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !call.getOperand(1).getType().isF32() ||
      !isTensorF32RecordABI(call.getResult(0).getType()))
    return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), rank);
  auto resultType = getTensorType(rewriter.getContext(), rank);
  auto result = createTensorCall(rewriter, loc, call, name,
                                 ValueRange{input, call.getOperand(1)},
                                 TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

static auto rewriteUnaryCall(func::CallOp call, PatternRewriter &rewriter,
                             StringRef name) -> LogicalResult {
  if (!isTensorMulberryNNCall(call, name))
    return failure();
  if (call.getNumOperands() != 1 || call.getNumResults() != 1)
    return failure();
  auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto resultType = llvm::dyn_cast<MemRefType>(call.getResult(0).getType());
  if (!inputType || !resultType)
    return failure();

  auto loc = call.getLoc();
  Value out;
  if (name == "transpose") {
    out = allocateTranspose(rewriter, loc, call.getOperand(0), resultType);
    if (!out)
      return failure();
    TransposeOp::create(rewriter, loc, call.getOperand(0), out);
  } else if (name == "exp") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    ExpOp::create(rewriter, loc, call.getOperand(0), out);
  } else if (name == "sigmoid") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    SigmoidOp::create(rewriter, loc, call.getOperand(0), out);
  } else if (name == "sigmoidPrime") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    SigmoidPrimeOp::create(rewriter, loc, call.getOperand(0), out);
  } else if (name == "relu") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    ReluOp::create(rewriter, loc, call.getOperand(0), out);
  } else if (name == "softmax") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    SoftmaxOp::create(rewriter, loc, call.getOperand(0), out);
  } else {
    return failure();
  }
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteUnaryRecordABICall(func::CallOp call,
                                      PatternRewriter &rewriter, StringRef name,
                                      size_t rank) -> LogicalResult {
  if (!isPublicMulberryNNCall(call, name))
    return failure();
  if (call.getNumOperands() != 1 || call.getNumResults() != 1)
    return failure();
  if (!isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !isTensorF32RecordABI(call.getResult(0).getType()))
    return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), rank);
  auto resultType = getTensorType(rewriter.getContext(), rank);
  auto result = createTensorCall(rewriter, loc, call, name, ValueRange{input},
                                 TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

static auto rewriteSoftmaxCrossEntropyCall(func::CallOp call,
                                           PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, "softmaxCrossEntropy") ||
      call.getNumOperands() != 2 || call.getNumResults() != 1 ||
      !call.getResult(0).getType().isF32())
    return failure();

  auto logitsType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto expectedType = llvm::dyn_cast<MemRefType>(call.getOperand(1).getType());
  if (!logitsType || logitsType.getRank() != 2 || !expectedType ||
      expectedType.getRank() != 2)
    return failure();

  auto result = SoftmaxCrossEntropyOp::create(
      rewriter, call.getLoc(), call.getResult(0).getType(),
      call.getOperand(0), call.getOperand(1));
  rewriter.replaceOp(call, result.getResult());
  return success();
}

static auto rewriteSoftmaxCrossEntropyRecordABICall(
    func::CallOp call, PatternRewriter &rewriter) -> LogicalResult {
  if (!isPublicMulberryNNCall(call, "softmaxCrossEntropy") ||
      call.getNumOperands() != 2 || call.getNumResults() != 1 ||
      !isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !isTensorF32RecordABI(call.getOperand(1).getType()) ||
      !call.getResult(0).getType().isF32())
    return failure();

  auto loc = call.getLoc();
  auto logits = createTensorView(rewriter, loc, call.getOperand(0), 2);
  auto expected = createTensorView(rewriter, loc, call.getOperand(1), 2);
  auto result = createTensorCall(
      rewriter, loc, call, "softmaxCrossEntropy",
      ValueRange{logits, expected}, TypeRange{call.getResult(0).getType()});
  rewriter.replaceOp(call, result.getResult(0));
  return success();
}

static auto rewriteConv2DCall(func::CallOp call, PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, "conv2d") || call.getNumOperands() != 3 ||
      call.getNumResults() != 1)
    return failure();

  auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto weightType = llvm::dyn_cast<MemRefType>(call.getOperand(1).getType());
  auto biasType = llvm::dyn_cast<MemRefType>(call.getOperand(2).getType());
  auto resultType = llvm::dyn_cast<MemRefType>(call.getResult(0).getType());
  if (!inputType || inputType.getRank() != 4 || !weightType ||
      weightType.getRank() != 4 || !biasType || biasType.getRank() != 1 ||
      !resultType || resultType.getRank() != 4)
    return failure();

  auto loc = call.getLoc();
  auto out = allocateConv2D(rewriter, loc, call.getOperand(0),
                            call.getOperand(1), resultType);
  if (!out)
    return failure();

  auto padding = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
  auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
  auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
  Conv2DOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                   call.getOperand(2), out, padding, strides, dilations);
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteConv2DRecordABICall(func::CallOp call,
                                       PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isPublicMulberryNNCall(call, "conv2d") || call.getNumOperands() != 3 ||
      call.getNumResults() != 1)
    return failure();
  if (!isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !isTensorF32RecordABI(call.getOperand(1).getType()) ||
      !isTensorF32RecordABI(call.getOperand(2).getType()) ||
      !isTensorF32RecordABI(call.getResult(0).getType()))
    return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), 4);
  auto weight = createTensorView(rewriter, loc, call.getOperand(1), 4);
  auto bias = createTensorView(rewriter, loc, call.getOperand(2), 1);
  auto resultType = getTensorType(rewriter.getContext(), 4);
  auto result =
      createTensorCall(rewriter, loc, call, "conv2d",
                       ValueRange{input, weight, bias}, TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

static auto rewriteMaxPool2DCall(func::CallOp call, PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, "maxPool2d") ||
      call.getNumOperands() != 3 || call.getNumResults() != 1)
    return failure();

  auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto resultType = llvm::dyn_cast<MemRefType>(call.getResult(0).getType());
  auto height = getPositiveI64Constant(call.getOperand(1));
  auto width = getPositiveI64Constant(call.getOperand(2));
  if (!inputType || inputType.getRank() != 4 || !resultType ||
      resultType.getRank() != 4 || failed(height) || failed(width))
    return failure();

  auto loc = call.getLoc();
  auto out = allocateMaxPool2D(rewriter, loc, call.getOperand(0), *height,
                               *width, resultType);
  if (!out)
    return failure();

  auto kernel = rewriter.getDenseI64ArrayAttr({*height, *width});
  auto padding = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
  auto strides = rewriter.getDenseI64ArrayAttr({*height, *width});
  MaxPool2DOp::create(rewriter, loc, call.getOperand(0), out, kernel, padding,
                      strides);
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteMaxPool2DRecordABICall(func::CallOp call,
                                          PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isPublicMulberryNNCall(call, "maxPool2d") ||
      call.getNumOperands() != 3 || call.getNumResults() != 1)
    return failure();
  if (!isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !call.getOperand(1).getType().isInteger(64) ||
      !call.getOperand(2).getType().isInteger(64) ||
      !isTensorF32RecordABI(call.getResult(0).getType()) ||
      failed(getPositiveI64Constant(call.getOperand(1))) ||
      failed(getPositiveI64Constant(call.getOperand(2))))
    return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), 4);
  auto resultType = getTensorType(rewriter.getContext(), 4);
  auto result = createTensorCall(
      rewriter, loc, call, "maxPool2d",
      ValueRange{input, call.getOperand(1), call.getOperand(2)},
      TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

static auto rewriteReluBackwardIntoCall(func::CallOp call,
                                        PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, "reluBackwardInto") ||
      call.getNumOperands() != 3 || call.getNumResults() != 0)
    return failure();
  for (Value operand : call.getOperands())
    if (!llvm::isa<MemRefType>(operand.getType()))
      return failure();

  ReluBackwardOp::create(rewriter, call.getLoc(), call.getOperand(0),
                         call.getOperand(1), call.getOperand(2));
  rewriter.eraseOp(call);
  return success();
}

static auto rewriteReluBackwardIntoRecordABICall(func::CallOp call,
                                                 PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isPublicMulberryNNCall(call, "reluBackwardInto") ||
      call.getNumOperands() != 3 || call.getNumResults() != 0)
    return failure();
  for (Value operand : call.getOperands())
    if (!isTensorF32RecordABI(operand.getType()))
      return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), 4);
  auto outputGradient = createTensorView(rewriter, loc, call.getOperand(1), 4);
  auto inputGradient = createTensorView(rewriter, loc, call.getOperand(2), 4);
  createTensorCall(rewriter, loc, call, "reluBackwardInto",
                   ValueRange{input, outputGradient, inputGradient},
                   TypeRange{});
  rewriter.eraseOp(call);
  return success();
}

static auto rewriteConv2DBackwardIntoCall(func::CallOp call,
                                          PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, "conv2dBackwardInto") ||
      call.getNumOperands() != 6 || call.getNumResults() != 0)
    return failure();

  auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto weightType = llvm::dyn_cast<MemRefType>(call.getOperand(1).getType());
  auto outputGradientType =
      llvm::dyn_cast<MemRefType>(call.getOperand(2).getType());
  auto inputGradientType =
      llvm::dyn_cast<MemRefType>(call.getOperand(3).getType());
  auto weightGradientType =
      llvm::dyn_cast<MemRefType>(call.getOperand(4).getType());
  auto biasGradientType =
      llvm::dyn_cast<MemRefType>(call.getOperand(5).getType());
  if (!inputType || inputType.getRank() != 4 || !weightType ||
      weightType.getRank() != 4 || !outputGradientType ||
      outputGradientType.getRank() != 4 || !inputGradientType ||
      inputGradientType.getRank() != 4 || !weightGradientType ||
      weightGradientType.getRank() != 4 || !biasGradientType ||
      biasGradientType.getRank() != 1)
    return failure();

  auto padding = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
  auto strides = rewriter.getDenseI64ArrayAttr({1, 1});
  auto dilations = rewriter.getDenseI64ArrayAttr({1, 1});
  Conv2DBackwardOp::create(rewriter, call.getLoc(), call.getOperand(0),
                           call.getOperand(1), call.getOperand(2),
                           call.getOperand(3), call.getOperand(4),
                           call.getOperand(5), padding, strides, dilations);
  rewriter.eraseOp(call);
  return success();
}

static auto rewriteConv2DBackwardIntoRecordABICall(func::CallOp call,
                                                   PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isPublicMulberryNNCall(call, "conv2dBackwardInto") ||
      call.getNumOperands() != 6 || call.getNumResults() != 0)
    return failure();
  for (Value operand : call.getOperands())
    if (!isTensorF32RecordABI(operand.getType()))
      return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), 4);
  auto weight = createTensorView(rewriter, loc, call.getOperand(1), 4);
  auto outputGradient = createTensorView(rewriter, loc, call.getOperand(2), 4);
  auto inputGradient = createTensorView(rewriter, loc, call.getOperand(3), 4);
  auto weightGradient = createTensorView(rewriter, loc, call.getOperand(4), 4);
  auto biasGradient = createTensorView(rewriter, loc, call.getOperand(5), 1);
  createTensorCall(rewriter, loc, call, "conv2dBackwardInto",
                   ValueRange{input, weight, outputGradient, inputGradient,
                              weightGradient, biasGradient},
                   TypeRange{});
  rewriter.eraseOp(call);
  return success();
}

static auto rewriteMaxPool2DBackwardIntoCall(func::CallOp call,
                                             PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isTensorMulberryNNCall(call, "maxPool2dBackwardInto") ||
      call.getNumOperands() != 5 || call.getNumResults() != 0)
    return failure();

  auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
  auto outputGradientType =
      llvm::dyn_cast<MemRefType>(call.getOperand(1).getType());
  auto height = getPositiveI64Constant(call.getOperand(2));
  auto width = getPositiveI64Constant(call.getOperand(3));
  auto inputGradientType =
      llvm::dyn_cast<MemRefType>(call.getOperand(4).getType());
  if (!inputType || inputType.getRank() != 4 || !outputGradientType ||
      outputGradientType.getRank() != 4 || failed(height) || failed(width) ||
      !inputGradientType || inputGradientType.getRank() != 4)
    return failure();

  auto kernel = rewriter.getDenseI64ArrayAttr({*height, *width});
  auto padding = rewriter.getDenseI64ArrayAttr({0, 0, 0, 0});
  auto strides = rewriter.getDenseI64ArrayAttr({*height, *width});
  MaxPool2DBackwardOp::create(rewriter, call.getLoc(), call.getOperand(0),
                              call.getOperand(1), call.getOperand(4), kernel,
                              padding, strides);
  rewriter.eraseOp(call);
  return success();
}

static auto rewriteMaxPool2DBackwardIntoRecordABICall(func::CallOp call,
                                                      PatternRewriter &rewriter)
    -> LogicalResult {
  if (!isPublicMulberryNNCall(call, "maxPool2dBackwardInto") ||
      call.getNumOperands() != 5 || call.getNumResults() != 0)
    return failure();
  if (!isTensorF32RecordABI(call.getOperand(0).getType()) ||
      !isTensorF32RecordABI(call.getOperand(1).getType()) ||
      !call.getOperand(2).getType().isInteger(64) ||
      !call.getOperand(3).getType().isInteger(64) ||
      failed(getPositiveI64Constant(call.getOperand(2))) ||
      failed(getPositiveI64Constant(call.getOperand(3))) ||
      !isTensorF32RecordABI(call.getOperand(4).getType()))
    return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0), 4);
  auto outputGradient = createTensorView(rewriter, loc, call.getOperand(1), 4);
  auto inputGradient = createTensorView(rewriter, loc, call.getOperand(4), 4);
  createTensorCall(rewriter, loc, call, "maxPool2dBackwardInto",
                   ValueRange{input, outputGradient, call.getOperand(2),
                              call.getOperand(3), inputGradient},
                   TypeRange{});
  rewriter.eraseOp(call);
  return success();
}

class MulberryNNCallRewrite : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  auto matchAndRewrite(func::CallOp call, PatternRewriter &rewriter) const
      -> LogicalResult final {
    auto loc = call.getLoc();

    if (isTensorMulberryNNCall(call, "argmax")) {
      if (call.getNumOperands() != 1 || call.getNumResults() != 1 ||
          !call.getResult(0).getType().isInteger(64))
        return failure();

      Value input = call.getOperand(0);
      if (llvm::isa<MemRefType>(input.getType())) {
        auto result =
            ArgmaxOp::create(rewriter, loc, call.getResult(0).getType(), input);
        rewriter.replaceOp(call, result.getResult());
        return success();
      }

      return failure();
    }

    if (isPublicMulberryNNCall(call, "argmax")) {
      if (call.getNumOperands() != 1 || call.getNumResults() != 1 ||
          !call.getResult(0).getType().isInteger(64) ||
          !isTensorF32RecordABI(call.getOperand(0).getType()))
        return failure();

      auto view = createTensorView(rewriter, loc, call.getOperand(0), 2);
      auto result =
          createTensorCall(rewriter, loc, call, "argmax", ValueRange{view},
                           TypeRange{call.getResult(0).getType()});
      rewriter.replaceOp(call, result.getResult(0));
      return success();
    }

    if (succeeded(rewriteBinaryCall(call, rewriter, "matmul",
                                    BinaryOperation::Matmul)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "matadd",
                                    BinaryOperation::Add)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "matsub",
                                    BinaryOperation::Subtract)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "hadamard",
                                    BinaryOperation::Hadamard)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "__tensorAdd1d",
                                    BinaryOperation::Add)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "__tensorSubtract1d",
                                    BinaryOperation::Subtract)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "__tensorAdd4d",
                                    BinaryOperation::Add)) ||
        succeeded(rewriteBinaryCall(call, rewriter, "__tensorSubtract4d",
                                    BinaryOperation::Subtract)) ||
        succeeded(rewriteBinaryRecordABICall(call, rewriter, "matmul", 2)) ||
        succeeded(rewriteBinaryRecordABICall(call, rewriter, "matadd", 2)) ||
        succeeded(rewriteBinaryRecordABICall(call, rewriter, "matsub", 2)) ||
        succeeded(rewriteBinaryRecordABICall(call, rewriter, "hadamard", 2)) ||
        succeeded(
            rewriteBinaryRecordABICall(call, rewriter, "__tensorAdd1d", 1)) ||
        succeeded(rewriteBinaryRecordABICall(call, rewriter,
                                             "__tensorSubtract1d", 1)) ||
        succeeded(
            rewriteBinaryRecordABICall(call, rewriter, "__tensorAdd4d", 4)) ||
        succeeded(rewriteBinaryRecordABICall(call, rewriter,
                                             "__tensorSubtract4d", 4)) ||
        succeeded(rewriteScaleCall(call, rewriter, "matscale")) ||
        succeeded(rewriteScaleCall(call, rewriter, "__tensorScale1d")) ||
        succeeded(rewriteScaleCall(call, rewriter, "__tensorScale4d")) ||
        succeeded(rewriteScaleRecordABICall(call, rewriter, "matscale", 2)) ||
        succeeded(
            rewriteScaleRecordABICall(call, rewriter, "__tensorScale1d", 1)) ||
        succeeded(
            rewriteScaleRecordABICall(call, rewriter, "__tensorScale4d", 4)) ||
        succeeded(rewriteUnaryCall(call, rewriter, "transpose")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "exp")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "sigmoid")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "sigmoidPrime")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "relu")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "softmax")) ||
        succeeded(rewriteUnaryRecordABICall(call, rewriter, "transpose", 2)) ||
        succeeded(rewriteUnaryRecordABICall(call, rewriter, "exp", 2)) ||
        succeeded(rewriteUnaryRecordABICall(call, rewriter, "sigmoid", 2)) ||
        succeeded(
            rewriteUnaryRecordABICall(call, rewriter, "sigmoidPrime", 2)) ||
        succeeded(rewriteUnaryRecordABICall(call, rewriter, "relu", 4)) ||
        succeeded(rewriteUnaryRecordABICall(call, rewriter, "softmax", 2)) ||
        succeeded(rewriteSoftmaxCrossEntropyCall(call, rewriter)) ||
        succeeded(
            rewriteSoftmaxCrossEntropyRecordABICall(call, rewriter)) ||
        succeeded(rewriteConv2DCall(call, rewriter)) ||
        succeeded(rewriteConv2DRecordABICall(call, rewriter)) ||
        succeeded(rewriteMaxPool2DCall(call, rewriter)) ||
        succeeded(rewriteMaxPool2DRecordABICall(call, rewriter)) ||
        succeeded(rewriteReluBackwardIntoCall(call, rewriter)) ||
        succeeded(rewriteReluBackwardIntoRecordABICall(call, rewriter)) ||
        succeeded(rewriteConv2DBackwardIntoCall(call, rewriter)) ||
        succeeded(rewriteConv2DBackwardIntoRecordABICall(call, rewriter)) ||
        succeeded(rewriteMaxPool2DBackwardIntoCall(call, rewriter)) ||
        succeeded(rewriteMaxPool2DBackwardIntoRecordABICall(call, rewriter)))
      return success();

    return failure();
  }
};

struct PrepareMulberryNNCalls
    : public impl::PrepareMulberryNNCallsBase<PrepareMulberryNNCalls> {
  using impl::PrepareMulberryNNCallsBase<
      PrepareMulberryNNCalls>::PrepareMulberryNNCallsBase;

  auto runOnOperation() -> void final {
    RewritePatternSet patterns(&getContext());
    patterns.add<MulberryNNCallRewrite>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

static auto convertMemRefShape(ArrayRef<int64_t> shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape)
    memrefShape.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  return memrefShape;
}

class MulberryNNTypeConverter : public TypeConverter {
public:
  MulberryNNTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](mlir::mulberry_core::TensorType type) -> Type {
      return MemRefType::get(convertMemRefShape(type.getShape()),
                             type.getElementType());
    });
  }
};

struct LowerMulberryNN : public impl::LowerMulberryNNBase<LowerMulberryNN> {
  using impl::LowerMulberryNNBase<LowerMulberryNN>::LowerMulberryNNBase;

  auto runOnOperation() -> void final {
    MulberryNNTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, math::MathDialect,
                           memref::MemRefDialect, scf::SCFDialect>();
    target.addIllegalDialect<MulberryNNDialect>();

    RewritePatternSet patterns(&getContext());
    populateMulberryNNToLinalgPatterns(typeConverter, patterns);

    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::mulberry_nn
