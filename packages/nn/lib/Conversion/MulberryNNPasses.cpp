//===--- MulberryNNPasses.cpp - Mulberry NN package passes ----------------===//

#include "MulberryNN/MulberryNNPasses.h"
#include "MulberryNN/MulberryNNToLinalgPatterns.h"
#include "MulberryNN/MulberryNNDialect.h"
#include "MulberryNN/MulberryNNOps.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

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

static auto getOrCreateTensorCallee(PatternRewriter& rewriter, Location loc,
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

static auto allocateLike(PatternRewriter& rewriter, Location loc, Value source,
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

static auto createDim(PatternRewriter& rewriter, Location loc, Value source,
                      int64_t index) -> Value {
  auto dimIndex = arith::ConstantIndexOp::create(rewriter, loc, index);
  return memref::DimOp::create(rewriter, loc, source, dimIndex);
}

static auto allocateMatmul(PatternRewriter& rewriter, Location loc, Value lhs,
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

static auto allocateTranspose(PatternRewriter& rewriter, Location loc,
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

static auto isTensorF32Record(Type type) -> bool {
  auto recordType = llvm::dyn_cast<mlir::mulberry::RecordType>(type);
  if (!recordType)
    return false;

  auto ptrType = llvm::dyn_cast<mlir::mulberry::PtrType>(
      recordType.getFieldType("data"));
  if (!ptrType || !ptrType.getPointeeType().isF32())
    return false;

  auto rankType = recordType.getFieldType("rank");
  auto numelType = recordType.getFieldType("numel");
  return rankType && rankType.isInteger(64) && numelType &&
         numelType.isInteger(64) && recordType.getFieldType("sizes") &&
         recordType.getFieldType("strides");
}

static auto getMatrixTensorType(MLIRContext* context)
    -> mlir::mulberry::TensorType {
  return mlir::mulberry::TensorType::get(
      context, ArrayRef<int64_t>{ShapedType::kDynamic, ShapedType::kDynamic},
      Float32Type::get(context));
}

static auto createTensorView(PatternRewriter& rewriter, Location loc,
                             Value tensorRecord) -> Value {
  auto tensorType = getMatrixTensorType(rewriter.getContext());
  return mlir::mulberry::TensorViewOp::create(rewriter, loc, tensorType,
                                              tensorRecord);
}

static auto createTensorCall(PatternRewriter& rewriter, Location loc,
                             func::CallOp call, StringRef name,
                             ValueRange inputs, TypeRange results)
    -> func::CallOp {
  auto callee = getOrCreateTensorCallee(rewriter, loc, call, name,
                                        inputs.getTypes(), results);
  return func::CallOp::create(rewriter, loc, callee, results, inputs);
}

static auto packTensor(PatternRewriter& rewriter, Location loc, Value tensor,
                       Type recordType) -> Value {
  return mlir::mulberry::TensorPackOp::create(rewriter, loc, recordType,
                                              tensor);
}

static auto rewriteBinaryCall(func::CallOp call, PatternRewriter& rewriter,
                              StringRef name) -> LogicalResult {
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
  if (name == "matmul") {
    out = allocateMatmul(rewriter, loc, call.getOperand(0), call.getOperand(1),
                         resultType);
    if (!out)
      return failure();
    MatmulOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1), out);
  } else if (name == "matadd") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    MataddOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1), out);
  } else if (name == "matsub") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    MatsubOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1), out);
  } else if (name == "hadamard") {
    out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
    HadamardOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                       out);
  } else {
    return failure();
  }
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteBinaryRecordCall(func::CallOp call, PatternRewriter& rewriter,
                                    StringRef name) -> LogicalResult {
  if (!isPublicMulberryNNCall(call, name))
    return failure();
  if (call.getNumOperands() != 2 || call.getNumResults() != 1)
    return failure();
  if (!isTensorF32Record(call.getOperand(0).getType()) ||
      !isTensorF32Record(call.getOperand(1).getType()) ||
      !isTensorF32Record(call.getResult(0).getType()))
    return failure();

  auto loc = call.getLoc();
  auto lhs = createTensorView(rewriter, loc, call.getOperand(0));
  auto rhs = createTensorView(rewriter, loc, call.getOperand(1));
  auto resultType = getMatrixTensorType(rewriter.getContext());
  auto result = createTensorCall(rewriter, loc, call, name, ValueRange{lhs, rhs},
                                 TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

static auto rewriteUnaryCall(func::CallOp call, PatternRewriter& rewriter,
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
  } else {
    return failure();
  }
  rewriter.replaceOp(call, out);
  return success();
}

static auto rewriteUnaryRecordCall(func::CallOp call, PatternRewriter& rewriter,
                                   StringRef name) -> LogicalResult {
  if (!isPublicMulberryNNCall(call, name))
    return failure();
  if (call.getNumOperands() != 1 || call.getNumResults() != 1)
    return failure();
  if (!isTensorF32Record(call.getOperand(0).getType()) ||
      !isTensorF32Record(call.getResult(0).getType()))
    return failure();

  auto loc = call.getLoc();
  auto input = createTensorView(rewriter, loc, call.getOperand(0));
  auto resultType = getMatrixTensorType(rewriter.getContext());
  auto result = createTensorCall(rewriter, loc, call, name, ValueRange{input},
                                 TypeRange{resultType});
  rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                      call.getResult(0).getType()));
  return success();
}

class MulberryNNCallRewrite : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern<func::CallOp>::OpRewritePattern;

  auto matchAndRewrite(func::CallOp call,
                       PatternRewriter& rewriter) const -> LogicalResult final {
    auto loc = call.getLoc();

    if (isTensorMulberryNNCall(call, "argmax")) {
      if (call.getNumOperands() != 1 || call.getNumResults() != 1 ||
          !call.getResult(0).getType().isInteger(64))
        return failure();

      Value input = call.getOperand(0);
      if (llvm::isa<MemRefType>(input.getType())) {
        auto result = ArgmaxOp::create(rewriter, loc,
                                       call.getResult(0).getType(), input);
        rewriter.replaceOp(call, result.getResult());
        return success();
      }

      return failure();
    }

    if (isPublicMulberryNNCall(call, "argmax")) {
      if (call.getNumOperands() != 1 || call.getNumResults() != 1 ||
          !call.getResult(0).getType().isInteger(64) ||
          !isTensorF32Record(call.getOperand(0).getType()))
        return failure();

      auto view = createTensorView(rewriter, loc, call.getOperand(0));
      auto result = createTensorCall(rewriter, loc, call, "argmax",
                                     ValueRange{view},
                                     TypeRange{call.getResult(0).getType()});
      rewriter.replaceOp(call, result.getResult(0));
      return success();
    }

    if (succeeded(rewriteBinaryCall(call, rewriter, "matmul")) ||
        succeeded(rewriteBinaryCall(call, rewriter, "matadd")) ||
        succeeded(rewriteBinaryCall(call, rewriter, "matsub")) ||
        succeeded(rewriteBinaryCall(call, rewriter, "hadamard")) ||
        succeeded(rewriteBinaryRecordCall(call, rewriter, "matmul")) ||
        succeeded(rewriteBinaryRecordCall(call, rewriter, "matadd")) ||
        succeeded(rewriteBinaryRecordCall(call, rewriter, "matsub")) ||
        succeeded(rewriteBinaryRecordCall(call, rewriter, "hadamard")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "transpose")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "exp")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "sigmoid")) ||
        succeeded(rewriteUnaryCall(call, rewriter, "sigmoidPrime")) ||
        succeeded(rewriteUnaryRecordCall(call, rewriter, "transpose")) ||
        succeeded(rewriteUnaryRecordCall(call, rewriter, "exp")) ||
        succeeded(rewriteUnaryRecordCall(call, rewriter, "sigmoid")) ||
        succeeded(rewriteUnaryRecordCall(call, rewriter, "sigmoidPrime")))
      return success();

    if (isTensorMulberryNNCall(call, "matscale")) {
      if (call.getNumOperands() != 2 || call.getNumResults() != 1)
        return failure();
      auto inputType = llvm::dyn_cast<MemRefType>(call.getOperand(0).getType());
      auto resultType = llvm::dyn_cast<MemRefType>(call.getResult(0).getType());
      if (!inputType || !resultType || !call.getOperand(1).getType().isF32())
        return failure();

      auto out = allocateLike(rewriter, loc, call.getOperand(0), resultType);
      MatscaleOp::create(rewriter, loc, call.getOperand(0), call.getOperand(1),
                         out);
      rewriter.replaceOp(call, out);
      return success();
    }

    if (isPublicMulberryNNCall(call, "matscale") &&
        call.getNumOperands() == 2 && call.getNumResults() == 1 &&
        isTensorF32Record(call.getOperand(0).getType()) &&
        isTensorF32Record(call.getResult(0).getType()) &&
        call.getOperand(1).getType().isF32()) {
      auto input = createTensorView(rewriter, loc, call.getOperand(0));
      auto resultType = getMatrixTensorType(rewriter.getContext());
      auto result = createTensorCall(rewriter, loc, call, "matscale",
                                     ValueRange{input, call.getOperand(1)},
                                     TypeRange{resultType});
      rewriter.replaceOp(call, packTensor(rewriter, loc, result.getResult(0),
                                          call.getResult(0).getType()));
      return success();
    }

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
    addConversion([](mlir::mulberry::TensorType type) -> Type {
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
                           memref::MemRefDialect>();
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
