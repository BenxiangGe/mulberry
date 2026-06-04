//===--- LowerMulberry.cpp -----------------------------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/Conversion/CherryPasses.h"
#include "CherryNNToLinalgPatterns.h"
#include "cherry/MLIRGen/IR/CherryNNOps.h"
#include "cherry/MLIRGen/IR/MulberryDialect.h"
#include "cherry/MLIRGen/IR/MulberryOps.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

#include <optional>

namespace mlir::cherry {

#define GEN_PASS_DEF_LOWERMULBERRY
#include "cherry/MLIRGen/Conversion/CherryPasses.h.inc"

namespace {

static auto getPtrType(MLIRContext* context) -> LLVM::LLVMPointerType {
  return LLVM::LLVMPointerType::get(context);
}

static auto isScalarStorageType(Type type) -> bool {
  return type.isIndex() || llvm::isa<IntegerType, FloatType>(type);
}

static auto convertRecordType(mulberry::RecordType type)
    -> std::optional<Type>;

static auto convertTensorType(mulberry::TensorType type)
    -> std::optional<Type> {
  return MemRefType::get(type.getShape(), type.getElementType());
}

// This pass is a transitional storage lowering, not the final
// Mulberry-to-LLVM ABI lowering. Tensor values become memrefs so cherry_nn can
// lower to linalg, while scalar/record stack storage still uses LLVM dialect.
static auto convertStorageType(Type type) -> std::optional<Type> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    return convertRecordType(recordType);

  if (isScalarStorageType(type))
    return type;

  return std::nullopt;
}

static auto convertSlotElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
    return convertTensorType(tensorType);

  return convertStorageType(type);
}

static auto convertRecordType(mulberry::RecordType type)
    -> std::optional<Type> {
  std::vector<Type> fieldTypes;
  for (auto field : type.getFields()) {
    // Keep tensor/list fields illegal until record storage can contain real
    // Mulberry value descriptors instead of backend ABI workaround values.
    auto fieldType = convertStorageType(field.type);
    if (!fieldType)
      return std::nullopt;
    fieldTypes.push_back(*fieldType);
  }

  return LLVM::LLVMStructType::getLiteral(type.getContext(), fieldTypes);
}

static auto convertPtrType(mulberry::PtrType type) -> std::optional<Type> {
  auto pointeeType = type.getPointeeType();
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(pointeeType)) {
    auto tensorStorageType = convertTensorType(tensorType);
    // ptr<Tensor> is a local tensor handle slot. After Tensor lowers to memref,
    // the slot becomes a 0-D memref storing that memref value; this is not a
    // function-boundary pointer bridge.
    if (tensorStorageType)
      return MemRefType::get({}, *tensorStorageType);
  }

  if (convertStorageType(pointeeType))
    return getPtrType(type.getContext());

  return std::nullopt;
}

// Safety net for Mulberry types that should not fall through to the identity
// conversion after their specific lowering conversion fails.
static auto rejectUnloweredMulberryType(Type type, SmallVectorImpl<Type>&)
    -> std::optional<LogicalResult> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    if (!convertRecordType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type))
    if (!convertPtrType(ptrType))
      return failure();

  if (llvm::isa<mulberry::ListType>(type))
    return failure();

  return std::nullopt;
}

class MulberryTypeConverter : public TypeConverter {
public:
  MulberryTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(rejectUnloweredMulberryType);
    addConversion(convertRecordType);
    addConversion(convertPtrType);
    addConversion(convertTensorType);
    // Keep unsupported Mulberry types illegal until each one has a real
    // lowering. The identity conversion above is only for non-Mulberry types.
  }
};

static auto getRecordType(Type type) -> mulberry::RecordType {
  auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type);
  if (!ptrType)
    return {};

  return llvm::dyn_cast<mulberry::RecordType>(ptrType.getPointeeType());
}

static auto getRecordFieldType(mulberry::RecordType recordType,
                               StringRef field) -> Type {
  return convertStorageType(recordType.getFieldType(field)).value_or(Type{});
}

static auto getPtrPointeeType(Type type) -> Type {
  auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type);
  if (!ptrType)
    return {};

  return ptrType.getPointeeType();
}

static auto getOne(Location location, ConversionPatternRewriter& rewriter)
    -> Value {
  return LLVM::ConstantOp::create(rewriter, location, rewriter.getI64Type(),
                                  1);
}

class AllocaOpLowering : public OpConversionPattern<mulberry::AllocaOp> {
public:
  using OpConversionPattern<mulberry::AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto elementType = convertSlotElementType(op.getElementType());
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "alloca needs a lowerable storage type");

    if (llvm::isa<MemRefType>(*elementType)) {
      auto slotType = MemRefType::get({}, *elementType);
      auto alloca = memref::AllocaOp::create(rewriter, op.getLoc(), slotType);
      rewriter.replaceOp(op, alloca.getResult());
      return success();
    }

    auto alloca = LLVM::AllocaOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *elementType,
        getOne(op.getLoc(), rewriter), /*alignment=*/0);
    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<mulberry::LoadOp> {
public:
  using OpConversionPattern<mulberry::LoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "load needs a lowerable result type");

    if (llvm::isa<MemRefType>(resultType)) {
      auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                         adaptor.getPtr(), ValueRange{});
      rewriter.replaceOp(op, load.getResult());
      return success();
    }

    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), resultType,
                                     adaptor.getPtr());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<mulberry::StoreOp> {
public:
  using OpConversionPattern<mulberry::StoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto elementType = convertSlotElementType(getPtrPointeeType(
        op.getPtr().getType()));
    if (!elementType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a lowered storage slot");

    if (llvm::isa<MemRefType>(*elementType)) {
      memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                              adaptor.getPtr(), ValueRange{});
      rewriter.eraseOp(op);
      return success();
    }

    LLVM::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                          adaptor.getPtr());
    rewriter.eraseOp(op);
    return success();
  }
};

class RecordGetFieldOpLowering
    : public OpConversionPattern<mulberry::RecordGetFieldOp> {
public:
  using OpConversionPattern<mulberry::RecordGetFieldOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordGetFieldOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto recordType = getRecordType(op.getRecord().getType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto llvmRecordType = getTypeConverter()->convertType(recordType);
    auto fieldType = getRecordFieldType(recordType, op.getField());
    if (!llvmRecordType || !fieldType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a lowerable record field");

    auto fieldIndex = static_cast<int32_t>(
        recordType.getFieldIndex(op.getField()));
    auto fieldPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), llvmRecordType,
        adaptor.getRecord(), ArrayRef<LLVM::GEPArg>{0, fieldIndex});
    rewriter.replaceOp(op, fieldPtr.getResult());
    return success();
  }
};

class RecordExtractOpLowering
    : public OpConversionPattern<mulberry::RecordExtractOp> {
public:
  using OpConversionPattern<mulberry::RecordExtractOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::RecordExtractOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto recordType = llvm::dyn_cast<mulberry::RecordType>(
        op.getRecord().getType());
    auto fieldType = getRecordFieldType(recordType, op.getField());
    if (!fieldType)
      return rewriter.notifyMatchFailure(
          op, "record extract needs a lowerable field");

    auto fieldIndex = static_cast<int64_t>(
        recordType.getFieldIndex(op.getField()));
    auto extracted = LLVM::ExtractValueOp::create(
        rewriter, op.getLoc(), fieldType, adaptor.getRecord(), fieldIndex);
    rewriter.replaceOp(op, extracted.getResult());
    return success();
  }
};

class TensorAllocOpLowering
    : public OpConversionPattern<mulberry::TensorAllocOp> {
public:
  using OpConversionPattern<mulberry::TensorAllocOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = llvm::dyn_cast_or_null<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "tensor alloc needs a lowerable memref result type");

    auto alloc = memref::AllocOp::create(rewriter, op.getLoc(), resultType,
                                         adaptor.getDynamicSizes());
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

class TensorDimOpLowering : public OpConversionPattern<mulberry::TensorDimOp> {
public:
  using OpConversionPattern<mulberry::TensorDimOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorDimOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto dim = memref::DimOp::create(rewriter, op.getLoc(),
                                     adaptor.getTensor(), adaptor.getIndex());
    rewriter.replaceOp(op, dim.getResult());
    return success();
  }
};

class TensorCastOpLowering
    : public OpConversionPattern<mulberry::TensorCastOp> {
public:
  using OpConversionPattern<mulberry::TensorCastOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto destType = getTypeConverter()->convertType(op.getDest().getType());
    if (!destType)
      return rewriter.notifyMatchFailure(
          op, "tensor cast needs a lowerable memref result type");

    auto cast = memref::CastOp::create(rewriter, op.getLoc(), destType,
                                       adaptor.getSource());
    rewriter.replaceOp(op, cast.getResult());
    return success();
  }
};

class TensorLoadOpLowering
    : public OpConversionPattern<mulberry::TensorLoadOp> {
public:
  using OpConversionPattern<mulberry::TensorLoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorLoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                       adaptor.getTensor(),
                                       adaptor.getIndices());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class TensorStoreOpLowering
    : public OpConversionPattern<mulberry::TensorStoreOp> {
public:
  using OpConversionPattern<mulberry::TensorStoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::TensorStoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            adaptor.getTensor(), adaptor.getIndices());
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerMulberry : public impl::LowerMulberryBase<LowerMulberry> {
  using impl::LowerMulberryBase<LowerMulberry>::LowerMulberryBase;

  auto runOnOperation() -> void final {
    MulberryTypeConverter typeConverter;

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, func::FuncDialect,
                           linalg::LinalgDialect, LLVM::LLVMDialect,
                           math::MathDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    target.addIllegalDialect<cherry_nn::CherryNNDialect>();
    target.addIllegalDialect<mulberry::MulberryDialect>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) { return typeConverter.isLegal(op); });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, LoadOpLowering, RecordExtractOpLowering,
                 RecordGetFieldOpLowering, StoreOpLowering,
                 TensorAllocOpLowering, TensorCastOpLowering,
                 TensorDimOpLowering, TensorLoadOpLowering,
                 TensorStoreOpLowering>(typeConverter, &getContext());
    populateCherryNNToLinalgPatterns(typeConverter, patterns);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    populateCallOpTypeConversionPattern(patterns, typeConverter);
    populateReturnOpTypeConversionPattern(patterns, typeConverter);

    // Keep unresolved Mulberry ops fail-fast until each one has an explicit
    // lowering. A silent no-op pass would make backend tests look lower than
    // they are.
    FrozenRewritePatternSet patternSet(std::move(patterns));
    if (failed(applyPartialConversion(getOperation(), target, patternSet)))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::cherry
