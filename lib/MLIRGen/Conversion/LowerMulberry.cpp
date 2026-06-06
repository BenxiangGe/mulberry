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

static auto convertRecordToBackendType(mulberry::RecordType type)
    -> std::optional<Type>;

static auto convertMemRefShape(ArrayRef<int64_t> shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape) {
    memrefShape.push_back(dim < 0 ? ShapedType::kDynamic : dim);
  }
  return memrefShape;
}

static auto convertTensorToMemRefType(mulberry::TensorType type)
    -> std::optional<Type> {
  return MemRefType::get(convertMemRefShape(type.getShape()),
                         type.getElementType());
}

// This pass is a transitional storage lowering, not the final
// Mulberry-to-LLVM ABI lowering. Tensor values become memrefs so cherry_nn can
// lower to linalg, while scalar/record stack storage still uses LLVM dialect.
static auto convertToBackendType(Type type) -> std::optional<Type> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    return convertRecordToBackendType(recordType);

  if (isScalarStorageType(type))
    return type;

  return std::nullopt;
}

static auto convertAllocaElementType(Type type) -> std::optional<Type> {
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  return convertToBackendType(type);
}

static auto convertListElementType(Type type) -> std::optional<Type> {
  if (isScalarStorageType(type))
    return type;

  // List storage contains already-lowered element values. For List<Tensor>,
  // that means storing the lowered memref handle, not the high-level tensor.
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(type))
    return convertTensorToMemRefType(tensorType);

  return std::nullopt;
}

static auto convertToListStorageType(mulberry::ListType type)
    -> std::optional<mulberry::ListStorageType> {
  auto elementType = convertListElementType(type.getElementType());
  if (!elementType)
    return std::nullopt;

  return mulberry::ListStorageType::get(type.getContext(), *elementType);
}

static auto convertToMemRefType(mulberry::ListStorageType type)
    -> std::optional<MemRefType> {
  auto elementType = type.getElementType();
  // List<Tensor> stores lowered tensor handles. This is still a memref-level
  // storage lowering, not the final Mulberry-to-LLVM ABI representation.
  if (!isScalarStorageType(elementType) && !llvm::isa<MemRefType>(elementType))
    return std::nullopt;

  return MemRefType::get({ShapedType::kDynamic}, elementType);
}

static auto containsListType(Type type) -> bool {
  if (llvm::isa<mulberry::ListType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsListType(field.type))
        return true;

  return false;
}

static auto containsListType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsListType(type))
      return true;

  return false;
}

static auto containsListType(FunctionType type) -> bool {
  return containsListType(type.getInputs()) ||
         containsListType(type.getResults());
}

static auto containsListStorageType(Type type) -> bool {
  if (llvm::isa<mulberry::ListStorageType>(type))
    return true;

  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    for (auto field : recordType.getFields())
      if (containsListStorageType(field.type))
        return true;

  return false;
}

static auto containsListStorageType(TypeRange types) -> bool {
  for (auto type : types)
    if (containsListStorageType(type))
      return true;

  return false;
}

static auto containsListStorageType(FunctionType type) -> bool {
  return containsListStorageType(type.getInputs()) ||
         containsListStorageType(type.getResults());
}

static auto convertRecordToBackendType(mulberry::RecordType type)
    -> std::optional<Type> {
  std::vector<Type> fieldTypes;
  for (auto field : type.getFields()) {
    // Keep tensor/list fields illegal until record storage can contain real
    // Mulberry value descriptors instead of backend ABI workaround values.
    auto fieldType = convertToBackendType(field.type);
    if (!fieldType)
      return std::nullopt;
    fieldTypes.push_back(*fieldType);
  }

  return LLVM::LLVMStructType::getLiteral(type.getContext(), fieldTypes);
}

static auto convertPtrType(mulberry::PtrType type) -> std::optional<Type> {
  auto pointeeType = type.getPointeeType();
  if (auto tensorType = llvm::dyn_cast<mulberry::TensorType>(pointeeType)) {
    auto tensorStorageType = convertTensorToMemRefType(tensorType);
    // ptr<Tensor> is a local tensor handle slot. After Tensor lowers to memref,
    // the slot becomes a 0-D memref storing that memref value; this is not a
    // function-boundary pointer bridge.
    if (tensorStorageType)
      return MemRefType::get({}, *tensorStorageType);
  }

  if (convertToBackendType(pointeeType))
    return getPtrType(type.getContext());

  return std::nullopt;
}

// Safety net for Mulberry types that should not fall through to the identity
// conversion after their specific lowering conversion fails.
static auto rejectUnloweredMulberryType(Type type, SmallVectorImpl<Type>&)
    -> std::optional<LogicalResult> {
  if (auto recordType = llvm::dyn_cast<mulberry::RecordType>(type))
    if (!convertRecordToBackendType(recordType))
      return failure();

  if (auto ptrType = llvm::dyn_cast<mulberry::PtrType>(type))
    if (!convertPtrType(ptrType))
      return failure();

  if (llvm::isa<mulberry::ListType>(type))
    return failure();

  if (llvm::isa<mulberry::ListStorageType>(type))
    return failure();

  return std::nullopt;
}

class MulberryTypeConverter : public TypeConverter {
public:
  MulberryTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion(rejectUnloweredMulberryType);
    addConversion([](mulberry::ListType type) -> std::optional<Type> {
      return convertToListStorageType(type);
    });
    addConversion([](mulberry::ListStorageType type) {
      return convertToMemRefType(type);
    });
    addConversion(convertRecordToBackendType);
    addConversion(convertPtrType);
    addConversion(convertTensorToMemRefType);
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
  return convertToBackendType(recordType.getFieldType(field)).value_or(Type{});
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
    auto elementType = convertAllocaElementType(op.getElementType());
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
    auto elementType = convertAllocaElementType(getPtrPointeeType(
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

class ListCreateOpLowering
    : public OpConversionPattern<mulberry::ListCreateOp> {
public:
  using OpConversionPattern<mulberry::ListCreateOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListCreateOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto listType =
        llvm::cast<mulberry::ListType>(op.getResult().getType());
    auto storageType = convertToListStorageType(listType);
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "list storage needs a lowerable element type");

    auto length = arith::ConstantIndexOp::create(
        rewriter, op.getLoc(), op.getElements().size());
    auto storage = mulberry::ListAllocOp::create(
        rewriter, op.getLoc(), *storageType, length);

    for (auto element : llvm::enumerate(adaptor.getElements())) {
      auto index = arith::ConstantIndexOp::create(
          rewriter, op.getLoc(), element.index());
      mulberry::ListStoreOp::create(rewriter, op.getLoc(), element.value(),
                                    storage, index);
    }

    rewriter.replaceOp(op, storage.getResult());
    return success();
  }
};

class ListAllocOpLowering : public OpConversionPattern<mulberry::ListAllocOp> {
public:
  using OpConversionPattern<mulberry::ListAllocOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto storageType = llvm::dyn_cast_or_null<MemRefType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "list storage needs a lowerable memref storage type");

    auto alloc = memref::AllocOp::create(rewriter, op.getLoc(), storageType,
                                         adaptor.getLength());
    rewriter.replaceOp(op, alloc.getResult());
    return success();
  }
};

class ListGetOpLowering : public OpConversionPattern<mulberry::ListGetOp> {
public:
  using OpConversionPattern<mulberry::ListGetOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListGetOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultType = getTypeConverter()->convertType(op.getResult().getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "list load needs a lowerable result type");

    auto loaded = mulberry::ListLoadOp::create(
        rewriter, op.getLoc(), resultType, adaptor.getList(),
        adaptor.getIndex());
    rewriter.replaceOp(op, loaded.getResult());
    return success();
  }
};

class ListLoadOpLowering : public OpConversionPattern<mulberry::ListLoadOp> {
public:
  using OpConversionPattern<mulberry::ListLoadOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListLoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto load = memref::LoadOp::create(rewriter, op.getLoc(),
                                       adaptor.getStorage(),
                                       adaptor.getIndex());
    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class ListSizeOpLowering : public OpConversionPattern<mulberry::ListSizeOp> {
public:
  using OpConversionPattern<mulberry::ListSizeOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListSizeOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto length = mulberry::ListLengthOp::create(
        rewriter, op.getLoc(), rewriter.getIndexType(), adaptor.getList());
    auto result = arith::IndexCastOp::create(
        rewriter, op.getLoc(), op.getResult().getType(), length);
    rewriter.replaceOp(op, result.getResult());
    return success();
  }
};

class ListStoreOpLowering : public OpConversionPattern<mulberry::ListStoreOp> {
public:
  using OpConversionPattern<mulberry::ListStoreOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListStoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    memref::StoreOp::create(rewriter, op.getLoc(), adaptor.getValue(),
                            adaptor.getStorage(), adaptor.getIndex());
    rewriter.eraseOp(op);
    return success();
  }
};

class ListLengthOpLowering
    : public OpConversionPattern<mulberry::ListLengthOp> {
public:
  using OpConversionPattern<mulberry::ListLengthOp>::OpConversionPattern;

  auto matchAndRewrite(mulberry::ListLengthOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto zero = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    auto length = memref::DimOp::create(rewriter, op.getLoc(),
                                        adaptor.getStorage(), zero);
    rewriter.replaceOp(op, length.getResult());
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
          if (containsListType(op.getFunctionType()) ||
              containsListStorageType(op.getFunctionType()))
            return false;
          return typeConverter.isSignatureLegal(op.getFunctionType());
        });
    target.addDynamicallyLegalOp<func::CallOp, func::ReturnOp>(
        [&](Operation* op) {
          if (containsListType(op->getOperandTypes()) ||
              containsListType(op->getResultTypes()) ||
              containsListStorageType(op->getOperandTypes()) ||
              containsListStorageType(op->getResultTypes()))
            return false;
          return typeConverter.isLegal(op);
        });

    RewritePatternSet patterns(&getContext());
    patterns.add<AllocaOpLowering, ListAllocOpLowering,
                 ListCreateOpLowering, ListGetOpLowering,
                 ListLengthOpLowering, ListLoadOpLowering,
                 ListSizeOpLowering, ListStoreOpLowering, LoadOpLowering,
                 RecordExtractOpLowering, RecordGetFieldOpLowering,
                 StoreOpLowering, TensorAllocOpLowering,
                 TensorCastOpLowering, TensorDimOpLowering,
                 TensorLoadOpLowering, TensorStoreOpLowering>(
        typeConverter, &getContext());
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
