//===--- LowerMulberryCore.cpp --------------------------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "LowerMulberrySupport.h"

#include "mulberry/MLIRGen/IR/MulberryOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

#include <vector>

namespace mlir::mulberry_core::detail {

namespace {

class FunctionConstantTypeConversion
    : public OpConversionPattern<func::ConstantOp> {
public:
  using OpConversionPattern<func::ConstantOp>::OpConversionPattern;

  auto matchAndRewrite(func::ConstantOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto convertedType = llvm::dyn_cast_or_null<FunctionType>(
        getTypeConverter()->convertType(op.getResult().getType()));
    if (!convertedType)
      return rewriter.notifyMatchFailure(
          op, "function constant needs a converted function type");

    auto constant = func::ConstantOp::create(
        rewriter, op.getLoc(), convertedType, op.getValueAttr());
    rewriter.replaceOp(op, constant.getResult());
    return success();
  }
};

class ResultTryTypeConversion : public OpConversionPattern<ResultTryOp> {
public:
  using OpConversionPattern<ResultTryOp>::OpConversionPattern;

  auto matchAndRewrite(ResultTryOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    std::vector<Type> resultTypes;
    for (auto result : op.getResults()) {
      auto converted = getTypeConverter()->convertType(result.getType());
      if (!converted)
        return rewriter.notifyMatchFailure(
            op, "result try value has no storage type");
      resultTypes.push_back(converted);
    }

    auto sourceErrorType =
        getTypeConverter()->convertType(op.getSourceErrorType());
    auto targetErrorType =
        getTypeConverter()->convertType(op.getTargetErrorType());
    if (!sourceErrorType || !targetErrorType)
      return rewriter.notifyMatchFailure(
          op, "result try error type has no storage type");

    auto convertedOp = ResultTryOp::create(
        rewriter, op.getLoc(), resultTypes, adaptor.getInput(),
        TypeAttr::get(sourceErrorType), TypeAttr::get(targetErrorType),
        op.getErrorConverterAttr());
    rewriter.replaceOp(op, convertedOp.getResults());
    return success();
  }
};

class CallIndirectTypeConversion
    : public OpConversionPattern<func::CallIndirectOp> {
public:
  using OpConversionPattern<func::CallIndirectOp>::OpConversionPattern;

  auto matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (!llvm::isa<FunctionType>(adaptor.getCallee().getType()))
      return rewriter.notifyMatchFailure(
          op, "indirect callee needs a converted function type");

    auto call = func::CallIndirectOp::create(
        rewriter, op.getLoc(), adaptor.getCallee(),
        adaptor.getCalleeOperands());
    call->setAttrs(op->getAttrDictionary());
    rewriter.replaceOp(op, call.getResults());
    return success();
  }
};

class DataConstructOpLowering
    : public OpConversionPattern<DataConstructOp> {
public:
  using OpConversionPattern<DataConstructOp>::OpConversionPattern;

  auto matchAndRewrite(DataConstructOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultPtrType = llvm::dyn_cast<PtrType>(op.getResult().getType());
    if (!resultPtrType ||
        !llvm::isa<DataType>(resultPtrType.getPointeeType()))
      return rewriter.notifyMatchFailure(
          op, "data constructor result must reference a data type");

    auto context = op.getContext();
    auto i64Type = getI64Type(context);
    std::vector<Type> fieldTypes{i64Type};
    std::vector<Value> payloads;
    for (auto [source, converted] :
         llvm::zip(op.getPayloads(), adaptor.getPayloads())) {
      auto fieldType = convertBackendValueType(source.getType());
      if (!fieldType || llvm::isa<TensorType>(source.getType()))
        return rewriter.notifyMatchFailure(
            op, "data constructor has an unsupported payload type");

      Value payload = converted;
      if (payload.getType() != *fieldType) {
        if (!llvm::isa<FunctionType>(source.getType()))
          return rewriter.notifyMatchFailure(
              op, "converted data payload does not match its storage type");
        payload = UnrealizedConversionCastOp::create(
                      rewriter, op.getLoc(), *fieldType, payload)
                      .getResult(0);
      }
      fieldTypes.push_back(*fieldType);
      payloads.push_back(payload);
    }

    auto layout = LLVM::LLVMStructType::getLiteral(context, fieldTypes);
    auto objectBytes = createSizeOf(op.getLoc(), rewriter, layout);
    auto object = callBoehmMalloc(op.getLoc(), rewriter, op, objectBytes);
    if (failed(object))
      return failure();

    auto tag = LLVM::ConstantOp::create(
        rewriter, op.getLoc(), i64Type,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(op.getTag())));
    Value value = LLVM::UndefOp::create(rewriter, op.getLoc(), layout);
    value = LLVM::InsertValueOp::create(
                rewriter, op.getLoc(), value, tag,
                ArrayRef<int64_t>{0})
                .getResult();
    for (const auto& indexedPayload : llvm::enumerate(payloads)) {
      value = LLVM::InsertValueOp::create(
                  rewriter, op.getLoc(), value, indexedPayload.value(),
                  ArrayRef<int64_t>{
                      static_cast<int64_t>(indexedPayload.index() + 1)})
                  .getResult();
    }
    LLVM::StoreOp::create(rewriter, op.getLoc(), value, *object);
    rewriter.replaceOp(op, *object);
    return success();
  }
};

class DataTagOpLowering : public OpConversionPattern<DataTagOp> {
public:
  using OpConversionPattern<DataTagOp>::OpConversionPattern;

  auto matchAndRewrite(DataTagOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto valuePtrType = llvm::dyn_cast<PtrType>(op.getValue().getType());
    if (!valuePtrType ||
        !llvm::isa<DataType>(valuePtrType.getPointeeType()))
      return rewriter.notifyMatchFailure(
          op, "data tag value must reference a data type");

    auto tag = LLVM::LoadOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getValue());
    rewriter.replaceOp(op, tag.getResult());
    return success();
  }
};

class DataUnpackOpLowering : public OpConversionPattern<DataUnpackOp> {
public:
  using OpConversionPattern<DataUnpackOp>::OpConversionPattern;

  auto matchAndRewrite(DataUnpackOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto valuePtrType = llvm::dyn_cast<PtrType>(op.getValue().getType());
    if (!valuePtrType ||
        !llvm::isa<DataType>(valuePtrType.getPointeeType()))
      return rewriter.notifyMatchFailure(
          op, "data unpack value must reference a data type");

    std::vector<Type> fieldTypes{getI64Type(op.getContext())};
    std::vector<Type> convertedResultTypes;
    for (auto result : op.getPayloads()) {
      auto sourceType = result.getType();
      auto fieldType = convertBackendValueType(sourceType);
      auto convertedType = getTypeConverter()->convertType(sourceType);
      if (!fieldType || !convertedType || llvm::isa<TensorType>(sourceType))
        return rewriter.notifyMatchFailure(
            op, "data unpack has an unsupported payload type");
      fieldTypes.push_back(*fieldType);
      convertedResultTypes.push_back(convertedType);
    }

    auto layout = LLVM::LLVMStructType::getLiteral(op.getContext(),
                                                   fieldTypes);
    std::vector<Value> payloads;
    for (size_t i = 0; i < op.getNumResults(); ++i) {
      auto fieldPtr = LLVM::GEPOp::create(
          rewriter, op.getLoc(), getPtrType(op.getContext()), layout,
          adaptor.getValue(),
          ArrayRef<LLVM::GEPArg>{0, static_cast<int32_t>(i + 1)});
      Value payload = LLVM::LoadOp::create(
          rewriter, op.getLoc(), fieldTypes[i + 1], fieldPtr);
      if (payload.getType() != convertedResultTypes[i]) {
        if (!llvm::isa<FunctionType>(op.getResult(i).getType()))
          return rewriter.notifyMatchFailure(
              op, "unpacked payload does not match its converted type");
        payload = UnrealizedConversionCastOp::create(
                      rewriter, op.getLoc(), convertedResultTypes[i], payload)
                      .getResult(0);
      }
      payloads.push_back(payload);
    }

    rewriter.replaceOp(op, payloads);
    return success();
  }
};

class AllocaOpLowering : public OpConversionPattern<AllocaOp> {
public:
  using OpConversionPattern<AllocaOp>::OpConversionPattern;

  auto matchAndRewrite(AllocaOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    if (llvm::isa<TensorType>(op.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto storageType = convertBackendValueType(op.getElementType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "alloca needs a lowerable storage type");

    auto elementCount = arith::ConstantIntOp::create(
        rewriter, op.getLoc(), 1, /*width=*/64);
    auto alloca = LLVM::AllocaOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *storageType,
        elementCount.getResult(), /*alignment=*/0);
    rewriter.replaceOp(op, alloca.getResult());
    return success();
  }
};

class ReplSlotOpLowering : public OpConversionPattern<ReplSlotOp> {
public:
  using OpConversionPattern<ReplSlotOp>::OpConversionPattern;

  auto matchAndRewrite(ReplSlotOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto resultPtrType = llvm::dyn_cast<PtrType>(op.getResult().getType());
    if (!resultPtrType)
      return rewriter.notifyMatchFailure(
          op, "REPL slot result must be a Mulberry pointer");

    auto storageType = convertBackendValueType(
        resultPtrType.getPointeeType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "REPL slot needs a lowerable storage type");

    auto sizeInBytes = createSizeOf(op.getLoc(), rewriter, *storageType);
    auto slot = callRuntimeValue(
        op.getLoc(), rewriter, op, "mulberry_repl_get_slot",
        getPtrType(op.getContext()),
        ValueRange{adaptor.getSlot(), sizeInBytes});
    if (failed(slot))
      return failure();

    rewriter.replaceOp(op, *slot);
    return success();
  }
};

class HeapAllocOpLowering : public OpConversionPattern<HeapAllocOp> {
public:
  using OpConversionPattern<HeapAllocOp>::OpConversionPattern;

  auto matchAndRewrite(HeapAllocOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    // Internal tensors lower to memrefs and are never heap-allocated through
    // Mulberry pointer storage.
    if (llvm::isa<TensorType>(op.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "internal tensors require memref allocation");

    auto storageType = convertBackendValueType(op.getElementType());
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "heap alloc needs a lowerable backend type");

    auto elementBytes = createSizeOf(op.getLoc(), rewriter, *storageType);
    auto sizeInBytes = arith::MulIOp::create(
        rewriter, op.getLoc(), elementBytes, adaptor.getCount());
    auto heapPtr = callBoehmMalloc(op.getLoc(), rewriter, op,
                                   sizeInBytes.getResult());
    if (failed(heapPtr))
      return failure();

    rewriter.replaceOp(op, *heapPtr);
    return success();
  }
};

class PtrIndexOpLowering : public OpConversionPattern<PtrIndexOp> {
public:
  using OpConversionPattern<PtrIndexOp>::OpConversionPattern;

  auto matchAndRewrite(PtrIndexOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<PtrType>(op.getPtr().getType());
    auto pointeeType = ptrType.getPointeeType();
    if (llvm::isa<TensorType>(pointeeType))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto storageType = convertBackendValueType(pointeeType);
    if (!storageType)
      return rewriter.notifyMatchFailure(
          op, "pointer index needs a lowerable backend element type");

    auto index = arith::IndexCastOp::create(
        rewriter, op.getLoc(), getI64Type(op.getContext()),
        adaptor.getIndex());
    auto elementPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *storageType,
        adaptor.getPtr(), ArrayRef<Value>{index.getResult()});
    rewriter.replaceOp(op, elementPtr.getResult());
    return success();
  }
};

class PtrCastOpLowering : public OpConversionPattern<PtrCastOp> {
public:
  using OpConversionPattern<PtrCastOp>::OpConversionPattern;

  auto matchAndRewrite(PtrCastOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    rewriter.replaceOp(op, adaptor.getPtr());
    return success();
  }
};

class LoadOpLowering : public OpConversionPattern<LoadOp> {
public:
  using OpConversionPattern<LoadOp>::OpConversionPattern;

  auto matchAndRewrite(LoadOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::cast<PtrType>(op.getPtr().getType());
    auto pointeeType = ptrType.getPointeeType();
    if (llvm::isa<TensorType>(pointeeType))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto valueType = convertBackendValueType(pointeeType);
    if (!valueType)
      return rewriter.notifyMatchFailure(
          op, "load needs a lowerable result type");

    auto load = LLVM::LoadOp::create(rewriter, op.getLoc(), *valueType,
                                     adaptor.getPtr());
    if (llvm::isa<FunctionType>(pointeeType)) {
      // FuncToLLVM later turns the SSA function value into the same opaque
      // pointer already used by LLVM storage; reconciliation removes this cast.
      auto sourceType = getTypeConverter()->convertType(pointeeType);
      auto cast = UnrealizedConversionCastOp::create(
          rewriter, op.getLoc(), sourceType, load.getResult());
      rewriter.replaceOp(op, cast.getResult(0));
      return success();
    }

    rewriter.replaceOp(op, load.getResult());
    return success();
  }
};

class StoreOpLowering : public OpConversionPattern<StoreOp> {
public:
  using OpConversionPattern<StoreOp>::OpConversionPattern;

  auto matchAndRewrite(StoreOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::dyn_cast<PtrType>(op.getPtr().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a Mulberry pointer");

    auto pointeeType = ptrType.getPointeeType();
    if (llvm::isa<TensorType>(pointeeType))
      return rewriter.notifyMatchFailure(
          op, "internal tensors cannot use pointer storage");

    auto valueType = convertBackendValueType(pointeeType);
    if (!valueType)
      return rewriter.notifyMatchFailure(
          op, "store target must be a lowered storage slot");

    auto value = adaptor.getValue();
    if (llvm::isa<FunctionType>(pointeeType)) {
      // This is the inverse bridge of the function-value load above.
      value = UnrealizedConversionCastOp::create(
                  rewriter, op.getLoc(), *valueType, value)
                  .getResult(0);
    }
    LLVM::StoreOp::create(rewriter, op.getLoc(), value, adaptor.getPtr());
    rewriter.eraseOp(op);
    return success();
  }
};

class RecordGetFieldOpLowering : public OpConversionPattern<RecordGetFieldOp> {
public:
  using OpConversionPattern<RecordGetFieldOp>::OpConversionPattern;

  auto matchAndRewrite(RecordGetFieldOp op, OpAdaptor adaptor,
                       ConversionPatternRewriter& rewriter) const
      -> LogicalResult final {
    auto ptrType = llvm::dyn_cast<PtrType>(op.getRecord().getType());
    if (!ptrType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordType =
        llvm::dyn_cast<RecordType>(ptrType.getPointeeType());
    if (!recordType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a pointer to a record");

    auto recordBackendType = convertRecordLayoutType(recordType);
    auto fieldType = convertRecordFieldStorageType(
        recordType.getFieldType(op.getField()));
    if (!recordBackendType || !fieldType)
      return rewriter.notifyMatchFailure(
          op, "field address needs a lowerable record field");

    auto fieldIndex = static_cast<int32_t>(
        recordType.getFieldIndex(op.getField()));
    auto fieldPtr = LLVM::GEPOp::create(
        rewriter, op.getLoc(), getPtrType(op.getContext()), *recordBackendType,
        adaptor.getRecord(), ArrayRef<LLVM::GEPArg>{0, fieldIndex});
    rewriter.replaceOp(op, fieldPtr.getResult());
    return success();
  }
};

} // namespace

void populateMulberryCoreLoweringPatterns(RewritePatternSet& patterns,
                                          TypeConverter& typeConverter,
                                          MLIRContext* context) {
  patterns.add<AllocaOpLowering, ReplSlotOpLowering,
               CallIndirectTypeConversion,
               DataConstructOpLowering, DataTagOpLowering,
               DataUnpackOpLowering, FunctionConstantTypeConversion,
               ResultTryTypeConversion,
               HeapAllocOpLowering, LoadOpLowering, PtrCastOpLowering,
               PtrIndexOpLowering, RecordGetFieldOpLowering, StoreOpLowering>(
      typeConverter, context);
}

} // namespace mlir::mulberry_core::detail
