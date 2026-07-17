//===--- MulberryOps.cpp - Mulberry core dialect ops ---------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/IR/MulberryOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mulberry_core;

#define GET_OP_CLASSES
#include "mulberry/MLIRGen/IR/MulberryOps.cpp.inc"

static auto getPtrPointeeType(Type type) -> Type {
  if (auto ptrType = llvm::dyn_cast<PtrType>(type))
    return ptrType.getPointeeType();
  return {};
}

static auto getTensorType(Type type) -> mulberry_core::TensorType {
  return llvm::dyn_cast<mulberry_core::TensorType>(type);
}

static auto isDataReference(Type type) -> bool {
  auto ptrType = llvm::dyn_cast<PtrType>(type);
  return ptrType && llvm::isa<DataType>(ptrType.getPointeeType());
}

auto DataConstructOp::verify() -> LogicalResult {
  if (!isDataReference(getResult().getType()))
    return emitOpError("result must be a pointer to a Mulberry data type");
  if (getConstructor().empty())
    return emitOpError("constructor name cannot be empty");
  return success();
}

auto DataTagOp::verify() -> LogicalResult {
  if (!isDataReference(getValue().getType()))
    return emitOpError("value must be a pointer to a Mulberry data type");
  return success();
}

auto DataUnpackOp::verify() -> LogicalResult {
  if (!isDataReference(getValue().getType()))
    return emitOpError("value must be a pointer to a Mulberry data type");
  if (getConstructor().empty())
    return emitOpError("constructor name cannot be empty");
  return success();
}

auto ResultTryOp::verify() -> LogicalResult {
  if (!isDataReference(getInput().getType()))
    return emitOpError("input must reference a Mulberry data type");
  if (getNumResults() > 1)
    return emitOpError("must produce zero or one payload value");
  if (getNumResults() == 1 && llvm::isa<NoneType>(getResult(0).getType()))
    return emitOpError("Unit payload must not produce an SSA value");

  auto function = getOperation()->getParentOfType<func::FuncOp>();
  if (!function || function.getFunctionType().getNumResults() != 1 ||
      !isDataReference(function.getFunctionType().getResult(0)))
    return emitOpError(
        "enclosing function must return a Mulberry data reference");
  return success();
}

auto AllocaOp::verify() -> LogicalResult {
  auto resultPointeeType = getPtrPointeeType(getResult().getType());
  if (resultPointeeType != getElementType())
    return emitOpError("result pointer pointee type must match alloca type");
  return success();
}

auto HeapAllocOp::verify() -> LogicalResult {
  auto resultPointeeType = getPtrPointeeType(getResult().getType());
  if (resultPointeeType != getElementType())
    return emitOpError("result pointer pointee type must match heap alloc type");
  return success();
}

static auto verifyTensorStorageRecord(Operation* op, RecordType storageType,
                                      Type elementType) -> LogicalResult {
  if (!storageType || !elementType)
    return op->emitOpError(
        " result must reference TensorStorage with an element type");

  auto allocatedType = getPtrPointeeType(storageType.getFieldType("allocated"));
  auto dataType = getPtrPointeeType(storageType.getFieldType("data"));
  auto disposedType = storageType.getFieldType("disposed");
  if (allocatedType != elementType || dataType != elementType ||
      !disposedType || !disposedType.isInteger(1) ||
      storageType.getFieldIndex("allocated") != 0 ||
      storageType.getFieldIndex("data") != 1 ||
      storageType.getFieldIndex("disposed") != 2)
    return op->emitOpError(
        " TensorStorage needs ordered `allocated`/`data` pointers and an i1 "
        "`disposed` field");
  return success();
}

auto TensorStorageAllocOp::verify() -> LogicalResult {
  auto storagePtr = llvm::dyn_cast<PtrType>(getStorage().getType());
  auto storageType = storagePtr
                         ? llvm::dyn_cast<RecordType>(storagePtr.getPointeeType())
                         : RecordType{};
  if (failed(verifyTensorStorageRecord(getOperation(), storageType,
                                       getElementType())))
    return failure();

  auto payloadType = llvm::dyn_cast<TensorType>(getPayload().getType());
  if (!payloadType || payloadType.getShape().size() != 1 ||
      !ShapedType::isDynamic(payloadType.getShape().front()) ||
      payloadType.getElementType() != getElementType())
    return emitOpError(
        "payload must be a dynamic flat Tensor with matching element type");
  return success();
}

auto TensorStorageAllocLoweredOp::verify() -> LogicalResult {
  if (!llvm::isa<LLVM::LLVMPointerType>(getStorage().getType()))
    return emitOpError("storage result must be an LLVM pointer");

  auto storageLayout = llvm::dyn_cast<LLVM::LLVMStructType>(getStorageLayout());
  if (!storageLayout || storageLayout.isOpaque() ||
      storageLayout.getBody().size() != 3)
    return emitOpError("storage layout must be a three-field LLVM struct");
  auto storageFields = storageLayout.getBody();
  if (!llvm::isa<LLVM::LLVMPointerType>(storageFields[0]) ||
      !llvm::isa<LLVM::LLVMPointerType>(storageFields[1]) ||
      !storageFields[2].isInteger(1))
    return emitOpError("storage layout must be `{ptr, ptr, i1}`");
  if (!getOperation()->hasAttr("bufferization.manual_deallocation"))
    return emitOpError(
        "must carry `bufferization.manual_deallocation`");

  auto payloadType = llvm::dyn_cast<MemRefType>(getPayload().getType());
  if (!payloadType || payloadType.getRank() != 1 ||
      !payloadType.isDynamicDim(0) ||
      payloadType.getElementType() != getElementType())
    return emitOpError(
        "payload must be a dynamic flat memref with matching element type");
  return success();
}

auto PtrIndexOp::verify() -> LogicalResult {
  auto inputPointeeType = getPtrPointeeType(getPtr().getType());
  auto resultPointeeType = getPtrPointeeType(getResult().getType());
  if (inputPointeeType != resultPointeeType)
    return emitOpError("result pointer pointee type must match input pointer");
  return success();
}

auto LoadOp::verify() -> LogicalResult {
  auto ptrPointeeType = getPtrPointeeType(getPtr().getType());
  if (ptrPointeeType != getResult().getType())
    return emitOpError("result type must match pointer pointee type");
  return success();
}

auto StoreOp::verify() -> LogicalResult {
  auto ptrPointeeType = getPtrPointeeType(getPtr().getType());
  if (ptrPointeeType != getValue().getType())
    return emitOpError("value type must match pointer pointee type");
  return success();
}

auto RecordGetFieldOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(
      getPtrPointeeType(getRecord().getType()));
  if (!recordType)
    return emitOpError("input must be a pointer to a Mulberry record");

  auto fieldType = recordType.getFieldType(getField());
  if (!fieldType)
    return emitOpError("unknown record field `") << getField() << "`";

  auto resultPointeeType = getPtrPointeeType(getResult().getType());
  if (resultPointeeType != fieldType)
    return emitOpError("result pointer pointee type must match field type");

  return success();
}

static auto verifyTensorMetadataListReference(Operation* op, Type type,
                                              StringRef fieldName)
    -> LogicalResult {
  auto ptrType = llvm::dyn_cast<PtrType>(type);
  auto listType = ptrType
                      ? llvm::dyn_cast<RecordType>(ptrType.getPointeeType())
                      : RecordType{};
  if (!listType)
    return op->emitOpError(" record needs a List<i64> reference `")
           << fieldName << "` field";

  auto lengthType = listType.getFieldType("length");
  auto capacityType = listType.getFieldType("capacity");
  auto dataType = listType.getFieldType("data");
  if (!lengthType || !lengthType.isInteger(64) ||
      !capacityType || !capacityType.isInteger(64) ||
      getPtrPointeeType(dataType) != IntegerType::get(op->getContext(), 64))
    return op->emitOpError(" record needs a List<i64> reference `")
           << fieldName << "` field";

  return success();
}

static auto verifyTensorStorage(Operation* op, RecordType tensorType)
    -> LogicalResult {
  auto storagePtr = llvm::dyn_cast<PtrType>(tensorType.getFieldType("_storage"));
  auto storageType = storagePtr
                         ? llvm::dyn_cast<RecordType>(storagePtr.getPointeeType())
                         : RecordType{};
  if (!storageType)
    return op->emitOpError(" Tensor record needs a shared `_storage` object");

  auto tensorElementType = getPtrPointeeType(tensorType.getFieldType("data"));
  return verifyTensorStorageRecord(op, storageType, tensorElementType);
}

static auto verifyTensorRecordABI(Operation* op, RecordType recordType,
                                  mulberry_core::TensorType tensorType,
                                  StringRef valueName) -> LogicalResult {
  if (!recordType)
    return op->emitOpError(valueName)
           << " must be a Mulberry record";

  auto dataType = recordType.getFieldType("data");
  if (getPtrPointeeType(dataType) != tensorType.getElementType())
    return op->emitOpError(valueName)
           << " record data pointee type must match tensor element type";

  auto rankType = recordType.getFieldType("rank");
  if (!rankType || !rankType.isInteger(64))
    return op->emitOpError(valueName)
           << " record needs an i64 `rank` field";

  auto numelType = recordType.getFieldType("numel");
  if (!numelType || !numelType.isInteger(64))
    return op->emitOpError(valueName)
           << " record needs an i64 `numel` field";

  if (failed(verifyTensorMetadataListReference(
          op, recordType.getFieldType("sizes"), "sizes")) ||
      failed(verifyTensorMetadataListReference(
          op, recordType.getFieldType("strides"), "strides")) ||
      failed(verifyTensorStorage(op, recordType)))
    return failure();

  return success();
}

auto TensorViewOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(getTensorRecord().getType());
  auto tensorType = getTensorType(getResult().getType());
  return verifyTensorRecordABI(getOperation(), recordType, tensorType,
                               "tensor");
}

auto TensorPackOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(getTensorRecord().getType());
  auto tensorType = getTensorType(getTensor().getType());
  return verifyTensorRecordABI(getOperation(), recordType, tensorType,
                               "result");
}

static auto verifyTensorObject(Operation* op, Type type) -> LogicalResult {
  auto ptrType = llvm::dyn_cast<PtrType>(type);
  auto recordType = ptrType
                        ? llvm::dyn_cast<RecordType>(ptrType.getPointeeType())
                        : RecordType{};
  if (!recordType)
    return op->emitOpError(" operand must reference a Tensor record");
  return verifyTensorStorage(op, recordType);
}

auto TensorDisposeOp::verify() -> LogicalResult {
  return verifyTensorObject(getOperation(), getTensor().getType());
}

auto TensorAssertAliveOp::verify() -> LogicalResult {
  return verifyTensorObject(getOperation(), getTensor().getType());
}
