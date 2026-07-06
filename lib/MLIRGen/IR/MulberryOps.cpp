//===--- MulberryOps.cpp - Mulberry core dialect ops ---------------------------===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "mulberry/MLIRGen/IR/MulberryOps.h"

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

static auto getTensorType(Type type) -> mlir::mulberry_core::TensorType {
  return llvm::dyn_cast<mlir::mulberry_core::TensorType>(type);
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

auto RecordExtractOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(getRecord().getType());
  if (!recordType)
    return emitOpError("input must be a Mulberry record");

  auto fieldType = recordType.getFieldType(getField());
  if (!fieldType)
    return emitOpError("unknown record field `") << getField() << "`";

  if (getResult().getType() != fieldType)
    return emitOpError("result type must match field type");

  return success();
}

static auto verifyTensorMetadataList(Operation* op, Type type,
                                     StringRef fieldName) -> LogicalResult {
  auto listType = llvm::dyn_cast<RecordType>(type);
  if (!listType)
    return op->emitOpError(" record needs a List<i64> `")
           << fieldName << "` field";

  auto lengthType = listType.getFieldType("length");
  auto capacityType = listType.getFieldType("capacity");
  auto dataType = listType.getFieldType("data");
  if (!lengthType || !lengthType.isInteger(64) ||
      !capacityType || !capacityType.isInteger(64) ||
      getPtrPointeeType(dataType) != IntegerType::get(op->getContext(), 64))
    return op->emitOpError(" record needs a List<i64> `")
           << fieldName << "` field";

  return success();
}

static auto verifyTensorRecord(Operation* op, RecordType recordType,
                               mlir::mulberry_core::TensorType tensorType,
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

  if (failed(verifyTensorMetadataList(op, recordType.getFieldType("sizes"),
                                      "sizes")) ||
      failed(verifyTensorMetadataList(op, recordType.getFieldType("strides"),
                                      "strides")))
    return failure();

  return success();
}

auto TensorViewOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(getTensor().getType());
  auto tensorType = getTensorType(getResult().getType());
  return verifyTensorRecord(getOperation(), recordType, tensorType, "tensor");
}

auto TensorPackOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(getResult().getType());
  auto tensorType = getTensorType(getTensor().getType());
  return verifyTensorRecord(getOperation(), recordType, tensorType, "result");
}
