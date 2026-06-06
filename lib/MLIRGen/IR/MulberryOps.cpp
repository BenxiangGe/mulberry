//===--- MulberryOps.cpp - Mulberry dialect ops ---------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/MulberryOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::mulberry;

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/MulberryOps.cpp.inc"

static auto getPtrPointeeType(Type type) -> Type {
  if (auto ptrType = llvm::dyn_cast<PtrType>(type))
    return ptrType.getPointeeType();
  return {};
}

static auto getTensorType(Type type) -> mlir::mulberry::TensorType {
  return llvm::dyn_cast<mlir::mulberry::TensorType>(type);
}

static auto getListType(Type type) -> mlir::mulberry::ListType {
  return llvm::dyn_cast<mlir::mulberry::ListType>(type);
}

static auto getListStorageType(Type type)
    -> mlir::mulberry::ListStorageType {
  return llvm::dyn_cast<mlir::mulberry::ListStorageType>(type);
}

static auto countDynamicDims(ArrayRef<int64_t> shape) -> size_t {
  size_t count = 0;
  for (auto dim : shape)
    if (dim < 0)
      ++count;
  return count;
}

static auto compatibleTensorShape(ArrayRef<int64_t> sourceShape,
                                  ArrayRef<int64_t> destShape) -> bool {
  if (sourceShape.size() != destShape.size())
    return false;

  for (size_t i = 0; i < sourceShape.size(); ++i) {
    auto sourceDim = sourceShape[i];
    auto destDim = destShape[i];
    if (sourceDim >= 0 && destDim >= 0 && sourceDim != destDim)
      return false;
  }

  return true;
}

auto AllocaOp::verify() -> LogicalResult {
  auto resultPointeeType = getPtrPointeeType(getResult().getType());
  if (resultPointeeType != getElementType())
    return emitOpError("result pointer pointee type must match alloca type");
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

auto TensorAllocOp::verify() -> LogicalResult {
  auto tensorType = getTensorType(getResult().getType());
  auto expectedDynamicSizeCount = countDynamicDims(tensorType.getShape());
  if (expectedDynamicSizeCount != getDynamicSizes().size())
    return emitOpError("dynamic size count must match dynamic tensor dims");

  return success();
}

auto TensorDimOp::verify() -> LogicalResult {
  if (!getTensorType(getTensor().getType()))
    return emitOpError("input must be a Mulberry tensor");

  return success();
}

auto TensorCastOp::verify() -> LogicalResult {
  auto sourceType = getTensorType(getSource().getType());
  auto destType = getTensorType(getDest().getType());

  if (sourceType.getElementType() != destType.getElementType())
    return emitOpError("source and destination element types must match");

  if (!compatibleTensorShape(sourceType.getShape(), destType.getShape()))
    return emitOpError("source and destination shapes are incompatible");

  return success();
}

auto TensorLoadOp::verify() -> LogicalResult {
  auto tensorType = getTensorType(getTensor().getType());
  if (tensorType.getShape().size() != getIndices().size())
    return emitOpError("index count must match tensor rank");

  if (tensorType.getElementType() != getResult().getType())
    return emitOpError("result type must match tensor element type");

  return success();
}

auto TensorStoreOp::verify() -> LogicalResult {
  auto tensorType = getTensorType(getTensor().getType());
  if (tensorType.getShape().size() != getIndices().size())
    return emitOpError("index count must match tensor rank");

  if (tensorType.getElementType() != getValue().getType())
    return emitOpError("value type must match tensor element type");

  return success();
}

auto ListCreateOp::verify() -> LogicalResult {
  auto listType = getListType(getResult().getType());
  auto elementType = listType.getElementType();
  for (auto element : getElements())
    if (element.getType() != elementType)
      return emitOpError("element type must match list element type");

  return success();
}

auto ListGetOp::verify() -> LogicalResult {
  auto listType = getListType(getList().getType());
  if (listType.getElementType() != getResult().getType())
    return emitOpError("result type must match list element type");

  return success();
}

auto ListLoadOp::verify() -> LogicalResult {
  auto storageType = getListStorageType(getStorage().getType());
  if (storageType.getElementType() != getResult().getType())
    return emitOpError("result type must match list storage element type");

  return success();
}

auto ListStoreOp::verify() -> LogicalResult {
  auto storageType = getListStorageType(getStorage().getType());
  if (storageType.getElementType() != getValue().getType())
    return emitOpError("value type must match list storage element type");

  return success();
}
