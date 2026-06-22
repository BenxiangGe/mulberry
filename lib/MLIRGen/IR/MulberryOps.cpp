//===--- MulberryOps.cpp - Mulberry dialect ops ---------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/MulberryOps.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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

static auto getTensorDescType(Type type)
    -> mlir::mulberry::TensorDescType {
  return llvm::dyn_cast<mlir::mulberry::TensorDescType>(type);
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

static auto tensorShapeFitsDesc(ArrayRef<int64_t> tensorShape,
                                ArrayRef<int64_t> descShape) -> bool {
  if (tensorShape.size() != descShape.size())
    return false;

  for (size_t i = 0; i < tensorShape.size(); ++i) {
    auto tensorDim = tensorShape[i];
    auto descDim = descShape[i];
    if (descDim >= 0 && tensorDim != descDim)
      return false;
  }

  return true;
}

static auto descShapeFitsTensor(ArrayRef<int64_t> descShape,
                                ArrayRef<int64_t> tensorShape) -> bool {
  if (descShape.size() != tensorShape.size())
    return false;

  for (size_t i = 0; i < descShape.size(); ++i) {
    auto descDim = descShape[i];
    auto tensorDim = tensorShape[i];
    // Unpack may widen static descriptor dims to dynamic tensor dims. Narrowing
    // a dynamic descriptor dim to a static tensor dim would need a runtime check.
    if (tensorDim >= 0 && descDim != tensorDim)
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

auto TensorDescPackOp::verify() -> LogicalResult {
  auto tensorType = getTensorType(getTensor().getType());
  auto descType = getTensorDescType(getResult().getType());

  if (tensorType.getElementType() != descType.getElementType())
    return emitOpError(
        "descriptor element type must match tensor element type");

  if (!tensorShapeFitsDesc(tensorType.getShape(), descType.getShape()))
    return emitOpError("descriptor shape must be compatible with tensor shape");

  return success();
}

auto TensorDescUnpackOp::verify() -> LogicalResult {
  auto descType = getTensorDescType(getDesc().getType());
  auto tensorType = getTensorType(getResult().getType());

  if (descType.getElementType() != tensorType.getElementType())
    return emitOpError(
        "tensor element type must match descriptor element type");

  if (!descShapeFitsTensor(descType.getShape(), tensorType.getShape()))
    return emitOpError("tensor shape must be compatible with descriptor shape");

  return success();
}

static auto isTensorABIDataPtrType(Type type) -> bool {
  return llvm::isa<LLVM::LLVMPointerType>(type);
}

static auto isTensorABIRecordType(Type type, unsigned rank) -> bool {
  auto structType = llvm::dyn_cast<LLVM::LLVMStructType>(type);
  if (!structType || structType.isOpaque())
    return false;

  auto fields = structType.getBody();
  if (fields.size() != 3 || !isTensorABIDataPtrType(fields[0]))
    return false;

  auto sizesType = llvm::dyn_cast<LLVM::LLVMArrayType>(fields[1]);
  auto stridesType = llvm::dyn_cast<LLVM::LLVMArrayType>(fields[2]);
  if (!sizesType || !stridesType)
    return false;

  return sizesType.getNumElements() == rank &&
         stridesType.getNumElements() == rank &&
         sizesType.getElementType().isInteger(64) &&
         stridesType.getElementType().isInteger(64);
}

auto TensorDescToABIOp::verify() -> LogicalResult {
  auto descType = getTensorDescType(getDesc().getType());
  if (!isTensorABIRecordType(getResult().getType(), descType.getShape().size()))
    return emitOpError(
        "result type must be a tensor ABI record `{data, sizes, strides}`");

  return success();
}
