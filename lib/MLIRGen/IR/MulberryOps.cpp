//===--- MulberryOps.cpp - Mulberry dialect ops ---------------------------===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#include "cherry/MLIRGen/IR/MulberryOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

#include <optional>

using namespace mlir;
using namespace mlir::mulberry;

#define GET_OP_CLASSES
#include "cherry/MLIRGen/IR/MulberryOps.cpp.inc"

static auto getPtrElementType(Type type) -> Type {
  if (auto ptrType = llvm::dyn_cast<PtrType>(type))
    return ptrType.getElementType();
  return {};
}

static auto getTensorDescriptorElementType(RecordType descriptorType)
    -> Type {
  auto allocatedPtrType = llvm::dyn_cast_if_present<PtrType>(
      descriptorType.getFieldType("allocated"));
  auto alignedPtrType = llvm::dyn_cast_if_present<PtrType>(
      descriptorType.getFieldType("aligned"));
  if (!allocatedPtrType || !alignedPtrType)
    return {};
  if (allocatedPtrType.getElementType() != alignedPtrType.getElementType())
    return {};
  return allocatedPtrType.getElementType();
}

static auto getTensorMetadataRank(RecordType descriptorType, StringRef fieldName)
    -> std::optional<int64_t> {
  auto metadataFieldType = descriptorType.getFieldType(fieldName);
  auto metadataType = llvm::dyn_cast_if_present<RecordType>(metadataFieldType);
  if (!metadataType)
    return std::nullopt;

  for (const auto& field : metadataType.getFields())
    if (!field.type.isInteger(64))
      return std::nullopt;
  return metadataType.getNumFields();
}

static auto verifyTensorDescriptorMatchesMemRef(Operation *op,
                                                RecordType descriptorType,
                                                MemRefType memRefType)
    -> LogicalResult {
  // Match MLIR memref metadata instead of only shape: this keeps sliced or
  // strided tensors representable when pack/unpack are materialized later.
  auto elementType = getTensorDescriptorElementType(descriptorType);
  if (!elementType)
    return op->emitOpError(
        "tensor descriptor must have matching allocated/aligned pointer fields");
  if (elementType != memRefType.getElementType())
    return op->emitOpError("tensor descriptor pointer element type must match "
                           "memref element type");

  auto offsetType = descriptorType.getFieldType("offset");
  if (!offsetType || !offsetType.isInteger(64))
    return op->emitOpError("tensor descriptor must have i64 offset field");

  auto sizesRank = getTensorMetadataRank(descriptorType, "sizes");
  if (!sizesRank)
    return op->emitOpError("tensor descriptor must have i64 sizes record");
  if (*sizesRank != memRefType.getRank())
    return op->emitOpError("tensor descriptor sizes rank must match memref rank");

  auto stridesRank = getTensorMetadataRank(descriptorType, "strides");
  if (!stridesRank)
    return op->emitOpError("tensor descriptor must have i64 strides record");
  if (*stridesRank != memRefType.getRank())
    return op->emitOpError(
        "tensor descriptor strides rank must match memref rank");

  return success();
}

auto TensorPackOp::verify() -> LogicalResult {
  return verifyTensorDescriptorMatchesMemRef(
      getOperation(), llvm::cast<RecordType>(getResult().getType()),
      llvm::cast<MemRefType>(getTensor().getType()));
}

auto TensorUnpackOp::verify() -> LogicalResult {
  return verifyTensorDescriptorMatchesMemRef(
      getOperation(), llvm::cast<RecordType>(getTensor().getType()),
      llvm::cast<MemRefType>(getResult().getType()));
}

namespace {

struct FoldUnpackOfPack final : public OpRewritePattern<TensorUnpackOp> {
  using OpRewritePattern<TensorUnpackOp>::OpRewritePattern;

  auto matchAndRewrite(TensorUnpackOp op,
                       PatternRewriter& rewriter) const
      -> LogicalResult final {
    auto packOp = op.getTensor().getDefiningOp<TensorPackOp>();
    if (!packOp)
      return failure();

    // Only erase the bridge when the exact memref type is preserved. Different
    // shapes/layouts need an explicit cast or descriptor materialization.
    if (packOp.getTensor().getType() != op.getResult().getType())
      return failure();

    rewriter.replaceOp(op, packOp.getTensor());
    return success();
  }
};

} // namespace

void TensorUnpackOp::getCanonicalizationPatterns(RewritePatternSet& patterns,
                                                 MLIRContext *context) {
  patterns.add<FoldUnpackOfPack>(context);
}

auto RecordCreateOp::verify() -> LogicalResult {
  auto recordType = llvm::cast<RecordType>(getResult().getType());
  if (getFields().size() != recordType.getNumFields())
    return emitOpError("field count must match record type");

  for (auto field : llvm::enumerate(recordType.getFields())) {
    auto valueType = getFields()[field.index()].getType();
    if (valueType != field.value().type)
      return emitOpError("field `")
             << field.value().name << "` type must match record type";
  }

  return success();
}

auto AllocaOp::verify() -> LogicalResult {
  auto resultElementType = getPtrElementType(getResult().getType());
  if (resultElementType != getElementType())
    return emitOpError("result pointer element type must match alloca type");
  return success();
}

auto LoadOp::verify() -> LogicalResult {
  auto ptrElementType = getPtrElementType(getPtr().getType());
  if (ptrElementType != getResult().getType())
    return emitOpError("result type must match pointer element type");
  return success();
}

auto StoreOp::verify() -> LogicalResult {
  auto ptrElementType = getPtrElementType(getPtr().getType());
  if (ptrElementType != getValue().getType())
    return emitOpError("value type must match pointer element type");
  return success();
}

auto RecordGetFieldOp::verify() -> LogicalResult {
  auto recordType = llvm::dyn_cast<RecordType>(
      getPtrElementType(getRecord().getType()));
  if (!recordType)
    return emitOpError("input must be a pointer to a Mulberry record");

  auto fieldType = recordType.getFieldType(getField());
  if (!fieldType)
    return emitOpError("unknown record field `") << getField() << "`";

  auto resultElementType = getPtrElementType(getResult().getType());
  if (resultElementType != fieldType)
    return emitOpError("result pointer element type must match field type");

  return success();
}
