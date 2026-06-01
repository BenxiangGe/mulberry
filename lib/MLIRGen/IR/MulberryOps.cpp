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
  auto dataPtrType =
      llvm::dyn_cast_if_present<PtrType>(descriptorType.getFieldType("data"));
  if (!dataPtrType)
    return {};
  return dataPtrType.getElementType();
}

static auto getTensorDescriptorRank(RecordType descriptorType)
    -> std::optional<int64_t> {
  auto shapeFieldType = descriptorType.getFieldType("shape");
  auto shapeType = llvm::dyn_cast_if_present<RecordType>(shapeFieldType);
  if (!shapeType)
    return std::nullopt;

  for (const auto& field : shapeType.getFields())
    if (!field.type.isInteger(64))
      return std::nullopt;
  return shapeType.getNumFields();
}

static auto verifyTensorDescriptorMatchesMemRef(Operation *op,
                                                RecordType descriptorType,
                                                MemRefType memRefType)
    -> LogicalResult {
  // Tensor descriptors are intentionally just Mulberry records at this layer.
  // Keep the verifier shape-based so future lowering can replace pack/unpack
  // with ordinary record field materialization.
  auto elementType = getTensorDescriptorElementType(descriptorType);
  if (!elementType)
    return op->emitOpError("tensor descriptor must have data pointer field");
  if (elementType != memRefType.getElementType())
    return op->emitOpError(
        "tensor descriptor data element type must match memref element type");

  auto rank = getTensorDescriptorRank(descriptorType);
  if (!rank)
    return op->emitOpError("tensor descriptor must have i64 shape record");
  if (*rank != memRefType.getRank())
    return op->emitOpError("tensor descriptor rank must match memref rank");

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
