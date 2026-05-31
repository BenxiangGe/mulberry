#ifndef CHERRY_MLIRGEN_CONVERSION_TENSORDESCRIPTORLOWERING_H
#define CHERRY_MLIRGEN_CONVERSION_TENSORDESCRIPTORLOWERING_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <vector>

namespace mlir::cherry {

inline auto getTensorDescriptorMemRefType(MemRefType memRefType)
    -> MemRefType {
  std::vector<int64_t> strides;
  for (int64_t i = 0; i < memRefType.getRank(); ++i)
    strides.push_back(ShapedType::kDynamic);

  auto layout = StridedLayoutAttr::get(memRefType.getContext(),
                                       ShapedType::kDynamic, strides);
  return MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                         layout, memRefType.getMemorySpace());
}

inline auto getTensorReinterpretSizes(MemRefType memRefType,
                                      ValueRange storedSizes,
                                      PatternRewriter& rewriter)
    -> SmallVector<OpFoldResult> {
  SmallVector<OpFoldResult> sizes;
  auto storedSize = storedSizes.begin();

  for (auto dim : memRefType.getShape()) {
    // memref.reinterpret_cast requires static sizes to be attributes even
    // though descriptor storage keeps every runtime size for the future ABI.
    if (ShapedType::isDynamic(dim))
      sizes.push_back(*storedSize);
    else
      sizes.push_back(rewriter.getIndexAttr(dim));
    ++storedSize;
  }

  return sizes;
}

inline auto getTensorReinterpretStrides(ValueRange storedStrides)
    -> SmallVector<OpFoldResult> {
  SmallVector<OpFoldResult> strides;
  for (auto stride : storedStrides)
    strides.push_back(stride);
  return strides;
}

inline auto reconstructTensorFromDescriptor(
    Location location, MemRefType resultType, Value baseBuffer, Value offset,
    ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides,
    PatternRewriter& rewriter) -> Value {
  auto reconstructedType = getTensorDescriptorMemRefType(resultType);
  auto reconstructed = memref::ReinterpretCastOp::create(
      rewriter, location, reconstructedType, baseBuffer, offset, sizes,
      strides);

  if (reconstructed.getType() == resultType)
    return reconstructed;

  return memref::CastOp::create(rewriter, location, resultType,
                                reconstructed);
}

inline auto reconstructTensorFromMetadata(
    Location location, MemRefType resultType,
    memref::ExtractStridedMetadataOp metadata, PatternRewriter& rewriter)
    -> Value {
  auto sizes = getTensorReinterpretSizes(resultType, metadata.getSizes(),
                                         rewriter);
  auto strides = getTensorReinterpretStrides(metadata.getStrides());
  return reconstructTensorFromDescriptor(
      location, resultType, metadata.getBaseBuffer(), metadata.getOffset(),
      sizes, strides, rewriter);
}

// Tensor lists store each MLIR memref descriptor field in a separate table so
// list[i] can dynamically load one tensor descriptor. This is a lowering-only
// bridge until Mulberry gets a real runtime List<T> storage ABI.
// See: https://mlir.llvm.org/docs/Dialects/MemRef/
struct TensorDescriptorTables {
  Value baseBuffers;
  Value offsets;
  Value sizes;
  Value strides;
};

inline auto createTensorDescriptorTables(Location location, size_t length,
                                         MemRefType tensorType,
                                         PatternRewriter& rewriter)
    -> TensorDescriptorTables {
  auto elementType = tensorType.getElementType();
  auto memorySpace = tensorType.getMemorySpace();
  auto baseBufferType = MemRefType::get(
      {}, elementType, MemRefLayoutAttrInterface{}, memorySpace);
  auto indexType = rewriter.getIndexType();
  auto listLength = static_cast<int64_t>(length);
  auto rank = tensorType.getRank();

  auto baseBufferStorageType = MemRefType::get({listLength}, baseBufferType);
  auto offsetStorageType = MemRefType::get({listLength}, indexType);
  auto sizesStorageType = MemRefType::get({listLength, rank}, indexType);
  auto stridesStorageType = MemRefType::get({listLength, rank}, indexType);
  return {
      memref::AllocOp::create(rewriter, location, baseBufferStorageType)
          .getResult(),
      memref::AllocOp::create(rewriter, location, offsetStorageType)
          .getResult(),
      memref::AllocOp::create(rewriter, location, sizesStorageType)
          .getResult(),
      memref::AllocOp::create(rewriter, location, stridesStorageType)
          .getResult(),
  };
}

inline void storeTensorDescriptor(Location location, size_t listIndex,
                                  Value tensor,
                                  const TensorDescriptorTables& tables,
                                  PatternRewriter& rewriter) {
  auto metadata =
      memref::ExtractStridedMetadataOp::create(rewriter, location, tensor);
  auto listIndexValue =
      arith::ConstantIndexOp::create(rewriter, location, listIndex);

  memref::StoreOp::create(rewriter, location, metadata.getBaseBuffer(),
                          tables.baseBuffers, ValueRange{listIndexValue});
  memref::StoreOp::create(rewriter, location, metadata.getOffset(),
                          tables.offsets, ValueRange{listIndexValue});

  for (const auto& size : llvm::enumerate(metadata.getSizes())) {
    auto dimIndex =
        arith::ConstantIndexOp::create(rewriter, location, size.index());
    memref::StoreOp::create(rewriter, location, size.value(), tables.sizes,
                            ValueRange{listIndexValue, dimIndex});
  }

  for (const auto& stride : llvm::enumerate(metadata.getStrides())) {
    auto dimIndex =
        arith::ConstantIndexOp::create(rewriter, location, stride.index());
    memref::StoreOp::create(rewriter, location, stride.value(), tables.strides,
                            ValueRange{listIndexValue, dimIndex});
  }
}

inline auto loadTensorDescriptorSizes(Location location, MemRefType memRefType,
                                      const TensorDescriptorTables& tables,
                                      Value listIndex,
                                      PatternRewriter& rewriter)
    -> SmallVector<OpFoldResult> {
  SmallVector<OpFoldResult> sizes;
  for (const auto& dim : llvm::enumerate(memRefType.getShape())) {
    // memref.reinterpret_cast requires static sizes to be attributes even
    // though descriptor storage keeps every runtime size for the future ABI.
    if (!ShapedType::isDynamic(dim.value())) {
      sizes.push_back(rewriter.getIndexAttr(dim.value()));
      continue;
    }

    auto dimIndex =
        arith::ConstantIndexOp::create(rewriter, location, dim.index());
    auto size = memref::LoadOp::create(
        rewriter, location, tables.sizes, ValueRange{listIndex, dimIndex});
    sizes.push_back(size.getResult());
  }
  return sizes;
}

inline auto loadTensorDescriptorStrides(Location location,
                                        MemRefType memRefType,
                                        const TensorDescriptorTables& tables,
                                        Value listIndex,
                                        PatternRewriter& rewriter)
    -> SmallVector<OpFoldResult> {
  SmallVector<Value> storedStrides;
  for (int64_t dim = 0; dim < memRefType.getRank(); ++dim) {
    auto dimIndex = arith::ConstantIndexOp::create(rewriter, location, dim);
    auto stride = memref::LoadOp::create(
        rewriter, location, tables.strides, ValueRange{listIndex, dimIndex});
    storedStrides.push_back(stride.getResult());
  }
  return getTensorReinterpretStrides(storedStrides);
}

inline auto loadTensorFromDescriptorTables(
    Location location, MemRefType resultType,
    const TensorDescriptorTables& tables, Value listIndex,
    PatternRewriter& rewriter) -> Value {
  auto baseBuffer = memref::LoadOp::create(
      rewriter, location, tables.baseBuffers, ValueRange{listIndex});
  auto offset = memref::LoadOp::create(rewriter, location, tables.offsets,
                                       ValueRange{listIndex});
  auto sizes = loadTensorDescriptorSizes(location, resultType, tables,
                                         listIndex, rewriter);
  auto strides = loadTensorDescriptorStrides(location, resultType, tables,
                                             listIndex, rewriter);
  return reconstructTensorFromDescriptor(location, resultType,
                                         baseBuffer.getResult(),
                                         offset.getResult(), sizes, strides,
                                         rewriter);
}

} // namespace mlir::cherry

#endif // CHERRY_MLIRGEN_CONVERSION_TENSORDESCRIPTORLOWERING_H
