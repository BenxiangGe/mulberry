#ifndef CHERRY_MLIRGEN_TYPECONVERTER_H
#define CHERRY_MLIRGEN_TYPECONVERTER_H

#include "cherry/Basic/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

#include <array>

namespace mlir {
class MemRefType;
} // namespace mlir

namespace cir {
class RecordType;
} // namespace cir

namespace cherry {

struct TensorStorageLayout {
  static constexpr unsigned allocatedPtrIndex = 0;
  static constexpr unsigned alignedPtrIndex = 1;
  static constexpr unsigned offsetIndex = 2;
  static constexpr unsigned sizesIndex = 3;
  static constexpr unsigned stridesIndex = 4;

  mlir::Type pointerType;
  mlir::Type offsetType;
  mlir::Type sizesType;
  mlir::Type stridesType;

  // Mirror MLIR's ranked memref descriptor layout:
  // {allocated pointer, aligned pointer, offset, sizes[rank], strides[rank]}.
  // See: https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
  auto fields() const -> std::array<mlir::Type, 5> {
    return {pointerType, pointerType, offsetType, sizesType, stridesType};
  }
};

struct ListStorageLayout {
  static constexpr unsigned lengthIndex = 0;
  static constexpr unsigned dataPtrIndex = 1;

  mlir::Type lengthType;
  mlir::Type dataPtrType;

  // List values lower to {length, dataPtr}. This is separate from MLIR's
  // ranked memref descriptor layout used by tensors.
  auto fields() const -> std::array<mlir::Type, 2> {
    return {lengthType, dataPtrType};
  }
};

class MLIRTypeConverter {
public:
  explicit MLIRTypeConverter(mlir::OpBuilder& builder) : _builder(builder) {}

  auto convert(const Type *type) const -> mlir::Type;

private:
  auto convertBuiltin(const BuiltinType& type) const -> mlir::Type;
  auto convertTensor(const TensorType& type) const -> mlir::MemRefType;
  auto convertTensorDescriptor(const TensorType& type) const
      -> cir::RecordType;
  auto convertTensorElement(const BuiltinType& type) const -> mlir::Type;
  auto convertListElement(const Type *type) const -> mlir::Type;
  auto convertList(const ListType& type) const -> cir::RecordType;
  auto convertStruct(const StructType& type) const -> cir::RecordType;

  mlir::OpBuilder& _builder;
};

} // namespace cherry

#endif // CHERRY_MLIRGEN_TYPECONVERTER_H
