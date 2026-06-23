#ifndef CHERRY_MLIRGEN_TYPECONVERTER_H
#define CHERRY_MLIRGEN_TYPECONVERTER_H

#include "cherry/Basic/Types.h"
#include "cherry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
class MemRefType;
} // namespace mlir

namespace cherry {

class MLIRTypeConverter {
public:
  explicit MLIRTypeConverter(mlir::OpBuilder& builder) : _builder(builder) {}

  auto convert(const Type *type) const -> mlir::Type;
  auto convertTensorToMemRefType(const TensorType& type) const
      -> mlir::MemRefType;

private:
  auto convert(const BuiltinType& type) const -> mlir::Type;
  auto convert(const TensorType& type) const -> mlir::mulberry::TensorType;
  auto convert(const PtrType& type) const -> mlir::Type;
  auto convertTensorElement(const BuiltinType& type) const -> mlir::Type;
  auto convert(const StructType& type) const -> mlir::mulberry::RecordType;

  mlir::OpBuilder& _builder;
};

} // namespace cherry

#endif // CHERRY_MLIRGEN_TYPECONVERTER_H
