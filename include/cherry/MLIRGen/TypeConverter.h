#ifndef CHERRY_MLIRGEN_TYPECONVERTER_H
#define CHERRY_MLIRGEN_TYPECONVERTER_H

#include "cherry/Basic/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mlir {
class MemRefType;
} // namespace mlir

namespace cir {
class RecordType;
} // namespace cir

namespace cherry {

class MLIRTypeConverter {
public:
  explicit MLIRTypeConverter(mlir::OpBuilder& builder) : _builder(builder) {}

  auto convert(const Type *type) const -> mlir::Type;

private:
  auto convert(const BuiltinType& type) const -> mlir::Type;
  auto convert(const TensorType& type) const -> mlir::MemRefType;
  auto convertTensorElement(const BuiltinType& type) const -> mlir::Type;
  auto convert(const StructType& type) const -> cir::RecordType;

  mlir::OpBuilder& _builder;
};

} // namespace cherry

#endif // CHERRY_MLIRGEN_TYPECONVERTER_H
