#ifndef MULBERRY_MLIRGEN_TYPECONVERTER_H
#define MULBERRY_MLIRGEN_TYPECONVERTER_H

#include "mulberry/Basic/Types.h"
#include "mulberry/MLIRGen/IR/MulberryTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"

namespace mulberry {

class MLIRTypeConverter {
public:
  explicit MLIRTypeConverter(mlir::OpBuilder& builder) : _builder(builder) {}

  // Keep object layout, source reference, and field/element storage contexts
  // explicit even when two representations currently happen to be identical.
  auto convertLayout(const Type *type) const -> mlir::Type;
  auto convertSource(const Type *type) const -> mlir::Type;
  auto convertStorage(const Type *type) const -> mlir::Type;

private:
  auto convertLayout(const BuiltinType& type) const -> mlir::Type;
  auto convertLayout(const ArrayType& type) const -> mlir::Type;
  auto convertLayout(const PtrType& type) const -> mlir::Type;
  auto convertLayout(const StructType& type) const
      -> mlir::mulberry_core::RecordType;

  mlir::OpBuilder& _builder;
};

} // namespace mulberry

#endif // MULBERRY_MLIRGEN_TYPECONVERTER_H
