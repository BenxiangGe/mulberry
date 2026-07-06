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

  auto convert(const Type *type) const -> mlir::Type;

private:
  auto convert(const BuiltinType& type) const -> mlir::Type;
  auto convert(const ArrayType& type) const -> mlir::Type;
  auto convert(const PtrType& type) const -> mlir::Type;
  auto convert(const StructType& type) const -> mlir::mulberry_core::RecordType;

  mlir::OpBuilder& _builder;
};

} // namespace mulberry

#endif // MULBERRY_MLIRGEN_TYPECONVERTER_H
