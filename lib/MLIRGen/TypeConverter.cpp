#include "cherry/MLIRGen/TypeConverter.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace cherry {

auto MLIRTypeConverter::convert(const BuiltinType& type) const
    -> mlir::Type {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::Unit:
    return _builder.getNoneType();
  case BuiltinTypeKind::UInt64:
    return cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  case BuiltinTypeKind::Float32:
    return cir::SingleType::get(_builder.getContext());
  case BuiltinTypeKind::Bool:
    return cir::BoolType::get(_builder.getContext());
  }
}

auto MLIRTypeConverter::convertTensorElement(const BuiltinType& type) const
    -> mlir::Type {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::UInt64:
    return _builder.getI64Type();
  case BuiltinTypeKind::Float32:
    return _builder.getF32Type();
  case BuiltinTypeKind::Bool:
    return _builder.getI1Type();
  case BuiltinTypeKind::Unit:
    llvm_unreachable("Unit cannot be used as a memref element type");
  }
}

auto MLIRTypeConverter::convert(const TensorType& type) const
    -> mlir::MemRefType {
  auto *elementType = cherry::getBuiltinType(type.elementType());
  if (!elementType)
    return {};

  auto mlirElementType = convertTensorElement(*elementType);
  if (!mlirElementType)
    return {};

  return mlir::MemRefType::get(type.shape(), mlirElementType);
}

auto MLIRTypeConverter::convert(const StructType& type) const
    -> cir::RecordType {
  std::vector<mlir::Type> fieldTypes;
  for (const auto& field : type.fields()) {
    auto fieldType = convert(field.type());
    if (!fieldType)
      return {};
    fieldTypes.push_back(fieldType);
  }

  auto attr = _builder.getStringAttr(type.name());
  auto recordType = cir::RecordType::get(
      _builder.getContext(), attr, cir::RecordType::RecordKind::Struct);
  recordType.complete(fieldTypes, false, false);
  return recordType;
}

auto MLIRTypeConverter::convert(const Type *type) const -> mlir::Type {
  if (auto *builtinType = cherry::getBuiltinType(type))
    return convert(*builtinType);

  if (auto *tensorType = cherry::getTensorType(type))
    return convert(*tensorType);

  if (auto *structType = cherry::getStructType(type))
    return convert(*structType);

  return {};
}

} // namespace cherry
