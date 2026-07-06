#include "mulberry/MLIRGen/TypeConverter.h"

#include "llvm/Support/ErrorHandling.h"

namespace mulberry {

auto MLIRTypeConverter::convert(const BuiltinType& type) const
    -> mlir::Type {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::Unit:
    return _builder.getNoneType();
  case BuiltinTypeKind::UInt8:
    return _builder.getI8Type();
  case BuiltinTypeKind::UInt64:
    return _builder.getI64Type();
  case BuiltinTypeKind::Float32:
    return _builder.getF32Type();
  case BuiltinTypeKind::Bool:
    return _builder.getI1Type();
  }
}

auto MLIRTypeConverter::convert(const ArrayType& type) const -> mlir::Type {
  auto mlirElementType = convert(type.elementType());
  if (!mlirElementType)
    return {};

  std::vector<mlir::mulberry_core::RecordType::Field> fields;
  fields.push_back({"length", _builder.getI64Type()});
  fields.push_back({
      "data",
      mlir::mulberry_core::PtrType::get(_builder.getContext(),
                                        mlirElementType),
  });
  return mlir::mulberry_core::RecordType::get(_builder.getContext(), "array",
                                              fields);
}

auto MLIRTypeConverter::convert(const PtrType& type) const
    -> mlir::Type {
  auto pointeeType = convert(type.pointeeType());
  if (!pointeeType)
    return {};

  return mlir::mulberry_core::PtrType::get(_builder.getContext(), pointeeType);
}

auto MLIRTypeConverter::convert(const StructType& type) const
    -> mlir::mulberry_core::RecordType {
  std::vector<mlir::mulberry_core::RecordType::Field> fields;
  for (const auto& field : type.fields()) {
    auto fieldType = convert(field.type());
    if (!fieldType)
      return {};
    fields.push_back({std::string(field.name()), fieldType});
  }

  return mlir::mulberry_core::RecordType::get(_builder.getContext(), type.name(),
                                         fields);
}

auto MLIRTypeConverter::convert(const Type *type) const -> mlir::Type {
  if (auto *builtinType = mulberry::getBuiltinType(type))
    return convert(*builtinType);

  if (auto *arrayType = mulberry::getArrayType(type))
    return convert(*arrayType);

  if (auto *ptrType = mulberry::getPtrType(type))
    return convert(*ptrType);

  if (auto *structType = mulberry::getStructType(type))
    return convert(*structType);

  return {};
}

} // namespace mulberry
