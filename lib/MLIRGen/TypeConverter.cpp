#include "mulberry/MLIRGen/TypeConverter.h"

#include "llvm/Support/ErrorHandling.h"

namespace mulberry {

namespace mulberry_core = mlir::mulberry_core;

namespace {

auto isObjectType(const Type *type) -> bool {
  return mulberry::getStructType(type) || mulberry::getArrayType(type);
}

} // namespace

auto MLIRTypeConverter::convertLayout(const BuiltinType& type) const
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

auto MLIRTypeConverter::convertLayout(const ArrayType& type) const
    -> mlir::Type {
  auto mlirElementType = convertStorage(type.elementType());
  if (!mlirElementType)
    return {};

  std::vector<mulberry_core::RecordType::Field> fields;
  fields.push_back({"length", _builder.getI64Type()});
  fields.push_back({
      "data",
      mulberry_core::PtrType::get(_builder.getContext(), mlirElementType),
  });
  return mulberry_core::RecordType::get(_builder.getContext(), "array", fields);
}

auto MLIRTypeConverter::convertLayout(const PtrType& type) const
    -> mlir::Type {
  auto pointeeType = convertStorage(type.pointeeType());
  if (!pointeeType)
    return {};

  return mulberry_core::PtrType::get(_builder.getContext(), pointeeType);
}

auto MLIRTypeConverter::convertLayout(const StructType& type) const
    -> mulberry_core::RecordType {
  std::vector<mulberry_core::RecordType::Field> fields;
  for (const auto& field : type.fields()) {
    auto fieldType = convertStorage(field.type());
    if (!fieldType)
      return {};
    fields.push_back({std::string(field.name()), fieldType});
  }

  return mulberry_core::RecordType::get(_builder.getContext(), type.name(),
                                        fields);
}

auto MLIRTypeConverter::convertLayout(const Type *type) const -> mlir::Type {
  if (auto *builtinType = mulberry::getBuiltinType(type))
    return convertLayout(*builtinType);

  if (auto *arrayType = mulberry::getArrayType(type))
    return convertLayout(*arrayType);

  if (auto *ptrType = mulberry::getPtrType(type))
    return convertLayout(*ptrType);

  if (auto *structType = mulberry::getStructType(type))
    return convertLayout(*structType);

  return {};
}

auto MLIRTypeConverter::convertSource(const Type *type) const -> mlir::Type {
  auto layoutType = convertLayout(type);
  if (!layoutType)
    return {};
  if (!isObjectType(type))
    return layoutType;
  return mulberry_core::PtrType::get(_builder.getContext(), layoutType);
}

auto MLIRTypeConverter::convertStorage(const Type *type) const -> mlir::Type {
  if (isObjectType(type))
    return convertSource(type);
  return convertLayout(type);
}

} // namespace mulberry
