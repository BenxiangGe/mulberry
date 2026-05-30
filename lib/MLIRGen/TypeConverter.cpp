#include "cherry/MLIRGen/TypeConverter.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace cherry {
namespace {

auto convertTensorShape(const std::vector<int64_t>& shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> result;
  for (auto dim : shape)
    result.push_back(dim < 0 ? mlir::ShapedType::kDynamic : dim);
  return result;
}

} // namespace

auto MLIRTypeConverter::convertBuiltin(const BuiltinType& type) const
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

auto MLIRTypeConverter::convertTensor(const TensorType& type) const
    -> mlir::MemRefType {
  auto mlirElementType = convertTensorElement(*type.elementType());
  if (!mlirElementType)
    return {};

  return mlir::MemRefType::get(convertTensorShape(type.shape()),
                               mlirElementType);
}

auto MLIRTypeConverter::convertTensorDescriptor(const TensorType& type) const
    -> cir::RecordType {
  auto mlirElementType = convertTensorElement(*type.elementType());
  if (!mlirElementType)
    return {};

  auto pointerType = cir::PointerType::get(mlirElementType);
  auto indexType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  auto sizesType = cir::ArrayType::get(indexType, type.shape().size());
  auto stridesType = cir::ArrayType::get(indexType, type.shape().size());

  auto layout =
      TensorStorageLayout{pointerType, indexType, sizesType, stridesType};
  return cir::RecordType::get(_builder.getContext(), layout.fields(),
                             /*packed=*/false, /*padded=*/false,
                             cir::RecordType::RecordKind::Struct);
}

auto MLIRTypeConverter::convertListElement(const Type *type) const
    -> mlir::Type {
  if (auto *tensorType = cherry::getTensorType(type))
    return convertTensorDescriptor(*tensorType);

  return convert(type);
}

auto MLIRTypeConverter::convertList(const ListType& type) const
    -> cir::RecordType {
  auto elementType = convertListElement(type.elementType());
  if (!elementType)
    return {};

  auto lengthType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  auto dataPtrType = cir::PointerType::get(elementType);
  auto layout = ListStorageLayout{lengthType, dataPtrType};
  return cir::RecordType::get(_builder.getContext(), layout.fields(),
                             /*packed=*/false, /*padded=*/false,
                             cir::RecordType::RecordKind::Struct);
}

auto MLIRTypeConverter::convertStruct(const StructType& type) const
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
    return convertBuiltin(*builtinType);

  if (auto *tensorType = cherry::getTensorType(type))
    return convertTensor(*tensorType);

  if (auto *listType = cherry::getListType(type))
    return convertList(*listType);

  if (auto *structType = cherry::getStructType(type))
    return convertStruct(*structType);

  return {};
}

} // namespace cherry
