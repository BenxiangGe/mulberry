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

  return mlir::MemRefType::get(convertTensorShape(type.shape()),
                               mlirElementType);
}

auto MLIRTypeConverter::convertTensorDescriptor(const TensorType& type) const
    -> cir::RecordType {
  auto memrefType = convert(type);
  if (!memrefType)
    return {};

  auto pointerType = cir::PointerType::get(memrefType.getElementType());
  auto indexType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);

  // Mirror MLIR's ranked memref descriptor layout:
  // {allocated pointer, aligned pointer, offset, sizes[rank], strides[rank]}.
  // See: https://mlir.llvm.org/docs/TargetLLVMIR/#ranked-memref-types
  std::vector<mlir::Type> fields{pointerType, pointerType, indexType};
  for (size_t i = 0; i < type.shape().size(); ++i)
    fields.push_back(indexType);
  for (size_t i = 0; i < type.shape().size(); ++i)
    fields.push_back(indexType);

  return cir::RecordType::get(_builder.getContext(), fields,
                             /*packed=*/false, /*padded=*/false,
                             cir::RecordType::RecordKind::Struct);
}

auto MLIRTypeConverter::convertListElement(const Type *type) const
    -> mlir::Type {
  if (auto *tensorType = cherry::getTensorType(type))
    return convertTensorDescriptor(*tensorType);

  return convert(type);
}

auto MLIRTypeConverter::convert(const ListType& type) const
    -> cir::RecordType {
  auto elementType = convertListElement(type.elementType());
  if (!elementType)
    return {};

  auto lengthType =
      cir::IntType::get(_builder.getContext(), 64, /*isSigned=*/false);
  auto storageType = cir::PointerType::get(elementType);
  mlir::Type fields[] = {lengthType, storageType};
  return cir::RecordType::get(_builder.getContext(), fields,
                             /*packed=*/false, /*padded=*/false,
                             cir::RecordType::RecordKind::Struct);
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

  if (auto *listType = cherry::getListType(type))
    return convert(*listType);

  if (auto *structType = cherry::getStructType(type))
    return convert(*structType);

  return {};
}

} // namespace cherry
