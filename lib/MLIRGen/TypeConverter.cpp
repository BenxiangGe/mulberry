#include "cherry/MLIRGen/TypeConverter.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

namespace cherry {

static auto convertMemRefShape(const std::vector<int64_t>& shape)
    -> std::vector<int64_t> {
  std::vector<int64_t> memrefShape;
  for (auto dim : shape) {
    memrefShape.push_back(dim < 0 ? mlir::ShapedType::kDynamic : dim);
  }
  return memrefShape;
}

auto MLIRTypeConverter::convert(const BuiltinType& type) const
    -> mlir::Type {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::Unit:
    return _builder.getNoneType();
  case BuiltinTypeKind::UInt64:
    return _builder.getI64Type();
  case BuiltinTypeKind::Float32:
    return _builder.getF32Type();
  case BuiltinTypeKind::Bool:
    return _builder.getI1Type();
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
    -> mlir::mulberry::TensorType {
  auto *elementType = cherry::getBuiltinType(type.elementType());
  if (!elementType)
    return {};

  auto mlirElementType = convertTensorElement(*elementType);
  if (!mlirElementType)
    return {};

  return mlir::mulberry::TensorType::get(_builder.getContext(), type.shape(),
                                         mlirElementType);
}

auto MLIRTypeConverter::convert(const ListType& type) const
    -> mlir::mulberry::ListType {
  auto elementType = convert(type.elementType());
  if (!elementType)
    return {};

  return mlir::mulberry::ListType::get(_builder.getContext(), elementType);
}

auto MLIRTypeConverter::convertTensorStorage(const TensorType& type) const
    -> mlir::MemRefType {
  auto *elementType = cherry::getBuiltinType(type.elementType());
  if (!elementType)
    return {};

  auto mlirElementType = convertTensorElement(*elementType);
  if (!mlirElementType)
    return {};

  // TODO: This is the temporary storage view for current cherry_nn/memref
  // codegen. High-level MLIRGen should use convert() and keep Tensor as a
  // Mulberry value until the real lowering pass decides the storage ABI.
  return mlir::MemRefType::get(convertMemRefShape(type.shape()),
                               mlirElementType);
}

auto MLIRTypeConverter::convert(const StructType& type) const
    -> mlir::mulberry::RecordType {
  std::vector<mlir::mulberry::RecordType::Field> fields;
  for (const auto& field : type.fields()) {
    auto fieldType = convert(field.type());
    if (!fieldType)
      return {};
    fields.push_back({std::string(field.name()), fieldType});
  }

  return mlir::mulberry::RecordType::get(_builder.getContext(), type.name(),
                                         fields);
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
