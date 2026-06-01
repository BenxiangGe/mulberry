#include "cherry/MLIRGen/TypeConverter.h"

#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>
#include <vector>

namespace cherry {

namespace {

auto tensorElementName(const BuiltinType& type) -> std::string {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::UInt64:
    return "UInt64";
  case BuiltinTypeKind::Float32:
    return "Float32";
  case BuiltinTypeKind::Bool:
    return "Bool";
  case BuiltinTypeKind::Unit:
    llvm_unreachable("Unit cannot be used as a tensor descriptor element");
  }
}

auto tensorShapeName(const TensorType& type) -> std::string {
  return "TensorShapeRank" + std::to_string(type.shape().size());
}

auto tensorDescriptorName(const TensorType& type) -> std::string {
  auto *elementType = cherry::getBuiltinType(type.elementType());
  if (!elementType)
    return {};

  return "TensorDescriptor" + tensorElementName(*elementType) + "Rank" +
         std::to_string(type.shape().size());
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

  return mlir::MemRefType::get(type.shape(), mlirElementType);
}

auto MLIRTypeConverter::convertTensorShape(const TensorType& type) const
    -> mlir::mulberry::RecordType {
  std::vector<mlir::mulberry::RecordType::Field> fields;
  auto dimType = _builder.getI64Type();
  for (size_t i = 0; i < type.shape().size(); ++i)
    fields.push_back({"dim" + std::to_string(i), dimType});

  return mlir::mulberry::RecordType::get(
      _builder.getContext(), tensorShapeName(type), fields);
}

auto MLIRTypeConverter::convertTensorDataPtr(const TensorType& type) const
    -> mlir::mulberry::PtrType {
  auto *elementType = cherry::getBuiltinType(type.elementType());
  if (!elementType)
    return {};

  auto mlirElementType = convertTensorElement(*elementType);
  if (!mlirElementType)
    return {};

  return mlir::mulberry::PtrType::get(_builder.getContext(),
                                      mlirElementType);
}

auto MLIRTypeConverter::convertTensorDescriptor(const TensorType& type) const
    -> mlir::mulberry::RecordType {
  auto dataPtrType = convertTensorDataPtr(type);
  auto shapeType = convertTensorShape(type);
  auto name = tensorDescriptorName(type);
  if (!dataPtrType || !shapeType || name.empty())
    return {};

  // Descriptor names include element type and rank so multiple tensor layouts
  // do not collapse into one named record during CIR/LLVM lowering.
  std::vector<mlir::mulberry::RecordType::Field> fields = {
      {"data", dataPtrType}, {"shape", shapeType}};
  return mlir::mulberry::RecordType::get(_builder.getContext(), name, fields);
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

  if (auto *structType = cherry::getStructType(type))
    return convert(*structType);

  return {};
}

} // namespace cherry
