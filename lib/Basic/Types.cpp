#include "cherry/Basic/Types.h"
#include "llvm/Support/Casting.h"

#include <memory>
#include <string_view>
#include <utility>

namespace cherry {

Type::Type(TypeKind kind) : _kind(kind) {}

auto Type::kind() const -> TypeKind { return _kind; }

BuiltinType::BuiltinType(BuiltinTypeKind kind)
    : Type(TypeKind::Builtin), _builtinKind(kind) {}

auto BuiltinType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Builtin;
}

auto BuiltinType::builtinKind() const -> BuiltinTypeKind {
  return _builtinKind;
}

auto BuiltinType::name() const -> std::string_view {
  switch (builtinKind()) {
  case BuiltinTypeKind::Unit:
    return "Unit";
  case BuiltinTypeKind::Bool:
    return "Bool";
  case BuiltinTypeKind::UInt64:
    return "UInt64";
  case BuiltinTypeKind::Float32:
    return "Float32";
  }
  return {};
}

StructField::StructField(std::string_view name, const Type *type,
                         unsigned index)
    : _name(name), _type(type), _index(index) {}

auto StructField::name() const -> std::string_view {
  return _name;
}

auto StructField::type() const -> const Type * {
  return _type;
}

auto StructField::index() const -> unsigned {
  return _index;
}

StructType::StructType(std::string_view name,
                       std::vector<StructField> fields)
    : Type(TypeKind::Struct),
      _name(name), _fields(std::move(fields)) {}

auto StructType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Struct;
}

auto StructType::name() const -> std::string_view {
  return _name;
}

auto StructType::fields() const -> const std::vector<StructField> & {
  return _fields;
}

auto StructType::field(std::string_view fieldName) const
    -> const StructField * {
  for (const auto &field : _fields)
    if (field.name() == fieldName)
      return &field;
  return nullptr;
}

ListType::ListType(const Type *elementType, std::vector<int64_t> shape)
    : Type(TypeKind::List), _elementType(elementType),
      _shape(std::move(shape)) {}

auto ListType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::List;
}

auto ListType::elementType() const -> const Type * {
  return _elementType;
}

auto ListType::shape() const -> const std::vector<int64_t> & {
  return _shape;
}

namespace {

auto formatBuiltinType(const BuiltinType& type) -> std::string {
  return std::string(type.name());
}

auto formatListType(const ListType& type) -> std::string {
  std::string result = formatType(type.elementType());
  result += "[";
  std::string separator;
  for (auto dim : type.shape()) {
    result += separator;
    result += dim < 0 ? "?" : std::to_string(dim);
    separator = ", ";
  }
  result += "]";
  return result;
}

auto sameBuiltinType(const BuiltinType& lhs, const BuiltinType& rhs)
    -> bool {
  return lhs.builtinKind() == rhs.builtinKind();
}

auto sameListShape(const std::vector<int64_t> &lhsShape,
                   const std::vector<int64_t> &rhsShape) -> bool {
  if (lhsShape.size() != rhsShape.size())
    return false;

  for (size_t i = 0; i < lhsShape.size(); ++i) {
    auto lhsDim = lhsShape[i];
    auto rhsDim = rhsShape[i];
    if (lhsDim >= 0 && rhsDim >= 0 && lhsDim != rhsDim)
      return false;
  }
  return true;
}

auto sameListType(const ListType& lhs, const ListType& rhs) -> bool {
  return sameType(lhs.elementType(), rhs.elementType()) &&
         sameListShape(lhs.shape(), rhs.shape());
}

auto structTypeStorage()
    -> std::vector<std::unique_ptr<StructType>> & {
  static auto &types = *new std::vector<std::unique_ptr<StructType>>();
  return types;
}

auto listTypeStorage()
    -> std::vector<std::unique_ptr<ListType>> & {
  static auto &types = *new std::vector<std::unique_ptr<ListType>>();
  return types;
}

auto findListType(const Type *elementType, const std::vector<int64_t> &shape)
    -> const ListType * {
  for (const auto &type : listTypeStorage())
    if (type->elementType() == elementType && type->shape() == shape)
      return type.get();
  return nullptr;
}

auto getUnitType() -> const BuiltinType * {
  static const BuiltinType type{BuiltinTypeKind::Unit};
  return &type;
}

auto getBoolType() -> const BuiltinType * {
  static const BuiltinType type{BuiltinTypeKind::Bool};
  return &type;
}

auto getUInt64Type() -> const BuiltinType * {
  static const BuiltinType type{BuiltinTypeKind::UInt64};
  return &type;
}

auto getFloat32Type() -> const BuiltinType * {
  static const BuiltinType type{BuiltinTypeKind::Float32};
  return &type;
}

} // namespace

auto sameType(const Type *lhs, const Type *rhs) -> bool {
  if (!lhs || !rhs)
    return false;

  if (lhs->kind() != rhs->kind())
    return false;

  switch (lhs->kind()) {
  case TypeKind::Builtin:
    return sameBuiltinType(*llvm::cast<BuiltinType>(lhs),
                           *llvm::cast<BuiltinType>(rhs));
  case TypeKind::Struct:
    return lhs == rhs;
  case TypeKind::List:
    return sameListType(*llvm::cast<ListType>(lhs),
                        *llvm::cast<ListType>(rhs));
  }

  return false;
}

auto getBuiltinType(const Type *type) -> const BuiltinType * {
  return llvm::dyn_cast_if_present<BuiltinType>(type);
}

auto getListType(const Type *type) -> const ListType * {
  return llvm::dyn_cast_if_present<ListType>(type);
}

auto getStructType(const Type *type) -> const StructType * {
  return llvm::dyn_cast_if_present<StructType>(type);
}

auto isBuiltinType(const Type *type, BuiltinTypeKind kind) -> bool {
  auto *builtinType = getBuiltinType(type);
  return builtinType && builtinType->builtinKind() == kind;
}

auto isUnitType(const Type *type) -> bool {
  return isBuiltinType(type, BuiltinTypeKind::Unit);
}

auto isUInt64Type(const Type *type) -> bool {
  return isBuiltinType(type, BuiltinTypeKind::UInt64);
}

auto isBoolType(const Type *type) -> bool {
  return isBuiltinType(type, BuiltinTypeKind::Bool);
}

auto isFloat32Type(const Type *type) -> bool {
  return isBuiltinType(type, BuiltinTypeKind::Float32);
}

auto isNumericType(const Type *type) -> bool {
  return isUInt64Type(type) || isFloat32Type(type);
}

auto isEquatableType(const Type *type) -> bool {
  return isUInt64Type(type) || isBoolType(type) || isFloat32Type(type);
}

auto isListType(const Type *type) -> bool {
  return getListType(type) != nullptr;
}

auto hasUnitType(const Type *type) -> bool {
  if (isUnitType(type))
    return true;

  auto *listType = getListType(type);
  if (!listType)
    return false;
  return hasUnitType(listType->elementType());
}

auto hasUnitElementType(const Type *type) -> bool {
  auto *listType = getListType(type);
  if (!listType)
    return false;
  return hasUnitType(listType->elementType());
}

auto formatType(const Type *type) -> std::string {
  if (!type)
    return "NULL";

  switch (type->kind()) {
  case TypeKind::Builtin:
    return formatBuiltinType(*llvm::cast<BuiltinType>(type));
  case TypeKind::Struct:
    return std::string(llvm::cast<StructType>(type)->name());
  case TypeKind::List:
    return formatListType(*llvm::cast<ListType>(type));
  }

  return "";
}

auto TypeContext::getBuiltinType(BuiltinTypeKind kind) const
    -> const BuiltinType * {
  switch (kind) {
  case BuiltinTypeKind::Unit:
    return getUnitType();
  case BuiltinTypeKind::Bool:
    return getBoolType();
  case BuiltinTypeKind::UInt64:
    return getUInt64Type();
  case BuiltinTypeKind::Float32:
    return getFloat32Type();
  }
  return nullptr;
}

auto TypeContext::createStructType(std::string_view name,
                                   std::vector<StructField> fields) const
    -> const StructType * {
  auto &structTypes = structTypeStorage();
  structTypes.push_back(std::make_unique<StructType>(
      name, std::move(fields)));
  return structTypes.back().get();
}

auto TypeContext::createListType(const Type *elementType,
                                 std::vector<int64_t> shape) const
    -> const ListType * {
  if (auto *type = findListType(elementType, shape))
    return type;

  auto &listTypes = listTypeStorage();
  listTypes.push_back(
      std::make_unique<ListType>(elementType, std::move(shape)));
  return listTypes.back().get();
}

} // namespace cherry
