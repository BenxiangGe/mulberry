#include "cherry/Basic/Types.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace cherry {

Type::Type(TypeKind kind) : _kind(kind) {}

auto Type::kind() const -> TypeKind { return _kind; }

ComptimeTypeValue::ComptimeTypeValue(const Type *type)
    : _kind(Kind::Type), _type(type) {}

ComptimeTypeValue::ComptimeTypeValue(uint64_t uint64Value)
    : _kind(Kind::UInt64), _uint64Value(uint64Value) {}

auto ComptimeTypeValue::kind() const -> Kind {
  return _kind;
}

auto ComptimeTypeValue::type() const -> const Type * {
  return _type;
}

auto ComptimeTypeValue::uint64Value() const -> uint64_t {
  return _uint64Value;
}

ComptimeAliasOrigin::ComptimeAliasOrigin(
    std::string_view aliasName, std::vector<ComptimeTypeValue> arguments)
    : _aliasName(aliasName), _arguments(std::move(arguments)) {}

auto ComptimeAliasOrigin::aliasName() const -> std::string_view {
  return _aliasName;
}

auto ComptimeAliasOrigin::arguments() const
    -> const std::vector<ComptimeTypeValue> & {
  return _arguments;
}

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
  case BuiltinTypeKind::UInt8:
    return "UInt8";
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

StructType::StructType(std::string_view name,
                       std::vector<StructField> fields,
                       ComptimeAliasOrigin origin)
    : Type(TypeKind::Struct), _name(name), _fields(std::move(fields)),
      _origin(std::move(origin)) {}

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

auto StructType::origin() const -> const ComptimeAliasOrigin * {
  if (!_origin)
    return nullptr;
  return &*_origin;
}

TensorType::TensorType(const Type *elementType, std::vector<int64_t> shape)
    : Type(TypeKind::Tensor), _elementType(elementType),
      _shape(std::move(shape)) {}

auto TensorType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Tensor;
}

auto TensorType::elementType() const -> const Type * {
  return _elementType;
}

auto TensorType::shape() const -> const std::vector<int64_t> & {
  return _shape;
}

ListType::ListType(const Type *elementType)
    : Type(TypeKind::List), _elementType(elementType) {}

auto ListType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::List;
}

auto ListType::elementType() const -> const Type * {
  return _elementType;
}

PtrType::PtrType(const Type *pointeeType)
    : Type(TypeKind::Ptr), _pointeeType(pointeeType) {}

auto PtrType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Ptr;
}

auto PtrType::pointeeType() const -> const Type * {
  return _pointeeType;
}

namespace {

auto formatBuiltinType(const BuiltinType& type) -> std::string {
  return std::string(type.name());
}

auto formatTensorType(const TensorType& type) -> std::string {
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

auto formatListType(const ListType& type) -> std::string {
  return "List<" + formatType(type.elementType()) + ">";
}

auto formatPtrType(const PtrType& type) -> std::string {
  return "Ptr<" + formatType(type.pointeeType()) + ">";
}

auto alignTo(uint64_t offset, uint64_t alignment) -> uint64_t {
  auto remainder = offset % alignment;
  if (remainder == 0)
    return offset;
  return offset + alignment - remainder;
}

auto sizeOfBuiltinType(const BuiltinType& type) -> std::optional<uint64_t> {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::Bool:
  case BuiltinTypeKind::UInt8:
    return 1;
  case BuiltinTypeKind::UInt64:
    return 8;
  case BuiltinTypeKind::Float32:
    return 4;
  case BuiltinTypeKind::Unit:
    return std::nullopt;
  }
  return std::nullopt;
}

auto alignOfBuiltinType(const BuiltinType& type) -> std::optional<uint64_t> {
  switch (type.builtinKind()) {
  case BuiltinTypeKind::Bool:
  case BuiltinTypeKind::UInt8:
    return 1;
  case BuiltinTypeKind::UInt64:
    return 8;
  case BuiltinTypeKind::Float32:
    return 4;
  case BuiltinTypeKind::Unit:
    return std::nullopt;
  }
  return std::nullopt;
}

auto sizeOfStructType(const StructType& type) -> std::optional<uint64_t> {
  uint64_t offset = 0;
  uint64_t maxAlignment = 1;
  for (const auto& field : type.fields()) {
    auto fieldSize = sizeOfType(field.type());
    auto fieldAlignment = alignOfType(field.type());
    if (!fieldSize || !fieldAlignment)
      return std::nullopt;

    offset = alignTo(offset, *fieldAlignment);
    offset += *fieldSize;
    maxAlignment = std::max(maxAlignment, *fieldAlignment);
  }
  return alignTo(offset, maxAlignment);
}

auto alignOfStructType(const StructType& type) -> std::optional<uint64_t> {
  uint64_t maxAlignment = 1;
  for (const auto& field : type.fields()) {
    auto fieldAlignment = alignOfType(field.type());
    if (!fieldAlignment)
      return std::nullopt;
    maxAlignment = std::max(maxAlignment, *fieldAlignment);
  }
  return maxAlignment;
}

auto sameBuiltinType(const BuiltinType& lhs, const BuiltinType& rhs)
    -> bool {
  return lhs.builtinKind() == rhs.builtinKind();
}

auto sameTensorShape(const std::vector<int64_t> &lhsShape,
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

auto sameTensorType(const TensorType& lhs, const TensorType& rhs) -> bool {
  return sameType(lhs.elementType(), rhs.elementType()) &&
         sameTensorShape(lhs.shape(), rhs.shape());
}

auto sameListType(const ListType& lhs, const ListType& rhs) -> bool {
  return sameType(lhs.elementType(), rhs.elementType());
}

auto samePtrType(const PtrType& lhs, const PtrType& rhs) -> bool {
  return sameType(lhs.pointeeType(), rhs.pointeeType());
}

auto structTypeStorage()
    -> std::vector<std::unique_ptr<StructType>> & {
  static auto &types = *new std::vector<std::unique_ptr<StructType>>();
  return types;
}

auto tensorTypeStorage()
    -> std::vector<std::unique_ptr<TensorType>> & {
  static auto &types = *new std::vector<std::unique_ptr<TensorType>>();
  return types;
}

auto listTypeStorage()
    -> std::vector<std::unique_ptr<ListType>> & {
  static auto &types = *new std::vector<std::unique_ptr<ListType>>();
  return types;
}

auto ptrTypeStorage()
    -> std::vector<std::unique_ptr<PtrType>> & {
  static auto &types = *new std::vector<std::unique_ptr<PtrType>>();
  return types;
}

auto findTensorType(const Type *elementType, const std::vector<int64_t> &shape)
    -> const TensorType * {
  for (const auto &type : tensorTypeStorage())
    if (type->elementType() == elementType && type->shape() == shape)
      return type.get();
  return nullptr;
}

auto findListType(const Type *elementType) -> const ListType * {
  for (const auto &type : listTypeStorage())
    if (type->elementType() == elementType)
      return type.get();
  return nullptr;
}

auto findPtrType(const Type *pointeeType) -> const PtrType * {
  for (const auto &type : ptrTypeStorage())
    if (type->pointeeType() == pointeeType)
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

auto getUInt8Type() -> const BuiltinType * {
  static const BuiltinType type{BuiltinTypeKind::UInt8};
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
  case TypeKind::Tensor:
    return sameTensorType(*llvm::cast<TensorType>(lhs),
                          *llvm::cast<TensorType>(rhs));
  case TypeKind::List:
    return sameListType(*llvm::cast<ListType>(lhs),
                        *llvm::cast<ListType>(rhs));
  case TypeKind::Ptr:
    return samePtrType(*llvm::cast<PtrType>(lhs),
                       *llvm::cast<PtrType>(rhs));
  }

  return false;
}

auto getBuiltinType(const Type *type) -> const BuiltinType * {
  return llvm::dyn_cast_if_present<BuiltinType>(type);
}

auto getTensorType(const Type *type) -> const TensorType * {
  return llvm::dyn_cast_if_present<TensorType>(type);
}

auto getStructType(const Type *type) -> const StructType * {
  return llvm::dyn_cast_if_present<StructType>(type);
}

auto getListType(const Type *type) -> const ListType * {
  return llvm::dyn_cast_if_present<ListType>(type);
}

auto getPtrType(const Type *type) -> const PtrType * {
  return llvm::dyn_cast_if_present<PtrType>(type);
}

auto isBuiltinType(const Type *type, BuiltinTypeKind kind) -> bool {
  auto *builtinType = getBuiltinType(type);
  return builtinType && builtinType->builtinKind() == kind;
}

auto isUnitType(const Type *type) -> bool {
  return isBuiltinType(type, BuiltinTypeKind::Unit);
}

auto isUInt8Type(const Type *type) -> bool {
  return isBuiltinType(type, BuiltinTypeKind::UInt8);
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

auto isFileType(const Type *type) -> bool {
  auto *structType = getStructType(type);
  if (!structType || structType->name() != "std.io.File")
    return false;

  auto &fields = structType->fields();
  if (fields.size() != 1 || fields[0].name() != "handle")
    return false;

  auto *handleType = getPtrType(fields[0].type());
  return handleType && isUInt8Type(handleType->pointeeType());
}

auto isNumericType(const Type *type) -> bool {
  return isUInt8Type(type) || isUInt64Type(type) || isFloat32Type(type);
}

auto isEquatableType(const Type *type) -> bool {
  return isNumericType(type) || isBoolType(type);
}

auto isTensorType(const Type *type) -> bool {
  return getTensorType(type) != nullptr;
}

auto isListType(const Type *type) -> bool {
  return getListType(type) != nullptr;
}

auto isPtrType(const Type *type) -> bool {
  return getPtrType(type) != nullptr;
}

auto hasUnitType(const Type *type) -> bool {
  if (isUnitType(type))
    return true;

  auto *tensorType = getTensorType(type);
  if (tensorType)
    return hasUnitType(tensorType->elementType());

  auto *listType = getListType(type);
  if (listType)
    return hasUnitType(listType->elementType());

  auto *ptrType = getPtrType(type);
  if (ptrType)
    return hasUnitType(ptrType->pointeeType());

  return false;
}

auto hasUnitElementType(const Type *type) -> bool {
  auto *tensorType = getTensorType(type);
  if (tensorType)
    return hasUnitType(tensorType->elementType());

  auto *listType = getListType(type);
  if (listType)
    return hasUnitType(listType->elementType());

  auto *ptrType = getPtrType(type);
  if (ptrType)
    return hasUnitType(ptrType->pointeeType());

  return false;
}

auto formatType(const Type *type) -> std::string {
  if (!type)
    return "NULL";

  switch (type->kind()) {
  case TypeKind::Builtin:
    return formatBuiltinType(*llvm::cast<BuiltinType>(type));
  case TypeKind::Struct:
    return std::string(llvm::cast<StructType>(type)->name());
  case TypeKind::Tensor:
    return formatTensorType(*llvm::cast<TensorType>(type));
  case TypeKind::List:
    return formatListType(*llvm::cast<ListType>(type));
  case TypeKind::Ptr:
    return formatPtrType(*llvm::cast<PtrType>(type));
  }

  return "";
}

auto sizeOfType(const Type *type) -> std::optional<uint64_t> {
  if (auto *builtinType = getBuiltinType(type))
    return sizeOfBuiltinType(*builtinType);
  if (auto *structType = getStructType(type))
    return sizeOfStructType(*structType);
  return std::nullopt;
}

auto alignOfType(const Type *type) -> std::optional<uint64_t> {
  if (auto *builtinType = getBuiltinType(type))
    return alignOfBuiltinType(*builtinType);
  if (auto *structType = getStructType(type))
    return alignOfStructType(*structType);
  return std::nullopt;
}

auto TypeContext::getBuiltinType(BuiltinTypeKind kind) const
    -> const BuiltinType * {
  switch (kind) {
  case BuiltinTypeKind::Unit:
    return getUnitType();
  case BuiltinTypeKind::Bool:
    return getBoolType();
  case BuiltinTypeKind::UInt8:
    return getUInt8Type();
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

auto TypeContext::createStructType(std::string_view name,
                                   std::vector<StructField> fields,
                                   ComptimeAliasOrigin origin) const
    -> const StructType * {
  auto &structTypes = structTypeStorage();
  structTypes.push_back(std::make_unique<StructType>(
      name, std::move(fields), std::move(origin)));
  return structTypes.back().get();
}

auto TypeContext::createTensorType(const Type *elementType,
                                   std::vector<int64_t> shape) const
    -> const TensorType * {
  if (auto *type = findTensorType(elementType, shape))
    return type;

  auto &tensorTypes = tensorTypeStorage();
  tensorTypes.push_back(
      std::make_unique<TensorType>(elementType, std::move(shape)));
  return tensorTypes.back().get();
}

auto TypeContext::createListType(const Type *elementType) const
    -> const ListType * {
  if (auto *type = findListType(elementType))
    return type;

  auto &listTypes = listTypeStorage();
  listTypes.push_back(std::make_unique<ListType>(elementType));
  return listTypes.back().get();
}

auto TypeContext::createPtrType(const Type *pointeeType) const
    -> const PtrType * {
  if (auto *type = findPtrType(pointeeType))
    return type;

  auto &ptrTypes = ptrTypeStorage();
  ptrTypes.push_back(std::make_unique<PtrType>(pointeeType));
  return ptrTypes.back().get();
}

} // namespace cherry
