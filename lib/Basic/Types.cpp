#include "mulberry/Basic/Types.h"
#include "llvm/Support/Casting.h"

#include <algorithm>
#include <cassert>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

namespace mulberry {

Type::Type(TypeKind kind) : _kind(kind) {}

auto Type::kind() const -> TypeKind { return _kind; }

ComptimeValue::ComptimeValue(const Type *type)
    : _kind(Kind::Type), _type(type) {}

ComptimeValue::ComptimeValue(bool boolValue)
    : _kind(Kind::Bool), _boolValue(boolValue) {}

ComptimeValue::ComptimeValue(uint64_t uint64Value)
    : _kind(Kind::UInt64), _uint64Value(uint64Value) {}

ComptimeValue::ComptimeValue(std::string_view stringValue)
    : _kind(Kind::String), _stringValue(stringValue) {}

auto ComptimeValue::kind() const -> Kind {
  return _kind;
}

auto ComptimeValue::type() const -> const Type * {
  return _type;
}

auto ComptimeValue::boolValue() const -> bool {
  return _boolValue;
}

auto ComptimeValue::uint64Value() const -> uint64_t {
  return _uint64Value;
}

auto ComptimeValue::stringValue() const -> std::string_view {
  return _stringValue;
}

ComptimeAliasOrigin::ComptimeAliasOrigin(
    std::string_view aliasName, std::vector<ComptimeValue> arguments)
    : _aliasName(aliasName), _arguments(std::move(arguments)) {}

auto ComptimeAliasOrigin::aliasName() const -> std::string_view {
  return _aliasName;
}

auto ComptimeAliasOrigin::arguments() const
    -> const std::vector<ComptimeValue> & {
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

DataConstructor::DataConstructor(std::string_view name,
                                 std::vector<const Type *> payloadTypes)
    : _name(name), _payloadTypes(std::move(payloadTypes)) {}

auto DataConstructor::name() const -> std::string_view {
  return _name;
}

auto DataConstructor::payloadTypes() const
    -> const std::vector<const Type *> & {
  return _payloadTypes;
}

DataType::DataType(std::string_view name, std::string_view declarationName,
                   std::vector<ComptimeValue> arguments)
    : Type(TypeKind::Data), _name(name),
      _declarationName(declarationName), _arguments(std::move(arguments)) {}

auto DataType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Data;
}

auto DataType::name() const -> std::string_view {
  return _name;
}

auto DataType::declarationName() const -> std::string_view {
  return _declarationName;
}

auto DataType::arguments() const -> const std::vector<ComptimeValue> & {
  return _arguments;
}

auto DataType::constructors() const
    -> const std::vector<DataConstructor> & {
  return _constructors;
}

auto DataType::setConstructors(std::vector<DataConstructor> constructors)
    -> void {
  assert(!_isComplete);
  _constructors = std::move(constructors);
  _isComplete = true;
}

ArrayType::ArrayType(const Type *elementType, uint64_t size)
    : Type(TypeKind::Array), _elementType(elementType), _size(size) {}

auto ArrayType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Array;
}

auto ArrayType::elementType() const -> const Type * {
  return _elementType;
}

auto ArrayType::size() const -> uint64_t {
  return _size;
}

FunctionType::FunctionType(
    std::vector<const Type *> parameterTypes,
    std::vector<bool> parameterCanMutateObject, const Type *returnType)
    : Type(TypeKind::Function), _parameterTypes(std::move(parameterTypes)),
      _parameterCanMutateObject(std::move(parameterCanMutateObject)),
      _returnType(returnType) {
  assert(_parameterTypes.size() == _parameterCanMutateObject.size());
}

auto FunctionType::classof(const Type *type) -> bool {
  return type && type->kind() == TypeKind::Function;
}

auto FunctionType::parameterTypes() const
    -> const std::vector<const Type *> & {
  return _parameterTypes;
}

auto FunctionType::parameterCanMutateObject() const
    -> const std::vector<bool> & {
  return _parameterCanMutateObject;
}

auto FunctionType::returnType() const -> const Type * {
  return _returnType;
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

auto formatArrayType(const ArrayType& type) -> std::string {
  return "Array<" + formatType(type.elementType()) + ", " +
         std::to_string(type.size()) + ">";
}

auto formatPtrType(const PtrType& type) -> std::string {
  return "Ptr<" + formatType(type.pointeeType()) + ">";
}

auto formatFunctionType(const FunctionType& type) -> std::string {
  std::string result = "fn(";
  std::string separator;
  for (size_t i = 0; i < type.parameterTypes().size(); ++i) {
    result += separator;
    if (type.parameterCanMutateObject()[i])
      result += "mut ";
    result += formatType(type.parameterTypes()[i]);
    separator = ", ";
  }
  result += "): ";
  result += isUnitType(type.returnType()) ? "()" : formatType(type.returnType());
  return result;
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

auto sameArrayType(const ArrayType& lhs, const ArrayType& rhs) -> bool {
  return sameType(lhs.elementType(), rhs.elementType()) &&
         lhs.size() == rhs.size();
}

auto samePtrType(const PtrType& lhs, const PtrType& rhs) -> bool {
  return sameType(lhs.pointeeType(), rhs.pointeeType());
}

auto sameFunctionType(const FunctionType& lhs, const FunctionType& rhs)
    -> bool {
  if (lhs.parameterTypes().size() != rhs.parameterTypes().size() ||
      lhs.parameterCanMutateObject() != rhs.parameterCanMutateObject() ||
      !sameType(lhs.returnType(), rhs.returnType()))
    return false;

  for (size_t i = 0; i < lhs.parameterTypes().size(); ++i)
    if (!sameType(lhs.parameterTypes()[i], rhs.parameterTypes()[i]))
      return false;
  return true;
}

auto structTypeStorage()
    -> std::vector<std::unique_ptr<StructType>> & {
  static auto &types = *new std::vector<std::unique_ptr<StructType>>();
  return types;
}

auto arrayTypeStorage()
    -> std::vector<std::unique_ptr<ArrayType>> & {
  static auto &types = *new std::vector<std::unique_ptr<ArrayType>>();
  return types;
}

auto ptrTypeStorage()
    -> std::vector<std::unique_ptr<PtrType>> & {
  static auto &types = *new std::vector<std::unique_ptr<PtrType>>();
  return types;
}

auto functionTypeStorage()
    -> std::vector<std::unique_ptr<FunctionType>> & {
  static auto &types = *new std::vector<std::unique_ptr<FunctionType>>();
  return types;
}

auto dataTypeStorage()
    -> std::vector<std::unique_ptr<DataType>> & {
  static auto &types = *new std::vector<std::unique_ptr<DataType>>();
  return types;
}

auto findArrayType(const Type *elementType, uint64_t size)
    -> const ArrayType * {
  for (const auto &type : arrayTypeStorage())
    if (type->elementType() == elementType && type->size() == size)
      return type.get();
  return nullptr;
}

auto findPtrType(const Type *pointeeType) -> const PtrType * {
  for (const auto &type : ptrTypeStorage())
    if (type->pointeeType() == pointeeType)
      return type.get();
  return nullptr;
}

auto findFunctionType(const std::vector<const Type *> &parameterTypes,
                      const std::vector<bool> &parameterCanMutateObject,
                      const Type *returnType) -> const FunctionType * {
  for (const auto &type : functionTypeStorage()) {
    if (type->parameterTypes().size() != parameterTypes.size() ||
        type->parameterCanMutateObject() != parameterCanMutateObject ||
        !sameType(type->returnType(), returnType))
      continue;

    bool matches = true;
    for (size_t i = 0; i < parameterTypes.size(); ++i)
      matches = matches && sameType(type->parameterTypes()[i], parameterTypes[i]);
    if (matches)
      return type.get();
  }
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
  case TypeKind::Data:
    return lhs == rhs;
  case TypeKind::Array:
    return sameArrayType(*llvm::cast<ArrayType>(lhs),
                         *llvm::cast<ArrayType>(rhs));
  case TypeKind::Function:
    return sameFunctionType(*llvm::cast<FunctionType>(lhs),
                            *llvm::cast<FunctionType>(rhs));
  case TypeKind::Ptr:
    return samePtrType(*llvm::cast<PtrType>(lhs),
                       *llvm::cast<PtrType>(rhs));
  }

  return false;
}

auto getBuiltinType(const Type *type) -> const BuiltinType * {
  return llvm::dyn_cast_if_present<BuiltinType>(type);
}

auto getArrayType(const Type *type) -> const ArrayType * {
  return llvm::dyn_cast_if_present<ArrayType>(type);
}

auto getFunctionType(const Type *type) -> const FunctionType * {
  return llvm::dyn_cast_if_present<FunctionType>(type);
}

auto getStructType(const Type *type) -> const StructType * {
  return llvm::dyn_cast_if_present<StructType>(type);
}

auto getDataType(const Type *type) -> const DataType * {
  return llvm::dyn_cast_if_present<DataType>(type);
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

auto isArrayElementType(const Type *type) -> bool {
  return type && !hasUnitType(type);
}

auto isArrayType(const Type *type) -> bool {
  return getArrayType(type) != nullptr;
}

auto isPtrType(const Type *type) -> bool {
  return getPtrType(type) != nullptr;
}

auto hasUnitType(const Type *type) -> bool {
  if (isUnitType(type))
    return true;

  auto *arrayType = getArrayType(type);
  if (arrayType)
    return hasUnitType(arrayType->elementType());

  auto *ptrType = getPtrType(type);
  if (ptrType)
    return hasUnitType(ptrType->pointeeType());

  return false;
}

auto hasUnitElementType(const Type *type) -> bool {
  auto *arrayType = getArrayType(type);
  if (arrayType)
    return hasUnitType(arrayType->elementType());

  auto *ptrType = getPtrType(type);
  if (ptrType)
    return hasUnitType(ptrType->pointeeType());

  return false;
}

auto getArrayShape(const Type *type) -> std::vector<int64_t> {
  std::vector<int64_t> shape;
  while (auto *arrayType = getArrayType(type)) {
    shape.push_back(static_cast<int64_t>(arrayType->size()));
    type = arrayType->elementType();
  }
  return shape;
}

auto getArrayLeafElementType(const Type *type) -> const Type * {
  auto *arrayType = getArrayType(type);
  if (!arrayType)
    return type;

  return getArrayLeafElementType(arrayType->elementType());
}

auto formatType(const Type *type) -> std::string {
  if (!type)
    return "NULL";

  switch (type->kind()) {
  case TypeKind::Builtin:
    return formatBuiltinType(*llvm::cast<BuiltinType>(type));
  case TypeKind::Struct:
    return std::string(llvm::cast<StructType>(type)->name());
  case TypeKind::Data: {
    auto &dataType = *llvm::cast<DataType>(type);
    std::string result(dataType.declarationName());
    if (dataType.arguments().empty())
      return result;
    result += "<";
    std::string separator;
    for (auto &argument : dataType.arguments()) {
      result += separator;
      if (argument.kind() == ComptimeValue::Kind::Type)
        result += formatType(argument.type());
      else if (argument.kind() == ComptimeValue::Kind::UInt64)
        result += std::to_string(argument.uint64Value());
      separator = ", ";
    }
    result += ">";
    return result;
  }
  case TypeKind::Array:
    return formatArrayType(*llvm::cast<ArrayType>(type));
  case TypeKind::Function:
    return formatFunctionType(*llvm::cast<FunctionType>(type));
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
  if (auto *arrayType = getArrayType(type)) {
    auto elementSize = sizeOfType(arrayType->elementType());
    if (!elementSize)
      return std::nullopt;
    return *elementSize * arrayType->size();
  }
  return std::nullopt;
}

auto alignOfType(const Type *type) -> std::optional<uint64_t> {
  if (auto *builtinType = getBuiltinType(type))
    return alignOfBuiltinType(*builtinType);
  if (auto *structType = getStructType(type))
    return alignOfStructType(*structType);
  if (auto *arrayType = getArrayType(type))
    return alignOfType(arrayType->elementType());
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

auto TypeContext::createDataType(
    std::string_view name, std::string_view declarationName,
    std::vector<ComptimeValue> arguments) const -> DataType * {
  auto &dataTypes = dataTypeStorage();
  dataTypes.push_back(std::make_unique<DataType>(
      name, declarationName, std::move(arguments)));
  return dataTypes.back().get();
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

auto TypeContext::createArrayType(const Type *elementType,
                                  uint64_t size) const -> const ArrayType * {
  if (auto *type = findArrayType(elementType, size))
    return type;

  auto &arrayTypes = arrayTypeStorage();
  arrayTypes.push_back(std::make_unique<ArrayType>(elementType, size));
  return arrayTypes.back().get();
}

auto TypeContext::createFunctionType(
    std::vector<const Type *> parameterTypes,
    std::vector<bool> parameterCanMutateObject,
    const Type *returnType) const -> const FunctionType * {
  if (auto *type = findFunctionType(parameterTypes, parameterCanMutateObject,
                                    returnType))
    return type;

  auto &functionTypes = functionTypeStorage();
  functionTypes.push_back(std::make_unique<FunctionType>(
      std::move(parameterTypes), std::move(parameterCanMutateObject),
      returnType));
  return functionTypes.back().get();
}

auto TypeContext::createPtrType(const Type *pointeeType) const
    -> const PtrType * {
  if (auto *type = findPtrType(pointeeType))
    return type;

  auto &ptrTypes = ptrTypeStorage();
  ptrTypes.push_back(std::make_unique<PtrType>(pointeeType));
  return ptrTypes.back().get();
}

} // namespace mulberry
