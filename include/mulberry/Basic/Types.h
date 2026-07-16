#ifndef MULBERRY_TYPES_H
#define MULBERRY_TYPES_H

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mulberry {

enum class TypeKind {
  Builtin,
  Struct,
  Array,
  Function,
  Ptr,
};

enum class BuiltinTypeKind {
  Unit,
  Bool,
  UInt8,
  UInt64,
  Float32,
};

class Type {
public:
  explicit Type(TypeKind kind);
  virtual ~Type() = default;

  auto kind() const -> TypeKind;

private:
  TypeKind _kind;
};

// Semantic values evaluated and consumed by Sema. Type values preserve the
// canonical Type identity; they are never represented by source-level names.
class ComptimeValue {
public:
  enum class Kind {
    Type,
    Bool,
    UInt64,
    String,
  };

  explicit ComptimeValue(const Type *type);
  explicit ComptimeValue(bool boolValue);
  explicit ComptimeValue(uint64_t uint64Value);
  explicit ComptimeValue(std::string_view stringValue);

  auto kind() const -> Kind;
  auto type() const -> const Type *;
  auto boolValue() const -> bool;
  auto uint64Value() const -> uint64_t;
  auto stringValue() const -> std::string_view;

private:
  Kind _kind;
  const Type *_type = nullptr;
  bool _boolValue = false;
  uint64_t _uint64Value = 0;
  std::string _stringValue;
};

class ComptimeAliasOrigin {
public:
  ComptimeAliasOrigin(std::string_view aliasName,
                      std::vector<ComptimeValue> arguments);

  auto aliasName() const -> std::string_view;
  auto arguments() const -> const std::vector<ComptimeValue> &;

private:
  std::string _aliasName;
  std::vector<ComptimeValue> _arguments;
};

class BuiltinType final : public Type {
public:
  explicit BuiltinType(BuiltinTypeKind kind);

  static auto classof(const Type *type) -> bool;

  auto builtinKind() const -> BuiltinTypeKind;

  auto name() const -> std::string_view;

private:
  BuiltinTypeKind _builtinKind;
};

class StructField {
public:
  StructField(std::string_view name, const Type *type, unsigned index);

  auto name() const -> std::string_view;

  auto type() const -> const Type *;

  auto index() const -> unsigned;

private:
  std::string _name;
  const Type *_type = nullptr;
  unsigned _index = 0;
};

class StructType final : public Type {
public:
  StructType(std::string_view name, std::vector<StructField> fields);
  StructType(std::string_view name, std::vector<StructField> fields,
             ComptimeAliasOrigin origin);

  static auto classof(const Type *type) -> bool;

  auto name() const -> std::string_view;

  auto fields() const -> const std::vector<StructField> &;

  auto field(std::string_view fieldName) const -> const StructField *;

  auto origin() const -> const ComptimeAliasOrigin *;

private:
  std::string _name;
  std::vector<StructField> _fields;
  std::optional<ComptimeAliasOrigin> _origin;
};

class ArrayType final : public Type {
public:
  ArrayType(const Type *elementType, uint64_t size);

  static auto classof(const Type *type) -> bool;

  auto elementType() const -> const Type *;

  auto size() const -> uint64_t;

private:
  const Type *_elementType;
  uint64_t _size = 0;
};

class FunctionType final : public Type {
public:
  FunctionType(std::vector<const Type *> parameterTypes,
               std::vector<bool> parameterCanMutateObject,
               const Type *returnType);

  static auto classof(const Type *type) -> bool;

  auto parameterTypes() const -> const std::vector<const Type *> &;

  auto parameterCanMutateObject() const -> const std::vector<bool> &;

  auto returnType() const -> const Type *;

private:
  std::vector<const Type *> _parameterTypes;
  std::vector<bool> _parameterCanMutateObject;
  const Type *_returnType = nullptr;
};

class PtrType final : public Type {
public:
  explicit PtrType(const Type *pointeeType);

  static auto classof(const Type *type) -> bool;

  auto pointeeType() const -> const Type *;

private:
  const Type *_pointeeType;
};

auto sameType(const Type *lhs, const Type *rhs) -> bool;
auto getBuiltinType(const Type *type) -> const BuiltinType *;
auto getArrayType(const Type *type) -> const ArrayType *;
auto getFunctionType(const Type *type) -> const FunctionType *;
auto getStructType(const Type *type) -> const StructType *;
auto getPtrType(const Type *type) -> const PtrType *;
auto isBuiltinType(const Type *type, BuiltinTypeKind kind) -> bool;
auto isUnitType(const Type *type) -> bool;
auto isUInt8Type(const Type *type) -> bool;
auto isUInt64Type(const Type *type) -> bool;
auto isBoolType(const Type *type) -> bool;
auto isFloat32Type(const Type *type) -> bool;
auto isFileType(const Type *type) -> bool;
auto isNumericType(const Type *type) -> bool;
auto isEquatableType(const Type *type) -> bool;
auto isArrayElementType(const Type *type) -> bool;
auto isArrayType(const Type *type) -> bool;
auto isPtrType(const Type *type) -> bool;
auto hasUnitType(const Type *type) -> bool;
auto hasUnitElementType(const Type *type) -> bool;
auto getArrayShape(const Type *type) -> std::vector<int64_t>;
auto getArrayLeafElementType(const Type *type) -> const Type *;
// Display-only formatter for diagnostics and dumps; not for type identity.
auto formatType(const Type *type) -> std::string;
auto sizeOfType(const Type *type) -> std::optional<uint64_t>;
auto alignOfType(const Type *type) -> std::optional<uint64_t>;

class TypeContext {
public:
  auto getBuiltinType(BuiltinTypeKind kind) const -> const BuiltinType *;

  auto createStructType(std::string_view name,
                        std::vector<StructField> fields) const
      -> const StructType *;

  auto createStructType(std::string_view name,
                        std::vector<StructField> fields,
                        ComptimeAliasOrigin origin) const
      -> const StructType *;

  auto createArrayType(const Type *elementType, uint64_t size) const
      -> const ArrayType *;

  auto createFunctionType(
      std::vector<const Type *> parameterTypes,
      std::vector<bool> parameterCanMutateObject,
      const Type *returnType) const -> const FunctionType *;

  auto createPtrType(const Type *pointeeType) const -> const PtrType *;
};

} // namespace mulberry

#endif // MULBERRY_TYPES_H
