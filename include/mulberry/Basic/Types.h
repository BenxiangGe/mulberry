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

class ComptimeTypeValue {
public:
  enum class Kind {
    Type,
    UInt64,
  };

  explicit ComptimeTypeValue(const Type *type);
  explicit ComptimeTypeValue(uint64_t uint64Value);

  auto kind() const -> Kind;
  auto type() const -> const Type *;
  auto uint64Value() const -> uint64_t;

private:
  Kind _kind;
  const Type *_type = nullptr;
  uint64_t _uint64Value = 0;
};

class ComptimeAliasOrigin {
public:
  ComptimeAliasOrigin(std::string_view aliasName,
                      std::vector<ComptimeTypeValue> arguments);

  auto aliasName() const -> std::string_view;
  auto arguments() const -> const std::vector<ComptimeTypeValue> &;

private:
  std::string _aliasName;
  std::vector<ComptimeTypeValue> _arguments;
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

  auto createPtrType(const Type *pointeeType) const -> const PtrType *;
};

} // namespace mulberry

#endif // MULBERRY_TYPES_H
