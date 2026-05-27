#ifndef CHERRY_TYPES_H
#define CHERRY_TYPES_H

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace cherry {

enum class TypeKind {
  Builtin,
  Struct,
  List,
  Tensor,
};

enum class BuiltinTypeKind {
  Unit,
  Bool,
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

  static auto classof(const Type *type) -> bool;

  auto name() const -> std::string_view;

  auto fields() const -> const std::vector<StructField> &;

  auto field(std::string_view fieldName) const -> const StructField *;

private:
  std::string _name;
  std::vector<StructField> _fields;
};

class TensorType final : public Type {
public:
  TensorType(const Type *elementType, std::vector<int64_t> shape);

  static auto classof(const Type *type) -> bool;

  auto elementType() const -> const Type *;

  auto shape() const -> const std::vector<int64_t> &;

private:
  const Type *_elementType;
  std::vector<int64_t> _shape;
};

class ListType final : public Type {
public:
  explicit ListType(const Type *elementType);

  static auto classof(const Type *type) -> bool;

  auto elementType() const -> const Type *;

private:
  const Type *_elementType;
};

auto sameType(const Type *lhs, const Type *rhs) -> bool;
auto getBuiltinType(const Type *type) -> const BuiltinType *;
auto getListType(const Type *type) -> const ListType *;
auto getTensorType(const Type *type) -> const TensorType *;
auto getStructType(const Type *type) -> const StructType *;
auto isBuiltinType(const Type *type, BuiltinTypeKind kind) -> bool;
auto isUnitType(const Type *type) -> bool;
auto isUInt64Type(const Type *type) -> bool;
auto isBoolType(const Type *type) -> bool;
auto isFloat32Type(const Type *type) -> bool;
auto isNumericType(const Type *type) -> bool;
auto isEquatableType(const Type *type) -> bool;
auto isListType(const Type *type) -> bool;
auto isTensorType(const Type *type) -> bool;
auto hasUnitType(const Type *type) -> bool;
auto hasUnitElementType(const Type *type) -> bool;
// Display-only formatter for diagnostics and dumps; not for type identity.
auto formatType(const Type *type) -> std::string;

class TypeContext {
public:
  auto getBuiltinType(BuiltinTypeKind kind) const -> const BuiltinType *;

  auto createStructType(std::string_view name,
                        std::vector<StructField> fields) const
      -> const StructType *;

  auto createListType(const Type *elementType) const -> const ListType *;

  auto createTensorType(const Type *elementType,
                        std::vector<int64_t> shape) const
      -> const TensorType *;
};

} // namespace cherry

#endif // CHERRY_TYPES_H
