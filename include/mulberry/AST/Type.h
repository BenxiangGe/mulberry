//===--- Type.h - Mulberry Language Type ASTs ---------------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_TYPE_H
#define MULBERRY_TYPE_H

#include "mulberry/AST/Expr.h"
#include "mulberry/AST/Node.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mulberry {

class FunctionDecl;
class TraitDecl;
class Type;
class VariableExpr;

struct ComptimeParam {
  enum class Kind {
    Type,
    UInt64,
  };

  explicit ComptimeParam(std::string_view name, Kind kind = Kind::Type,
                         std::string_view traitName = {})
      : name(name), kind(kind), traitName(traitName) {}

  auto hasTraitConstraint() const -> bool { return !traitName.empty(); }

  std::string name;
  Kind kind = Kind::Type;
  std::string traitName;
  const TraitDecl *trait = nullptr;
};

class TypeNode : public Node {
public:
  enum class Kind {
    Unit,
    Named,
    Computed,
    Array,
    Function,
    Ptr,
    Generic,
    Struct,
  };

  auto kind() const -> Kind { return _kind; }

protected:
  explicit TypeNode(llvm::SMLoc location, Kind kind)
      : Node(location), _kind(kind) {}

private:
  Kind _kind;
};

class UnitTypeNode final : public TypeNode {
public:
  explicit UnitTypeNode(llvm::SMLoc location)
      : TypeNode(location, TypeNode::Kind::Unit) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Unit;
  }
};

class NamedTypeNode final : public TypeNode {
public:
  explicit NamedTypeNode(llvm::SMLoc location, std::string_view name)
      : TypeNode(location, TypeNode::Kind::Named),
        _name(name) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Named;
  }

  auto name() const -> std::string_view { return _name; }

private:
  std::string _name;
};

// A comptime expression whose result is used as a source type.
class ComputedTypeNode final : public TypeNode {
public:
  ComputedTypeNode(llvm::SMLoc location, std::unique_ptr<Expr> expression)
      : TypeNode(location, TypeNode::Kind::Computed),
        _expression(std::move(expression)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Computed;
  }

  auto expression() const -> const std::unique_ptr<Expr> & {
    return _expression;
  }

private:
  std::unique_ptr<Expr> _expression;
};

// Bracket-shaped Array type node, e.g. `Float32[10]`.
// Multi-dimensional or dynamic `T[...]` source spelling has been removed; use
// Tensor<T> plus explicit tensor construction for ndarray-style values.
class ArrayTypeNode final : public TypeNode {
public:
  ArrayTypeNode(std::unique_ptr<TypeNode> elementType,
                std::vector<int64_t> shape, llvm::SMLoc location)
      : TypeNode(location, TypeNode::Kind::Array),
        _elementType(std::move(elementType)), _shape(std::move(shape)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Array;
  }

  auto elementTypeNode() const -> const TypeNode * {
    return _elementType.get();
  }

  auto shape() const -> const std::vector<int64_t> & { return _shape; }

private:
  std::unique_ptr<TypeNode> _elementType;
  std::vector<int64_t> _shape;
};

// Typed pointer node. e.g. `Ptr<UInt64>`
class PtrTypeNode final : public TypeNode {
public:
  PtrTypeNode(std::unique_ptr<TypeNode> pointeeType,
              llvm::SMLoc location)
      : TypeNode(location, TypeNode::Kind::Ptr),
        _pointeeType(std::move(pointeeType)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Ptr;
  }

  auto pointeeTypeNode() const -> const TypeNode * {
    return _pointeeType.get();
  }

private:
  std::unique_ptr<TypeNode> _pointeeType;
};

class FunctionTypeNode final : public TypeNode {
public:
  FunctionTypeNode(llvm::SMLoc location,
                   VectorUniquePtr<TypeNode> parameterTypes,
                   std::vector<bool> parameterCanMutateObject,
                   std::unique_ptr<TypeNode> returnType)
      : TypeNode(location, TypeNode::Kind::Function),
        _parameterTypes(std::move(parameterTypes)),
        _parameterCanMutateObject(std::move(parameterCanMutateObject)),
        _returnType(std::move(returnType)) {
    assert(_parameterTypes.size() == _parameterCanMutateObject.size());
  }

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Function;
  }

  auto parameterTypes() const -> const VectorUniquePtr<TypeNode> & {
    return _parameterTypes;
  }

  auto parameterCanMutateObject() const -> const std::vector<bool> & {
    return _parameterCanMutateObject;
  }

  auto returnTypeNode() const -> const TypeNode * { return _returnType.get(); }

private:
  VectorUniquePtr<TypeNode> _parameterTypes;
  std::vector<bool> _parameterCanMutateObject;
  std::unique_ptr<TypeNode> _returnType;
};

class ComptimeArg final {
public:
  enum class Kind {
    Type,
    UInt64,
  };

  explicit ComptimeArg(std::unique_ptr<TypeNode> typeNode)
      : _kind(Kind::Type), _typeNode(std::move(typeNode)) {}

  ComptimeArg(llvm::SMLoc location, uint64_t uint64Value)
      : _kind(Kind::UInt64), _location(location), _uint64Value(uint64Value) {}

  auto kind() const -> Kind { return _kind; }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto uint64Value() const -> uint64_t { return _uint64Value; }

  auto location() const -> llvm::SMLoc {
    return _typeNode ? _typeNode->location() : _location;
  }

private:
  Kind _kind;
  llvm::SMLoc _location;
  std::unique_ptr<TypeNode> _typeNode;
  uint64_t _uint64Value = 0;
};

// Generic type application node. e.g. `List<UInt64>` or `Tensor<Float32>`.
class GenericTypeNode final : public TypeNode {
public:
  GenericTypeNode(llvm::SMLoc location, std::string_view name,
                  std::unique_ptr<TypeNode> argumentType)
      : TypeNode(location, TypeNode::Kind::Generic), _name(name) {
    _arguments.push_back(ComptimeArg(std::move(argumentType)));
  }

  GenericTypeNode(llvm::SMLoc location, std::string_view name,
                  std::vector<ComptimeArg> arguments)
      : TypeNode(location, TypeNode::Kind::Generic), _name(name),
        _arguments(std::move(arguments)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Generic;
  }

  auto name() const -> std::string_view { return _name; }

  auto argumentTypeNode() const -> const TypeNode * {
    assert(_arguments.size() == 1);
    assert(_arguments.front().kind() == ComptimeArg::Kind::Type);
    return _arguments.front().typeNode();
  }

  auto arguments() const -> const std::vector<ComptimeArg> & {
    return _arguments;
  }

private:
  std::string _name;
  std::vector<ComptimeArg> _arguments;
};

class FieldDecl final : public Node {
public:
  FieldDecl(llvm::SMLoc location, std::unique_ptr<VariableExpr> variable,
            std::unique_ptr<TypeNode> typeNode)
      : Node{location}, _variable(std::move(variable)),
        _typeNode(std::move(typeNode)) {}

  auto variable() const -> const std::unique_ptr<VariableExpr> & {
    return _variable;
  }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

private:
  std::unique_ptr<VariableExpr> _variable;
  std::unique_ptr<TypeNode> _typeNode;
  const Type *_type = nullptr;
};

class ParameterDecl final : public Node {
public:
  ParameterDecl(llvm::SMLoc location, std::unique_ptr<VariableExpr> variable,
                std::unique_ptr<TypeNode> typeNode,
                bool canMutateObject = false)
      : Node{location}, _variable(std::move(variable)),
        _typeNode(std::move(typeNode)),
        _canMutateObject(canMutateObject) {}

  auto variable() const -> const std::unique_ptr<VariableExpr> & {
    return _variable;
  }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

  auto canMutateObject() const -> bool { return _canMutateObject; }

private:
  std::unique_ptr<VariableExpr> _variable;
  std::unique_ptr<TypeNode> _typeNode;
  const Type *_type = nullptr;
  bool _canMutateObject = false;
};

class StructTypeNode final : public TypeNode {
public:
  explicit StructTypeNode(llvm::SMLoc location,
                          VectorUniquePtr<FieldDecl> fields)
      : TypeNode(location, TypeNode::Kind::Struct),
        _fields(std::move(fields)) {}

  StructTypeNode(llvm::SMLoc location, VectorUniquePtr<FieldDecl> fields,
                 VectorUniquePtr<FunctionDecl> methods)
      : TypeNode(location, TypeNode::Kind::Struct),
        _fields(std::move(fields)), _methods(std::move(methods)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Struct;
  }

  auto fields() const -> const VectorUniquePtr<FieldDecl> & {
    return _fields;
  }

  auto methods() const -> const VectorUniquePtr<FunctionDecl> & {
    return _methods;
  }

private:
  VectorUniquePtr<FieldDecl> _fields;
  VectorUniquePtr<FunctionDecl> _methods;
};

} // end namespace mulberry

#endif // MULBERRY_TYPE_H
