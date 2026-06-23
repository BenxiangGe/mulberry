//===--- Type.h - Cherry Language Type ASTs ---------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TYPE_H
#define CHERRY_TYPE_H

#include "cherry/AST/Node.h"
#include <cassert>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cherry {

class FunctionDecl;
class VariableStat;

struct ComptimeParam {
  enum class Kind {
    Type,
    UInt64,
  };

  explicit ComptimeParam(std::string_view name, Kind kind = Kind::Type)
      : name(name), kind(kind) {}

  std::string name;
  Kind kind = Kind::Type;
};

class TypeNode : public Node {
public:
  enum class Kind {
    Unit,
    Named,
    Tensor,
    List,
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

// Tensor type node. e.g. `Float32[10, 20]`
class TensorTypeNode final : public TypeNode {
public:
  TensorTypeNode(std::unique_ptr<TypeNode> elementType,
                 std::vector<int64_t> shape, llvm::SMLoc location)
      : TypeNode(location, TypeNode::Kind::Tensor),
        _elementType(std::move(elementType)), _shape(std::move(shape)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Tensor;
  }

  auto elementTypeNode() const -> const TypeNode * {
    return _elementType.get();
  }

  auto shape() const -> const std::vector<int64_t> & { return _shape; }

private:
  std::unique_ptr<TypeNode> _elementType;
  std::vector<int64_t> _shape;
};

// Generic list type node. e.g. `List<UInt64>` or `List<Tensor<Float32, 2>>`.
class ListTypeNode final : public TypeNode {
public:
  ListTypeNode(std::unique_ptr<TypeNode> elementType,
               llvm::SMLoc location)
      : TypeNode(location, TypeNode::Kind::List),
        _elementType(std::move(elementType)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::List;
  }

  auto elementTypeNode() const -> const TypeNode * {
    return _elementType.get();
  }

private:
  std::unique_ptr<TypeNode> _elementType;
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

// Generic type application node. e.g. `Vector<UInt64>` or `Matrix<Float32>`.
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

class StructTypeNode final : public TypeNode {
public:
  explicit StructTypeNode(llvm::SMLoc location,
                          VectorUniquePtr<VariableStat> fields)
      : TypeNode(location, TypeNode::Kind::Struct),
        _fields(std::move(fields)) {}

  StructTypeNode(llvm::SMLoc location, VectorUniquePtr<VariableStat> fields,
                 VectorUniquePtr<FunctionDecl> methods)
      : TypeNode(location, TypeNode::Kind::Struct),
        _fields(std::move(fields)), _methods(std::move(methods)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Struct;
  }

  auto fields() const -> const VectorUniquePtr<VariableStat> & {
    return _fields;
  }

  auto methods() const -> const VectorUniquePtr<FunctionDecl> & {
    return _methods;
  }

private:
  VectorUniquePtr<VariableStat> _fields;
  VectorUniquePtr<FunctionDecl> _methods;
};

} // end namespace cherry

#endif // CHERRY_TYPE_H
