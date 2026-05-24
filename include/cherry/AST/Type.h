//===--- Type.h - Cherry Language Type ASTs ---------------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TYPE_H
#define CHERRY_TYPE_H

#include "cherry/AST/Node.h"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cherry {

class TypeNode : public Node {
public:
  enum class Kind {
    Unit,
    Named,
    List,
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
      : TypeNode(location, TypeNode::Kind::Named), _name(name) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::Named;
  }

  auto name() const -> std::string_view { return _name; }

private:
  std::string _name;
};

class ListTypeNode final : public TypeNode {
public:
  ListTypeNode(std::unique_ptr<TypeNode> elementType,
               std::vector<int64_t> shape, llvm::SMLoc location)
      : TypeNode(location, TypeNode::Kind::List),
        _elementType(std::move(elementType)), _shape(std::move(shape)) {}

  static auto classof(const TypeNode *node) -> bool {
    return node->kind() == TypeNode::Kind::List;
  }

  auto elementTypeNode() const -> const TypeNode * {
    return _elementType.get();
  }

  auto shape() const -> const std::vector<int64_t> & { return _shape; }

private:
  std::unique_ptr<TypeNode> _elementType;
  std::vector<int64_t> _shape;
};

} // end namespace cherry

#endif // CHERRY_TYPE_H
