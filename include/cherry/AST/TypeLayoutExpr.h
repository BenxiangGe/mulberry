//===--- TypeLayoutExpr.h - Cherry type layout expressions ------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TYPE_LAYOUT_EXPR_H
#define CHERRY_TYPE_LAYOUT_EXPR_H

#include "cherry/AST/Expr.h"
#include "cherry/AST/Type.h"
#include <cstdint>
#include <memory>

namespace cherry {

class TypeLayoutExpr final : public Expr {
public:
  enum class Query {
    SizeOf,
    AlignOf,
  };

  TypeLayoutExpr(llvm::SMLoc location, Query query,
                 std::unique_ptr<TypeNode> typeNode)
      : Expr{Expr_TypeLayout, location}, _query(query),
        _typeNode(std::move(typeNode)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_TypeLayout;
  }

  auto query() const -> Query { return _query; }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto queriedType() const -> const Type * { return _queriedType; }

  auto setQueriedType(const Type *type) -> void { _queriedType = type; }

  auto value() const -> uint64_t { return _value; }

  auto setValue(uint64_t value) -> void { _value = value; }

private:
  Query _query;
  std::unique_ptr<TypeNode> _typeNode;
  const Type *_queriedType = nullptr;
  uint64_t _value = 0;
};

} // end namespace cherry

#endif // CHERRY_TYPE_LAYOUT_EXPR_H
