//===--- StructLiteralExpr.h - Mulberry Struct Literal Expression ASTs -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_STRUCT_LITERAL_EXPR_H
#define MULBERRY_STRUCT_LITERAL_EXPR_H

#include "mulberry/AST/Expr.h"
#include "mulberry/AST/Type.h"
#include <memory>
#include <string>
#include <string_view>

namespace mulberry {

class StructType;

class StructLiteralExpr final : public Expr {
public:
  explicit StructLiteralExpr(llvm::SMLoc location,
                             std::unique_ptr<TypeNode> typeNode,
                             VectorUniquePtr<Expr> expressions)
      : Expr{Expr_StructLiteral, location}, _typeNode(std::move(typeNode)),
        _expressions(std::move(expressions)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_StructLiteral;
  }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto setStructType(const StructType *type) -> void { _structType = type; }

  auto structType() const -> const StructType * { return _structType; }

  auto expressions() const -> const VectorUniquePtr<Expr> & {
    return _expressions;
  }

private:
  std::unique_ptr<TypeNode> _typeNode;
  const StructType *_structType = nullptr;
  VectorUniquePtr<Expr> _expressions;

public:
  auto begin() const -> decltype(_expressions.begin()) {
    return _expressions.begin();
  }
  auto end() const -> decltype(_expressions.end()) {
    return _expressions.end();
  }
};

} // end namespace mulberry

#endif // MULBERRY_STRUCT_LITERAL_EXPR_H
