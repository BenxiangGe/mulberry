//===--- StructLiteralExpr.h - Cherry Struct Literal Expression ASTs -*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_STRUCT_LITERAL_EXPR_H
#define CHERRY_STRUCT_LITERAL_EXPR_H

#include "cherry/AST/Expr.h"
#include <string>
#include <string_view>

namespace cherry {

class StructType;

class StructLiteralExpr final : public Expr {
public:
  explicit StructLiteralExpr(llvm::SMLoc location, std::string_view name,
                             VectorUniquePtr<Expr> expressions)
      : Expr{Expr_StructLiteral, location}, _name(name),
        _expressions(std::move(expressions)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_StructLiteral;
  }

  auto name() const -> std::string_view { return _name; }

  auto setStructType(const StructType *type) -> void { _structType = type; }

  auto structType() const -> const StructType * { return _structType; }

  auto expressions() const -> const VectorUniquePtr<Expr> & {
    return _expressions;
  }

private:
  std::string _name;
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

} // end namespace cherry

#endif // CHERRY_STRUCT_LITERAL_EXPR_H
