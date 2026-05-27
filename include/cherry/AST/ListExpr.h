//===--- ListExpr.h - Cherry Language List Expression ASTs ------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_LIST_EXPR_H
#define CHERRY_LIST_EXPR_H

#include "cherry/AST/Expr.h"
#include <memory>
#include <vector>

namespace cherry {

class ListLiteralExpr final : public Expr {
public:
  ListLiteralExpr(llvm::SMLoc loc, std::vector<std::unique_ptr<Expr>> elements)
      : Expr(Expr_ListLiteral, loc), _elements(std::move(elements)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_ListLiteral;
  }

  auto elements() const -> const std::vector<std::unique_ptr<Expr>> & {
    return _elements;
  }
  auto elements() -> std::vector<std::unique_ptr<Expr>> & { return _elements; }

private:
  std::vector<std::unique_ptr<Expr>> _elements;
};

} // end namespace cherry

#endif // CHERRY_LIST_EXPR_H
