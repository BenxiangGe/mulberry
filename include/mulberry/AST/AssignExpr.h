//===--- AssignExpr.h - Mulberry Language Assignment Expression ASTs -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_ASSIGN_EXPR_H
#define MULBERRY_ASSIGN_EXPR_H

#include "mulberry/AST/Expr.h"
#include <memory>
#include <utility>

namespace mulberry {

class AssignExpr final : public Expr {
public:
  explicit AssignExpr(llvm::SMLoc location, std::unique_ptr<Expr> lhs,
                      std::unique_ptr<Expr> rhs)
      : Expr{Expr_Assign, location}, _lhs{std::move(lhs)},
        _rhs{std::move(rhs)} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Assign;
  }

  auto lhs() const -> const std::unique_ptr<Expr> & { return _lhs; }

  auto rhs() const -> const std::unique_ptr<Expr> & { return _rhs; }

private:
  std::unique_ptr<Expr> _lhs;
  std::unique_ptr<Expr> _rhs;
};

} // end namespace mulberry

#endif // MULBERRY_ASSIGN_EXPR_H
