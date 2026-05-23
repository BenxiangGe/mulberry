//===--- BinaryExpr.h - Cherry Language Binary Expression ASTs -*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_BINARY_EXPR_H
#define CHERRY_BINARY_EXPR_H

#include "cherry/AST/Expr.h"
#include <memory>
#include <string_view>
#include <utility>

namespace cherry {

class BinaryExpr final : public Expr {
public:
  enum class Operator {
    Add,
    Mul,
    Diff,
    Div,
    Rem,
    And,
    Or,
    LT,
    LE,
    GT,
    GE,
    EQ,
    NEQ,
  };

  explicit BinaryExpr(llvm::SMLoc location, BinaryExpr::Operator op,
                      std::unique_ptr<Expr> lhs, std::unique_ptr<Expr> rhs)
      : Expr{Expr_Binary, location}, _op{op}, _lhs{std::move(lhs)},
        _rhs{std::move(rhs)} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Binary;
  }

  auto lhs() const -> const std::unique_ptr<Expr> & { return _lhs; }

  auto rhs() const -> const std::unique_ptr<Expr> & { return _rhs; }

  auto op() const -> std::string_view {
    switch (_op) {
    case Operator::Add:
      return "+";
    case Operator::Diff:
      return "-";
    case Operator::Mul:
      return "*";
    case Operator::Div:
      return "/";
    case Operator::Rem:
      return "%";
    case Operator::And:
      return "and";
    case Operator::Or:
      return "or";
    case Operator::LT:
      return "lt";
    case Operator::LE:
      return "le";
    case Operator::GT:
      return "gt";
    case Operator::GE:
      return "ge";
    case Operator::EQ:
      return "eq";
    case Operator::NEQ:
      return "neq";
    }
  }

  auto opEnum() const -> Operator { return _op; }

private:
  Operator _op;
  std::unique_ptr<Expr> _lhs;
  std::unique_ptr<Expr> _rhs;
};

} // end namespace cherry

#endif // CHERRY_BINARY_EXPR_H
