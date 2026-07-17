//===--- MatchExpr.h - Mulberry Match Expression ASTs ------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_MATCH_EXPR_H
#define MULBERRY_MATCH_EXPR_H

#include "mulberry/AST/Expr.h"
#include "mulberry/AST/Stat.h"

namespace mulberry {

class MatchExprArm final : public Node {
public:
  MatchExprArm(llvm::SMLoc location,
               std::unique_ptr<DataPattern> pattern,
               std::unique_ptr<BlockExpr> bodyBlock,
               std::unique_ptr<Expr> resultExpr)
      : Node(location), _pattern(std::move(pattern)),
        _bodyBlock(std::move(bodyBlock)), _resultExpr(std::move(resultExpr)) {}

  auto pattern() const -> const std::unique_ptr<DataPattern> & {
    return _pattern;
  }

  auto bodyBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _bodyBlock;
  }

  auto resultExpr() const -> const std::unique_ptr<Expr> & {
    return _resultExpr;
  }

private:
  std::unique_ptr<DataPattern> _pattern;
  std::unique_ptr<BlockExpr> _bodyBlock;
  std::unique_ptr<Expr> _resultExpr;
};

class MatchExpr final : public Expr {
public:
  MatchExpr(llvm::SMLoc location, std::unique_ptr<Expr> value,
            VectorUniquePtr<MatchExprArm> arms)
      : Expr(Expr_Match, location), _value(std::move(value)),
        _arms(std::move(arms)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Match;
  }

  auto value() const -> const std::unique_ptr<Expr> & { return _value; }

  auto arms() const -> const VectorUniquePtr<MatchExprArm> & { return _arms; }

  auto canMutateObject() const -> bool { return _canMutateObject; }

  auto setCanMutateObject(bool canMutateObject) -> void {
    _canMutateObject = canMutateObject;
  }

private:
  std::unique_ptr<Expr> _value;
  VectorUniquePtr<MatchExprArm> _arms;
  bool _canMutateObject = true;
};

} // end namespace mulberry

#endif // MULBERRY_MATCH_EXPR_H
