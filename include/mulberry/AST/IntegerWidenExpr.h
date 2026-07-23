//===--- IntegerWidenExpr.h - Integer Widening Expression AST ---*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_INTEGER_WIDEN_EXPR_H
#define MULBERRY_INTEGER_WIDEN_EXPR_H

#include "mulberry/AST/Expr.h"
#include <memory>
#include <utility>

namespace mulberry {

// Sema inserts this only for the one implicit numeric conversion Mulberry
// permits: UInt8/UInt64 to the arbitrary-precision Integer object.
class IntegerWidenExpr final : public Expr {
public:
  IntegerWidenExpr(llvm::SMLoc location, std::unique_ptr<Expr> value)
      : Expr(Expr_IntegerWiden, location), _value(std::move(value)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_IntegerWiden;
  }

  auto value() const -> const std::unique_ptr<Expr> & { return _value; }

private:
  std::unique_ptr<Expr> _value;
};

} // namespace mulberry

#endif // MULBERRY_INTEGER_WIDEN_EXPR_H
