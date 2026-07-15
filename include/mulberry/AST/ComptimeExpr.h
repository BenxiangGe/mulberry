//===--- ComptimeExpr.h - Mulberry comptime expressions ----------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_COMPTIME_EXPR_H
#define MULBERRY_COMPTIME_EXPR_H

#include "mulberry/AST/Expr.h"
#include "mulberry/AST/Type.h"
#include <memory>

namespace mulberry {

// typeInfo() accepts a source type directly. Sema consumes this node before
// MLIRGen because its result only exists at compile time.
class TypeInfoExpr final : public Expr {
public:
  TypeInfoExpr(llvm::SMLoc location, std::unique_ptr<TypeNode> typeNode)
      : Expr{Expr_TypeInfo, location}, _typeNode(std::move(typeNode)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_TypeInfo;
  }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

private:
  std::unique_ptr<TypeNode> _typeNode;
};

} // end namespace mulberry

#endif // MULBERRY_COMPTIME_EXPR_H
