//===--- HeapExpr.h - Mulberry heap expression ASTs --------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_HEAP_EXPR_H
#define MULBERRY_HEAP_EXPR_H

#include "mulberry/AST/Expr.h"
#include "mulberry/AST/Type.h"
#include <memory>

namespace mulberry {

class HeapAllocExpr final : public Expr {
public:
  HeapAllocExpr(llvm::SMLoc location, std::unique_ptr<TypeNode> typeNode,
                std::unique_ptr<Expr> count)
      : Expr{Expr_HeapAlloc, location}, _typeNode(std::move(typeNode)),
        _count(std::move(count)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_HeapAlloc;
  }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto count() const -> const std::unique_ptr<Expr> & { return _count; }

  auto allocatedType() const -> const Type * { return _allocatedType; }

  auto setAllocatedType(const Type *type) -> void { _allocatedType = type; }

private:
  std::unique_ptr<TypeNode> _typeNode;
  std::unique_ptr<Expr> _count;
  const Type *_allocatedType = nullptr;
};

} // end namespace mulberry

#endif // MULBERRY_HEAP_EXPR_H
