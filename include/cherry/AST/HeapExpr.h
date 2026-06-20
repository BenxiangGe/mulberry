//===--- HeapExpr.h - Cherry heap expression ASTs --------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_HEAP_EXPR_H
#define CHERRY_HEAP_EXPR_H

#include "cherry/AST/Expr.h"
#include "cherry/AST/Type.h"
#include <memory>

namespace cherry {

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

class DerefExpr final : public Expr {
public:
  DerefExpr(llvm::SMLoc location, std::unique_ptr<Expr> pointer)
      : Expr{Expr_Deref, location}, _pointer(std::move(pointer)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Deref;
  }

  auto pointer() const -> const std::unique_ptr<Expr> & { return _pointer; }

  auto isLvalue() const -> bool override { return true; }

private:
  std::unique_ptr<Expr> _pointer;
};

} // end namespace cherry

#endif // CHERRY_HEAP_EXPR_H
