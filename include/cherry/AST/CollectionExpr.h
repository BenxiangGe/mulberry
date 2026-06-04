//===--- CollectionExpr.h - Cherry collection expressions -------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_COLLECTION_EXPR_H
#define CHERRY_COLLECTION_EXPR_H

#include "cherry/AST/Expr.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace cherry {

// Source `[...]` is type-neutral. Sema classifies it as Tensor or List using
// expected type where available.
class ArrayLiteralExpr final : public Expr {
public:
  ArrayLiteralExpr(llvm::SMLoc loc,
                   std::vector<std::unique_ptr<Expr>> elements)
      : Expr(Expr_ArrayLiteral, loc), _elements(std::move(elements)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_ArrayLiteral;
  }

  auto getElements() const
      -> const std::vector<std::unique_ptr<Expr>> & {
    return _elements;
  }

  auto setInferredShape(std::vector<int64_t> shape) -> void {
    _inferredShape = std::move(shape);
  }

  auto getInferredShape() const -> const std::vector<int64_t> & {
    return _inferredShape;
  }

private:
  std::vector<std::unique_ptr<Expr>> _elements;
  std::vector<int64_t> _inferredShape;
};

// Source `base[...]` is type-neutral. Sema classifies it by base type.
class IndexExpr final : public Expr {
public:
  IndexExpr(llvm::SMLoc loc, std::unique_ptr<Expr> base,
            std::vector<std::unique_ptr<Expr>> indices)
      : Expr(Expr_Index, loc), _base(std::move(base)),
        _indices(std::move(indices)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Index;
  }

  auto base() const -> const std::unique_ptr<Expr> & { return _base; }

  auto indices() const -> const std::vector<std::unique_ptr<Expr>> & {
    return _indices;
  }

  auto isLvalue() const -> bool override { return _isLvalue; }

  auto setLvalue(bool isLvalue) -> void { _isLvalue = isLvalue; }

private:
  std::unique_ptr<Expr> _base;
  std::vector<std::unique_ptr<Expr>> _indices;
  bool _isLvalue = false;
};

} // end namespace cherry

#endif // CHERRY_COLLECTION_EXPR_H
