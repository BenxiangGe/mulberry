//===--- TensorExpr.h - Cherry Language Tensor Expression ASTs --*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TENSOR_EXPR_H
#define CHERRY_TENSOR_EXPR_H

#include "cherry/AST/Expr.h"
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace cherry {

// Tensor literal. e.g. `[[1.0, 2.0], [3.0, 4.0]]`
class TensorLiteralExpr final : public Expr {
public:
  TensorLiteralExpr(llvm::SMLoc loc,
                    std::vector<std::unique_ptr<Expr>> elements)
      : Expr(Expr_TensorLiteral, loc), _elements(std::move(elements)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_TensorLiteral;
  }

  auto getElements() const
      -> const std::vector<std::unique_ptr<Expr>> & {
    return _elements;
  }

  auto takeElements() -> std::vector<std::unique_ptr<Expr>> {
    return std::move(_elements);
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

class IndexExpr final : public Expr {
public:
  IndexExpr(llvm::SMLoc loc, std::string_view varName,
            std::vector<std::unique_ptr<Expr>> indices)
      : Expr(Expr_Index, loc), _varName(varName),
        _indices(std::move(indices)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Index;
  }

  auto getVarName() const -> std::string_view { return _varName; }

  auto getIndices() const -> const std::vector<std::unique_ptr<Expr>> & {
    return _indices;
  }

  auto isLvalue() const -> bool override { return true; }

private:
  std::string _varName;
  std::vector<std::unique_ptr<Expr>> _indices;
};

} // end namespace cherry

#endif // CHERRY_TENSOR_EXPR_H
