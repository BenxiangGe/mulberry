//===--- TensorExpr.h - Cherry tensor expressions ---------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_TENSOR_EXPR_H
#define CHERRY_TENSOR_EXPR_H

#include "cherry/AST/Expr.h"
#include <memory>

namespace cherry {

class TensorZerosExpr final : public Expr {
public:
  explicit TensorZerosExpr(llvm::SMLoc location)
      : Expr{Expr_TensorZeros, location} {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_TensorZeros;
  }
};

class TensorPackExpr final : public Expr {
public:
  TensorPackExpr(llvm::SMLoc location, std::unique_ptr<Expr> tensor)
      : Expr{Expr_TensorPack, location}, _tensor(std::move(tensor)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_TensorPack;
  }

  auto tensor() const -> const std::unique_ptr<Expr> & { return _tensor; }

private:
  std::unique_ptr<Expr> _tensor;
};

class TensorViewExpr final : public Expr {
public:
  TensorViewExpr(llvm::SMLoc location, std::unique_ptr<Expr> tensorRecord)
      : Expr{Expr_TensorView, location},
        _tensorRecord(std::move(tensorRecord)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_TensorView;
  }

  auto tensorRecord() const -> const std::unique_ptr<Expr> & {
    return _tensorRecord;
  }

private:
  std::unique_ptr<Expr> _tensorRecord;
};

} // end namespace cherry

#endif // CHERRY_TENSOR_EXPR_H
