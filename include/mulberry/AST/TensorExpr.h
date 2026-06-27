//===--- TensorExpr.h - Mulberry tensor expressions ---------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_TENSOR_EXPR_H
#define MULBERRY_TENSOR_EXPR_H

#include "mulberry/AST/Expr.h"
#include <memory>

namespace mulberry {

class ZeroInitExpr final : public Expr {
public:
  explicit ZeroInitExpr(llvm::SMLoc location)
      : Expr{Expr_ZeroInit, location} {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_ZeroInit;
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

} // end namespace mulberry

#endif // MULBERRY_TENSOR_EXPR_H
