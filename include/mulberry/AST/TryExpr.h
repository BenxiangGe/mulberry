//===--- TryExpr.h - Mulberry Try Expression ASTs --------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_TRY_EXPR_H
#define MULBERRY_TRY_EXPR_H

#include "mulberry/AST/Expr.h"
#include <string>
#include <string_view>

namespace mulberry {

class TryExpr final : public Expr {
public:
  TryExpr(llvm::SMLoc location, std::unique_ptr<Expr> value)
      : Expr(Expr_Try, location), _value(std::move(value)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Try;
  }

  auto value() const -> const std::unique_ptr<Expr> & { return _value; }

  auto canMutateObject() const -> bool { return _canMutateObject; }

  auto setCanMutateObject(bool canMutateObject) -> void {
    _canMutateObject = canMutateObject;
  }

  auto setErrorTypes(const Type *sourceErrorType,
                     const Type *targetErrorType) -> void {
    _sourceErrorType = sourceErrorType;
    _targetErrorType = targetErrorType;
  }

  auto sourceErrorType() const -> const Type * { return _sourceErrorType; }

  auto targetErrorType() const -> const Type * { return _targetErrorType; }

  auto setErrorConverter(std::string functionName) -> void {
    _errorConverter = std::move(functionName);
  }

  auto errorConverter() const -> std::string_view { return _errorConverter; }

private:
  std::unique_ptr<Expr> _value;
  bool _canMutateObject = true;
  const Type *_sourceErrorType = nullptr;
  const Type *_targetErrorType = nullptr;
  std::string _errorConverter;
};

} // end namespace mulberry

#endif // MULBERRY_TRY_EXPR_H
