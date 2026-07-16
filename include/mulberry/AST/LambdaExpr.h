//===--- LambdaExpr.h - Mulberry Lambda Expression AST ----------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_LAMBDAEXPR_H
#define MULBERRY_LAMBDAEXPR_H

#include "mulberry/AST/Expr.h"
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mulberry {

class LambdaExpr final : public Expr {
public:
  struct Parameter {
    Parameter(llvm::SMLoc location, std::string name)
        : location(location), name(std::move(name)) {}

    llvm::SMLoc location;
    std::string name;
    const Type *type = nullptr;
  };

  LambdaExpr(llvm::SMLoc location, std::vector<Parameter> parameters,
             std::unique_ptr<Expr> body)
      : Expr{Expr_Lambda, location}, _parameters(std::move(parameters)),
        _body(std::move(body)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Lambda;
  }

  auto parameters() -> std::vector<Parameter> & { return _parameters; }

  auto parameters() const -> const std::vector<Parameter> & {
    return _parameters;
  }

  auto body() const -> const std::unique_ptr<Expr> & { return _body; }

  auto takeBody() -> std::unique_ptr<Expr> { return std::move(_body); }

  auto functionName() const -> std::string_view { return _functionName; }

  auto setFunctionName(std::string_view name) -> void { _functionName = name; }

private:
  std::vector<Parameter> _parameters;
  std::unique_ptr<Expr> _body;
  std::string _functionName;
};

} // namespace mulberry

#endif // MULBERRY_LAMBDAEXPR_H
