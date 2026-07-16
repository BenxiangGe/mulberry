//===--- DataConstructorExpr.h - Mulberry Data Constructor Expr -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_DATA_CONSTRUCTOR_EXPR_H
#define MULBERRY_DATA_CONSTRUCTOR_EXPR_H

#include "mulberry/AST/Expr.h"
#include <string>
#include <string_view>

namespace mulberry {

class DataConstructorExpr final : public Expr {
public:
  DataConstructorExpr(llvm::SMLoc location, std::string_view name,
                      VectorUniquePtr<Expr> expressions)
      : Expr(Expr_DataConstructor, location), _name(name),
        _expressions(std::move(expressions)) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_DataConstructor;
  }

  auto name() const -> std::string_view { return _name; }

  auto setName(std::string_view name) -> void { _name = name; }

  auto expressions() const -> const VectorUniquePtr<Expr> & {
    return _expressions;
  }

  auto setConstructorIndex(unsigned constructorIndex) -> void {
    _constructorIndex = constructorIndex;
  }

  auto constructorIndex() const -> unsigned { return _constructorIndex; }

private:
  std::string _name;
  VectorUniquePtr<Expr> _expressions;
  unsigned _constructorIndex = 0;
};

} // end namespace mulberry

#endif // MULBERRY_DATA_CONSTRUCTOR_EXPR_H
