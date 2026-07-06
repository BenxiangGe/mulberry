//===--- Expr.h - Mulberry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_EXPR_H
#define MULBERRY_EXPR_H

#include "mulberry/AST/Node.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace mulberry {
class Stat;
class Type;
// _____________________________________________________________________________
// Expression

class Expr : public Node {
public:
  enum ExpressionKind {
    Expr_Unit,
    Expr_Call,
    Expr_StructLiteral,
    Expr_DecimalLiteral,
    Expr_FloatLiteral,
    Expr_BoolLiteral,
    Expr_StringLiteral,
    Expr_CharLiteral,
    Expr_ArrayLiteral,
    Expr_Index,
    Expr_Member,
    Expr_Variable,
    Expr_Assign,
    Expr_Binary,
    Expr_Block,
    Expr_TypeLayout,
    Expr_HeapAlloc,
  };

  explicit Expr(ExpressionKind kind, llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> ExpressionKind { return _kind; }

  virtual auto isLvalue() const -> bool { return false; }
  virtual auto isStatement() -> bool { return false; };

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

private:
  const ExpressionKind _kind;
  const Type *_type = nullptr;
};

// _____________________________________________________________________________
// Block expression

class BlockExpr final : public Expr {
public:
  explicit BlockExpr(llvm::SMLoc location, VectorUniquePtr<Stat> statements)
      : Expr{Expr_Block, location}, _statements(std::move(statements)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Block;
  }

  auto statements() const -> const VectorUniquePtr<Stat> & {
    return _statements;
  }

private:
  VectorUniquePtr<Stat> _statements;

public:
  auto begin() const -> decltype(_statements.begin()) {
    return _statements.begin();
  }
  auto end() const -> decltype(_statements.end()) { return _statements.end(); }
};

} // end namespace mulberry

#endif // MULBERRY_EXPR_H
