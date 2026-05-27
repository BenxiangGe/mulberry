//===--- Expr.h - Cherry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_EXPR_H
#define CHERRY_EXPR_H

#include "cherry/AST/Node.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>

namespace cherry {
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
    Expr_ListLiteral,
    Expr_TensorLiteral,
    Expr_Index,
    Expr_Member,
    Expr_Variable,
    Expr_Assign,
    Expr_Binary,
    Expr_Block,
    Expr_If,
    Expr_While,
    Expr_For,
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
  explicit BlockExpr(llvm::SMLoc location, VectorUniquePtr<Stat> statements,
                     std::unique_ptr<Expr> expression)
      : Expr{Expr_Block, location}, _statements(std::move(statements)),
        _expression(std::move(expression)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Block;
  }

  auto statements() const -> const VectorUniquePtr<Stat> & {
    return _statements;
  }

  auto expression() const -> const std::unique_ptr<Expr> & {
    return _expression;
  }

private:
  VectorUniquePtr<Stat> _statements;
  std::unique_ptr<Expr> _expression;

public:
  auto begin() const -> decltype(_statements.begin()) {
    return _statements.begin();
  }
  auto end() const -> decltype(_statements.end()) { return _statements.end(); }
};

// _____________________________________________________________________________
// If expression

class IfExpr final : public Expr {
public:
  explicit IfExpr(llvm::SMLoc location, std::unique_ptr<Expr> condition,
                  std::unique_ptr<BlockExpr> thenExpr,
                  std::unique_ptr<BlockExpr> elseExpr)
      : Expr{Expr_If, location}, _condition(std::move(condition)),
        _thenExpr(std::move(thenExpr)), _elseExpr(std::move(elseExpr)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_If;
  }

  auto conditionExpr() const -> const std::unique_ptr<Expr> & {
    return _condition;
  }

  auto thenBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _thenExpr;
  }

  auto elseBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _elseExpr;
  }

private:
  std::unique_ptr<Expr> _condition;
  std::unique_ptr<BlockExpr> _thenExpr;
  std::unique_ptr<BlockExpr> _elseExpr;
};

// _____________________________________________________________________________
// While expression

class WhileExpr final : public Expr {
public:
  explicit WhileExpr(llvm::SMLoc location, std::unique_ptr<Expr> condition,
                     std::unique_ptr<BlockExpr> bodyBlock)
      : Expr{Expr_While, location}, _condition(std::move(condition)),
        _bodyBlock(std::move(bodyBlock)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_While;
  }

  auto conditionExpr() const -> const std::unique_ptr<Expr> & {
    return _condition;
  }

  auto bodyBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _bodyBlock;
  }

private:
  std::unique_ptr<Expr> _condition;
  std::unique_ptr<BlockExpr> _bodyBlock;
};

// _____________________________________________________________________________
// For expression

class ForExpr final : public Expr {
public:
  explicit ForExpr(llvm::SMLoc location, std::string_view variableName,
                   std::unique_ptr<Expr> startExpr,
                   std::unique_ptr<Expr> endExpr,
                   std::unique_ptr<BlockExpr> bodyBlock)
      : Expr{Expr_For, location}, _variableName(variableName),
        _startExpr(std::move(startExpr)), _endExpr(std::move(endExpr)),
        _bodyBlock(std::move(bodyBlock)) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_For;
  }

  auto variableName() const -> std::string_view { return _variableName; }

  auto startExpr() const -> const std::unique_ptr<Expr> & {
    return _startExpr;
  }

  auto endExpr() const -> const std::unique_ptr<Expr> & {
    return _endExpr;
  }

  auto bodyBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _bodyBlock;
  }

private:
  std::string _variableName;
  std::unique_ptr<Expr> _startExpr;
  std::unique_ptr<Expr> _endExpr;
  std::unique_ptr<BlockExpr> _bodyBlock;
};

} // end namespace cherry

#endif // CHERRY_EXPR_H
