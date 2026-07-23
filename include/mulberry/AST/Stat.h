//===--- Stat.h - Mulberry Language Expression ASTs ---------------*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_STAT_H
#define MULBERRY_STAT_H

#include "mulberry/AST/Node.h"
#include "mulberry/AST/Type.h"
#include "mulberry/Basic/Types.h"
#include <optional>
#include <string>
#include <string_view>

namespace mulberry {

class BlockExpr;
class Expr;
class Type;
class TypeNode;
class VariableExpr;

// _____________________________________________________________________________
// Expression

class Stat : public Node {
public:
  enum StatementKind {
    Stat_VariableDecl,
    Stat_Expression,
    Stat_If,
    Stat_Match,
    Stat_While,
    Stat_For,
    Stat_Break,
    Stat_Continue,
    Stat_Return,
  };

  explicit Stat(StatementKind kind, llvm::SMLoc location)
      : Node{location}, _kind{kind} {};

  auto getKind() const -> StatementKind { return _kind; }

private:
  const StatementKind _kind;
};

// _____________________________________________________________________________
// Variable statement

class VariableStat final : public Stat {
public:
  explicit VariableStat(llvm::SMLoc location,
                        std::unique_ptr<VariableExpr> variable,
                        std::unique_ptr<TypeNode> typeNode,
                        std::unique_ptr<Expr> init,
                        bool isConstBinding = false,
                        bool canMutateObject = true)
      : Stat{Stat_VariableDecl, location}, _variable(std::move(variable)),
        _typeNode(std::move(typeNode)), _init{std::move(init)},
        _isConstBinding(isConstBinding),
        _canMutateObject(canMutateObject) {};

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_VariableDecl;
  }

  auto variable() const -> const std::unique_ptr<VariableExpr> & {
    return _variable;
  }

  auto typeNode() const -> const TypeNode * { return _typeNode.get(); }

  auto hasExplicitType() const -> bool { return _typeNode != nullptr; }

  auto setType(const Type *type) -> void { _type = type; }

  auto type() const -> const Type * { return _type; }

  auto init() const -> const std::unique_ptr<Expr> & { return _init; }

  auto init() -> std::unique_ptr<Expr> & { return _init; }

  auto isConstBinding() const -> bool { return _isConstBinding; }

  auto canMutateObject() const -> bool { return _canMutateObject; }

  auto comptimeValue() const -> const std::optional<ComptimeValue> & {
    return _comptimeValue;
  }

  auto setComptimeValue(ComptimeValue value) -> void {
    _comptimeValue = std::move(value);
  }

private:
  std::unique_ptr<VariableExpr> _variable;
  std::unique_ptr<TypeNode> _typeNode;
  const Type *_type = nullptr;
  std::unique_ptr<Expr> _init;
  bool _isConstBinding;
  bool _canMutateObject;
  std::optional<ComptimeValue> _comptimeValue;
};

// _____________________________________________________________________________
// Expression statement

class ExprStat final : public Stat {
public:
  explicit ExprStat(llvm::SMLoc location, std::unique_ptr<Expr> expression)
      : Stat{Stat_Expression, location}, _expression{std::move(expression)} {};

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_Expression;
  }

  auto expression() const -> const std::unique_ptr<Expr> & {
    return _expression;
  }

  auto expression() -> std::unique_ptr<Expr> & { return _expression; }

private:
  std::unique_ptr<Expr> _expression;
};

// _____________________________________________________________________________
// If statement

class IfStat final : public Stat {
public:
  explicit IfStat(llvm::SMLoc location, std::unique_ptr<Expr> condition,
                  std::unique_ptr<BlockExpr> thenBlock,
                  std::unique_ptr<BlockExpr> elseBlock)
      : Stat{Stat_If, location}, _condition(std::move(condition)),
        _thenBlock(std::move(thenBlock)), _elseBlock(std::move(elseBlock)) {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_If;
  }

  auto conditionExpr() const -> const std::unique_ptr<Expr> & {
    return _condition;
  }

  auto thenBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _thenBlock;
  }

  auto elseBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _elseBlock;
  }

  auto hasElseBlock() const -> bool { return static_cast<bool>(_elseBlock); }

  auto comptimeValue() const -> const std::optional<bool> & {
    return _comptimeValue;
  }

  auto setComptimeValue(bool value) -> void { _comptimeValue = value; }

private:
  std::unique_ptr<Expr> _condition;
  std::unique_ptr<BlockExpr> _thenBlock;
  std::unique_ptr<BlockExpr> _elseBlock;
  std::optional<bool> _comptimeValue;
};

// _____________________________________________________________________________
// Match statement

class DataPattern final : public Node {
public:
  DataPattern(llvm::SMLoc location, std::string_view constructorName,
              VectorUniquePtr<VariableExpr> bindings)
      : Node(location), _constructorName(constructorName),
        _bindings(std::move(bindings)) {}

  auto constructorName() const -> std::string_view {
    return _constructorName;
  }

  auto setConstructorName(std::string_view constructorName) -> void {
    _constructorName = constructorName;
  }

  auto bindings() const -> const VectorUniquePtr<VariableExpr> & {
    return _bindings;
  }

  auto constructorIndex() const -> unsigned { return _constructorIndex; }

  auto setConstructorIndex(unsigned constructorIndex) -> void {
    _constructorIndex = constructorIndex;
  }

private:
  std::string _constructorName;
  VectorUniquePtr<VariableExpr> _bindings;
  unsigned _constructorIndex = 0;
};

class MatchArm final : public Node {
public:
  MatchArm(llvm::SMLoc location, std::unique_ptr<DataPattern> pattern,
           std::unique_ptr<BlockExpr> bodyBlock)
      : Node(location), _pattern(std::move(pattern)),
        _bodyBlock(std::move(bodyBlock)) {}

  auto pattern() const -> const std::unique_ptr<DataPattern> & {
    return _pattern;
  }

  auto bodyBlock() const -> const std::unique_ptr<BlockExpr> & {
    return _bodyBlock;
  }

private:
  std::unique_ptr<DataPattern> _pattern;
  std::unique_ptr<BlockExpr> _bodyBlock;
};

class MatchStat final : public Stat {
public:
  MatchStat(llvm::SMLoc location, std::unique_ptr<Expr> value,
            VectorUniquePtr<MatchArm> arms)
      : Stat(Stat_Match, location), _value(std::move(value)),
        _arms(std::move(arms)) {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_Match;
  }

  auto value() const -> const std::unique_ptr<Expr> & { return _value; }

  auto arms() const -> const VectorUniquePtr<MatchArm> & { return _arms; }

private:
  std::unique_ptr<Expr> _value;
  VectorUniquePtr<MatchArm> _arms;
};

// _____________________________________________________________________________
// While statement

class WhileStat final : public Stat {
public:
  explicit WhileStat(llvm::SMLoc location, std::unique_ptr<Expr> condition,
                     std::unique_ptr<BlockExpr> bodyBlock)
      : Stat{Stat_While, location}, _condition(std::move(condition)),
        _bodyBlock(std::move(bodyBlock)) {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_While;
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
// For statement

class ForStat final : public Stat {
public:
  explicit ForStat(llvm::SMLoc location, std::string_view variableName,
                   std::unique_ptr<Expr> startExpr,
                   std::unique_ptr<Expr> endExpr,
                   std::unique_ptr<BlockExpr> bodyBlock)
      : Stat{Stat_For, location}, _variableName(variableName),
        _startExpr(std::move(startExpr)), _endExpr(std::move(endExpr)),
        _bodyBlock(std::move(bodyBlock)) {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_For;
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

// _____________________________________________________________________________
// Loop control statements

class BreakStat final : public Stat {
public:
  explicit BreakStat(llvm::SMLoc location) : Stat{Stat_Break, location} {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_Break;
  }
};

class ContinueStat final : public Stat {
public:
  explicit ContinueStat(llvm::SMLoc location)
      : Stat{Stat_Continue, location} {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_Continue;
  }
};

// _____________________________________________________________________________
// Return statement

class ReturnStat final : public Stat {
public:
  explicit ReturnStat(llvm::SMLoc location, std::unique_ptr<Expr> expression)
      : Stat{Stat_Return, location}, _expression(std::move(expression)) {}

  static auto classof(const Stat *node) -> bool {
    return node->getKind() == Stat_Return;
  }

  auto expression() const -> const std::unique_ptr<Expr> & {
    return _expression;
  }

  auto expression() -> std::unique_ptr<Expr> & { return _expression; }

  auto hasExpression() const -> bool { return static_cast<bool>(_expression); }

private:
  std::unique_ptr<Expr> _expression;
};

} // end namespace mulberry

#endif // MULBERRY_STAT_H
