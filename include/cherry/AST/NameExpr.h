//===--- NameExpr.h - Cherry Language Name Expression ASTs ------*- C++ -*-===//
//
// This source file is part of the Cherry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef CHERRY_NAME_EXPR_H
#define CHERRY_NAME_EXPR_H

#include "cherry/AST/Expr.h"
#include <string>
#include <string_view>

namespace cherry {

class VariableExpr final : public Expr {
public:
  explicit VariableExpr(llvm::SMLoc location, std::string_view name)
      : Expr{Expr_Variable, location}, _name(name){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Variable;
  }

  auto name() const -> std::string_view { return _name; }

  auto isLvalue() const -> bool override { return true; }

private:
  std::string _name;
};

class MemberExpr final : public Expr {
public:
  MemberExpr(llvm::SMLoc location, std::unique_ptr<Expr> base,
             std::string_view fieldName)
      : Expr{Expr_Member, location}, _base(std::move(base)),
        _fieldName(fieldName) {}

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Member;
  }

  auto base() const -> const std::unique_ptr<Expr> & { return _base; }

  auto fieldName() const -> std::string_view { return _fieldName; }

  auto isLvalue() const -> bool override { return _isLValue; }

  auto setLvalue(bool isLvalue) -> void { _isLValue = isLvalue; }

  auto fieldIndex() const -> unsigned { return _fieldIndex; }

  auto setFieldIndex(unsigned fieldIndex) -> void {
    _fieldIndex = fieldIndex;
  }

private:
  std::unique_ptr<Expr> _base;
  std::string _fieldName;
  unsigned _fieldIndex = 0;
  bool _isLValue = false;
};

class CallExpr final : public Expr {
public:
  explicit CallExpr(llvm::SMLoc location, std::string_view name,
                    VectorUniquePtr<Expr> expressions)
      : Expr{Expr_Call, location}, _name(name),
        _expressions(std::move(expressions)){};

  CallExpr(llvm::SMLoc location, std::unique_ptr<Expr> receiver,
           std::string_view name, VectorUniquePtr<Expr> expressions)
      : Expr{Expr_Call, location}, _name(name),
        _receiver(std::move(receiver)),
        _expressions(std::move(expressions)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Call;
  }

  auto name() const -> std::string_view { return _name; }

  auto setName(std::string_view name) -> void { _name = name; }

  auto expressions() const -> const VectorUniquePtr<Expr> & {
    return _expressions;
  }

  auto hasReceiver() const -> bool { return _receiver != nullptr; }

  auto receiver() const -> const std::unique_ptr<Expr> & {
    return _receiver;
  }

  auto setReceiver(std::unique_ptr<Expr> receiver,
                   std::string_view methodName) -> void {
    _receiver = std::move(receiver);
    _name = methodName;
  }

  auto lowerMethodCall(std::string_view name) -> void {
    _name = name;
    _isLoweredMethodCall = true;
    _expressions.insert(_expressions.begin(), std::move(_receiver));
  }

  auto isLoweredMethodCall() const -> bool { return _isLoweredMethodCall; }

private:
  std::string _name;
  std::unique_ptr<Expr> _receiver;
  VectorUniquePtr<Expr> _expressions;
  bool _isLoweredMethodCall = false;

public:
  auto begin() const -> decltype(_expressions.begin()) {
    return _expressions.begin();
  }
  auto end() const -> decltype(_expressions.end()) {
    return _expressions.end();
  }
};

} // end namespace cherry

#endif // CHERRY_NAME_EXPR_H
