//===--- LiteralExpr.h - Mulberry Language Literal Expression ASTs -*- C++ -*-===//
//
// This source file is part of the Mulberry open source project
// See LICENSE.txt for license information
//
//===----------------------------------------------------------------------===//

#ifndef MULBERRY_LITERAL_EXPR_H
#define MULBERRY_LITERAL_EXPR_H

#include "mulberry/AST/Expr.h"
#include "llvm/ADT/APFloat.h"
#include <cstdint>
#include <string>
#include <string_view>

namespace mulberry {

class UnitExpr final : public Expr {
public:
  explicit UnitExpr(llvm::SMLoc location) : Expr{Expr_Unit, location} {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_Unit;
  }
};

class DecimalLiteralExpr final : public Expr {
public:
  explicit DecimalLiteralExpr(llvm::SMLoc location, uint64_t value)
      : Expr{Expr_DecimalLiteral, location}, _value(value){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_DecimalLiteral;
  }

  auto value() const -> uint64_t { return _value; }

private:
  uint64_t _value;
};

class FloatLiteralExpr final : public Expr {
public:
  explicit FloatLiteralExpr(llvm::SMLoc location, llvm::APFloat value)
      : Expr{Expr_FloatLiteral, location}, _value(std::move(value)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_FloatLiteral;
  }

  auto value() const -> const llvm::APFloat & { return _value; }

private:
  llvm::APFloat _value;
};

class BoolLiteralExpr final : public Expr {
public:
  explicit BoolLiteralExpr(llvm::SMLoc location, bool value)
      : Expr{Expr_BoolLiteral, location}, _value(value){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_BoolLiteral;
  }

  auto value() const -> bool { return _value; }

private:
  bool _value;
};

class StringLiteralExpr final : public Expr {
public:
  explicit StringLiteralExpr(llvm::SMLoc location, std::string value)
      : Expr{Expr_StringLiteral, location}, _value(std::move(value)){};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_StringLiteral;
  }

  auto value() const -> std::string_view { return _value; }

private:
  std::string _value;
};

class CharLiteralExpr final : public Expr {
public:
  explicit CharLiteralExpr(llvm::SMLoc location, uint8_t value)
      : Expr{Expr_CharLiteral, location}, _value(value) {};

  static auto classof(const Expr *node) -> bool {
    return node->getKind() == Expr_CharLiteral;
  }

  auto value() const -> uint8_t { return _value; }

private:
  uint8_t _value;
};

} // end namespace mulberry

#endif // MULBERRY_LITERAL_EXPR_H
